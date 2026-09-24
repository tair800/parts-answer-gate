"""The two tables, the vector column, and the indexes the effectivity predicate needs.

The tables mirror `domain.Document` and `domain.Chunk`. Where they differ from the domain models,
the difference is one of these three, and nothing else:

- **`Effectivity` is flattened** into `variant_id`, `serial_first` and `serial_last`. A composite
  type or a JSON column cannot be indexed usefully by a btree, and the entire sole-home skill is
  that the variant and serial predicate constrains the candidate set *in SQL* before a score exists.
- **`identifiers`** holds the part numbers `domain.part_numbers_in` finds in the chunk text, so the
  deterministic lookup is an indexed containment test rather than a sequential `LIKE` over every
  row. It is derived, never authored.
- **`content_hash`, `content_changed_at`, `embedded_at`, `embedding_model`** are the index
  lifecycle. Without them "re-embed only what changed" has nothing to compare against and
  "staleness" has no clock.

`chunk.embedding` is a real `vector(384)`, not an array of floats, and the HNSW index over it is
declared here rather than left to a hand-run DDL statement. ADR-001 kill condition K is the claim
that vector search executes inside PostgreSQL; an index somebody has to remember to create is an
index that is missing in CI.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Final

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Date,
    DateTime,
    Engine,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text as sql_text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from parts_answer_gate.domain import (
    Chunk,
    Document,
    Effectivity,
    Language,
    SerialRange,
    part_numbers_in,
)
from parts_answer_gate.retrieval.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

__all__ = [
    "CHUNK_READ_COLUMNS",
    "EFFECTIVITY_INDEXES",
    "HNSW_BUILD_PARAMETERS",
    "STORED_EMBEDDING_MODEL",
    "VECTOR_COLUMN",
    "Base",
    "ChunkRow",
    "DocumentRow",
    "chunk_from_mapping",
    "chunk_row_values",
    "create_schema",
    "document_row_values",
    "to_domain_chunk",
    "to_domain_document",
]

#: One committed configuration, as ADR-001 records under "HNSW index tuning sweep": not a swept
#: comparison, but the parameters are written down so the published recall figure means something.
#: m=16 and ef_construction=200 are pgvector's documented recall-favouring build, roughly doubling
#: build time against the ef_construction=64 default for a corpus this size — trivial at 5k rows and
#: the right trade when the number being published is recall@10.
#: `vector_cosine_ops` because the encoder emits unit vectors and `EMBEDDING_POLICY` says cosine.
HNSW_BUILD_PARAMETERS: Final[dict[str, Any]] = {
    "index_name": "ix_chunk_embedding_hnsw",
    "method": "hnsw",
    "opclass": "vector_cosine_ops",
    "m": 16,
    "ef_construction": 200,
}

#: The column kill condition K is about, named once so the diagnostic asks the catalogue about the
#: same column the retriever queries.
VECTOR_COLUMN: Final = ("chunk", "embedding")

#: The indexes that exist to serve the effectivity predicate, named so the diagnostic can assert
#: they are present rather than assume it.
EFFECTIVITY_INDEXES: Final = (
    "ix_chunk_effectivity",
    "ix_chunk_in_force",
    "ix_chunk_identifiers",
)


class Base(DeclarativeBase):
    """Declarative base. Alembic's autogenerate compares against this metadata."""


class DocumentRow(Base):
    """One revision of one document, mirroring `domain.Document`."""

    __tablename__ = "document"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    family_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(2), nullable=False)
    revision: Mapped[str] = mapped_column(String(32), nullable=False)
    valid_from: Mapped[date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    # Deliberately NOT a foreign key onto document.document_id. A supersession edge points forward
    # in time and the successor is frequently inserted after the revision it replaces; a
    # self-referential FK would force the loader to topologically sort a graph whose only consumer
    # is a query that tolerates a dangling id. The generator owns edge integrity and the corpus
    # artifact counts the edges.
    superseded_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        Index("ix_document_family", "family_id"),
        Index("ix_document_in_force", "valid_from", "valid_to"),
    )


class ChunkRow(Base):
    """A retrievable passage, mirroring `domain.Chunk`, plus effectivity and lifecycle columns."""

    __tablename__ = "chunk"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("document.document_id", ondelete="CASCADE"), nullable=False
    )
    family_id: Mapped[str] = mapped_column(String(64), nullable=False)
    language: Mapped[str] = mapped_column(String(2), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[str] = mapped_column(String(64), nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)

    # --- effectivity, flattened so a btree can read it -------------------------------------------
    variant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    #: NULL means open at that end, exactly as `domain.SerialRange` means it. NULL rather than a
    #: sentinel like 0 or 2**31 because "applies from serial 4200 onwards" has no upper bound, and a
    #: synthetic one would eventually stop applying to machines it applies to.
    serial_first: Mapped[int | None] = mapped_column(Integer, nullable=True)
    serial_last: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- validity, denormalised from the document so the filter needs no join --------------------
    valid_from: Mapped[date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    superseded_by: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # --- derived ---------------------------------------------------------------------------------
    #: Part numbers found in `text`, for the deterministic lookup that runs before any embedding.
    identifiers: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=sql_text("'{}'::text[]")
    )

    # --- index lifecycle --------------------------------------------------------------------------
    #: Hash of exactly what was embedded. Compared on load to decide whether this chunk needs a new
    #: vector; see `store.loader` for why metadata is excluded from it.
    content_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    content_changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # The effectivity predicate leads with variant equality because that is the selective term:
        # a query naming a variant discards roughly (variants - 1)/variants of the corpus before the
        # date range is consulted at all.
        Index("ix_chunk_effectivity", "variant_id", "valid_from", "valid_to"),
        # The as-of half on its own, for questions that name no variant.
        Index("ix_chunk_in_force", "valid_from", "valid_to"),
        # GIN, because the deterministic lookup asks "does this row's identifier array overlap the
        # query's", and that is an array containment test a btree cannot answer.
        Index("ix_chunk_identifiers", "identifiers", postgresql_using="gin"),
        Index("ix_chunk_document", "document_id"),
        Index("ix_chunk_family_language", "family_id", "language"),
        Index(
            HNSW_BUILD_PARAMETERS["index_name"],
            "embedding",
            postgresql_using=HNSW_BUILD_PARAMETERS["method"],
            postgresql_with={
                "m": HNSW_BUILD_PARAMETERS["m"],
                "ef_construction": HNSW_BUILD_PARAMETERS["ef_construction"],
            },
            postgresql_ops={"embedding": HNSW_BUILD_PARAMETERS["opclass"]},
        ),
    )


# ------------------------------------------------------------------ domain <-> row, one way each


def document_row_values(document: Document) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "family_id": document.family_id,
        "title": document.title,
        "language": document.language.value,
        "revision": document.revision,
        "valid_from": document.valid_from,
        "valid_to": document.valid_to,
        "superseded_by": document.superseded_by,
        "source_uri": document.source_uri,
        "checksum": document.checksum,
    }


def chunk_row_values(chunk: Chunk) -> dict[str, Any]:
    """Row values for a chunk, minus everything the loader computes (hash, vector, timestamps)."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "family_id": chunk.family_id,
        "language": chunk.language.value,
        "text": chunk.text,
        "section": chunk.section,
        "page": chunk.page,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "variant_id": chunk.effectivity.variant_id,
        "serial_first": chunk.effectivity.serials.first,
        "serial_last": chunk.effectivity.serials.last,
        "valid_from": chunk.valid_from,
        "valid_to": chunk.valid_to,
        "superseded_by": chunk.superseded_by,
        "identifiers": sorted(part_numbers_in(chunk.text)),
    }


def to_domain_document(row: DocumentRow) -> Document:
    return Document(
        document_id=row.document_id,
        family_id=row.family_id,
        title=row.title,
        language=Language(row.language),
        revision=row.revision,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        superseded_by=row.superseded_by,
        source_uri=row.source_uri,
        checksum=row.checksum,
    )


#: The columns a retrieval query needs. `embedding` is deliberately absent: a candidate fetch that
#: dragged 384 floats per row across the wire would spend more time serialising vectors than the
#: index spends searching them, and nothing downstream of the database reads a stored vector.
CHUNK_READ_COLUMNS: Final = (
    "chunk_id",
    "document_id",
    "family_id",
    "language",
    "text",
    "section",
    "page",
    "start_offset",
    "end_offset",
    "variant_id",
    "serial_first",
    "serial_last",
    "valid_from",
    "valid_to",
    "superseded_by",
)


def chunk_from_mapping(row: Mapping[str, Any]) -> Chunk:
    """Build a domain chunk from a result row.

    Takes a mapping rather than an ORM instance because the retrieval path runs raw SQL — the
    effectivity predicate is a textual `WHERE` fragment shared with the pgvector query, and routing
    it through the ORM would mean two spellings of the one rule this project is about.
    """
    return Chunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        family_id=row["family_id"],
        language=Language(row["language"]),
        text=row["text"],
        section=row["section"],
        page=row["page"],
        start_offset=row["start_offset"],
        end_offset=row["end_offset"],
        effectivity=Effectivity(
            variant_id=row["variant_id"],
            serials=SerialRange(first=row["serial_first"], last=row["serial_last"]),
        ),
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        superseded_by=row["superseded_by"],
    )


def to_domain_chunk(row: ChunkRow) -> Chunk:
    return chunk_from_mapping({name: getattr(row, name) for name in CHUNK_READ_COLUMNS})


def create_schema(engine: Engine) -> None:
    """Create the extension and the tables directly, for tests that want a throwaway database.

    Alembic owns the schema that ships. This exists so a test fixture does not have to run a
    migration chain to get a table, and it creates the same objects from the same metadata — if the
    two ever disagree, `alembic check` says so.
    """
    with engine.begin() as connection:
        connection.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


#: Recorded next to the schema so a lifecycle report can say which model the stored vectors came
#: from without importing the encoder.
STORED_EMBEDDING_MODEL: Final = EMBEDDING_MODEL_NAME
