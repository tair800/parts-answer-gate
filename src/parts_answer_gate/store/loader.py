"""Writing documents and chunks, and re-embedding only the chunks whose text actually changed.

The loader is idempotent by construction: everything is an upsert keyed on the primary key, so a
half-finished load is fixed by running it again rather than by truncating a table. That matters more
than it looks — the corpus is regenerated from a seed, and a loader that could only ever insert
would force a drop-and-rebuild, which in turn would re-embed 4,500 chunks to change forty of them.

**What `content_hash` covers, and why it is only the text.** A chunk's validity window, its
supersession edge and its variant are not inputs to the encoder. Hashing them would re-embed a chunk
every time a bulletin was withdrawn — the most common metadata edit in this domain — and pay the
full embedding cost for a vector that would come out bit-identical. The model id *is* in the hash,
because changing the encoder does invalidate every stored vector.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import blake2b
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from parts_answer_gate.domain import Chunk, Document
from parts_answer_gate.retrieval.embeddings import EMBEDDING_MODEL_NAME, Embedder
from parts_answer_gate.store.schema import (
    ChunkRow,
    DocumentRow,
    chunk_row_values,
    document_row_values,
)

__all__ = ["LoadReport", "chunk_count", "content_hash", "load_chunks", "load_documents"]

#: 16 bytes -> 32 hex characters, which is the declared width of `chunk.content_hash`. Truncated
#: blake2b rather than sha256 because this is a change detector, not a signature: a collision costs
#: one stale embedding, and nothing security-relevant depends on it.
_HASH_BYTES = 16


def content_hash(text: str, *, model: str = EMBEDDING_MODEL_NAME) -> str:
    digest = blake2b(digest_size=_HASH_BYTES)
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadReport:
    """What a load actually did. Every field is a count of rows, not an estimate."""

    chunks_seen: int
    chunks_new: int
    chunks_changed: int
    chunks_unchanged: int
    chunks_re_embedded: int
    embed_seconds: float
    write_seconds: float

    @property
    def chunks_written(self) -> int:
        return self.chunks_new + self.chunks_changed + self.chunks_unchanged


def load_documents(session: Session, documents: Sequence[Document]) -> int:
    if not documents:
        return 0
    values = [document_row_values(document) for document in documents]
    statement = insert(DocumentRow).values(values)
    updatable = {
        column: statement.excluded[column]
        for column in values[0]
        if column != "document_id"
    }
    session.execute(
        statement.on_conflict_do_update(index_elements=[DocumentRow.document_id], set_=updatable)
    )
    return len(values)


def load_chunks(
    session: Session,
    chunks: Sequence[Chunk],
    embedder: Embedder,
    *,
    batch_size: int = 256,
    changed_at: datetime | None = None,
) -> LoadReport:
    """Upsert chunks, embedding only those whose text is new or different.

    `changed_at` is injectable because the index-lifecycle measurement needs to time from the moment
    the new content existed, not from the moment this function happened to be called. A clock read
    inside a function cannot be replayed.
    """
    if not chunks:
        return LoadReport(0, 0, 0, 0, 0, 0.0, 0.0)

    content_changed_at = changed_at or datetime.now(UTC)
    rows = [chunk_row_values(chunk) for chunk in chunks]
    hashes = [content_hash(chunk.text) for chunk in chunks]

    existing = _existing_state(session, [row["chunk_id"] for row in rows])

    stale_indices = [
        index
        for index, (row, digest) in enumerate(zip(rows, hashes, strict=True))
        if _needs_embedding(existing.get(str(row["chunk_id"])), digest)
    ]
    new_indices = [
        index for index, row in enumerate(rows) if str(row["chunk_id"]) not in existing
    ]

    embed_started = time.perf_counter()
    vectors = _embed_in_batches(
        embedder, [str(rows[index]["text"]) for index in stale_indices], batch_size
    )
    embed_seconds = time.perf_counter() - embed_started
    # Read after the encoder returns, so `embedded_at - content_changed_at` is the real interval a
    # chunk spent with a vector that no longer described it.
    embedded_at = datetime.now(UTC)

    vector_by_index = dict(zip(stale_indices, vectors, strict=True))
    fresh_rows: list[dict[str, Any]] = []
    unchanged_rows: list[dict[str, Any]] = []
    for index, (row, digest) in enumerate(zip(rows, hashes, strict=True)):
        row["content_hash"] = digest
        row["content_changed_at"] = content_changed_at
        vector = vector_by_index.get(index)
        if vector is not None:
            row["embedding"] = vector
            row["embedding_model"] = EMBEDDING_MODEL_NAME
            row["embedded_at"] = embedded_at
            fresh_rows.append(row)
        else:
            unchanged_rows.append(row)

    write_started = time.perf_counter()
    _upsert_chunks(session, fresh_rows, include_embedding=True)
    _upsert_chunks(session, unchanged_rows, include_embedding=False)
    write_seconds = time.perf_counter() - write_started

    new_count = len(new_indices)
    return LoadReport(
        chunks_seen=len(chunks),
        chunks_new=new_count,
        chunks_changed=len(stale_indices) - new_count,
        chunks_unchanged=len(unchanged_rows),
        chunks_re_embedded=len(stale_indices),
        embed_seconds=embed_seconds,
        write_seconds=write_seconds,
    )


def chunk_count(session: Session) -> int:
    return int(session.execute(select(func.count()).select_from(ChunkRow)).scalar_one())


# --------------------------------------------------------------------------------- internals


@dataclass(frozen=True)
class _StoredState:
    content_hash: str
    has_embedding: bool
    embedding_model: str | None


def _existing_state(session: Session, chunk_ids: Sequence[str]) -> dict[str, _StoredState]:
    if not chunk_ids:
        return {}
    statement = select(
        ChunkRow.chunk_id,
        ChunkRow.content_hash,
        ChunkRow.embedding.is_not(None),
        ChunkRow.embedding_model,
    ).where(ChunkRow.chunk_id.in_(list(chunk_ids)))
    return {
        chunk_id: _StoredState(digest, bool(has_embedding), model)
        for chunk_id, digest, has_embedding, model in session.execute(statement).all()
    }


def _needs_embedding(stored: _StoredState | None, digest: str) -> bool:
    """A row needs a vector when it is new, its text changed, or its vector is missing or foreign.

    The `embedding_model` comparison is what makes a model upgrade safe: the stored vectors become
    stale by definition, and this detects that without anyone remembering to truncate.
    """
    if stored is None:
        return True
    if stored.content_hash != digest:
        return True
    if not stored.has_embedding:
        return True
    return stored.embedding_model != EMBEDDING_MODEL_NAME


def _embed_in_batches(
    embedder: Embedder, texts: Sequence[str], batch_size: int
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embedder.embed_documents(texts[start : start + batch_size]))
    return vectors


def _upsert_chunks(
    session: Session, rows: Sequence[dict[str, Any]], *, include_embedding: bool
) -> None:
    """Two statements rather than one, because the `SET` clause differs.

    A single upsert would have to write `embedding` for every row, which means either re-embedding
    everything or overwriting a good vector with NULL. Splitting by whether the content changed is
    what makes "re-embed only what changed" true at the SQL level and not just in the Python above.
    """
    if not rows:
        return
    skip = (
        set()
        if include_embedding
        else {"content_hash", "content_changed_at", "embedding", "embedding_model", "embedded_at"}
    )
    for start in range(0, len(rows), 500):
        batch = rows[start : start + 500]
        statement = insert(ChunkRow).values(batch)
        updatable = {
            column: statement.excluded[column]
            for column in batch[0]
            if column != "chunk_id" and column not in skip
        }
        session.execute(
            statement.on_conflict_do_update(index_elements=[ChunkRow.chunk_id], set_=updatable)
        )
