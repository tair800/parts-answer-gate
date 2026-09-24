"""The two reads that are not vector searches: the candidate fetch and the deterministic lookup.

Both take the same `CandidateFilter` the pgvector query takes, which is the point. If the lexical
stage saw a wider set than the dense stage, "the effectivity filter constrains the candidate set
before ranking" would be true of one stage and false of the other, and the fused result would
contain chunks that were never legally candidates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from sqlalchemy import text
from sqlalchemy.orm import Session

from parts_answer_gate.domain import Chunk, Query, part_numbers_in
from parts_answer_gate.store.effectivity import CandidateFilter
from parts_answer_gate.store.schema import CHUNK_READ_COLUMNS, chunk_from_mapping

__all__ = [
    "candidate_count",
    "exact_identifier_lookup",
    "fetch_candidates",
    "identifiers_in_query",
]

_COLUMNS: Final = ", ".join(f"chunk.{name}" for name in CHUNK_READ_COLUMNS)

# ORDER BY chunk_id is not cosmetic. The lexical index is built from this list in the order it
# arrives, and an unordered SELECT in PostgreSQL may return rows in a different physical order after
# an update rewrites a page. Kill condition J compares two runs byte for byte.
_CANDIDATE_SQL: Final = f"SELECT {_COLUMNS} FROM chunk WHERE ({{where}}) ORDER BY chunk.chunk_id"

_EXACT_SQL: Final = (
    f"SELECT {_COLUMNS} FROM chunk "
    "WHERE ({where}) AND chunk.identifiers && CAST(:identifiers AS text[]) "
    "ORDER BY chunk.chunk_id"
)


def identifiers_in_query(query: Query) -> tuple[str, ...]:
    """Part numbers the technician typed, in sorted order so the SQL parameter is reproducible."""
    return tuple(sorted(part_numbers_in(query.text)))


def fetch_candidates(session: Session, candidate_filter: CandidateFilter) -> list[Chunk]:
    """Every chunk the effectivity predicate admits. Scoring happens only over this list."""
    rows = session.execute(
        text(_CANDIDATE_SQL.format(where=candidate_filter.sql)),  # noqa: S608
        candidate_filter.params,
    ).mappings()
    return [chunk_from_mapping(row) for row in rows]


def candidate_count(session: Session, candidate_filter: CandidateFilter) -> int:
    statement = f"SELECT count(*) FROM chunk WHERE ({candidate_filter.sql})"  # noqa: S608
    return int(session.execute(text(statement), candidate_filter.params).scalar_one())


def exact_identifier_lookup(
    session: Session,
    query: Query,
    candidate_filter: CandidateFilter,
    *,
    identifiers: Sequence[str] | None = None,
) -> list[Chunk]:
    """Chunks containing a part number the question names. No embedding is consulted.

    This is the deterministic-first stage ADR-001 requires: sending `XP-4021-B` through a
    384-dimensional multilingual encoder to find the passage that literally contains `XP-4021-B`
    costs money and accuracy at the same time. The `&&` array-overlap test uses the GIN index on
    `chunk.identifiers`, so it is an index lookup rather than a scan over the text of every chunk.

    Serial numbers are not searched here because a serial is not text in this corpus: it arrives as
    `Query.serial` and is already enforced by the candidate filter, in SQL, before anything is
    ranked. Looking for it a second time would ask a question the `WHERE` clause has answered.
    """
    terms = list(identifiers) if identifiers is not None else list(identifiers_in_query(query))
    if not terms:
        return []
    params: dict[str, Any] = {**candidate_filter.params, "identifiers": terms}
    rows = session.execute(
        text(_EXACT_SQL.format(where=candidate_filter.sql)), params
    ).mappings()
    return [chunk_from_mapping(row) for row in rows]
