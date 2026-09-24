"""The pgvector half of retrieval, and the evidence that it really is pgvector.

ADR-001 kill condition K is the claim that vector search executes inside PostgreSQL rather than in a
Python loop that sorts a list of tuples. Importing `pgvector` proves nothing, so this module keeps
two things next to each other: the SQL that is actually executed, and an `EXPLAIN` of *that same
string*, run by the database. The artifact quotes the plan the server produced. Nothing in the
artifact is derived from reading the code that built the query.

**Why raw SQL and not the ORM.** The effectivity predicate is a textual `WHERE` fragment, because it
has to be identical in the lexical candidate fetch and here — the whole claim is that both stages
rank the same filtered set. Expressing it twice, once as ORM criteria and once as text, would make
"the same filter" an assertion instead of a fact.

**Why the vector is bound as a string with an explicit cast.** psycopg adapts a Python list of
floats to `float8[]`, not to `vector`, and the operator would then fail to resolve or, worse, fall
back to a plan that never touches the index. `CAST(:query_vector AS vector)` is unambiguous and the
literal round-trips exactly, so two runs of one query bind byte-identical text — which is what kill
condition J needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from sqlalchemy import text
from sqlalchemy.orm import Session

from parts_answer_gate.store.effectivity import CandidateFilter
from parts_answer_gate.store.schema import HNSW_BUILD_PARAMETERS

__all__ = [
    "DISTANCE_OPERATOR",
    "DenseHit",
    "ExplainEvidence",
    "dense_search",
    "dense_sql",
    "explain_dense",
    "vector_literal",
]

#: Cosine distance. The encoder emits unit vectors, so `<->` (L2) would rank identically, but the
#: HNSW index is built with `vector_cosine_ops` and an operator that does not match the index's
#: operator class is an operator the planner cannot use the index for. The two must agree, and they
#: agree here because both read the same constant.
DISTANCE_OPERATOR: Final = "<=>"

_SQL_TEMPLATE: Final = """
SELECT chunk.chunk_id AS chunk_id,
       chunk.embedding {operator} CAST(:query_vector AS vector) AS distance
FROM chunk
WHERE ({where}) AND chunk.embedding IS NOT NULL
ORDER BY chunk.embedding {operator} CAST(:query_vector AS vector)
LIMIT :limit
"""


@dataclass(frozen=True)
class DenseHit:
    chunk_id: str
    #: Cosine distance in [0, 2] as the database computed it.
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity, which is what a reader expects to see in a report."""
        return 1.0 - self.distance


@dataclass(frozen=True)
class ExplainEvidence:
    """What the server said about the plan. Every field is read out of the plan text."""

    plan: str
    mentions_index: bool
    index_names: tuple[str, ...]
    uses_vector_operator: bool


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def dense_sql(candidate_filter: CandidateFilter) -> str:
    """The executed statement, for the searcher and for `EXPLAIN` alike."""
    # The `where` fragment is assembled from module constants in `store.effectivity`; every value a
    # caller supplies arrives through a bound parameter, which is why this is not an injection site
    # despite being a formatted string.
    return _SQL_TEMPLATE.format(  # noqa: S608
        operator=DISTANCE_OPERATOR, where=candidate_filter.sql
    )


def _params(candidate_filter: CandidateFilter, vector: Sequence[float], limit: int) -> dict[str, Any]:
    return {**candidate_filter.params, "query_vector": vector_literal(vector), "limit": limit}


def dense_search(
    session: Session,
    candidate_filter: CandidateFilter,
    vector: Sequence[float],
    *,
    limit: int,
) -> list[DenseHit]:
    """Nearest neighbours **within the filtered candidate set**, ranked by the database."""
    rows = session.execute(
        text(dense_sql(candidate_filter)), _params(candidate_filter, vector, limit)
    ).all()
    return [DenseHit(chunk_id=str(chunk_id), distance=float(distance)) for chunk_id, distance in rows]


def explain_dense(
    session: Session,
    candidate_filter: CandidateFilter,
    vector: Sequence[float],
    *,
    limit: int,
    analyze: bool = True,
) -> ExplainEvidence:
    """Ask the database what it did with the same statement `dense_search` runs.

    `ANALYZE` by default so the plan is the one that executed, not the one the planner would have
    chosen if asked hypothetically. A plan without ANALYZE is an estimate, and an estimate is not
    evidence that the index was consulted.
    """
    options = "ANALYZE, BUFFERS, VERBOSE" if analyze else "VERBOSE"
    statement = f"EXPLAIN ({options}) {dense_sql(candidate_filter)}"
    rows = session.execute(text(statement), _params(candidate_filter, vector, limit)).all()
    plan = "\n".join(str(row[0]) for row in rows)

    index_name = str(HNSW_BUILD_PARAMETERS["index_name"])
    found = tuple(name for name in (index_name,) if name in plan)
    return ExplainEvidence(
        plan=plan,
        mentions_index=bool(found),
        index_names=found,
        uses_vector_operator=DISTANCE_OPERATOR in plan,
    )
