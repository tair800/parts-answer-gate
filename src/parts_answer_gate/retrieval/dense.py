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

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Final

from sqlalchemy import text
from sqlalchemy.orm import Session

from parts_answer_gate.store.effectivity import CandidateFilter
from parts_answer_gate.store.schema import HNSW_BUILD_PARAMETERS

__all__ = [
    "ANN_PLAN_SETTINGS",
    "DISTANCE_OPERATOR",
    "DenseHit",
    "ExplainEvidence",
    "ann_plan_scope",
    "dense_search",
    "dense_sql",
    "explain_dense",
    "vector_literal",
]

#: Cosine distance, and the choice is not arbitrary: the encoder does **not** emit unit vectors
#: (`EMBEDDING_POLICY["normalization"]` records the measurement), so `<->` and `<#>` would rank
#: differently rather than equivalently. Cosine normalises inside the operator, which is what makes
#: it the right one for vectors of varying magnitude.
#:
#: The HNSW index is built with `vector_cosine_ops`, and an operator that does not match the index's
#: operator class is an operator the planner cannot use the index for. The two must agree, and they
#: agree here because both read the same constant.
DISTANCE_OPERATOR: Final = "<=>"

#: Turned off for the ANN statement and for nothing else.
#:
#: PostgreSQL prices `cosine_distance` at `procost = 1`, the cost of an integer comparison, when a
#: 384-dimension distance is three orders of magnitude more work than that. Left alone, the planner
#: therefore computes the distance for every row the effectivity filter admits rather than walking
#: the HNSW graph.
#:
#: **At this corpus size that choice is the faster one, and the artifact says so.** Measured in the
#: call that builds `pgvector.json`, on 1,941 chunks with a filter admitting 188: ANN plan ~21 ms,
#: planner's own choice ~8 ms. Scanning 188 rows is simply cheaper than an approximate-neighbour
#: traversal, and it would be dishonest to present the index as a speed-up here.
#:
#: The settings are applied anyway, for a reason that is about evidence rather than latency: ADR-001
#: kill condition K asks whether vector retrieval *executes inside the database through the index*,
#: and a plan that never touches the index cannot answer that question either way. The crossover is
#: a property of corpus size — the scan is linear in admitted rows and the graph walk is not — so a
#: production corpus two orders of magnitude larger inverts it. `pgvector.json` publishes both plans
#: and both timings so a reader can see the trade rather than be told about it.
#:
#: pgvector's own troubleshooting note is to disable sequential scans when the index is not being
#: used; a bitmap heap scan needs `enable_bitmapscan` off as well. Both are scoped to the dense
#: statement because the deterministic identifier lookup reads a GIN index, which PostgreSQL can
#: only reach through a bitmap scan — disabling it session-wide would break the stage that is
#: supposed to be the cheapest one in the pipeline.
#:
#: **These two settings cannot guarantee the index is used, and the artifact does not pretend they
#: can.** Measured on a 4,536-chunk store: when a question names a variant the candidate set falls
#: to ~60 rows, and PostgreSQL then serves the query with a plain index scan on
#: `ix_chunk_effectivity` plus a top-N sort — a plan neither setting disables, and the right plan
#: for sixty rows. `explain_mentions_index` reports what the planner actually did, so a caller
#: building `pgvector.json` has to hand it a query whose effectivity filter admits enough rows for
#: an approximate index to be the sensible choice. A `false` there is a fact about the query, not a
#: defect in the index, and `candidates_after_effectivity_filter` in the same artifact says which.
#:
#: The cost is one extra round trip to set and one to reset. Both settings go in a single statement
#: because psycopg accepts several statements in one `execute` when no parameters are bound, and on
#: this host a round trip to the container costs 1.5-3.8 ms — measured — which is the same order as
#: the query itself. `build_pgvector_artifact` measures the query both ways and publishes both
#: numbers, so this trade is evidenced rather than asserted.
ANN_PLAN_SETTINGS: Final = ("enable_seqscan", "enable_bitmapscan")

_ANN_PLAN_SET: Final = "; ".join(f"SET LOCAL {name} = off" for name in ANN_PLAN_SETTINGS)
_ANN_PLAN_RESET: Final = "; ".join(f"RESET {name}" for name in ANN_PLAN_SETTINGS)

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


@contextmanager
def ann_plan_scope(session: Session, *, enabled: bool = True) -> Iterator[None]:
    """Plan the enclosed statements as ANN queries, then put the session back as it was.

    `SET LOCAL` rather than `SET`: the setting dies with the transaction even if the reset below
    never runs, so a raised exception cannot leave a connection in the pool that plans every
    subsequent query badly. `RESET` restores the session value rather than assuming it was `on`.
    """
    if not enabled:
        yield
        return
    session.execute(text(_ANN_PLAN_SET))
    try:
        yield
    finally:
        session.execute(text(_ANN_PLAN_RESET))


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def dense_sql(candidate_filter: CandidateFilter) -> str:
    """The executed statement, for the searcher and for `EXPLAIN` alike."""
    # The `where` fragment is assembled from module constants in `store.effectivity`; every value a
    # caller supplies arrives through a bound parameter, which is why this is not an injection site
    # despite being a formatted string.
    return _SQL_TEMPLATE.format(operator=DISTANCE_OPERATOR, where=candidate_filter.sql)


def _params(
    candidate_filter: CandidateFilter, vector: Sequence[float], limit: int
) -> dict[str, Any]:
    return {**candidate_filter.params, "query_vector": vector_literal(vector), "limit": limit}


def dense_search(
    session: Session,
    candidate_filter: CandidateFilter,
    vector: Sequence[float],
    *,
    limit: int,
    use_ann_plan: bool = True,
) -> list[DenseHit]:
    """Nearest neighbours **within the filtered candidate set**, ranked by the database.

    `use_ann_plan=False` exists for one caller: the diagnostic that measures what the planner's own
    choice costs. Nothing in the retrieval path passes it.
    """
    with ann_plan_scope(session, enabled=use_ann_plan):
        rows = session.execute(
            text(dense_sql(candidate_filter)), _params(candidate_filter, vector, limit)
        ).all()
    return [
        DenseHit(chunk_id=str(chunk_id), distance=float(distance)) for chunk_id, distance in rows
    ]


def explain_dense(
    session: Session,
    candidate_filter: CandidateFilter,
    vector: Sequence[float],
    *,
    limit: int,
    analyze: bool = True,
    use_ann_plan: bool = True,
) -> ExplainEvidence:
    """Ask the database what it did with the same statement `dense_search` runs.

    `ANALYZE` by default so the plan is the one that executed, not the one the planner would have
    chosen if asked hypothetically. A plan without ANALYZE is an estimate, and an estimate is not
    evidence that the index was consulted.

    The scope is the same one `dense_search` enters, for the same reason: a plan explained under
    different settings than the query executes under is evidence about a different query.
    """
    options = "ANALYZE, BUFFERS, VERBOSE" if analyze else "VERBOSE"
    statement = f"EXPLAIN ({options}) {dense_sql(candidate_filter)}"
    with ann_plan_scope(session, enabled=use_ann_plan):
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
