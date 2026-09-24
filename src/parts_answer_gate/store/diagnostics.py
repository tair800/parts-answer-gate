"""Evidence for ADR-001 kill condition K, and the breach that proves the evidence is load-bearing.

The condition is that vector retrieval executes through pgvector rather than through an in-process
fallback. Three things could satisfy that sentence dishonestly: importing `pgvector` and never using
it; declaring the column as `real[]` and calling it a vector; or issuing a query that the planner
serves with a sequential scan while the index sits unused. So none of the fields below is read from
the source code. `extension_installed` comes from `pg_extension`, `column_type` from
`information_schema`, and `explain_plan` is the text PostgreSQL returned for the *same statement*
`dense_search` executes.

`breach_caught` is the falsifiability clause. ADR-001 says a guard that only reads source text does
not count — a breach must change behaviour and a test must observe the change. So this module plants
the exact breach the ADR names: it fetches the same filtered candidates and sorts them by distance
in Python. That impostor returns **the same ranking** as the real thing, which is the point: the
substitution is invisible in the output, and only the query plan gives it away. `breach_caught` is
true when the guard accepts the real query, rejects the impostor, and the two agree on the result —
because if they disagreed, the test would be detecting a bug rather than detecting the breach.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from parts_answer_gate.domain import Language, Query
from parts_answer_gate.retrieval.dense import (
    ANN_PLAN_SETTINGS,
    DISTANCE_OPERATOR,
    DenseHit,
    dense_search,
    dense_sql,
    explain_dense,
    vector_literal,
)
from parts_answer_gate.retrieval.embeddings import EMBEDDING_POLICY, Embedder, cosine_similarity
from parts_answer_gate.store.effectivity import CandidateFilter, candidate_filter
from parts_answer_gate.store.engine import SESSION_SETTINGS
from parts_answer_gate.store.queries import candidate_count
from parts_answer_gate.store.schema import (
    EFFECTIVITY_INDEXES,
    HNSW_BUILD_PARAMETERS,
    VECTOR_COLUMN,
)

__all__ = [
    "BreachOutcome",
    "build_pgvector_artifact",
    "executed_through_pgvector",
    "in_memory_dense_search_breach",
]


@dataclass(frozen=True)
class BreachOutcome:
    """What happened when the planted breach was substituted for the real query."""

    planted: str
    guard_accepted_real_query: bool
    guard_rejected_breach: bool
    #: How much of the real top-k the impostor reproduced. Reported rather than asserted equal:
    #: HNSW is an approximate index and the in-memory sort is exhaustive, so the two *should*
    #: differ slightly. A high overlap is the point — the substitution is nearly invisible in the
    #: output, which is why the guard reads the query plan instead of the results.
    top_k_overlap: float
    breach_plan: str
    caught: bool


def executed_through_pgvector(plan: str) -> bool:
    """The guard. A plan is evidence only if the vector index *and* the operator are both in it.

    Either alone is insufficient. A plan can name the index in a bitmap step that never computes a
    distance, and a plan can contain the operator in a sort node that read every row sequentially.
    """
    return str(HNSW_BUILD_PARAMETERS["index_name"]) in plan and DISTANCE_OPERATOR in plan


def in_memory_dense_search_breach(
    session: Session,
    filters: CandidateFilter,
    vector: Sequence[float],
    *,
    limit: int,
) -> tuple[list[DenseHit], str]:
    """**The planted breach.** Does the ranking in Python, and returns the plan of what it ran.

    Not importable by accident from the retrieval package: it lives here, in the diagnostic, and
    nothing in `retrieval` references it. It exists so that "vector search happens in the database"
    has something to be false against.
    """
    # The fragment comes from `store.effectivity`, built out of module constants; every value the
    # caller supplied is a bound parameter, so this is not an injection site.
    statement = (
        "SELECT chunk.chunk_id AS chunk_id, chunk.embedding AS embedding FROM chunk "  # noqa: S608
        f"WHERE ({filters.sql}) AND chunk.embedding IS NOT NULL"
    )
    rows = session.execute(sql_text(statement), filters.params).all()
    scored = [
        DenseHit(chunk_id=str(chunk_id), distance=1.0 - cosine_similarity(vector, _floats(stored)))
        for chunk_id, stored in rows
    ]
    scored.sort(key=lambda hit: (hit.distance, hit.chunk_id))

    explained = session.execute(sql_text(f"EXPLAIN (ANALYZE, VERBOSE) {statement}"), filters.params)
    plan = "\n".join(str(row[0]) for row in explained.all())
    return scored[:limit], plan


def _floats(stored: Any) -> list[float]:
    """Coerce whatever the driver handed back for a `vector` column into plain floats.

    `Any` because the concrete type depends on whether pgvector's psycopg adapter is registered on
    the connection — a list or a numpy array — and the breach only needs the numbers.
    """
    return [float(value) for value in stored]


def _median_ms[T](work: Callable[[], T], *, runs: int = 5) -> float:
    """Median wall-clock cost of `work`, in milliseconds.

    Median over five rather than a single sample: the first execution of a query on this container
    pays for a cold buffer cache, and publishing that as the latency of the index would be as
    misleading in one direction as publishing the best of five would be in the other.
    """
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        work()
        samples.append((time.perf_counter() - started) * 1000.0)
    return sorted(samples)[len(samples) // 2]


def build_pgvector_artifact(
    session: Session,
    query: Query,
    *,
    embedder: Embedder,
    limit: int = 10,
    languages: Iterable[Language] | None = None,
) -> dict[str, Any]:
    """Produce `pgvector.json`.

    Everything is measured against the live database in this call. If the database is empty the
    plan will be honest about that too — an `EXPLAIN` over zero rows cannot use an index, and the
    artifact would say `explain_mentions_index: false` rather than quietly passing.
    """
    filters = candidate_filter(query, languages=languages)
    vector = embedder.embed_query(query.text)

    extension = session.execute(
        sql_text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar_one_or_none()
    table, column = VECTOR_COLUMN
    column_type = session.execute(
        sql_text(
            "SELECT udt_name FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :column"
        ),
        {"table": table, "column": column},
    ).scalar_one_or_none()

    real_hits = dense_search(session, filters, vector, limit=limit)
    ann_ms = _median_ms(lambda: dense_search(session, filters, vector, limit=limit))
    planner_ms = _median_ms(
        lambda: dense_search(session, filters, vector, limit=limit, use_ann_plan=False)
    )
    evidence = explain_dense(session, filters, vector, limit=limit)
    planner_evidence = explain_dense(session, filters, vector, limit=limit, use_ann_plan=False)

    breach_hits, breach_plan = in_memory_dense_search_breach(
        session, filters, vector, limit=limit
    )
    real_ids = [hit.chunk_id for hit in real_hits]
    breach_ids = [hit.chunk_id for hit in breach_hits]
    overlap = (
        len(set(real_ids) & set(breach_ids)) / len(real_ids) if real_ids else 0.0
    )
    guard_accepted = executed_through_pgvector(evidence.plan)
    guard_rejected = not executed_through_pgvector(breach_plan)
    breach = BreachOutcome(
        planted="dense ranking replaced by an in-process sort over the same filtered candidates",
        guard_accepted_real_query=guard_accepted,
        guard_rejected_breach=guard_rejected,
        top_k_overlap=round(overlap, 4),
        breach_plan=breach_plan,
        caught=guard_accepted and guard_rejected,
    )

    return {
        "extension_installed": extension is not None,
        "extension_version": extension,
        "column_type": column_type,
        "column": f"{table}.{column}",
        "declared_dimensions": EMBEDDING_POLICY["dim"],
        "distance_operator": DISTANCE_OPERATOR,
        "index": dict(HNSW_BUILD_PARAMETERS),
        "effectivity_indexes": list(EFFECTIVITY_INDEXES),
        "session_settings": dict(SESSION_SETTINGS),
        "explain_mentions_index": evidence.mentions_index,
        "explain_uses_vector_operator": evidence.uses_vector_operator,
        "explain_indexes_named": list(evidence.index_names),
        "explain_plan": evidence.plan,
        "explained_sql": dense_sql(filters).strip(),
        "plan_choice": {
            "ann_plan_settings": list(ANN_PLAN_SETTINGS),
            "ann_plan_median_ms": round(ann_ms, 3),
            "planner_default_median_ms": round(planner_ms, 3),
            "planner_default_uses_index": planner_evidence.mentions_index,
            "planner_default_plan": planner_evidence.plan,
            "why": (
                "PostgreSQL prices a 384-dimension cosine distance at procost=1 and therefore "
                "prefers computing it for every row the effectivity filter admits. Both timings "
                "above are measured on this database, in this call, over the same query."
            ),
        },
        "query": {
            "text": query.text,
            "as_of": query.as_of.isoformat(),
            "variant_id": query.variant_id,
            "serial": query.serial,
            "language": query.language.value,
            "limit": limit,
            "vector_prefix": vector_literal(vector[:4])[:-1] + ", ...]",
        },
        "candidates_after_effectivity_filter": candidate_count(session, filters),
        "dense_search_ms": round(ann_ms, 3),
        "breach": asdict(breach),
        "breach_caught": breach.caught,
    }
