"""The retrieval pipeline, in the one order ADR-001 permits.

    deterministic identifier lookup
      -> effectivity filter, in SQL, constraining the candidate set
      -> BM25 over the survivors
      -> pgvector over the same survivors
      -> reciprocal rank fusion
      -> deterministic rerank

The order is the argument. Steps 3 and 4 never see a row step 2 rejected, so the effectivity
guarantee is a property of the candidate set rather than of a post-processing pass that could be
skipped, misconfigured or quietly reordered. `RetrievalTrace.filter_applied` says `before_ranking`
because there is no code path in which it could say anything else.

Every score is carried separately to the end. A single fused number cannot tell a question that
failed with a strong lexical signal and a weak semantic one from a question where both were weak,
and the evaluation's job is to attribute error to a stage.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final

from sqlalchemy.orm import Session

from parts_answer_gate.domain import Chunk, Language, Query, RetrievedChunk
from parts_answer_gate.retrieval.bm25 import bm25_scores
from parts_answer_gate.retrieval.dense import DenseHit, dense_search, dense_sql
from parts_answer_gate.retrieval.embeddings import Embedder
from parts_answer_gate.retrieval.fusion import order_by_score, reciprocal_rank_fusion
from parts_answer_gate.retrieval.rerank import RerankCandidate, rerank
from parts_answer_gate.store.effectivity import FILTER_STAGE, CandidateFilter, candidate_filter
from parts_answer_gate.store.queries import (
    exact_identifier_lookup,
    fetch_candidates,
    identifiers_in_query,
)

__all__ = [
    "MIN_STAGE_DEPTH",
    "STAGE_DEPTH_MULTIPLIER",
    "STAGE_WEIGHTS",
    "RetrievalResult",
    "RetrievalTrace",
    "Retriever",
]

#: Each stage returns more than the caller asked for, because fusion needs depth to fuse. A stage
#: truncated to top_k can only ever confirm what the other stage already found at the same rank.
STAGE_DEPTH_MULTIPLIER: Final = 5
MIN_STAGE_DEPTH: Final = 50

#: Fusion weights. The deterministic stage counts double because a literal catalogue match is
#: evidence of a different kind from a similarity, not a stronger version of the same thing. The two
#: statistical stages are equal: preferring one over the other would be a tuned constant, and the
#: predeclared baselines exist precisely to measure what each contributes on its own.
STAGE_WEIGHTS: Final[dict[str, float]] = {
    "exact_identifier": 2.0,
    "lexical": 1.0,
    "dense": 1.0,
}


@dataclass(frozen=True)
class RetrievalTrace:
    """What the pipeline did, in the terms the evaluation artifacts report.

    `filter_applied` and `candidate_set_constrained_in_sql` are the two fields `effectivity.json`
    quotes for kill condition "filter before ranking". They are produced by the code that ran, not
    written into the artifact by hand.
    """

    filter_applied: str
    candidate_set_constrained_in_sql: bool
    candidates: int
    exact_identifier_hit: bool
    exact_identifier_terms: tuple[str, ...]
    exact_identifier_chunk_ids: tuple[str, ...]
    embedding_consulted: bool
    variant_constrained: bool
    serial_constrained: bool
    language_constrained: bool
    dense_sql: str
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    chunks: tuple[RetrievedChunk, ...]
    trace: RetrievalTrace


class Retriever:
    """Holds the encoder; takes a `Session` per call so the caller owns the transaction."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()

    # ------------------------------------------------------------------ the deterministic stage

    def exact_lookup(
        self,
        session: Session,
        query: Query,
        *,
        languages: Iterable[Language] | None = None,
    ) -> list[Chunk]:
        """Identifier match only, with no embedding involved at any point.

        Public in its own right because ADR-001 promises the README will publish what share of the
        question set never needs a model at all. That share cannot be measured from inside a hybrid
        pipeline that always embeds.
        """
        filters = candidate_filter(query, languages=languages)
        return exact_identifier_lookup(session, query, filters)

    # ---------------------------------------------------------------------------- the pipeline

    def retrieve(
        self,
        session: Session,
        query: Query,
        *,
        languages: Iterable[Language] | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> RetrievalResult:
        """Retrieve, optionally with the stage weights overridden.

        `weights` exists for the predeclared baselines and for nothing else. Setting `dense` to zero
        is what `bm25_only` *is*; setting `lexical` to zero is `dense_only`. Expressing them as a
        weight through this one code path keeps the comparison honest — a baseline built from a
        second retriever would measure the differences between two retrievers rather than the
        contribution of one signal.

        It is a parameter rather than a constructor argument so the same `Retriever`, holding the
        same loaded encoder, serves every arm. Five encoders would be five copies of 220MB and a
        chance for two arms to embed differently.
        """
        call_started = time.perf_counter()
        timings: dict[str, float] = {}
        depth = max(query.top_k * STAGE_DEPTH_MULTIPLIER, MIN_STAGE_DEPTH)
        filters = candidate_filter(query, languages=languages)

        exact_ids, exact_terms = self._exact_stage(session, query, filters, timings)
        candidates = _timed(timings, "candidates", lambda: fetch_candidates(session, filters))
        by_id = {chunk.chunk_id: chunk for chunk in candidates}

        lexical = _timed(
            timings,
            "lexical",
            lambda: bm25_scores(query.text, [(c.chunk_id, c.text) for c in candidates]),
        )
        lexical_order = order_by_score(lexical)[:depth]

        dense_hits = self._dense_stage(session, query, filters, depth, timings)
        dense_by_id = {hit.chunk_id: hit.similarity for hit in dense_hits}

        fused = _timed(
            timings,
            "fusion",
            lambda: reciprocal_rank_fusion(
                {
                    "exact_identifier": exact_ids,
                    "lexical": lexical_order,
                    "dense": [hit.chunk_id for hit in dense_hits],
                },
                weights=STAGE_WEIGHTS if weights is None else {**STAGE_WEIGHTS, **weights},
            ),
        )

        exact_set = set(exact_ids)
        pool = [
            RerankCandidate(
                chunk_id=chunk_id,
                text=by_id[chunk_id].text,
                language=by_id[chunk_id].language,
                fused_score=score,
                exact_identifier_hit=chunk_id in exact_set,
            )
            # `by_id` is the filtered candidate set; a fused id outside it would mean a stage scored
            # a row the effectivity predicate rejected, which cannot happen and is asserted rather
            # than assumed.
            for chunk_id, score in fused.items()
            if chunk_id in by_id
        ]
        if len(pool) != len(fused):
            raise RuntimeError(
                "a fused chunk id is outside the filtered candidate set; one ranking stage is not "
                "reading the same WHERE clause as the others"
            )

        reranked = _timed(timings, "rerank", lambda: rerank(query, pool))
        top = reranked[: query.top_k]

        chunks = tuple(
            RetrievedChunk(
                chunk=by_id[item.chunk_id],
                lexical_score=lexical.get(item.chunk_id, 0.0),
                dense_score=dense_by_id.get(item.chunk_id, 0.0),
                fused_score=item.score,
                rank=position,
                exact_identifier_hit=item.chunk_id in exact_set,
            )
            for position, item in enumerate(top, start=1)
        )

        # Measured end to end rather than summed from the stages: the difference between the two
        # is the overhead nobody instrumented, and hiding it would flatter the published p95.
        timings["total"] = (time.perf_counter() - call_started) * 1000.0
        return RetrievalResult(
            chunks=chunks,
            trace=RetrievalTrace(
                filter_applied=FILTER_STAGE,
                candidate_set_constrained_in_sql=True,
                candidates=len(candidates),
                exact_identifier_hit=bool(exact_ids),
                exact_identifier_terms=exact_terms,
                exact_identifier_chunk_ids=tuple(exact_ids),
                embedding_consulted=True,
                variant_constrained=filters.variant_constrained,
                serial_constrained=filters.serial_constrained,
                language_constrained=filters.language_constrained,
                dense_sql=dense_sql(filters).strip(),
                timings_ms=timings,
            ),
        )

    # ------------------------------------------------------------------------------- internals

    def _exact_stage(
        self,
        session: Session,
        query: Query,
        filters: CandidateFilter,
        timings: dict[str, float],
    ) -> tuple[list[str], tuple[str, ...]]:
        """Runs first, and finishes before the encoder is touched.

        Ordering is by how many of the question's identifiers a chunk contains, then by chunk id. A
        chunk naming both parts the question named is better evidence than one naming either, and
        the id keeps the rest reproducible.
        """
        terms = identifiers_in_query(query)
        started = time.perf_counter()
        hits = exact_identifier_lookup(session, query, filters, identifiers=list(terms))
        timings["exact_identifier"] = (time.perf_counter() - started) * 1000.0
        wanted = set(terms)
        ordered = sorted(
            hits, key=lambda chunk: (-len(wanted & set(chunk.part_numbers)), chunk.chunk_id)
        )
        return [chunk.chunk_id for chunk in ordered], terms

    def _dense_stage(
        self,
        session: Session,
        query: Query,
        filters: CandidateFilter,
        depth: int,
        timings: dict[str, float],
    ) -> list[DenseHit]:
        started = time.perf_counter()
        vector = self.embedder.embed_query(query.text)
        timings["embed_query"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        hits = dense_search(session, filters, vector, limit=depth)
        timings["dense"] = (time.perf_counter() - started) * 1000.0
        return hits


def _timed[T](timings: dict[str, float], name: str, work: Callable[[], T]) -> T:
    """Run `work`, record its wall-clock cost in milliseconds, return what it returned.

    Per stage rather than one total, because the published p95 has to be attributable: a slow query
    is a different engineering problem depending on whether the time went to the encoder, the index
    or the candidate fetch.
    """
    started = time.perf_counter()
    result = work()
    timings[name] = (time.perf_counter() - started) * 1000.0
    return result
