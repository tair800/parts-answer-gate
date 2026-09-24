"""A cheap, deterministic reranker. No model, no network, no learned weights.

A cross-encoder would rerank better. It would also put a neural model between the evidence and the
answer, and ADR-001's gate is built entirely from signals that are not model outputs — the moment a
transformer decides the final ordering, "the decision to answer is deterministic" stops being true
of the thing the decision was made about. So this stage is four features and four constants, all
readable, all reproducible, and all cheap enough that they cost less than the round trip they follow.

The features exist because RRF throws information away on purpose. Fusion knows only positions, so
it cannot tell that a candidate ranked third by both stages happens to contain the exact part number
the question asked for. These four put the discarded evidence back, without re-deriving a score
scale that RRF deliberately avoided.
"""

from __future__ import annotations

from collections.abc import Sequence, Set as AbstractSet
from dataclasses import dataclass, field
from typing import Final

from parts_answer_gate.domain import Language, Query, part_numbers_in
from parts_answer_gate.retrieval.bm25 import tokenize

__all__ = ["RERANK_WEIGHTS", "RerankCandidate", "RerankedChunk", "rerank"]

#: Additive over the min-max normalised fusion score, which is therefore worth 1.0 at the top of the
#: pool. Every weight below is smaller than that on purpose: reranking is a tie-break over the
#: fusion, not a replacement for it.
#:
#: `exact_identifier` is the largest because it is the only feature that is evidence of a different
#: kind — a literal catalogue match rather than a similarity — and ADR-001 makes the deterministic
#: identifier hit a gate signal in its own right.
#: `language_match` is small because the candidate filter already constrains language by default;
#: the weight only does work in the widened cross-lingual ablation, where it prefers a passage in
#: the technician's own language when two passages say the same thing.
RERANK_WEIGHTS: Final[dict[str, float]] = {
    "exact_identifier": 0.50,
    "identifier_coverage": 0.20,
    "term_coverage": 0.15,
    "language_match": 0.05,
}

#: Tokens shorter than this are ignored when measuring term coverage. A length floor rather than a
#: stopword list because a stopword list for EN, TR and RU is three lists to maintain and three
#: chances to hand one language a different question than the other two.
_MIN_CONTENT_TOKEN = 3


@dataclass(frozen=True)
class RerankCandidate:
    chunk_id: str
    text: str
    language: Language
    fused_score: float
    exact_identifier_hit: bool = False


@dataclass(frozen=True)
class RerankedChunk:
    chunk_id: str
    score: float
    #: Every feature that contributed, so a bad ranking can be explained instead of re-run.
    features: dict[str, float] = field(default_factory=dict)


def _content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) >= _MIN_CONTENT_TOKEN}


def _normalised(scores: Sequence[float]) -> list[float]:
    """Min-max onto [0, 1]. A pool where every score is equal normalises to 1.0, not to 0.0.

    Mapping a flat pool to zero would delete the fusion's contribution entirely and let the smallest
    feature decide the order, which is the opposite of what the weights above say should happen.
    """
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high <= low:
        return [1.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def rerank(query: Query, candidates: Sequence[RerankCandidate]) -> list[RerankedChunk]:
    """Re-score and re-order. Ties break on chunk id, as everywhere else in this pipeline."""
    if not candidates:
        return []

    query_tokens = _content_tokens(query.text)
    query_identifiers = part_numbers_in(query.text)
    fused = _normalised([candidate.fused_score for candidate in candidates])

    scored: list[RerankedChunk] = []
    for candidate, fused_score in zip(candidates, fused, strict=True):
        chunk_tokens = _content_tokens(candidate.text)
        chunk_identifiers = part_numbers_in(candidate.text)
        features: dict[str, float] = {
            "fused_normalised": fused_score,
            "exact_identifier": 1.0 if candidate.exact_identifier_hit else 0.0,
            "identifier_coverage": _coverage(query_identifiers, chunk_identifiers),
            "term_coverage": _coverage(query_tokens, chunk_tokens),
            "language_match": 1.0 if candidate.language is query.language else 0.0,
        }
        score = fused_score + sum(
            weight * features[name] for name, weight in RERANK_WEIGHTS.items()
        )
        scored.append(RerankedChunk(chunk_id=candidate.chunk_id, score=score, features=features))

    scored.sort(key=lambda item: (-item.score, item.chunk_id))
    return scored


def _coverage(wanted: AbstractSet[str], present: AbstractSet[str]) -> float:
    """Share of the question's terms the chunk contains. An empty question covers nothing."""
    if not wanted:
        return 0.0
    return len(wanted & present) / len(wanted)
