"""Reciprocal Rank Fusion, and the tie-break that makes two runs identical.

RRF combines rankings by position rather than by score, which is the whole reason it is used here:
BM25 returns an unbounded score whose scale depends on the candidate set, pgvector returns a cosine
distance in [0, 2], and normalising the two onto a common scale means inventing a relationship
between them. Positions need no such invention.

Kill condition J requires two runs of the pipeline to produce byte-identical retrieval. Ties in RRF
are common — two chunks at the same rank in both lists score exactly the same — so the ordering is
`(-score, chunk_id)` everywhere. Sorting by score alone leaves the order of a tie to whatever the
database happened to return first, which is not a guarantee Postgres makes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

__all__ = ["RRF_K", "order_by_score", "reciprocal_rank_fusion"]

#: The constant from Cormack et al. 2009, unchanged. It damps the contribution of the very top of
#: each list so a single stage cannot dominate the fusion; raising it flattens the stages towards
#: equal weight and lowering it lets rank 1 decide the result. Left at 60 because tuning it against
#: the development split would make the published recall a tuned number and the baselines unfair.
RRF_K: Final = 60


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = RRF_K,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Fuse named rankings of chunk ids into one score per chunk.

    `weights` exists so the deterministic identifier ranking can be given more say than a semantic
    one without collapsing into a hand-tuned linear combination of raw scores. Absent, every stage
    counts equally.
    """
    fused: dict[str, float] = {}
    for name, ordered in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        if weight == 0.0:
            continue
        for position, chunk_id in enumerate(ordered, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + position)
    return fused


def order_by_score(scores: Mapping[str, float], *, drop_zero: bool = True) -> list[str]:
    """Descending by score, ascending by chunk id, with the unscored dropped.

    The tie-break is the determinism guarantee: kill condition J requires two runs to produce
    byte-identical retrieval, and a set iterated in whatever order it was built is how that is lost.

    `drop_zero` is the other half, and it matters more than it looks. BM25 scores a chunk sharing no
    term with the query at exactly 0.0, and with the zeros kept the tail of this ranking is an
    alphabetical list of chunks the stage found nothing in — which reciprocal rank fusion then
    rewards with real reciprocal-rank weight purely for sorting early. A stage that says "no
    opinion" should contribute no opinion, not an opinion ordered by identifier.
    """
    items = [(k, v) for k, v in scores.items() if v > 0.0] if drop_zero else list(scores.items())
    ordered = sorted(items, key=lambda item: (-item[1], item[0]))
    return [chunk_id for chunk_id, _ in ordered]
