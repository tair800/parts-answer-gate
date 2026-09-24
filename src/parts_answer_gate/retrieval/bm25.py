"""Lexical scoring over the chunks the effectivity filter left standing.

BM25 is not a legacy stage here. A technician's question is full of exact tokens — a part number, a
torque figure, a section name — and a 384-dimensional multilingual encoder is measurably worse at
"does this passage contain XP-4021-B" than counting is. Dense retrieval wins on paraphrase, lexical
wins on identifiers, and the fusion in `fusion.py` is what keeps both.

**Why the index is built per query rather than kept warm.** `rank-bm25` builds its statistics from
the corpus it is handed, and the corpus here is *the filtered candidate set* — different for every
as-of date, variant and language. A persistent index would have to be built over everything, which
means the IDF weights would be computed over documents that are not candidates, and, worse, that the
lexical stage would score rows the effectivity filter rejected. That is the post-filtering failure
this project exists to avoid, reintroduced one layer down. At this corpus size the rebuild costs
single-digit milliseconds; at a scale where it did not, the correct move is PostgreSQL's own
`ts_rank` inside the same filtered query, not a global index.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Final

__all__ = ["BM25_B", "BM25_K1", "TOKEN_PATTERN", "bm25_scores", "tokenize"]

#: Unicode word characters, so Turkish and Cyrillic tokens survive. An ASCII-only `\w` would reduce
#: every Russian question to punctuation and hand the multilingual evaluation a lexical arm that
#: scores zero on two of its three languages.
TOKEN_PATTERN: Final = re.compile(r"\w+", re.UNICODE)

#: rank-bm25's defaults, stated rather than inherited. k1 controls term-frequency saturation and b
#: the length normalisation; these are the Okapi values the literature reports, and this project
#: does not tune them — a lexical arm tuned on the development split is a baseline that no longer
#: measures what a plain lexical arm does.
BM25_K1: Final = 1.5
BM25_B: Final = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens.

    `str.lower()` rather than `casefold()`: casefold maps Turkish dotted and dotless i in ways that
    differ from `lower()`, and the only property that matters is that the query and the passage go
    through the identical transformation. Choosing the simpler one keeps that obvious.
    """
    return TOKEN_PATTERN.findall(text.lower())


def bm25_scores(query_text: str, candidates: Sequence[tuple[str, str]]) -> dict[str, float]:
    """Score `(chunk_id, text)` pairs against the query. Higher is better; 0.0 means no overlap.

    Returns every candidate, including the zeros, because the fusion stage needs to know a chunk was
    considered and scored nothing — that is a different signal from a chunk that was never a
    candidate, and the error attribution in the evaluation depends on telling them apart.
    """
    if not candidates:
        return {}
    tokens = tokenize(query_text)
    corpus = [tokenize(text) for _, text in candidates]
    if not tokens or not any(corpus):
        return {chunk_id: 0.0 for chunk_id, _ in candidates}

    # rank-bm25 ships no type information; the import is annotated rather than silenced globally so
    # the untyped surface stays visible at its single point of entry.
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]  # noqa: PLC0415

    index: Any = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
    raw = index.get_scores(tokens)
    return {chunk_id: float(score) for (chunk_id, _), score in zip(candidates, raw, strict=True)}
