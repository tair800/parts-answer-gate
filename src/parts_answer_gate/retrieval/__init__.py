"""Ranking: embeddings, lexical scoring, the pgvector search, fusion and the rerank.

Nothing here decides what a query is *allowed* to see. That is `store.effectivity`, and it runs
first. A ranking module that could widen its own candidate set would make the effectivity guarantee
a convention rather than a structure.
"""

from __future__ import annotations

__all__: list[str] = []
