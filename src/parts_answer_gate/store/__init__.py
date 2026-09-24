"""Persistence: the schema, the connection, the loader and the effectivity predicate.

Kept separate from `retrieval` because the two answer different questions. This package knows what a
chunk looks like on disk and which rows a query is allowed to consider; `retrieval` knows how to
rank the rows it is given. The split is what lets the managed-vector-store comparison put a second
backend behind the same ranking code.
"""

from __future__ import annotations

__all__: list[str] = []
