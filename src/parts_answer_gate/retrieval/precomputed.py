"""Vectors computed once, at build time, and looked up by content hash at load time.

Embedding this corpus takes about eleven minutes on nine cores. A free-tier container has a
fraction of one, so a deployment that embedded at start would either never finish or would finish
long after the platform's health check gave up — and the alternative, embedding lazily on first
request, would put eleven minutes in front of the first visitor instead of in front of the operator.

So the work moves to build time. `scripts/precompute_embeddings.py` writes the vectors next to the
corpus; `CachedEmbedder` serves them to `store.loader.load_chunks`, which cannot tell the difference
and does not need to.

**Keyed by content hash, not by chunk id.** It is the same key `load_chunks` uses to decide what
needs re-embedding, so a cache entry exists exactly when the loader would have skipped the work
anyway. Keying by chunk id would let an edited passage keep the vector of the text it replaced —
the chunk id is stable across an edit and the whole point of the hash is that it is not.

**Float32, exactly as the encoder emitted them.** Half precision would halve the file and would
change the ranking in the tail, and a deployment whose ordering differs from the evaluated one is
not the evaluated system. Nineteen megabytes is the right trade.

A miss falls through to the real encoder rather than raising. Query embedding is a miss by
definition — nobody precomputes a question nobody has asked yet — and the service needs it to work.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from parts_answer_gate.retrieval.embeddings import EMBEDDING_DIM, Embedder

__all__ = ["HASH_FILE", "VECTOR_FILE", "CachedEmbedder", "load_cache", "write_cache"]

#: Beside the corpus, because the two are only meaningful together.
VECTOR_FILE = "embeddings.f32.npy"
HASH_FILE = "embeddings.index.json"


def write_cache(directory: Path, hashes: Sequence[str], vectors: Sequence[Sequence[float]]) -> Path:
    """Write the vectors and the hash order they correspond to."""
    if len(hashes) != len(vectors):
        raise ValueError(f"{len(hashes)} hashes against {len(vectors)} vectors")
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"expected (n, {EMBEDDING_DIM}) vectors, got {array.shape}")

    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / VECTOR_FILE, array, allow_pickle=False)
    (directory / HASH_FILE).write_text(
        json.dumps(
            {
                "model_keyed_by": "content_hash(text) from store.loader",
                "dimensions": EMBEDDING_DIM,
                "count": len(hashes),
                "hashes": list(hashes),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return directory / VECTOR_FILE


def load_cache(directory: Path) -> dict[str, list[float]]:
    """Read the cache, or return empty when it is absent or does not match the corpus."""
    vector_path = directory / VECTOR_FILE
    hash_path = directory / HASH_FILE
    if not vector_path.is_file() or not hash_path.is_file():
        return {}

    manifest: dict[str, Any] = json.loads(hash_path.read_text(encoding="utf-8"))
    array = np.load(vector_path, allow_pickle=False)
    hashes: list[str] = list(manifest["hashes"])
    if array.shape[0] != len(hashes):
        raise ValueError(
            f"{vector_path.name} holds {array.shape[0]} vectors and {hash_path.name} names "
            f"{len(hashes)}; the pair was written by different runs"
        )
    return {digest: array[index].tolist() for index, digest in enumerate(hashes)}


class CachedEmbedder:
    """An `Embedder` that answers from the cache and falls through to the encoder on a miss.

    Deliberately *not* a subclass. `Embedder` loads the ONNX session on first use, and inheriting
    would make a cache hit carry the cost of a model that a fully-cached load never needs to open.
    The real encoder is constructed only when something misses.
    """

    def __init__(self, cache: dict[str, list[float]], *, fallback: Embedder | None = None) -> None:
        self._cache = cache
        self._fallback = fallback
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def _encoder(self) -> Embedder:
        if self._fallback is None:
            self._fallback = Embedder()
        return self._fallback

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        from parts_answer_gate.store.loader import content_hash  # noqa: PLC0415 - cycle

        wanted = [content_hash(text) for text in texts]
        missing = [index for index, digest in enumerate(wanted) if digest not in self._cache]
        self._hits += len(texts) - len(missing)
        self._misses += len(missing)

        computed: dict[int, list[float]] = {}
        if missing:
            fresh = self._encoder().embed_documents([texts[index] for index in missing])
            for index, vector in zip(missing, fresh, strict=True):
                computed[index] = vector
                self._cache[wanted[index]] = vector

        return [
            computed[index] if index in computed else self._cache[digest]
            for index, digest in enumerate(wanted)
        ]

    def embed_query(self, text: str) -> list[float]:
        """Always the real encoder. A question nobody has asked has no precomputed vector."""
        return self._encoder().embed_query(text)

    def measured_norm(self, probe: str = "hydraulic pump seal replacement") -> float:
        return self._encoder().measured_norm(probe)
