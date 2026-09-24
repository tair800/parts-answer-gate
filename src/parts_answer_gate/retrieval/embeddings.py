"""The embedding model, and the policy that has to stay the same on both sides of the index.

An embedding is only comparable to another embedding produced the same way. The model, the
dimension, whether vectors are normalised, what text was fed in and what was done to the query
before it was encoded are all part of the contract between the loader and the retriever, and a
mismatch in any of them degrades recall silently — nothing raises, the numbers just get worse.
`EMBEDDING_POLICY` exists so that contract is one importable object rather than a habit spread over
two modules.

`paraphrase-multilingual-MiniLM-L12-v2` is chosen because this corpus is parallel EN/TR/RU and the
multilingual evaluation compares the three against each other. A stronger English-only encoder would
win on the English split and make the cross-lingual result meaningless, which is the one result the
blueprint actually asks for. Runs as quantised ONNX through fastembed: no PyTorch, no CUDA, no
network call at query time, which is also what makes CI able to score retrieval without a key.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, Final, Protocol

__all__ = [
    "EMBEDDING_CACHE_DIR",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_POLICY",
    "EMBEDDING_THREADS",
    "DocumentEncoder",
    "Embedder",
    "TextEncoder",
    "cosine_similarity",
]

#: The fastembed model id. fastembed resolves it to the quantised ONNX mirror it publishes, which is
#: what `.fastembed_cache/` already contains.
EMBEDDING_MODEL_NAME: Final = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#: Must equal the `vector(...)` width declared in `store.schema`. Imported there rather than
#: repeated, so the two cannot drift apart without a type error.
EMBEDDING_DIM: Final = 384

#: Kept inside the repository, not in the user profile: a CI job and a developer laptop should hit
#: the same bytes, and a cache in `~` is invisible when a run suddenly needs the network.
EMBEDDING_CACHE_DIR: Final = ".fastembed_cache"

#: The whole contract, in one place, so an artifact can quote it instead of describing it.
EMBEDDING_POLICY: Final[dict[str, Any]] = {
    "model": EMBEDDING_MODEL_NAME,
    "runtime": "fastembed / ONNX, quantised, CPU, offline",
    "dim": EMBEDDING_DIM,
    "normalization": (
        "None. The quantised ONNX build fastembed serves for this model does NOT L2-normalise its "
        "output: `Embedder.measured_norm()` on the probe string returns about 5.4, not 1.0, and "
        "that is a measurement rather than an assumption. Vectors are therefore stored exactly as "
        "the encoder emits them and compared with cosine distance, which normalises internally — "
        "so retrieval is unaffected, but `<->` (L2) and `<#>` (inner product) would NOT rank "
        "identically to `<=>` here, and the HNSW index is built with `vector_cosine_ops` to match."
    ),
    "chunking": (
        "One embedding per stored chunk, over exactly `chunk.text` and nothing else: no "
        "re-window, no overlap added at embed time, no title or metadata prepended. Chunk "
        "boundaries belong to the corpus generator. Prepending metadata here would put text into "
        "the vector that the citation check cannot locate in the source document, which is kill "
        "condition D's failure mode arriving through the back door."
    ),
    "query": (
        "Identical treatment to a document: this model has no asymmetric query prefix, unlike E5 "
        "or BGE. Adding one would encode queries into a different region of the space than the "
        "passages they must match."
    ),
    "distance": "cosine",
}


class TextEncoder(Protocol):
    """What the retrieval path needs: one method for passages, one for a question.

    A protocol because two unrelated classes satisfy it — `Embedder`, which opens an ONNX session,
    and `retrieval.precomputed.CachedEmbedder`, which serves vectors computed at build time and on
    a constrained instance never opens one at all.
    """

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class DocumentEncoder(Protocol):
    """What `store.loader.load_chunks` needs, which is one method.

    A protocol rather than the concrete `Embedder`, because the loader is served by two things that
    are not related by inheritance: the encoder itself, and `retrieval.precomputed.CachedEmbedder`,
    which answers from vectors computed at build time and opens no ONNX session at all. Making the
    cache a subclass would give every cache hit the cost of a model it never uses.
    """

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Plain cosine, used only to check the database agrees with the local arithmetic."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return 0.0 if norm == 0.0 else dot / norm


#: How many threads the ONNX session may use. `None` lets onnxruntime size a pool from the host's
#: core count, which is right on a builder and wrong on a 512MB shared-CPU instance: each thread
#: takes its own arena, and the process is killed for memory long before the extra cores help a
#: single query. `PAG_EMBEDDING_THREADS` exists so the serving configuration can say one without
#: the batch path — which encodes thousands of passages and genuinely wants the cores — being
#: slowed down by a constant nobody can override.
EMBEDDING_THREADS: Final[int | None] = (
    int(os.environ["PAG_EMBEDDING_THREADS"]) if os.environ.get("PAG_EMBEDDING_THREADS") else None
)


class Embedder:
    """A lazily-loaded fastembed encoder.

    Lazy because importing this module must stay free: `store.schema` imports `EMBEDDING_DIM` from
    here, so a migration, a `--help` or a unit test that never embeds anything would otherwise pay
    several seconds of ONNX session startup.
    """

    def __init__(
        self,
        *,
        model_name: str = EMBEDDING_MODEL_NAME,
        cache_dir: str | Path = EMBEDDING_CACHE_DIR,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)

    @cached_property
    def _model(self) -> Any:
        # Imported here rather than at module scope for the same reason the property is cached:
        # `import fastembed` alone costs onnxruntime's import, and `store.schema` imports this
        # module purely for an integer.
        from fastembed import TextEmbedding  # noqa: PLC0415

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return TextEmbedding(
            model_name=self.model_name,
            cache_dir=str(self.cache_dir),
            threads=EMBEDDING_THREADS,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode passages, in the order given.

        The order guarantee is load-bearing: the loader zips the result against the chunk ids it
        sent, and a reordering would attach every embedding to the wrong chunk while every test that
        only counts rows kept passing.
        """
        if not texts:
            return []
        vectors = [[float(value) for value in vector] for vector in self._model.embed(list(texts))]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors for {len(texts)} texts; the loader's "
                "zip of ids to vectors would be silently wrong"
            )
        for vector in vectors:
            if len(vector) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"embedder returned {len(vector)} dimensions, schema declares {EMBEDDING_DIM}"
                )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def measured_norm(self, probe: str = "hydraulic pump seal replacement") -> float:
        """The real L2 norm of a real vector, for an artifact that claims vectors are normalised."""
        vector = self.embed_query(probe)
        return math.sqrt(sum(value * value for value in vector))
