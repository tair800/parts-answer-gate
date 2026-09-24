"""The build-time vector cache, and the one claim the deployed service rests on.

The public instance runs on 512MB. The encoder is a multilingual model whose loaded session
measures about 671MB resident, so the deployment sets `PAG_QUERY_CACHE_ONLY` and serves query
vectors from the same cache the index was loaded from. That is only honest if a cached vector is
the vector the encoder would have produced, so this file holds that claim to the source rather
than to a comment: `Embedder.embed_query` must remain `embed_documents([text])[0]`, with no query
prefix and no second transformation. The day somebody adds one, the deployed system would start
searching with vectors that no longer match its index, and nothing else in the suite would notice.

Nothing here loads the model. The `fast` lane has no encoder cache, and a test that downloaded
220MB of ONNX to assert a property of five lines of source would be paying for the wrong thing.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from collections.abc import Sequence
from pathlib import Path

import pytest

from parts_answer_gate.retrieval import embeddings
from parts_answer_gate.retrieval.embeddings import EMBEDDING_DIM
from parts_answer_gate.retrieval.precomputed import (
    HASH_FILE,
    VECTOR_FILE,
    CachedEmbedder,
    QueryNotPrecomputedError,
    load_cache,
    write_cache,
)
from parts_answer_gate.store.loader import content_hash


class RefusingEncoder:
    """Stands in for the real encoder and fails if it is reached.

    A cache hit that quietly opened an ONNX session would pass every assertion about the *value*
    returned while destroying the only property the deployment needs, which is that no session is
    opened at all.
    """

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError(f"the encoder was opened for {list(texts)!r}")

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError(f"the encoder was opened for {text!r}")


def a_vector(seed: int) -> list[float]:
    return [float(seed + index) / 1000.0 for index in range(EMBEDDING_DIM)]


def test_embed_query_is_embed_documents_of_one_text() -> None:
    """The source property that makes a build-time query vector equal to a fresh one.

    Asserted over the AST because the claim is about what the method *is*, not about what it
    returned once. A prefix, a normalisation or an instruction template added here would make every
    cached query vector wrong in a way that degrades ranking silently.
    """
    source = textwrap.dedent(inspect.getsource(embeddings.Embedder.embed_query))
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    body = [node for node in function.body if not isinstance(node, ast.Expr)]
    assert len(body) == 1, "embed_query grew a step the precomputed cache does not reproduce"

    returned = body[0]
    assert isinstance(returned, ast.Return)
    assert (
        ast.unparse(returned.value) == "self.embed_documents([text])[0]"  # type: ignore[arg-type]
    ), "a query is no longer encoded exactly as a passage is; the cached vectors are now wrong"


def test_a_cached_query_never_opens_the_encoder() -> None:
    text = "torque for the main bearing cap"
    cache = {content_hash(text): a_vector(1)}
    embedder = CachedEmbedder(cache, fallback=RefusingEncoder())  # type: ignore[arg-type]

    assert embedder.embed_query(text) == a_vector(1)
    assert (embedder.hits, embedder.misses) == (1, 0)


def test_an_unknown_query_is_refused_rather_than_answered_from_nothing() -> None:
    """The cost of the constrained deployment, stated rather than hidden.

    The alternative — returning a zero vector, or an empty result — would read to a visitor as
    "the system found nothing", which is a claim about the corpus. It is not; it is a claim about
    this instance's memory, and the two must not be confused.
    """
    embedder = CachedEmbedder({}, fallback=RefusingEncoder(), encoder_allowed=False)  # type: ignore[arg-type]

    with pytest.raises(QueryNotPrecomputedError):
        embedder.embed_query("a question nobody wrote into the corpus")
    assert (embedder.hits, embedder.misses) == (0, 1)


def test_an_unknown_query_falls_through_when_the_encoder_is_allowed() -> None:
    """Off the constrained deployment the cache is an optimisation, not a boundary."""
    embedder = CachedEmbedder({}, fallback=RefusingEncoder())  # type: ignore[arg-type]

    with pytest.raises(AssertionError, match="the encoder was opened"):
        embedder.embed_query("a question nobody wrote into the corpus")


def test_the_cache_round_trips_by_content_hash(tmp_path: Path) -> None:
    hashes = [content_hash("first passage"), content_hash("second passage")]
    write_cache(tmp_path, hashes, [a_vector(1), a_vector(2)])

    loaded = load_cache(tmp_path)
    assert loaded[content_hash("first passage")] == pytest.approx(a_vector(1))
    assert loaded[content_hash("second passage")] == pytest.approx(a_vector(2))


def test_a_cache_whose_halves_disagree_raises_rather_than_mismatching(tmp_path: Path) -> None:
    """Two files written by different runs would silently pair text with another text's vector."""
    hashes = [content_hash("first passage"), content_hash("second passage")]
    write_cache(tmp_path, hashes, [a_vector(1), a_vector(2)])

    manifest = json.loads((tmp_path / HASH_FILE).read_text(encoding="utf-8"))
    manifest["hashes"] = manifest["hashes"][:1]
    (tmp_path / HASH_FILE).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="written by different runs"):
        load_cache(tmp_path)


def test_an_absent_cache_is_empty_rather_than_an_error(tmp_path: Path) -> None:
    """A development checkout has no cache, and must still be able to open the encoder."""
    assert load_cache(tmp_path) == {}
    assert not (tmp_path / VECTOR_FILE).exists()
