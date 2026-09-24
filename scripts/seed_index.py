"""Load the corpus into PostgreSQL and embed it into the pgvector column.

    python scripts/seed_index.py
    python scripts/seed_index.py --skip-if-populated   # what the container entrypoint runs

Incremental by construction rather than by a flag. `load_chunks` hashes each chunk's text and
re-embeds only what is new or changed, so running this twice over an unchanged corpus embeds
nothing — which is both the fast path and the measurement kill condition "index lifecycle" reports.

Run at container start on the free deployment, because a pre-deploy command and a one-off job are
both paid features there. That exemption is a property of a single-instance plan and not of the
design: `docker-compose.yml` keeps migration and loading out of the serving process, and anything
with more than one replica must go back to that arrangement or two replicas will race through the
same embedding work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sqlalchemy import func, select  # noqa: E402

from parts_answer_gate.domain import Chunk, Document  # noqa: E402
from parts_answer_gate.retrieval.embeddings import Embedder  # noqa: E402
from parts_answer_gate.retrieval.precomputed import (  # noqa: E402
    CachedEmbedder,
    load_cache,
)
from parts_answer_gate.store.engine import (  # noqa: E402
    build_engine,
    database_url,
    session_scope,
    wait_for_database,
)
from parts_answer_gate.store.loader import load_chunks, load_documents  # noqa: E402
from parts_answer_gate.store.schema import ChunkRow  # noqa: E402

DEFAULT_CORPUS = REPO_ROOT / "data" / "generated"


def _log(message: str) -> None:
    print(f"[index] {message}", flush=True)


def _project(record: dict[str, object], model: type[Document] | type[Chunk]) -> dict[str, object]:
    """Only the keys the model declares."""
    return {key: value for key, value in record.items() if key in model.model_fields}


def _read(corpus_dir: Path) -> tuple[list[Document], list[Chunk]]:
    documents_path = corpus_dir / "documents.json"
    chunks_path = corpus_dir / "chunks.json"
    if not documents_path.is_file() or not chunks_path.is_file():
        raise SystemExit(
            f"no corpus in {corpus_dir}. Run `python scripts/generate_corpus.py` first."
        )

    raw_documents = json.loads(documents_path.read_text(encoding="utf-8"))
    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    documents = raw_documents["documents"] if isinstance(raw_documents, dict) else raw_documents
    chunks = raw_chunks["chunks"] if isinstance(raw_chunks, dict) else raw_chunks

    # Projected onto the declared fields before validating.
    #
    # The corpus records more than the domain models do — a document carries its full text, its
    # series and the split it landed in, all of which the generator and the evaluation need and the
    # retrieval schema does not. `extra="forbid"` refuses them, and that is the right refusal: it is
    # what stops a mistyped field arriving as a silently ignored attribute. So the projection is
    # explicit here rather than the models being loosened there.
    #
    # Validated rather than passed through as dictionaries, because `Document` refuses to exist
    # half-superseded — a corpus that recorded a withdrawal with no successor fails at this line
    # instead of becoming a row an as-of query keeps returning for ever.
    return (
        [Document.model_validate(_project(item, Document)) for item in documents],
        [Chunk.model_validate(_project(item, Chunk)) for item in chunks],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--skip-if-populated",
        action="store_true",
        help="exit early when chunks already exist; what the container entrypoint runs",
    )
    args = parser.parse_args(argv)

    engine = build_engine(database_url())
    _log(f"waiting for {engine.url.render_as_string(hide_password=True)}")
    waited = wait_for_database(engine)
    _log(f"database answered after {waited:.1f}s")

    with session_scope(engine) as session:
        existing = session.execute(select(func.count()).select_from(ChunkRow)).scalar_one()
    if args.skip_if_populated and existing:
        _log(f"{existing} chunks already indexed; nothing to do")
        return 0

    documents, chunks = _read(args.corpus)
    _log(f"{len(documents)} documents, {len(chunks)} chunks to load")

    # Precomputed vectors when the build left any, the encoder when it did not.
    #
    # `CachedEmbedder` opens the ONNX session lazily, so a fully-cached load never pays for the
    # model at all. That is what makes a free-tier container start: embedding this corpus takes
    # about eleven minutes on nine cores, and a shared-CPU instance has a fraction of one.
    started = time.perf_counter()
    cache = load_cache(args.corpus)
    embedder: Embedder | CachedEmbedder
    if cache:
        embedder = CachedEmbedder(cache)
        _log(f"{len(cache)} precomputed vectors loaded in {time.perf_counter() - started:.1f}s")
    else:
        embedder = Embedder()
        _log(f"no precomputed vectors; encoder ready in {time.perf_counter() - started:.1f}s")

    changed_at = datetime.now(UTC)
    with session_scope(engine) as session:
        written = load_documents(session, documents)
        report = load_chunks(session, chunks, embedder, changed_at=changed_at)

    _log(f"documents written {written}")
    _log(
        f"chunks seen {report.chunks_seen}  new {report.chunks_new}  "
        f"changed {report.chunks_changed}  unchanged {report.chunks_unchanged}"
    )
    _log(
        f"re-embedded {report.chunks_re_embedded} in {report.embed_seconds:.1f}s, "
        f"written in {report.write_seconds:.1f}s"
    )
    if report.chunks_unchanged and not report.chunks_re_embedded:
        _log("nothing changed, so nothing was embedded — which is the point of hashing the text")
    if isinstance(embedder, CachedEmbedder):
        _log(f"vector cache: {embedder.hits} hits, {embedder.misses} misses")
        if embedder.misses:
            _log(
                "a miss means the corpus changed after the cache was written; those chunks were "
                "embedded now, which is correct but slow"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
