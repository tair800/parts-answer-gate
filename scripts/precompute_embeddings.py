"""Embed the corpus once, at build time, and write the vectors next to it.

    python scripts/precompute_embeddings.py

Run by the Docker build, after the corpus is generated and before the image is sealed. The
container then loads an index without opening the encoder at all, which is the difference between a
free-tier deployment that starts and one that is still embedding when the platform's health check
gives up.

See `parts_answer_gate.retrieval.precomputed` for why the cache is keyed by content hash and why
the vectors are float32.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from parts_answer_gate.retrieval.embeddings import Embedder  # noqa: E402
from parts_answer_gate.retrieval.precomputed import write_cache  # noqa: E402
from parts_answer_gate.store.loader import content_hash  # noqa: E402

DEFAULT_CORPUS = REPO_ROOT / "data" / "generated"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args(argv)

    path = args.corpus / "chunks.json"
    if not path.is_file():
        print(f"no corpus in {args.corpus}; run scripts/generate_corpus.py", file=sys.stderr)
        return 1

    raw = json.loads(path.read_text(encoding="utf-8"))
    chunks = raw["chunks"] if isinstance(raw, dict) else raw

    # Deduplicated by hash before encoding. Two chunks with identical text embed identically, and
    # paying twice for that would be paying for the corpus's redundancy rather than its size.
    by_hash: dict[str, str] = {}
    for chunk in chunks:
        by_hash.setdefault(content_hash(str(chunk["text"])), str(chunk["text"]))

    # The corpus's own questions, cached alongside the passages.
    #
    # `Embedder.embed_query` is `embed_documents([text])[0]` — this model has no query or passage
    # prefix — so the vector for a question is the same vector whenever it is computed. Caching it
    # lets a memory-constrained deployment run **real** hybrid retrieval for every question the
    # corpus contains, rather than loading a 671MB session it cannot hold. It changes nothing about
    # what retrieval does; it moves where the arithmetic happens, exactly as it already does for the
    # passages.
    questions_path = args.corpus / "questions.json"
    if questions_path.is_file():
        raw_questions = json.loads(questions_path.read_text(encoding="utf-8"))
        questions = raw_questions["questions"] if isinstance(raw_questions, dict) else raw_questions
        for question in questions:
            by_hash.setdefault(content_hash(str(question["text"])), str(question["text"]))
        print(f"[precompute] {len(questions)} question texts included")

    hashes = sorted(by_hash)
    texts = [by_hash[digest] for digest in hashes]
    print(f"[precompute] {len(chunks)} chunks, {len(hashes)} distinct texts")

    started = time.perf_counter()
    vectors = Embedder().embed_documents(texts)
    elapsed = time.perf_counter() - started

    written = write_cache(args.corpus, hashes, vectors)
    size_mb = written.stat().st_size / 1_000_000
    print(
        f"[precompute] embedded in {elapsed:.1f}s "
        f"({len(hashes) / elapsed:.0f}/s), wrote {size_mb:.1f}MB to {written.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
