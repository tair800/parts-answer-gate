"""Materialise the hold-out membership and commit it, before any score over it exists.

    python scripts/freeze_holdout.py            # write artifacts/holdout.json
    python scripts/freeze_holdout.py --check    # fail if the corpus would now produce another

ADR-001 fixes the rule — a product family is held out iff `blake2b(family_id) % 100 < 34` — and a
rule is not yet a hold-out. A rule plus a corpus implies a membership only while both are unchanged,
so a corpus edit could quietly move a family across the line and every score either side of it would
be measured over a different set.

This writes the membership out: every family, document and question identifier, with a digest. Once
that file is **committed**, the hold-out is a fact in git history with a timestamp, and a reader who
does not trust the author can check any later score against it — including checking that the freeze
commit precedes the first commit carrying a score.

`--check` is what CI runs on every push. It recomputes and compares, and fails on any drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from parts_answer_gate.holdout import (  # noqa: E402
    HOLDOUT_RULE,
    HoldoutDriftError,
    compute,
    freeze,
    load_frozen,
    verify_partition,
)

DEFAULT_CORPUS = REPO_ROOT / "data" / "generated"
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"


def _load(corpus_dir: Path) -> tuple[list[dict], list[dict], dict[str, str]]:
    documents_path = corpus_dir / "documents.json"
    questions_path = corpus_dir / "questions.json"
    if not documents_path.is_file() or not questions_path.is_file():
        raise SystemExit(
            f"no corpus in {corpus_dir}. Run `python scripts/generate_corpus.py` first — it is "
            "rebuilt from a committed seed rather than kept in git."
        )
    documents = json.loads(documents_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    chunks_path = corpus_dir / "chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8")) if chunks_path.is_file() else []
    chunks = chunks["chunks"] if isinstance(chunks, dict) else chunks

    # The chunk -> document map, without which the leak check has nothing to resolve: the corpus
    # names supporting evidence per chunk, and a question's split has to be compared against the
    # split of the document that chunk belongs to.
    return (
        documents["documents"] if isinstance(documents, dict) else documents,
        questions["questions"] if isinstance(questions, dict) else questions,
        {str(c["chunk_id"]): str(c["document_id"]) for c in chunks},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if the committed freeze no longer matches the corpus",
    )
    parser.add_argument(
        "--allow-refreeze",
        action="store_true",
        help=(
            "deliberately re-draw the hold-out. This invalidates every score taken against the old "
            "membership; ADR-001 requires those to be re-run rather than carried over."
        ),
    )
    args = parser.parse_args(argv)

    documents, questions, chunk_to_document = _load(args.corpus)

    leaks = verify_partition(documents, questions, chunk_to_document)
    if leaks:
        print("the split does not partition cleanly:", file=sys.stderr)
        for line in leaks[:10]:
            print(f"  {line}", file=sys.stderr)
        return 2

    if args.check:
        existing = load_frozen(args.artifacts)
        if existing is None:
            print("artifacts/holdout.json does not exist; nothing is frozen", file=sys.stderr)
            return 1
        computed = compute(documents, questions)
        if existing.digest != computed.digest:
            print(
                "the committed hold-out no longer matches the corpus.\n"
                f"  frozen:   {len(existing.families)} families, {len(existing.documents)} docs, "
                f"{len(existing.questions)} questions, digest {existing.digest[:16]}\n"
                f"  computed: {len(computed.families)} families, {len(computed.documents)} docs, "
                f"{len(computed.questions)} questions, digest {computed.digest[:16]}\n"
                "Every score taken against the old membership is now measuring a different set.",
                file=sys.stderr,
            )
            return 1
        print(
            f"the hold-out is unchanged: {len(existing.families)} families, "
            f"{len(existing.documents)} documents, {len(existing.questions)} questions, "
            f"digest {existing.digest[:16]}"
        )
        return 0

    try:
        frozen = freeze(args.artifacts, documents, questions, allow_refreeze=args.allow_refreeze)
    except HoldoutDriftError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"hold-out frozen into {(args.artifacts / 'holdout.json').relative_to(REPO_ROOT)}")
    print(f"  rule       {HOLDOUT_RULE}")
    print(f"  families   {len(frozen.families)}  {', '.join(frozen.families)}")
    print(f"  documents  {len(frozen.documents)}")
    print(f"  questions  {len(frozen.questions)}")
    print(f"  digest     {frozen.digest}")
    print()
    print("Commit this file before running any evaluation. Its position in git history is the")
    print("evidence that the split was not chosen after seeing a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
