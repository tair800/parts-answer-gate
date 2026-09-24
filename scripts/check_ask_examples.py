"""Refuse a build whose demo links the deployment cannot answer.

    python scripts/check_ask_examples.py

The public instance serves query vectors computed at build time, because the encoder needs about
671MB and the free tier provides 512 (`retrieval.precomputed`). That covers every question the
corpus contains and nothing outside it.

Which makes the Ask screen's example links load-bearing in a way they were not before. They are the
first thing anyone clicks, and the day query-vector caching was deployed all four of them were
hand-written strings that no longer resolved: a visitor arriving at the demo got the panel
explaining what the instance cannot do, four times, and never saw the system answer anything. The
deployment was correct, the evidence was correct, and the front page was useless.

So the examples are corpus questions, and this says so out of the corpus rather than out of a
comment. It runs in the evidence lane, where the corpus exists.

It checks the question text only. The variant, the dates and the language are deliberately free:
those are arguments to the SQL predicate rather than to the embedding, and varying them is the whole
demonstration — the same question at two dates is two different revisions and two different figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from parts_answer_gate.api.app import _EXAMPLE_QUERIES  # noqa: E402

DEFAULT_CORPUS = REPO_ROOT / "data" / "generated"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args(argv)

    path = args.corpus / "questions.json"
    if not path.is_file():
        print(f"no corpus in {args.corpus}; run scripts/generate_corpus.py", file=sys.stderr)
        return 1

    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    questions = raw["questions"] if isinstance(raw, dict) else raw
    known = {str(question["text"]) for question in questions}

    missing = [
        (label, params["q"]) for label, params in _EXAMPLE_QUERIES if params["q"] not in known
    ]
    for label, _ in _EXAMPLE_QUERIES:
        print(f"  {'FAIL' if label in {name for name, _ in missing} else 'ok  '} {label}")

    if missing:
        print(
            f"\n{len(missing)} example question(s) are not in the corpus, so the deployed instance "
            "has no precomputed vector for them and the demo link shows a refusal panel:",
            file=sys.stderr,
        )
        for label, text in missing:
            print(f"  {label}: {text!r}", file=sys.stderr)
        return 1

    print(f"\nall {len(_EXAMPLE_QUERIES)} Ask examples are questions the corpus contains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
