"""Refuse a build whose committed evidence describes a corpus that no longer exists.

    python scripts/check_artifacts_current.py

Every artifact carries a provenance block naming the hold-out it was scored against. This compares
that against the hold-out currently frozen, and against the set of artifacts the kill test requires.
Three ways the evidence can go stale, and all three have happened in this repository:

- **an artifact is missing.** `pgvector.json`, `storage_comparison.json` and `index_lifecycle.json`
  were absent for the whole of the first iteration while helper functions that could have produced
  them sat uncalled, and the numbers quoted for them came from scripts run by hand. ADR-002.
- **an artifact is stale.** The first benchmark's seven score artifacts described a corpus that had
  been regenerated underneath them. A reader comparing two files could not tell which corpus either
  one measured.
- **an artifact scored a different question set.** The likeliest version of this is the quietest:
  one file rebuilt, the rest left behind, and a report assembled from both.

This does not rebuild anything. A check that took twenty minutes would not run in a pre-commit hook
and would be deleted within a week; this one reads nine files and compares two strings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from parts_answer_gate.holdout import load_frozen  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"

#: Every artifact `tests/test_kill_criteria.py` loads, plus the two the second iteration added.
#: Kept here rather than imported from the test, because the test is predeclared and must not gain
#: a dependency on anything written after it.
REQUIRED = (
    "corpus.json",
    "determinism.json",
    "effectivity.json",
    "evaluation.json",
    "gate.json",
    "groundedness.json",
    "index_lifecycle.json",
    "multilingual.json",
    "pgvector.json",
    "storage_comparison.json",
)

#: Produced by their own scripts rather than by `build_artifacts.py`, so they are checked for
#: existence but not for provenance.
SIDECARS = ("breaches.json", "candidate_profile.json")


def main() -> int:
    frozen = load_frozen(ARTIFACTS)
    if frozen is None:
        print("artifacts/holdout.json is missing; nothing is frozen", file=sys.stderr)
        return 1

    problems: list[str] = []

    missing = [name for name in REQUIRED if not (ARTIFACTS / name).is_file()]
    for name in missing:
        problems.append(f"{name} is missing; run `make artifacts`")

    for name in SIDECARS:
        if not (ARTIFACTS / name).is_file():
            problems.append(f"{name} is missing; run `make breaches` / `make candidates`")

    for name in REQUIRED:
        path = ARTIFACTS / name
        if not path.is_file():
            continue
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.get("provenance")
        if provenance is None:
            # corpus.json is written by the generator and carries no provenance block; it is
            # checked by its own split rule instead.
            if name == "corpus.json":
                continue
            problems.append(f"{name} carries no provenance block")
            continue
        digest = provenance.get("holdout_digest")
        if digest is None:
            problems.append(f"{name} does not record which hold-out it scored")
        elif digest != frozen.digest:
            problems.append(
                f"{name} was scored against hold-out {str(digest)[:16]}, but the frozen hold-out "
                f"is {frozen.digest[:16]}"
            )

    if problems:
        print("the committed evidence is not current:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(
        f"all {len(REQUIRED)} graded artifacts and {len(SIDECARS)} sidecars are present and were "
        f"scored against hold-out {frozen.digest[:16]} "
        f"({len(frozen.questions)} questions, {len(frozen.families)} families)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
