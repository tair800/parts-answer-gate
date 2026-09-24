"""Run the predeclared kill test and report the release gate, without laundering the result.

    python scripts/release_gate.py

**The original release gate FAILED.** Four of ADR-001's twelve kill conditions do not hold and a
fifth passes for a reason that makes the pass nearly worthless, and this project is closed as a
pre-registered negative result rather than as a shipped system.

That fact creates a problem this script exists to solve, and the problem is not cosmetic. If the
kill test simply runs inside the ordinary suite, the repository's build is red for ever — and a
permanently red build is one nobody reads, which would eventually hide a *regression* behind the
disclosed failure. But the obvious alternative is worse: marking the three failures `xfail` or
deleting them converts a failed criterion into an accepted pass, which is the single thing ADR-001
forbids and the whole reason this project is interesting.

So neither. The failures are **pinned**. `DISCLOSED_FAILURES` below records exactly which
conditions fail and why, and this script exits non-zero whenever reality diverges from that
record in *either* direction:

- a condition that was passing starts failing — a genuine regression, and the build goes red;
- a condition recorded as failing starts passing — which is good news, and still fails the build,
  because the record is now wrong and somebody must decide whether the fix was legitimate or
  whether the hold-out was tuned against.

The engineering build stays green while the software is working. The release gate stays FAILED
because it failed. Nothing is converted into anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# `scripts/` is not a package, so the sibling is loaded by path rather than imported by name.
# The mapping of condition letters to test names lives in one place and this is that place's
# only consumer; duplicating it here is how the two would come to disagree.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from readme_numbers import KILL_CONDITIONS  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"

#: Which conditions fail, and the one-line reason each fails. Pinned so that a regression and a
#: disclosed failure cannot be mistaken for one another. Argued in full in DECISIONS.md ADR-003.
DISCLOSED_FAILURES: dict[str, str] = {
    "E": (
        "the gate answers 47 of 306 unsupportable questions. The weakness is concentrated: it "
        "refuses 44 of 45 questions about a product family that does not exist, and only 17 of 45 "
        "where the product exists and the attribute does not"
    ),
    "F": (
        "the ungated_rag baseline removes only the gate and so runs the identical retriever, "
        "making 'above every baseline' on a retrieval metric impossible rather than hard; and "
        "recall@10 is the wrong measure of what effectivity filtering buys, because removing the "
        "predicate enlarges the candidate pool"
    ),
    "I": (
        "abstention on the unanswerable set is 0.8464 against a floor of 0.90. Same mechanism as "
        "E: gate.term_is_covered approximates stemming with a bidirectional prefix match and errs "
        "towards covering, and attribute_absent_for_existing_product is where it costs most"
    ),
    "K": (
        "the query plan does not mention a vector index. pgvector 0.8.6 is installed, the column "
        "is a real `vector`, and the executed statement uses `<=>` -- but the effectivity filter "
        "has already reduced the candidate set to 21 rows, for which PostgreSQL correctly prefers "
        "a sequential scan: forcing the ANN plan measures 5.6ms against the planner's 3.1ms. The "
        "criterion asked for an index scan on a query that should not have one"
    ),
}

#: Conditions that pass, where the pass means less than it appears to. Recorded because a verdict
#: table printing only PASS would be true and misleading at once.
VACUOUS_PASSES: dict[str, str] = {
    "G": (
        "near-vacuous. For the 297 hold-out questions that name a variant -- 297 of 312 -- the "
        "numerator is empty by construction: the effectivity predicate excludes revision- and "
        "variant-incorrect passages upstream of the gate, so no answer the system can give for "
        "those is capable of being wrong in either sense. The measured 0.0047 comes entirely from "
        "the 15 questions about a product family the corpus does not contain, where the filter has "
        "nothing to constrain. G measures the gate only on the class of question the filter cannot "
        "help with"
    ),
}


def _run() -> dict[str, str]:
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            str(REPO_ROOT / "tests" / "test_kill_criteria.py"),
            "-q",
            "--no-header",
            "--tb=no",
            "-rA",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    output = completed.stdout + completed.stderr
    verdicts: dict[str, str] = {}
    for letter, _, test in KILL_CONDITIONS:
        if f"PASSED tests/test_kill_criteria.py::{test}" in output:
            verdicts[letter] = "PASS"
        elif f"FAILED tests/test_kill_criteria.py::{test}" in output:
            verdicts[letter] = "FAIL"
        else:
            verdicts[letter] = "NOT RUN"
    if any(v == "NOT RUN" for v in verdicts.values()):
        print(output[-4000:], file=sys.stderr)
    return verdicts


def main() -> int:
    verdicts = _run()
    failing = {letter for letter, verdict in verdicts.items() if verdict == "FAIL"}
    not_run = {letter for letter, verdict in verdicts.items() if verdict == "NOT RUN"}

    width = max(len(description) for _, description, _ in KILL_CONDITIONS)
    print()
    for letter, description, _ in KILL_CONDITIONS:
        verdict = verdicts[letter]
        note = ""
        if letter in VACUOUS_PASSES and verdict == "PASS":
            verdict = "PASS BUT VACUOUS"
            note = "  (see ADR-003)"
        print(f"  {letter}  {description:<{width}}  {verdict}{note}")

    print()
    print("  ORIGINAL RELEASE GATE: FAILED" if failing else "  ORIGINAL RELEASE GATE: PASSED")
    print(f"  {len(failing)} of {len(KILL_CONDITIONS)} predeclared kill conditions do not hold.")
    print()

    payload = {
        "release_gate": "FAILED" if failing else "PASSED",
        "conditions": {
            letter: {
                "description": description,
                "verdict": verdicts[letter],
                "vacuous": letter in VACUOUS_PASSES and verdicts[letter] == "PASS",
                "why": VACUOUS_PASSES.get(letter) or DISCLOSED_FAILURES.get(letter),
            }
            for letter, description, _ in KILL_CONDITIONS
        },
        "failing": sorted(failing),
        "disclosed_failures": sorted(DISCLOSED_FAILURES),
        "vacuous_passes": sorted(VACUOUS_PASSES),
        "note": (
            "The thresholds in tests/test_kill_criteria.py were fixed before any source file "
            "existed and none has been lowered. Nothing here marks a failure as accepted: the "
            "failing set is pinned, and this script exits non-zero if reality diverges from it in "
            "either direction."
        ),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "release_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    if not_run:
        print(
            f"  {sorted(not_run)} did not run at all; the gate cannot be reported", file=sys.stderr
        )
        return 2

    regressions = sorted(failing - set(DISCLOSED_FAILURES))
    recovered = sorted(set(DISCLOSED_FAILURES) - failing)

    if regressions:
        print(
            f"  REGRESSION: {regressions} now fail and are not in the disclosed set. A condition "
            "that was holding has stopped holding.",
            file=sys.stderr,
        )
    if recovered:
        print(
            f"  {recovered} now pass but are recorded as failing. That is good news and it still "
            "fails this check, because somebody has to decide whether the fix was legitimate or "
            "whether the hold-out was tuned against — and then update DISCLOSED_FAILURES and "
            "DECISIONS.md.",
            file=sys.stderr,
        )
    return 1 if (regressions or recovered) else 0


if __name__ == "__main__":
    raise SystemExit(main())
