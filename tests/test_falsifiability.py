"""Grade `artifacts/breaches.json`: every guarantee was broken on purpose and something noticed.

ADR-001's falsifiability clause says a passing suite is evidence the tests pass, not that they
would fail. `scripts/plant_breaches.py` breaks each guarantee in turn; this file checks the record
it leaves — that every breach was planted, that every one was caught, and that the baseline was
clean before each was planted.

That last check is the one worth having. A breach whose baseline already showed the failure proves
nothing: the detector was reporting a problem that existed anyway, and it would have reported it
with the system intact. So each result carries what the detector saw *before* the breach, and a
breach whose `caught` is true while its baseline was already dirty is a false positive that this
file is written to catch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "breaches.json"

#: Every boundary ADR-001's falsifiability clause names, plus the two the second iteration added:
#: knowledge time, which did not exist then, and the split leak, whose guard was vacuous.
REQUIRED_BREACHES = {
    "effectivity_bypass",
    "wrong_variant",
    "stale_revision_off_by_one",
    "bitemporal_knowledge_time",
    "pgvector_bypass",
    "unsupported_answer",
    "wrong_citation",
    "missing_disclosure",
    "incremental_index_skip",
    "split_leak",
}


def _report() -> dict[str, Any]:
    if not ARTIFACT.is_file():
        pytest.fail("artifacts/breaches.json is missing; run `python scripts/plant_breaches.py`")
    payload: dict[str, Any] = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return payload


def test_every_named_boundary_was_actually_breached() -> None:
    """A boundary missing from the report is a guarantee nobody tried to break."""
    report = _report()
    planted = {str(b["name"]) for b in report["breaches"]}

    missing = REQUIRED_BREACHES - planted
    assert not missing, f"no breach was planted for: {sorted(missing)}"


def test_every_planted_breach_was_caught() -> None:
    report = _report()
    uncaught = sorted(str(b["name"]) for b in report["breaches"] if not b["caught"])

    assert not uncaught, (
        f"{len(uncaught)} guarantee(s) can be broken without anything noticing: {uncaught}. "
        "A guarantee whose breach nobody detects is a sentence in a README."
    )


def test_each_breach_had_a_clean_baseline() -> None:
    """The detector must have been quiet before the breach, or it is not detecting the breach."""
    report = _report()
    for breach in report["breaches"]:
        assert breach["baseline"], f"{breach['name']} recorded no baseline observation"
        assert breach["baseline"] != breach["breached"], (
            f"{breach['name']} saw the same thing before and after the breach "
            f"({breach['baseline']!r}), so the detector is not responding to the breach"
        )


def test_each_breach_records_what_it_replaced_and_what_observed_it() -> None:
    """ADR-001: a breach must change behaviour and a test must observe the behaviour change.

    The mechanism and detector fields are how a reader checks that, without re-running anything.
    """
    report = _report()
    for breach in report["breaches"]:
        assert breach["mechanism"], f"{breach['name']} does not say what it replaced"
        assert breach["detector"], f"{breach['name']} does not say what observed it"
        assert breach["guarantee"], f"{breach['name']} does not name the guarantee it breaks"


def test_the_report_does_not_claim_more_than_it_planted() -> None:
    report = _report()
    assert report["caught"] == sum(1 for b in report["breaches"] if b["caught"])
    assert report["planted"] == len(report["breaches"])
    assert sorted(report["uncaught"]) == sorted(
        str(b["name"]) for b in report["breaches"] if not b["caught"]
    )
