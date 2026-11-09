"""Every screen states the predeclared verdict, including when it cannot be read.

The banner is this console's headline disclosure: the original release gate failed. It used to
render only for FAILED and PASSED, so an instance whose `release_gate.json` was missing or
unparseable served every screen with no banner at all and logged nothing — the disclosure failing
open, which is the selective reporting the project exists to argue against.

These tests pin all four states on all three screens that carry the banner, assert that the FAILED
and PASSED markup is unchanged, and assert that an unavailable verdict is never dressed as a pass.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from parts_answer_gate.api import app as app_module
from parts_answer_gate.api.app import app, settings
from parts_answer_gate.config import get_settings

SCREENS = ("/", "/evidence", "/failures", "/provenance")
COMMITTED = Path(__file__).resolve().parents[1] / "artifacts"
GATE = "release_gate.json"
#: The class an unavailable verdict must never borrow: `ok` is what a cleared gate renders.
OK_BANNER_CLASS = 'class="gate-banner ok"'


def _client(tmp_path: Path, content: str | None) -> TestClient:
    """A client reading a copy of the committed artifacts, with the verdict file replaced."""
    artifacts = tmp_path / "artifacts"
    shutil.copytree(COMMITTED, artifacts, ignore=shutil.ignore_patterns(GATE))
    if content is not None:
        (artifacts / GATE).write_text(content, encoding="utf-8")
    app.dependency_overrides[settings] = lambda: replace(
        get_settings(), artifacts_dir=str(artifacts)
    )
    _forget_warned_states()
    return TestClient(app)


def _forget_warned_states() -> None:
    """Reset the once-per-process warning record, so each case starts from nothing reported."""
    app_module._GATE_STATES_WARNED.clear()


@pytest.fixture(autouse=True)
def _restore() -> object:
    yield
    app.dependency_overrides.pop(settings, None)
    _forget_warned_states()


def _committed_gate() -> str:
    return (COMMITTED / GATE).read_text(encoding="utf-8")


def _passed_gate() -> str:
    verdict = json.loads(_committed_gate())
    verdict.update({"release_gate": "PASSED", "failing": [], "vacuous_passes": []})
    return json.dumps(verdict)


#: Every way the verdict can be unavailable, and the state each one has to report. Parsing is not
#: understanding: JSON that is not a verdict leaves the console just as unable to show one.
UNAVAILABLE = [
    pytest.param(None, "NOT RUN", id="absent"),
    pytest.param("{ not json", "UNREADABLE", id="not-json"),
    pytest.param("{}", "UNREADABLE", id="empty-object"),
    pytest.param("null", "UNREADABLE", id="null"),
    pytest.param("[]", "UNREADABLE", id="list"),
    pytest.param('{"failing": []}', "UNREADABLE", id="no-state"),
    pytest.param('{"release_gate": "  "}', "UNREADABLE", id="blank-state"),
]


@pytest.mark.parametrize(("content", "state"), UNAVAILABLE)
def test_an_unreadable_verdict_is_named_on_every_screen(
    tmp_path: Path, content: str | None, state: str
) -> None:
    client = _client(tmp_path, content)
    for screen in SCREENS:
        body = client.get(screen).text
        assert f"ORIGINAL RELEASE GATE: {state}" in body, screen
        assert "Treat the verdict as unknown, not as a pass." in body, screen
        # An unknown verdict must not borrow the PASSED banner's styling.
        assert OK_BANNER_CLASS not in body, screen


@pytest.mark.parametrize(("content", "state"), UNAVAILABLE)
def test_an_unreadable_verdict_is_logged_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, content: str | None, state: str
) -> None:
    client = _client(tmp_path, content)
    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        for screen in SCREENS:
            client.get(screen)
    messages = [r.getMessage() for r in caplog.records if state in r.getMessage()]
    assert len(messages) == 1, messages
    assert str(tmp_path / "artifacts" / GATE) in messages[0]


def test_the_failed_banner_is_unchanged(tmp_path: Path) -> None:
    """The committed verdict is FAILED; its banner is what production serves today."""
    served = _client(tmp_path, _committed_gate()).get(SCREENS[0]).text
    assert "ORIGINAL RELEASE GATE: FAILED" in served
    assert "predeclared kill conditions do not hold" in served
    assert "Thresholds were fixed before any source file existed and none was lowered." in served
    assert "Treat the verdict as unknown, not as a pass." not in served
    assert OK_BANNER_CLASS not in served


def test_a_passing_verdict_still_renders_the_pass_banner(tmp_path: Path) -> None:
    served = _client(tmp_path, _passed_gate()).get(SCREENS[0]).text
    assert f"<div {OK_BANNER_CLASS}><strong>ORIGINAL RELEASE GATE: PASSED</strong></div>" in served
    assert "Treat the verdict as unknown, not as a pass." not in served


def test_the_committed_verdict_is_untouched_by_these_tests() -> None:
    """The evidence this console reports is read-only here; the thresholds are not ours to move."""
    verdict = json.loads(_committed_gate())
    assert verdict["release_gate"] == "FAILED"
    assert verdict["failing"] == ["E", "F", "I", "K"]
