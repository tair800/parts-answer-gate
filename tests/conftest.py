"""A session-level guard that asks pytest what it is about to run.

`test_predeclaration.py` checks the kill test by parsing it. Neither is enough alone: banning one
spelling of `importorskip` bans one spelling, and a `pytest.mark.skip` applied from a plugin or a
conftest never appears in the file's own source at all. So the real guard is here, and it inspects
collected items rather than text.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

KILL_TEST_FILE = "test_kill_criteria.py"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Refuse the whole session if the predeclared kill test has been disabled by a mark.

    `tryfirst` so this sees items **before** `-m` deselection removes them. Deselection is not
    disablement, and a lane that legitimately deselects these must still not be able to silence
    them. A `UsageError` rather than a failing test, because a failing test can be deselected too.
    """
    if importlib.util.find_spec("parts_answer_gate") is None:
        return

    kill = [item for item in items if Path(str(item.fspath)).name == KILL_TEST_FILE]
    if not kill:
        return

    disabled = sorted(
        item.nodeid
        for item in kill
        if item.get_closest_marker("skip") is not None
        or item.get_closest_marker("skipif") is not None
    )
    if disabled:
        raise pytest.UsageError(
            "the predeclared kill test has been disabled by a mark, so the criterion that decides "
            "whether this project ships is not being evaluated: " + ", ".join(disabled)
        )
