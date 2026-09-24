"""Guards on the predeclared kill test itself.

Projects 4, 5 and 6 each learned a version of the same lesson: a kill test that can disable itself
is decoration, and the ways it disables itself are not always visible in its source. So this file
parses `test_kill_criteria.py` rather than grepping it, and `conftest.py` asks pytest what it is
about to run rather than trusting either.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

KILL_TEST = Path(__file__).resolve().parent / "test_kill_criteria.py"

#: Every way a test can decline to run that has been seen in this portfolio.
_SKIPPING = {
    "pytest.skip",
    "pytest.importorskip",
    "pytest.xfail",
    "pytest.mark.skip",
    "pytest.mark.skipif",
    "pytest.mark.xfail",
    "unittest.skip",
}


def _tree() -> ast.Module:
    return ast.parse(KILL_TEST.read_text(encoding="utf-8"))


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_the_thresholds_are_module_level_constants() -> None:
    """A threshold computed at run time is a threshold that can be computed from the result."""
    assignments = {
        target.id
        for node in _tree().body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    required = {
        "MAX_SUPERSEDED_RETURNED",
        "MAX_WRONG_VARIANT_RETURNED",
        "MAX_UNGROUNDED_PART_NUMBERS",
        "MAX_UNFAITHFUL_CITATIONS",
        "MAX_ANSWERED_WITHOUT_SUPPORT",
        "MIN_HOLDOUT_RECALL_AT_10",
        "MAX_HOLDOUT_WRONG_ANSWER_RATE",
        "MIN_UNGATED_WRONG_ANSWER_MULTIPLE",
        "MIN_ABSTENTION_ON_UNANSWERABLE",
    }
    missing = required - assignments
    assert not missing, f"these thresholds are no longer module-level constants: {sorted(missing)}"


def test_the_zero_thresholds_are_actually_zero() -> None:
    """Read out of the AST, so a value edited to 1 is caught even if every test still passes."""
    values = {
        target.id: node.value.value
        for node in _tree().body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for name in (
        "MAX_SUPERSEDED_RETURNED",
        "MAX_WRONG_VARIANT_RETURNED",
        "MAX_UNGROUNDED_PART_NUMBERS",
        "MAX_UNFAITHFUL_CITATIONS",
        "MAX_ANSWERED_WITHOUT_SUPPORT",
        "MAX_SPLIT_LEAKS",
    ):
        assert values[name] == 0, f"{name} is {values[name]}; ADR-001 fixes it at 0"


def test_the_declared_floors_have_not_been_lowered() -> None:
    """ADR-001 permits raising a threshold and forbids lowering one."""
    values = {
        target.id: node.value.value
        for node in _tree().body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert values["MIN_HOLDOUT_RECALL_AT_10"] >= 0.85
    assert values["MAX_HOLDOUT_WRONG_ANSWER_RATE"] <= 0.02
    assert values["MIN_UNGATED_WRONG_ANSWER_MULTIPLE"] >= 5.0
    assert values["MIN_ABSTENTION_ON_UNANSWERABLE"] >= 0.90
    assert values["MIN_CHUNKS"] >= 1_500
    assert values["MIN_QUESTIONS"] >= 300
    assert values["MIN_UNANSWERABLE"] >= 90


def test_the_kill_test_does_not_import_the_implementation() -> None:
    """It grades artifacts. A kill test that imports the code can be satisfied by editing it."""
    imported = {
        node.module
        for node in ast.walk(_tree())
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("parts_answer_gate" in name for name in imported), (
        f"the kill test imports {sorted(imported)}; it must read artifacts/ and nothing else"
    )


def test_the_predeclaration_skip_does_not_outlive_the_package() -> None:
    """Once `parts_answer_gate` imports, the kill test must actually run."""
    if importlib.util.find_spec("parts_answer_gate") is None:
        pytest.skip("the package does not exist yet; the kill test is still predeclared")

    found: list[str] = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) and (name := _dotted(node.func)) in _SKIPPING:
            found.append(f"{name}() at line {node.lineno}")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if (name := _dotted(target)) in _SKIPPING:
                    found.append(f"@{name} on {node.name}")

    assert not found, (
        "parts_answer_gate is importable, so nothing in the kill test may disable itself. "
        f"Found: {', '.join(found)}."
    )
