"""The Ask screen's date inputs are named by the labels a sighted user reads.

`valid at` and `known at` are the visible labels of the two date inputs, and each input sits inside
its `<label>`. An `aria-label` on the input replaced that label as the accessible name with the long
explanatory sentence, so assistive technology named the field with the sentence, and a voice-control
user who says "valid at" may not be able to target it (WCAG 2.5.3, Label in Name). The sentence
belongs in the description, which the input's `title` supplies, and which is also its tooltip.

The accessible name and description are computed from the rendered page rather than matched as
text, by the steps of the W3C Accessible Name and Description Computation 1.2 that HTML-AAM gives a
date input: `aria-labelledby`, then `aria-label`, then the `<label>` elements whose labeled control
it is, then `title`; the description comes from `aria-describedby`, else from `title` when `title`
did not give the name. Only those steps are implemented, and the synthetic cases at the end pin each
of them. The screen is rendered as an instance without a database serves it.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from parts_answer_gate.api.app import app
from parts_answer_gate.store.engine import DATABASE_URL_ENV

# One piece of markup: a comment, a declaration, a whole script or style element, or a tag.
# Attribute values may hold ">" inside their quotes.
_MARKUP = (
    r"<!--.*?-->|<![^>]*>|<(script|style)\b[^>]*>.*?</\1\s*>"
    r"|<(/?)([A-Za-z][\w-]*)((?:[^>\"']|\"[^\"]*\"|'[^']*')*)>"
)
_ATTRIBUTE = r"([^\s=/>]+)(?:\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+))?"
_VOID = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "source",
        "track",
        "wbr",
    }
)
# The elements a <label> can label (HTML's "labelable elements"); a hidden input is not one.
_LABELABLE = frozenset({"button", "input", "meter", "output", "progress", "select", "textarea"})

# name -> (visible label, explanation)
DATE_INPUTS = {
    "as_of": ("valid at", "What was true of the machine on this date."),
    "known_as_of": ("known at", "What we knew on this date. Leave blank for what we know now."),
}


@dataclass(eq=False, repr=False)
class Element:
    tag: str
    attributes: dict[str, str]
    parent: Element | None = None
    children: list[Element | str] = field(default_factory=list)

    def descendants(self) -> Iterator[Element]:
        for child in self.children:
            if isinstance(child, Element):
                yield child
                yield from child.descendants()

    def text(self, leave_out: Element | None = None) -> str:
        """The text content, without `leave_out`: the control a label is naming."""
        return "".join(
            child if isinstance(child, str) else child.text(leave_out)
            for child in self.children
            if child is not leave_out
        )


def _unquote(value: str | None) -> str:
    """An attribute's value: its delimiting quotes removed, its character references decoded."""
    if value is None:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return html.unescape(value)


def parse(page: str) -> Element:
    """The page's elements and text as a tree, enough to find a control and its labels."""
    document = Element("#document", {})
    current = document
    position = 0
    for piece in re.finditer(_MARKUP, page, re.DOTALL | re.IGNORECASE):
        current.children.append(html.unescape(page[position : piece.start()]))
        position = piece.end()
        closing, tag, raw = piece.group(2), piece.group(3), piece.group(4)
        if tag is None:  # a comment, a declaration, or script or style content
            continue
        tag = tag.lower()
        if closing:
            opened: Element | None = current
            while opened is not None and opened.tag != tag:
                opened = opened.parent
            if opened is not None and opened.parent is not None:
                current = opened.parent
            continue
        attributes: dict[str, str] = {}
        for match in re.finditer(_ATTRIBUTE, raw):
            attributes.setdefault(match.group(1).lower(), _unquote(match.group(2)))
        element = Element(tag, attributes, current)
        current.children.append(element)
        if tag not in _VOID and not raw.rstrip().endswith("/"):
            current = element
    current.children.append(html.unescape(page[position:]))
    return document


def _flat(text: str) -> str:
    return " ".join(text.split())


def _by_id(document: Element, ident: str) -> Element | None:
    """The first element with this id, the one a browser resolves it to. An empty id is none."""
    if not ident:
        return None
    return next((e for e in document.descendants() if e.attributes.get("id") == ident), None)


def _referenced(document: Element, ids: str) -> list[Element]:
    """The elements an `aria-labelledby` or `aria-describedby` value names, in its order."""
    found = (_by_id(document, ident) for ident in ids.split())
    return [element for element in found if element is not None]


def _labelable(element: Element) -> bool:
    return element.tag in _LABELABLE and not (
        element.tag == "input" and element.attributes.get("type", "").lower() == "hidden"
    )


def _labeled_control(document: Element, label: Element) -> Element | None:
    """A label's labeled control: the element its `for` names, else its first labelable child."""
    if "for" in label.attributes:
        target = _by_id(document, label.attributes["for"])
        return target if target is not None and _labelable(target) else None
    return next((e for e in label.descendants() if _labelable(e)), None)


def _labels(document: Element, control: Element) -> list[Element]:
    """The `<label>` elements whose labeled control is `control`, in document order."""
    return [
        e
        for e in document.descendants()
        if e.tag == "label" and _labeled_control(document, e) is control
    ]


def accessible_name(document: Element, control: Element) -> tuple[str, str]:
    """A date input's accessible name, and the attribute or element that gave it."""
    labelled_by = _referenced(document, control.attributes.get("aria-labelledby", ""))
    if name := _flat(" ".join(e.text() for e in labelled_by)):
        return name, "aria-labelledby"
    if name := _flat(control.attributes.get("aria-label", "")):
        return name, "aria-label"
    if name := _flat(" ".join(label.text(control) for label in _labels(document, control))):
        return name, "label"
    if name := _flat(control.attributes.get("title", "")):
        return name, "title"
    return "", ""


def accessible_description(document: Element, control: Element, name_from: str) -> str:
    """`aria-describedby`'s text, else the `title`, unless the `title` gave the name."""
    described_by = _referenced(document, control.attributes.get("aria-describedby", ""))
    if description := _flat(" ".join(e.text() for e in described_by)):
        return description
    return "" if name_from == "title" else _flat(control.attributes.get("title", ""))


@pytest.fixture(scope="module")
def ask_screen() -> Element:
    """The Ask screen as an instance without a database serves it, whatever this run configures."""
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv(DATABASE_URL_ENV, raising=False)
        response = TestClient(app).get("/")
    assert response.status_code == 200
    return parse(response.text)


def _date_input(screen: Element, name: str) -> Element:
    found = [
        e for e in screen.descendants() if e.tag == "input" and e.attributes.get("name") == name
    ]
    assert len(found) == 1, f"expected one input named {name!r}, found {len(found)}"
    assert found[0].attributes.get("type") == "date"
    return found[0]


DATE_CASES = [
    pytest.param(name, label, explanation, id=name)
    for name, (label, explanation) in DATE_INPUTS.items()
]


@pytest.mark.parametrize(("name", "label", "explanation"), DATE_CASES)
def test_a_date_input_is_named_by_its_visible_label(
    ask_screen: Element, name: str, label: str, explanation: str
) -> None:
    control = _date_input(ask_screen, name)
    computed, name_from = accessible_name(ask_screen, control)
    assert (computed, name_from) == (label, "label")
    assert accessible_description(ask_screen, control, name_from) == explanation


@pytest.mark.parametrize(("name", "label", "explanation"), DATE_CASES)
def test_a_date_input_keeps_its_explanation_as_a_tooltip(
    ask_screen: Element, name: str, label: str, explanation: str
) -> None:
    assert _date_input(ask_screen, name).attributes.get("title") == explanation


# Each step above on minimal markup, as id: (markup, name, where the name came from, description).
# The control is always the input named "x".
STEPS = {
    "wrapping-label": ('<label>shown <input name="x"></label>', "shown", "label", ""),
    "label-for": ('<label for="i">shown</label><input id="i" name="x">', "shown", "label", ""),
    "label-for-counts-once": ('<label for="i">s <input id="i" name="x"></label>', "s", "label", ""),
    "first-labelable-only": (
        '<label>s <input name="y"><input name="x" title="T"></label>',
        "T",
        "title",
        "",
    ),
    "hidden-not-labelable": (
        '<label>s <input type="HIDDEN" name="y"><input name="x"></label>',
        "s",
        "label",
        "",
    ),
    "first-repeated-attribute": (
        '<label>s <input name="x" aria-label="a" aria-label="b"></label>',
        "a",
        "aria-label",
        "",
    ),
    "aria-label-before-label": (
        '<label>s <input name="x" aria-label="said"></label>',
        "said",
        "aria-label",
        "",
    ),
    "labelledby-first-repeated-id": (
        '<i id="a">one</i><i id="a">two</i>'
        '<label>s <input name="x" aria-labelledby="a" aria-label="b"></label>',
        "one",
        "aria-labelledby",
        "",
    ),
    "title-is-description": (
        '<label>s <input name="x" title="the \'y\'"></label>',
        "s",
        "label",
        "the 'y'",
    ),
    "describedby-before-title": (
        '<p id="d">more</p><label>s <input name="x" aria-describedby="d" title="T"></label>',
        "s",
        "label",
        "more",
    ),
    "title-names-unlabelled": ('<input name="x" title="T">', "T", "title", ""),
}


@pytest.mark.parametrize(
    ("markup", "name", "name_from", "description"),
    [pytest.param(*case, id=step) for step, case in STEPS.items()],
)
def test_each_step_of_the_computation(
    markup: str, name: str, name_from: str, description: str
) -> None:
    document = parse(markup)
    control = next(e for e in document.descendants() if e.attributes.get("name") == "x")
    assert accessible_name(document, control) == (name, name_from)
    assert accessible_description(document, control, name_from) == description
