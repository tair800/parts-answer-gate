"""The failures screen shows each wrong answer's fields as text, never as Python values.

The questions about a product family the corpus does not contain are the only ones the question
builder leaves without a variant (`corpus/questions.py`), so the committed evidence records their
`asked_variant` as `null`; it records each answer's cited revisions as a list. Printed straight
into the template, those became the literal `None` and a list repr: the public page's only wrong
answer read as if a machine called "None" had been asked about, citing
`['fam-vb2-cooling-unit/E']`.

The screen is rendered from the committed artifacts, as the deployment serves it, with no
database. The experiment is closed and its evidence is frozen, so the wrong answer it lists is
always there to render.
"""

from __future__ import annotations

import html
import re

from fastapi.testclient import TestClient

from parts_answer_gate.api.app import app


def _wrong_answer_cells() -> list[list[str]]:
    """The visible text of every cell in the "answers that were wrong" table, row by row."""
    response = TestClient(app).get("/failures")
    assert response.status_code == 200
    table = response.text.partition("Answers that were wrong, in full")[2].partition("</table>")[0]
    rows = re.findall(r"<tr>.*?</tr>", table, re.DOTALL)
    cells = [re.findall(r"<td.*?>(.*?)</td>", row, re.DOTALL) for row in rows]
    return [
        [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).split()) for c in r] for r in cells if r
    ]


def test_no_wrong_answer_cell_prints_a_python_value() -> None:
    rows = _wrong_answer_cells()
    assert rows, "the committed evidence lists a wrong answer, and the screen must render it"
    for question, asked, cited, _reason in rows:
        assert not asked.startswith("None"), (question, asked)
        assert cited and "[" not in cited and "'" not in cited, (question, cited)


def test_the_committed_wrong_answer_says_what_was_asked_about() -> None:
    rows = {row[0]: row for row in _wrong_answer_cells()}
    _, asked, cited, _reason = rows["q-neg-otherfamily-14-tr"]
    assert asked == "product family not in the corpus as of 2025-09-11"
    assert cited == "fam-vb2-cooling-unit/E"
