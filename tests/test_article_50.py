"""EU AI Act Article 50, implemented exactly as scoped and no wider.

**What is in force and what this implements.** Article 50's transparency obligation applies from
2 August 2026, with a grace period to 2 December 2026, and it applies here because this surface
talks to end users: a technician receives machine-generated text. The obligation is that they are
told so. That is the whole of it, and it is the whole of what this project implements.

**What this is not.** Implementing a transparency disclosure asserts no conformity with any other
part of the AI Act, with any other instrument, or with anything at all beyond the sentence it puts
on a response. The project makes no broader compliance claim anywhere, and
`test_no_broader_compliance_claim_is_made` asserts that the repository's own prose does not drift
into one.

**Why it lives on the response object.** The obligation follows the content, not the page the
content is drawn on. A disclosure implemented in a Jinja template is absent the moment somebody
calls the JSON API, adds a second surface, or embeds an answer somewhere else — and each of those
is a normal thing to do that would quietly drop the obligation. Putting it on `Answer` means every
consumer gets it and a test can assert it cannot be removed.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from parts_answer_gate.domain import (
    Answer,
    Citation,
    GateDecision,
    GateOutcome,
    GateSignals,
    Query,
)

ROOT = Path(__file__).resolve().parents[1]

#: The only claim this project makes about the AI Act. Anything stronger in the repository's
#: prose is a compliance assertion nobody is entitled to make, and the last test looks for it.
_OVERCLAIMS = (
    "ai act compliant",
    "fully compliant",
    "gdpr compliant",
    "certified",
    "conformity assessment passed",
    "legally compliant",
    "guarantees compliance",
)


def _decision(outcome: GateOutcome) -> GateDecision:
    return GateDecision(
        outcome=outcome,
        reason="test",
        signals=GateSignals(
            top_fused_score=0.9,
            supporting_chunks=2,
            term_coverage=0.9,
            variant_agreement=True,
            superseded_present=False,
            conflicting_evidence=False,
            exact_identifier_hit=True,
        ),
    )


def _query() -> Query:
    return Query(text="torque for the impeller bolt?", as_of=date(2026, 3, 15))


def _citation() -> Citation:
    quote = "Tighten the impeller retaining bolt HX-4410-B to 48 Nm."
    return Citation(
        chunk_id="CH-1",
        document_id="DOC-A",
        revision="A",
        section="4.2",
        page=17,
        quote=quote,
        start_offset=1200,
        end_offset=1200 + len(quote),
    )


class TestTheDisclosureIsOnTheResponse:
    def test_an_answer_carries_it(self) -> None:
        answer = Answer(
            query=_query(),
            decision=_decision(GateOutcome.ANSWER),
            text="Tighten bolt HX-4410-B to 48 Nm.",
            citations=(_citation(),),
        )
        assert answer.ai_disclosure

    def test_an_abstention_carries_it_too(self) -> None:
        """A refusal is machine-generated text as much as an answer is.

        It is the easy one to forget, because it feels like the absence of output rather than
        output — and it is still a sentence a person reads and acts on.
        """
        answer = Answer(query=_query(), decision=_decision(GateOutcome.ABSTAIN))
        assert answer.ai_disclosure

    def test_a_review_carries_it_too(self) -> None:
        answer = Answer(query=_query(), decision=_decision(GateOutcome.REVIEW))
        assert answer.ai_disclosure

    def test_it_says_the_text_was_machine_generated(self) -> None:
        """The substance of the obligation, not merely the presence of a field."""
        disclosure = Answer(query=_query(), decision=_decision(GateOutcome.ABSTAIN)).ai_disclosure
        assert re.search(r"\bAI\b", disclosure)
        assert "generated" in disclosure.lower()

    def test_it_tells_the_reader_what_to_do_with_that(self) -> None:
        """A disclosure that only labels is a label. This one says to verify against the cited
        revision, which is the action the domain actually needs."""
        disclosure = Answer(query=_query(), decision=_decision(GateOutcome.ABSTAIN)).ai_disclosure
        assert "erify" in disclosure


class TestItCannotBeRemoved:
    def test_blanking_it_is_refused(self) -> None:
        """The structural half. A field with a good default that any caller can empty is a default,
        not an obligation."""
        with pytest.raises(ValidationError):
            Answer(
                query=_query(),
                decision=_decision(GateOutcome.ABSTAIN),
                ai_disclosure="",
            )

    def test_it_cannot_be_mutated_after_construction(self) -> None:
        """`Answer` is frozen, so a middleware cannot strip the disclosure on the way out."""
        answer = Answer(query=_query(), decision=_decision(GateOutcome.ABSTAIN))
        with pytest.raises(ValidationError):
            answer.ai_disclosure = ""  # type: ignore[misc]

    def test_it_survives_serialisation(self) -> None:
        """The JSON API is a consumer too, and the obligation follows the content."""
        answer = Answer(query=_query(), decision=_decision(GateOutcome.ABSTAIN))
        assert answer.model_dump()["ai_disclosure"]
        assert "ai_disclosure" in answer.model_dump_json()


def test_no_broader_compliance_claim_is_made() -> None:
    """The scope guard.

    Implementing one in-force transparency obligation is a narrow, checkable thing. Describing the
    project as compliant with an instrument is a legal assertion, and this repository is not
    entitled to make one. Reads the prose rather than trusting that nobody will write it.
    """
    offenders: list[str] = []
    for path in (*ROOT.glob("*.md"), *(ROOT / "docs").glob("**/*.md")):
        text = path.read_text(encoding="utf-8").lower()
        offenders.extend(
            f"{path.relative_to(ROOT)}: {phrase!r}" for phrase in _OVERCLAIMS if phrase in text
        )
    assert not offenders, (
        "the repository claims a compliance status it is not entitled to claim. Article 50 "
        "transparency is implemented; nothing else is asserted. Offenders: " + "; ".join(offenders)
    )
