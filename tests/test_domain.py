"""The two intervals, and the invariants that make the headline claim expressible at all.

The supersession tests are the ones that matter. A document with `superseded_by` and no `valid_to`
stays in force for ever, so an as-of query keeps returning a withdrawn bulletin — which is precisely
the failure the project exists to prevent, arriving through the data rather than through the query.
The model refuses to be constructed that way, so the failure cannot enter the system at all.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from parts_answer_gate.domain import (
    Answer,
    Chunk,
    Citation,
    Document,
    Effectivity,
    GateDecision,
    GateOutcome,
    GateSignals,
    Language,
    Query,
    RetrievedChunk,
    SerialRange,
    part_numbers_in,
)

MARCH = date(2026, 3, 15)
JUNE = date(2026, 6, 15)
SEPTEMBER = date(2026, 9, 15)


def _document(**overrides: object) -> Document:
    base: dict[str, object] = {
        "document_id": "DOC-A",
        "family_id": "FAM-1",
        "title": "XP-400 Service Manual",
        "language": Language.EN,
        "revision": "A",
        "valid_from": date(2026, 1, 1),
        "source_uri": "synthetic://xp-400/rev-a",
        "checksum": "0" * 64,
    }
    return Document(**(base | overrides))  # type: ignore[arg-type]


def _chunk(**overrides: object) -> Chunk:
    base: dict[str, object] = {
        "chunk_id": "CH-1",
        "document_id": "DOC-A",
        "family_id": "FAM-1",
        "language": Language.EN,
        "text": "Tighten the impeller retaining bolt HX-4410-B to 48 Nm.",
        "section": "4.2",
        "page": 17,
        "start_offset": 1200,
        "end_offset": 1255,
        "effectivity": Effectivity(variant_id="XP-400", serials=SerialRange(first=1000, last=4199)),
        "valid_from": date(2026, 1, 1),
    }
    return Chunk(**(base | overrides))  # type: ignore[arg-type]


class TestSupersession:
    def test_a_half_superseded_document_cannot_exist(self) -> None:
        """`superseded_by` without `valid_to` leaves a withdrawn revision in force for ever."""
        with pytest.raises(ValidationError, match="withdrawn revision needs both"):
            _document(superseded_by="DOC-B")

    def test_an_end_date_without_a_successor_is_also_refused(self) -> None:
        """The other direction. A revision that stopped applying and names no replacement is a gap
        in the corpus, not a fact about the world."""
        with pytest.raises(ValidationError, match="withdrawn revision needs both"):
            _document(valid_to=JUNE)

    def test_a_withdrawn_revision_is_still_in_force_before_its_end_date(self) -> None:
        """The reason revisions are withdrawn rather than deleted.

        A warranty dispute about an incident in March needs the procedure approved in March, and
        that procedure is in a document nobody should have deleted.
        """
        withdrawn = _document(valid_to=JUNE, superseded_by="DOC-B")
        assert withdrawn.in_force_on(MARCH) is True
        assert withdrawn.in_force_on(SEPTEMBER) is False

    def test_the_boundary_day_belongs_to_the_successor(self) -> None:
        """Half-open on purpose: `[valid_from, valid_to)`.

        Closed at both ends would make two revisions in force on the changeover day, and an as-of
        query on that day would legitimately return both — a conflict the corpus invented.
        """
        withdrawn = _document(valid_to=JUNE, superseded_by="DOC-B")
        assert withdrawn.in_force_on(date(2026, 6, 14)) is True
        assert withdrawn.in_force_on(JUNE) is False

    def test_a_current_revision_has_no_end(self) -> None:
        assert _document().in_force_on(SEPTEMBER) is True


class TestEffectivity:
    def test_a_procedure_for_one_variant_does_not_apply_to_another(self) -> None:
        """The right revision of the wrong machine is still the wrong answer."""
        effectivity = Effectivity(variant_id="XP-400")
        assert effectivity.applies_to("XP-400", None) is True
        assert effectivity.applies_to("XP-400S", None) is False

    @pytest.mark.parametrize(
        ("serial", "expected"),
        [(999, False), (1000, True), (2500, True), (4199, True), (4200, False)],
    )
    def test_the_serial_range_is_inclusive_at_both_ends(self, serial: int, expected: bool) -> None:
        effectivity = Effectivity(variant_id="XP-400", serials=SerialRange(first=1000, last=4199))
        assert effectivity.applies_to("XP-400", serial) is expected

    def test_an_open_ended_range_covers_everything_above_its_start(self) -> None:
        """`from serial 4200 onwards` is how a bulletin is actually written."""
        effectivity = Effectivity(variant_id="XP-400", serials=SerialRange(first=4200))
        assert effectivity.applies_to("XP-400", 4200) is True
        assert effectivity.applies_to("XP-400", 99_999) is True
        assert effectivity.applies_to("XP-400", 4199) is False

    def test_an_unknown_serial_is_covered_only_by_an_unbounded_range(self) -> None:
        """The safe direction, and the asymmetry is deliberate.

        A technician who did not say which machine they have must not be handed a procedure that
        applies to half the fleet.
        """
        bounded = Effectivity(variant_id="XP-400", serials=SerialRange(first=1000, last=4199))
        unbounded = Effectivity(variant_id="XP-400")
        assert bounded.applies_to("XP-400", None) is False
        assert unbounded.applies_to("XP-400", None) is True

    def test_an_inverted_range_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="inverted"):
            SerialRange(first=5000, last=1000)


class TestPartNumbers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Use HX-4410-B.", {"HX-4410-B"}),
            ("HX-4410-B supersedes HX-4410-A", {"HX-4410-B", "HX-4410-A"}),
            ("Torque to 48 Nm", set()),
            ("Part 4410 is not a part number", set()),
            ("lowercase hx-4410-b is not matched", set()),
        ],
    )
    def test_only_a_part_number_shaped_token_counts(self, text: str, expected: set[str]) -> None:
        """Strict on purpose.

        A loose pattern that also matched `48 Nm` would make kill condition C pass by never finding
        anything to check — the most comfortable way to satisfy a zero.
        """
        assert part_numbers_in(text) == expected


class TestTheGateDecision:
    def _signals(self, **overrides: object) -> GateSignals:
        base: dict[str, object] = {
            "top_fused_score": 0.9,
            "supporting_chunks": 2,
            "term_coverage": 0.8,
            "variant_agreement": True,
            "superseded_present": False,
            "conflicting_evidence": False,
            "exact_identifier_hit": False,
        }
        return GateSignals(**(base | overrides))  # type: ignore[arg-type]

    def test_an_abstention_cannot_carry_approved_evidence(self) -> None:
        """An answerer that received evidence for a refused question could be asked to use it."""
        retrieved = RetrievedChunk(chunk=_chunk(), fused_score=0.9, rank=1)
        with pytest.raises(ValidationError, match="must not be able to see evidence"):
            GateDecision(
                outcome=GateOutcome.ABSTAIN,
                reason="nothing found",
                signals=self._signals(),
                approved_chunks=(retrieved,),
            )

    def test_a_review_may_carry_evidence(self) -> None:
        """REVIEW is a person deciding between passages, so they need the passages."""
        retrieved = RetrievedChunk(chunk=_chunk(), fused_score=0.9, rank=1)
        decision = GateDecision(
            outcome=GateOutcome.REVIEW,
            reason="two variants disagree",
            signals=self._signals(variant_agreement=False),
            approved_chunks=(retrieved,),
        )
        assert decision.approved_chunks


class TestTheAnswer:
    def _decision(self, outcome: GateOutcome) -> GateDecision:
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

    def _query(self) -> Query:
        return Query(text="torque for the impeller bolt?", as_of=MARCH, variant_id="XP-400")

    def _citation(self, quote: str) -> Citation:
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

    def test_an_abstention_carries_no_text(self) -> None:
        with pytest.raises(ValidationError, match="abstention carries answer text"):
            Answer(
                query=self._query(),
                decision=self._decision(GateOutcome.ABSTAIN),
                text="48 Nm",
            )

    def test_an_answer_without_a_citation_is_refused(self) -> None:
        """Every factual answer cites. An uncited one is the thing this project exists against."""
        with pytest.raises(ValidationError, match="every factual answer must cite"):
            Answer(
                query=self._query(),
                decision=self._decision(GateOutcome.ANSWER),
                text="48 Nm",
            )

    def test_grounded_is_true_when_every_part_number_is_cited(self) -> None:
        answer = Answer(
            query=self._query(),
            decision=self._decision(GateOutcome.ANSWER),
            text="Tighten bolt HX-4410-B to 48 Nm.",
            citations=(self._citation("the impeller retaining bolt HX-4410-B to 48 Nm"),),
        )
        assert answer.grounded is True

    def test_grounded_is_false_when_a_part_number_is_not_in_any_quote(self) -> None:
        """Kill condition C, as a property of the object rather than a check somebody remembers."""
        answer = Answer(
            query=self._query(),
            decision=self._decision(GateOutcome.ANSWER),
            text="Tighten bolt HX-9999-Z to 48 Nm.",
            citations=(self._citation("the impeller retaining bolt HX-4410-B to 48 Nm"),),
        )
        assert answer.grounded is False

    def test_an_abstention_is_trivially_grounded(self) -> None:
        answer = Answer(query=self._query(), decision=self._decision(GateOutcome.ABSTAIN))
        assert answer.grounded is True

    def test_every_answer_carries_the_article_50_disclosure(self) -> None:
        """In force since 2 August 2026, and a property of the response rather than of the UI."""
        answer = Answer(query=self._query(), decision=self._decision(GateOutcome.ABSTAIN))
        assert "AI system" in answer.ai_disclosure
        assert answer.ai_disclosure


class TestTheQuery:
    def test_the_as_of_date_is_required(self) -> None:
        """No default, deliberately.

        `date.today()` buried in a model is a hidden clock, and a hidden clock makes a replayed
        evaluation irreproducible — the service fills it at the edge, where it is visible.
        """
        with pytest.raises(ValidationError):
            Query(text="anything")  # type: ignore[call-arg]

    def test_an_empty_question_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Query(text="", as_of=MARCH)
