"""The two arms, and the rule that outlives both of them.

Claim 3 of ADR-001 — a part number the evidence does not contain cannot appear in an answer — is
tested here twice over, because it is held twice over. The extractive arm cannot break it, so those
tests assert a structural property. The post-validator can catch an arm that does break it, so that
one is tested against a planted breach: an answerer that invents an identifier, which must be
rejected rather than returned.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import pytest

from parts_answer_gate import gate
from parts_answer_gate.answerer import (
    ANSWER_SEPARATOR,
    AbstractiveAnswerer,
    AbstractiveArmUnavailableError,
    Answerer,
    EmptyEvidenceError,
    ExtractiveAnswerer,
    PostValidatedAnswerer,
    UngroundedAnswerError,
    post_validate,
    ungrounded_part_numbers,
)
from parts_answer_gate.citations import build_citations
from parts_answer_gate.domain import (
    Answer,
    Chunk,
    Effectivity,
    GateDecision,
    GateOutcome,
    Language,
    Query,
    RetrievedChunk,
    SerialRange,
    part_numbers_in,
)

AS_OF = date(2026, 3, 1)

TORQUE_EN = (
    "Tighten the drive coupling bolts on the XP-400 to 48 Nm in a cross pattern. "
    "Use only the AB-1234-C fastener kit; a lower grade bolt will not hold this value."
)
TORQUE_EN_SECOND_SOURCE = (
    "The drive coupling bolts on the XP-400 are torqued to 48 Nm. "
    "Replace the CD-5678-K gasket at every second service."
)
TORQUE_EN_CONFLICTING = (
    "Tighten the drive coupling bolts on the XP-400 to 52 Nm in a cross pattern. "
    "Recheck the torque after the first 50 operating hours."
)

REVISIONS = {f"DOC-{index}": chr(ord("A") + index) for index in range(1, 10)}

TORQUE_QUESTION = "What torque for the XP-400 drive coupling bolts?"


def make_chunk(
    chunk_id: str,
    text: str = TORQUE_EN,
    *,
    document_id: str = "DOC-1",
    variant: str = "XP-400",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        family_id="FAM-1",
        language=Language.EN,
        text=text,
        section="3.2",
        page=7,
        start_offset=0,
        end_offset=len(text),
        effectivity=Effectivity(variant_id=variant, serials=SerialRange(first=1000, last=4999)),
        valid_from=date(2024, 1, 1),
    )


def found(chunk: Chunk, *, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk, lexical_score=0.9, dense_score=0.9, fused_score=0.9, rank=rank
    )


def ask(text: str = TORQUE_QUESTION) -> Query:
    return Query(text=text, language=Language.EN, as_of=AS_OF, variant_id="XP-400", serial=2200)


def answering_decision(question: str = TORQUE_QUESTION) -> GateDecision:
    decision = gate.decide(
        ask(question),
        [
            found(make_chunk("c1"), rank=1),
            found(make_chunk("c2", TORQUE_EN_SECOND_SOURCE, document_id="DOC-2"), rank=2),
        ],
    )
    assert decision.outcome is GateOutcome.ANSWER, decision.reason
    return decision


# ------------------------------------------------------------------------ the extractive arm


def test_the_answer_is_its_citations_character_for_character() -> None:
    decision = answering_decision()

    answer = ExtractiveAnswerer().answer(ask(), decision, REVISIONS)

    assert answer.text == ANSWER_SEPARATOR.join(c.quote for c in answer.citations)
    assert answer.answerer == "extractive"
    for citation in answer.citations:
        assert citation.quote.strip()
        assert citation.quote in (answer.text or "")


def test_no_part_number_in_the_answer_is_absent_from_the_quotes() -> None:
    """Kill condition C, as a structural property rather than as a measurement."""
    for question in (
        TORQUE_QUESTION,
        "Which fastener kit for the XP-400 drive coupling?",
        "Which gasket does the XP-400 drive coupling use?",
    ):
        decision = answering_decision(question)
        answer = ExtractiveAnswerer().answer(ask(question), decision, REVISIONS)

        cited: frozenset[str] = frozenset()
        for citation in answer.citations:
            cited |= part_numbers_in(citation.quote)
        assert part_numbers_in(answer.text or "") <= cited, question
        assert answer.grounded is True
        assert ungrounded_part_numbers(answer) == frozenset()


def test_a_part_number_in_the_question_cannot_leak_into_the_answer() -> None:
    """The extractive arm copies from the evidence, so the question is not a source of parts."""
    question = "Is ZZ-9999-X the correct kit for the XP-400 drive coupling bolts?"
    decision = answering_decision(question)

    answer = ExtractiveAnswerer().answer(ask(question), decision, REVISIONS)

    assert "ZZ-9999-X" not in (answer.text or "")


def test_an_abstention_carries_no_text_and_no_evidence() -> None:
    decision = gate.decide(ask("What is the reservoir capacity of the ZR-900?"), [])
    assert decision.outcome is GateOutcome.ABSTAIN

    answer = ExtractiveAnswerer().answer(ask(), decision, REVISIONS)

    assert answer.text is None
    assert answer.citations == ()
    assert answer.decision.outcome is GateOutcome.ABSTAIN


def test_a_review_shows_the_passages_that_disagree_and_states_nothing() -> None:
    """A technician told "these two bulletins disagree, here they are" has something to act on."""
    decision = gate.decide(
        ask(),
        [
            found(make_chunk("c1"), rank=1),
            found(make_chunk("c2", TORQUE_EN_CONFLICTING, document_id="DOC-2"), rank=2),
        ],
    )
    assert decision.outcome is GateOutcome.REVIEW

    answer = ExtractiveAnswerer().answer(ask(), decision, REVISIONS)

    assert answer.text is None
    assert len(answer.citations) == 2
    assert len({c.quote for c in answer.citations}) == 2, (
        "a review that shows the same passage twice shows no disagreement"
    )


def test_the_answerer_never_changes_the_gates_decision() -> None:
    """No model may move a question from an abstention to an answer, and neither may this."""
    for retrieved in (
        [],
        [found(make_chunk("c1"))],
        [found(make_chunk("c1")), found(make_chunk("c2", TORQUE_EN_CONFLICTING))],
        [found(make_chunk("c1")), found(make_chunk("c2", TORQUE_EN_SECOND_SOURCE))],
    ):
        decision = gate.decide(ask(), retrieved)
        answer = ExtractiveAnswerer().answer(ask(), decision, REVISIONS)
        assert answer.decision.outcome is decision.outcome
        assert answer.decision.approved_chunks == decision.approved_chunks


# --------------------------------------------------------------------- Article 50 disclosure


def test_every_answer_carries_the_disclosure() -> None:
    corroborated = [found(make_chunk("c1")), found(make_chunk("c2", TORQUE_EN_SECOND_SOURCE))]
    for retrieved in ([], corroborated):
        decision = gate.decide(ask(), retrieved)
        answer = ExtractiveAnswerer().answer(ask(), decision, REVISIONS)
        assert "AI system" in answer.ai_disclosure
        assert answer.ai_disclosure.strip()


def test_an_answer_with_the_disclosure_blanked_is_refused() -> None:
    """The obligation follows the content, so it is enforced where the content is made."""
    decision = answering_decision()
    answer = ExtractiveAnswerer().answer(ask(), decision, REVISIONS)
    stripped = answer.model_copy(update={"ai_disclosure": "   "})

    with pytest.raises(ValueError, match="Article 50"):
        post_validate(stripped)


# ------------------------------------------------------------------ the post-validator itself


def test_an_invented_part_number_is_rejected() -> None:
    decision = answering_decision()
    honest = ExtractiveAnswerer().answer(ask(), decision, REVISIONS)
    invented = honest.model_copy(update={"text": "Fit the ZZ-9999-X kit and torque to 48 Nm."})

    assert ungrounded_part_numbers(invented) == frozenset({"ZZ-9999-X"})
    with pytest.raises(UngroundedAnswerError, match="ZZ-9999-X"):
        post_validate(invented)


class _InventingAnswerer:
    """A planted breach: an arm that names a part its evidence does not contain.

    ADR-001 requires breaches that change behaviour rather than guards that read source text. This
    one produces a real, well-formed `Answer` that would be served to a technician, and the only
    thing standing between it and the caller is the post-validator.
    """

    @property
    def name(self) -> str:
        return "inventing"

    def answer(
        self,
        query: Query,
        decision: GateDecision,
        revisions: Mapping[str, str],
    ) -> Answer:
        return Answer(
            query=query,
            decision=decision,
            text="Fit the ZZ-9999-X kit.",
            citations=build_citations(decision.approved_chunks, revisions, query),
            answerer=self.name,
        )


def test_the_post_validator_catches_an_arm_that_invents() -> None:
    decision = answering_decision()
    guarded: Answerer = PostValidatedAnswerer(_InventingAnswerer())

    with pytest.raises(UngroundedAnswerError, match="kill condition C"):
        guarded.answer(ask(), decision, REVISIONS)


def test_the_post_validator_passes_an_honest_arm_through_unchanged() -> None:
    decision = answering_decision()
    inner = ExtractiveAnswerer()

    direct = inner.answer(ask(), decision, REVISIONS)
    guarded = PostValidatedAnswerer(inner).answer(ask(), decision, REVISIONS)

    assert guarded == direct
    assert PostValidatedAnswerer(inner).name == "extractive"


# ----------------------------------------------------------------------- the abstractive arm


def test_the_abstractive_arm_raises_rather_than_returning_a_stub() -> None:
    """No key exists in this build. A stub would put a number no model produced into the results."""
    decision = answering_decision()

    with pytest.raises(AbstractiveArmUnavailableError, match="no model is configured"):
        AbstractiveAnswerer().answer(ask(), decision, REVISIONS)


def test_the_abstractive_prompt_carries_the_approved_chunks_and_nothing_else() -> None:
    retrieved = [
        found(
            make_chunk(f"c{index}", TORQUE_EN_SECOND_SOURCE, document_id=f"DOC-{index}"),
            rank=index,
        )
        for index in range(1, 8)
    ]
    decision = gate.decide(ask(), retrieved)
    assert decision.outcome is GateOutcome.ANSWER
    assert len(decision.approved_chunks) < len(retrieved), "the cap is what makes this test mean it"

    payload = AbstractiveAnswerer().prompt_payload(ask(), decision, REVISIONS)

    approved = [chunk.chunk.chunk_id for chunk in decision.approved_chunks]
    assert [item.chunk_id for item in payload.evidence] == approved
    retrieved_only = {chunk.chunk.chunk_id for chunk in retrieved} - set(approved)
    assert retrieved_only, "the fixture no longer withholds anything"
    for item in payload.evidence:
        assert item.chunk_id not in retrieved_only
    assert payload.question == TORQUE_QUESTION
    assert payload.as_of == AS_OF.isoformat()


def test_the_abstractive_prompt_refuses_to_be_built_without_evidence() -> None:
    decision = gate.decide(ask(), [])
    assert decision.approved_chunks == ()

    with pytest.raises(EmptyEvidenceError, match="own memory"):
        AbstractiveAnswerer().prompt_payload(ask(), decision, REVISIONS)


def test_a_prompt_may_not_describe_a_revision_it_was_not_given() -> None:
    decision = answering_decision()

    with pytest.raises(KeyError, match="DOC-1"):
        AbstractiveAnswerer().prompt_payload(ask(), decision, {})


# ------------------------------------------------------------------------- the port itself


def test_both_arms_satisfy_the_port() -> None:
    """Checked by assignment, so mypy --strict verifies the shape as well as the runtime."""
    arms: list[Answerer] = [
        ExtractiveAnswerer(),
        AbstractiveAnswerer(),
        PostValidatedAnswerer(ExtractiveAnswerer()),
    ]

    assert [arm.name for arm in arms] == ["extractive", "abstractive", "extractive"]
