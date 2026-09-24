"""The gate, exercised without a database, a retriever or an embedding model.

Every fixture here is built by hand. That is the property the gate was designed for: if these tests
needed a corpus to run, the decision logic and the retrieval stack would fail together and nobody
could tell which had broken.

The first test in the file is the one ADR-001 names — the answering outcome is constructed at
exactly one place — and it is asserted over the module's AST rather than by reading it.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest

from parts_answer_gate import gate
from parts_answer_gate.domain import (
    Chunk,
    Effectivity,
    GateOutcome,
    Language,
    Query,
    RetrievedChunk,
    SerialRange,
)

GATE_SOURCE = Path(gate.__file__)

AS_OF = date(2026, 3, 1)

TORQUE_EN = (
    "Tighten the drive coupling bolts on the XP-400 to 48 Nm in a cross pattern. "
    "Use only the AB-1234-C fastener kit; a lower grade bolt will not hold this value."
)
TORQUE_EN_CONFLICTING = (
    "Tighten the drive coupling bolts on the XP-400 to 52 Nm in a cross pattern. "
    "Recheck the torque after the first 50 operating hours."
)
TORQUE_EN_SECOND_SOURCE = (
    "The drive coupling bolts on the XP-400 are torqued to 48 Nm. "
    "Replace the AB-1234-C kit at every second service."
)


def make_chunk(
    chunk_id: str,
    text: str = TORQUE_EN,
    *,
    variant: str = "XP-400",
    document_id: str = "DOC-1",
    family_id: str = "FAM-1",
    language: Language = Language.EN,
    valid_from: date = date(2024, 1, 1),
    valid_to: date | None = None,
    superseded_by: str | None = None,
    start_offset: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        family_id=family_id,
        language=language,
        text=text,
        section="3.2",
        page=7,
        start_offset=start_offset,
        end_offset=start_offset + len(text),
        effectivity=Effectivity(variant_id=variant, serials=SerialRange(first=1000, last=4999)),
        valid_from=valid_from,
        valid_to=valid_to,
        superseded_by=superseded_by,
    )


def found(
    chunk: Chunk,
    *,
    fused: float = 0.9,
    rank: int = 1,
    exact: bool = False,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk,
        lexical_score=fused,
        dense_score=fused,
        fused_score=fused,
        rank=rank,
        exact_identifier_hit=exact,
    )


def ask(text: str, *, language: Language = Language.EN, as_of: date = AS_OF) -> Query:
    return Query(text=text, language=language, as_of=as_of, variant_id="XP-400", serial=2200)


TORQUE_QUESTION = "What torque for the XP-400 drive coupling bolts?"


# ------------------------------------------------------------------ the structural guarantee


def test_the_answering_outcome_is_constructed_in_exactly_one_place() -> None:
    """ADR-001: "`ANSWER` is reachable from exactly one place in the gate module."

    Asserted over the AST, not by grep and not by review. A second construction site is a second
    place a guard can be forgotten, and the only version of this check that survives a refactor is
    one that counts the nodes.
    """
    tree = ast.parse(GATE_SOURCE.read_text(encoding="utf-8"))
    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "ANSWER"
        and isinstance(node.value, ast.Name)
        and node.value.id == "GateOutcome"
    ]
    assert len(sites) == 1, (
        f"the answering outcome is named at {len(sites)} places in {GATE_SOURCE.name}, at lines "
        f"{[node.lineno for node in sites]}; ADR-001 fixes it at one"
    )

    decide = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "decide"
    )
    assert sites[0].lineno in range(decide.lineno, decide.end_lineno or decide.lineno), (
        "the single construction site is not inside decide(), so some other function can answer"
    )

    # The attribute form is not the only way to name the outcome. `GateOutcome` is a `StrEnum` and
    # `GateDecision` is a pydantic model, so `GateDecision(outcome="answer", ...)` builds exactly
    # the same answering decision and is invisible to the node count above — an AST guard that
    # looks for one spelling guards one spelling. Every string literal equal to the enum's value is
    # therefore counted too.
    literals = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == GateOutcome.ANSWER.value
    ]
    assert not literals, (
        f"{GATE_SOURCE.name} names the answering outcome as a bare string at lines "
        f"{[node.lineno for node in literals]}; a StrEnum makes that the same construction, and it "
        "would not be counted by the attribute check above"
    )


def test_no_gate_signal_is_derived_from_a_model() -> None:
    """The gate module imports nothing that could generate text.

    ADR-001 allows the model to phrase an approved answer and nothing else. An import of an answerer
    here would be the first step towards a signal a model can influence.
    """
    tree = ast.parse(GATE_SOURCE.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        name.startswith(("parts_answer_gate.answerer", "parts_answer_gate.retrieval"))
        for name in imported
    ), f"the gate imports {sorted(imported)}"


# ------------------------------------------------------------------------- abstention paths


def test_nothing_retrieved_abstains_and_approves_nothing() -> None:
    decision = gate.decide(ask(TORQUE_QUESTION), [])

    assert decision.outcome is GateOutcome.ABSTAIN
    assert decision.approved_chunks == ()
    assert decision.signals.supporting_chunks == 0
    assert "no passage" in decision.reason


def test_one_lucky_chunk_is_not_corroboration() -> None:
    decision = gate.decide(ask(TORQUE_QUESTION), [found(make_chunk("c1"))])

    assert decision.outcome is GateOutcome.ABSTAIN
    assert decision.approved_chunks == ()
    assert decision.signals.supporting_chunks == 1


def test_evidence_about_something_adjacent_abstains() -> None:
    """The failure the coverage signal exists for: the right machine, the wrong attribute."""
    retrieved = [found(make_chunk("c1"), rank=1), found(make_chunk("c2"), rank=2)]

    decision = gate.decide(
        ask("What is the hydraulic reservoir capacity of the XP-400?"), retrieved
    )

    assert decision.outcome is GateOutcome.ABSTAIN
    assert decision.approved_chunks == ()
    assert decision.signals.term_coverage < gate.DEFAULT_MIN_TERM_COVERAGE


def test_nothing_scored_and_no_exact_hit_abstains() -> None:
    retrieved = [
        found(make_chunk("c1"), fused=0.0, rank=1),
        found(make_chunk("c2", TORQUE_EN_SECOND_SOURCE), fused=0.0, rank=2),
    ]

    decision = gate.decide(ask(TORQUE_QUESTION), retrieved)

    assert decision.outcome is GateOutcome.ABSTAIN
    assert decision.signals.top_fused_score == 0.0


def test_a_question_of_only_function_words_abstains() -> None:
    retrieved = [found(make_chunk("c1")), found(make_chunk("c2", TORQUE_EN_SECOND_SOURCE))]

    decision = gate.decide(ask("what is it for"), retrieved)

    assert decision.outcome is GateOutcome.ABSTAIN
    assert decision.signals.term_coverage == 0.0


# ------------------------------------------------------------------------------ review paths


def test_a_passage_not_in_force_on_the_as_of_date_goes_to_review() -> None:
    """The one failure this project exists to prevent, seen from the gate's side.

    It should be unreachable — the effectivity filter runs in SQL before ranking — so reaching it
    means the filter did not run. Abstaining would hide that; a review puts it in front of a person.
    """
    withdrawn = make_chunk(
        "c1",
        valid_to=date(2025, 6, 1),
        superseded_by="DOC-2",
    )
    retrieved = [found(withdrawn, rank=1), found(make_chunk("c2", TORQUE_EN_SECOND_SOURCE), rank=2)]

    decision = gate.decide(ask(TORQUE_QUESTION), retrieved)

    assert decision.outcome is GateOutcome.REVIEW
    assert decision.signals.superseded_present is True
    assert decision.approved_chunks, "a review must show the passages it is reviewing"


def test_a_superseded_passage_outranks_the_weak_evidence_rules() -> None:
    """Documented precedence: the supersession leak is reported even when support is thin.

    Every other conflict is checked after the strength rules, because a disagreement between two
    passages too thin to support anything is not worth a person's time. This one is checked first
    because it is evidence that a guarantee failed.
    """
    withdrawn = make_chunk("c1", valid_to=date(2025, 6, 1), superseded_by="DOC-2")

    decision = gate.decide(ask(TORQUE_QUESTION), [found(withdrawn)])

    assert decision.outcome is GateOutcome.REVIEW
    assert decision.signals.supporting_chunks < gate.DEFAULT_MIN_SUPPORTING_CHUNKS


def test_evidence_from_two_variants_is_a_conflict_not_a_consensus() -> None:
    retrieved = [
        found(make_chunk("c1"), rank=1),
        found(make_chunk("c2", TORQUE_EN_SECOND_SOURCE, variant="XP-400S"), rank=2),
    ]

    decision = gate.decide(ask(TORQUE_QUESTION), retrieved)

    assert decision.outcome is GateOutcome.REVIEW
    assert decision.signals.variant_agreement is False


def test_two_passages_with_different_values_go_to_review() -> None:
    retrieved = [
        found(make_chunk("c1"), rank=1),
        found(make_chunk("c2", TORQUE_EN_CONFLICTING, document_id="DOC-2"), rank=2),
    ]

    decision = gate.decide(ask(TORQUE_QUESTION), retrieved)

    assert decision.outcome is GateOutcome.REVIEW
    assert decision.signals.conflicting_evidence is True


def test_passages_whose_values_overlap_are_not_in_conflict() -> None:
    """A passage listing one of several stated torques agrees about that one."""
    retrieved = [
        found(make_chunk("c1"), rank=1),
        found(make_chunk("c2", TORQUE_EN_SECOND_SOURCE, document_id="DOC-2"), rank=2),
    ]

    decision = gate.decide(ask(TORQUE_QUESTION), retrieved)

    assert decision.signals.conflicting_evidence is False
    assert decision.outcome is GateOutcome.ANSWER


def test_a_conflict_in_russian_is_the_same_conflict() -> None:
    """The units table carries localised spellings, or the conflict signal is false off English."""
    first = make_chunk(
        "c1",
        "Затяните болты муфты привода XP-400 моментом 48 Н·м крест-накрест.",  # noqa: RUF001
        language=Language.RU,
    )
    second = make_chunk(
        "c2",
        "Болты муфты привода XP-400 затягиваются моментом 52 Н·м.",  # noqa: RUF001
        language=Language.RU,
        document_id="DOC-2",
    )

    decision = gate.decide(
        ask("Какой момент затяжки болтов муфты привода XP-400?", language=Language.RU),
        [found(first, rank=1), found(second, rank=2)],
    )

    assert decision.signals.conflicting_evidence is True
    assert decision.outcome is GateOutcome.REVIEW


# ------------------------------------------------------------------------------ answer paths


def test_corroborated_in_force_agreeing_evidence_answers() -> None:
    retrieved = [
        found(make_chunk("c1"), rank=1),
        found(make_chunk("c2", TORQUE_EN_SECOND_SOURCE, document_id="DOC-2"), rank=2),
    ]

    decision = gate.decide(ask(TORQUE_QUESTION), retrieved)

    assert decision.outcome is GateOutcome.ANSWER
    assert len(decision.approved_chunks) == 2
    assert decision.signals.variant_agreement is True
    assert decision.signals.superseded_present is False
    assert decision.signals.term_coverage >= gate.DEFAULT_MIN_TERM_COVERAGE


def test_an_exact_identifier_hit_stands_alone() -> None:
    """A catalogue match is a different kind of evidence and does not need a second opinion."""
    decision = gate.decide(
        ask("Which fastener kit is AB-1234-C?"),
        [found(make_chunk("c1"), fused=0.2, exact=True)],
    )

    assert decision.outcome is GateOutcome.ANSWER
    assert decision.signals.exact_identifier_hit is True
    assert decision.signals.supporting_chunks == 1


def test_an_exact_hit_that_the_fusion_ranked_low_is_still_supporting() -> None:
    strong = found(make_chunk("c1", TORQUE_EN_SECOND_SOURCE), fused=0.9, rank=1)
    weak_exact = found(make_chunk("c2"), fused=0.01, rank=9, exact=True)

    approved = gate.supporting_chunks([strong, weak_exact])

    assert {chunk.chunk.chunk_id for chunk in approved} == {"c1", "c2"}


def test_turkish_coverage_survives_inflection() -> None:
    """Exact token matching would score near zero here and abstain on the whole language."""
    text = (
        "XP-400 tahrik kaplini cıvatalarını 48 Nm değerine sıkın. "  # noqa: RUF001
        "Yalnızca AB-1234-C bağlantı elemanı takımını kullanın."  # noqa: RUF001
    )
    retrieved = [
        found(make_chunk("c1", text, language=Language.TR), rank=1),
        found(make_chunk("c2", text, language=Language.TR, document_id="DOC-2"), rank=2),
    ]

    decision = gate.decide(
        ask("XP-400 tahrik kaplin cıvata torku nedir?", language=Language.TR),  # noqa: RUF001
        retrieved,
    )

    assert decision.signals.term_coverage >= gate.DEFAULT_MIN_TERM_COVERAGE
    assert decision.outcome is GateOutcome.ANSWER


# --------------------------------------------------------------------- containment and order


def test_the_approved_set_is_capped() -> None:
    retrieved = [
        found(make_chunk(f"c{index}", TORQUE_EN_SECOND_SOURCE), rank=index) for index in range(1, 9)
    ]

    decision = gate.decide(ask(TORQUE_QUESTION), retrieved)

    assert len(decision.approved_chunks) == gate.DEFAULT_MAX_APPROVED_CHUNKS
    assert decision.signals.supporting_chunks == gate.DEFAULT_MAX_APPROVED_CHUNKS


def test_the_decision_does_not_depend_on_the_order_the_retriever_emitted() -> None:
    """Kill condition J in miniature: two runs must agree byte for byte."""
    chunks = [
        found(make_chunk("c1"), fused=0.9, rank=1),
        found(make_chunk("c2", TORQUE_EN_SECOND_SOURCE, document_id="DOC-2"), fused=0.9, rank=2),
        found(make_chunk("c3", TORQUE_EN_SECOND_SOURCE, document_id="DOC-3"), fused=0.9, rank=3),
    ]

    first = gate.decide(ask(TORQUE_QUESTION), chunks)
    second = gate.decide(ask(TORQUE_QUESTION), list(reversed(chunks)))

    assert first.model_dump_json() == second.model_dump_json()


def test_an_abstention_can_never_carry_evidence() -> None:
    """Belt and braces: the domain model forbids it, and every abstaining path is checked here."""
    cases = [
        ([], TORQUE_QUESTION),
        ([found(make_chunk("c1"))], TORQUE_QUESTION),
        (
            [found(make_chunk("c1")), found(make_chunk("c2"))],
            "What is the hydraulic reservoir capacity of the XP-400?",
        ),
        (
            [found(make_chunk("c1"), fused=0.0), found(make_chunk("c2"), fused=0.0)],
            TORQUE_QUESTION,
        ),
    ]
    for retrieved, question in cases:
        decision = gate.decide(ask(question), retrieved)
        assert decision.outcome is GateOutcome.ABSTAIN, question
        assert decision.approved_chunks == ()


# ------------------------------------------------------------------------------- thresholds


def test_the_primary_threshold_can_be_swept_without_touching_the_others() -> None:
    swept = gate.thresholds_at(0.25)

    assert swept.min_term_coverage == 0.25
    assert swept.min_supporting_chunks == gate.DEFAULT_THRESHOLDS.min_supporting_chunks
    assert swept.max_approved_chunks == gate.DEFAULT_THRESHOLDS.max_approved_chunks
    assert gate.DEFAULT_THRESHOLDS.min_term_coverage == gate.DEFAULT_MIN_TERM_COVERAGE


def test_the_declared_sweep_has_enough_points_for_a_curve() -> None:
    """ADR-001 requires a curve, and the kill test requires at least five points on it."""
    assert len(gate.PRIMARY_THRESHOLD_SWEEP) >= 5
    assert tuple(sorted(gate.PRIMARY_THRESHOLD_SWEEP)) == gate.PRIMARY_THRESHOLD_SWEEP
    assert gate.PRIMARY_THRESHOLD_NAME in gate.GateThresholds.model_fields


def test_raising_the_primary_threshold_never_increases_coverage() -> None:
    """The sweep must be monotone or the curve it draws is not a trade-off, it is noise."""
    corroborated = [
        found(make_chunk("c1"), rank=1),
        found(make_chunk("c2", TORQUE_EN_SECOND_SOURCE, document_id="DOC-2"), rank=2),
    ]
    questions = [
        "What torque for the XP-400 drive coupling bolts?",
        "What torque for the drive coupling bolts?",
        "Which fastener kit for the XP-400 drive coupling?",
        "What is the hydraulic reservoir capacity of the XP-400?",
        "What is the recommended lubricant for the ZR-900 gearbox?",
    ]

    answered_at: list[int] = []
    for threshold in gate.PRIMARY_THRESHOLD_SWEEP:
        thresholds = gate.thresholds_at(threshold)
        answered_at.append(
            sum(
                1
                for question in questions
                if gate.decide(ask(question), corroborated, thresholds).outcome
                is GateOutcome.ANSWER
            )
        )

    assert answered_at == sorted(answered_at, reverse=True), answered_at
    assert answered_at[0] > answered_at[-1], "the sweep changed nothing; it is not a sweep"


@pytest.mark.parametrize("outcome", [GateOutcome.ABSTAIN, GateOutcome.REVIEW])
def test_both_non_answering_outcomes_withhold(outcome: GateOutcome) -> None:
    """Coverage is measured against this function, so the curve and the service agree on REVIEW."""
    assert gate.withholds_answer(outcome) is True


def test_the_answering_outcome_does_not_withhold() -> None:
    assert gate.withholds_answer(GateOutcome.ANSWER) is False


# --------------------------------------------------------------------------- signal details


def test_the_signals_are_recorded_even_when_they_did_not_decide() -> None:
    """A decision that cannot be argued with is a decision nobody will trust."""
    signals = gate.compute_signals(
        ask(TORQUE_QUESTION),
        [
            found(make_chunk("c1"), rank=1),
            found(make_chunk("c2", TORQUE_EN_SECOND_SOURCE, document_id="DOC-2"), rank=2),
        ],
    )

    assert signals.top_fused_score == 0.9
    assert signals.supporting_chunks == 2
    assert 0.0 <= signals.term_coverage <= 1.0
    assert signals.variant_agreement is True
    assert signals.superseded_present is False
    assert signals.conflicting_evidence is False
    assert signals.exact_identifier_hit is False


def test_a_not_yet_effective_passage_counts_as_out_of_force() -> None:
    """A revision that comes into force next month is not the text approved on the asked date."""
    future = make_chunk("c1", valid_from=date(2027, 1, 1))

    signals = gate.compute_signals(ask(TORQUE_QUESTION), [found(future), found(make_chunk("c2"))])

    assert signals.superseded_present is True
