"""Unit tests for the scoring functions, against inputs written by hand.

No corpus, no database, no embedding model. Every input here is a few strings chosen so that the
expected number can be worked out on paper and written into the assertion as a literal — `0.5`
rather than `len(found) / len(relevant)`, which would restate the implementation and pass whatever
the implementation did.

The edge cases carry as much weight as the happy paths, because the edge cases are where a metric
silently starts measuring something else: an empty relevant set, a `k` past the end of the list, a
question with several supporting passages, a system that answered nothing at all.
"""

from __future__ import annotations

import pytest

from parts_answer_gate.domain import GateOutcome, GateSignals, Language
from parts_answer_gate.evaluation.metrics import (
    AnswerOutcome,
    RetrievalCase,
    abstention_rate,
    attribute_failure,
    by_language,
    coverage,
    coverage_curve,
    default_thresholds,
    failure_attribution,
    mrr,
    ndcg_at_k,
    non_answer_rate,
    recall_at_k,
    review_rate,
    summarise_retrieval,
    ungated_wrong_answer_rate,
    wrong_answer_rate,
)


def case(
    question_id: str,
    retrieved: tuple[str, ...],
    relevant: frozenset[str],
    language: Language = Language.EN,
) -> RetrievalCase:
    return RetrievalCase(
        question_id=question_id,
        retrieved_ids=retrieved,
        relevant_ids=relevant,
        language=language,
    )


def outcome(
    question_id: str = "q",
    *,
    decided: GateOutcome = GateOutcome.ANSWER,
    gate_score: float = 1.0,
    cited_revisions: frozenset[str] = frozenset({"C"}),
    in_force_revisions: frozenset[str] = frozenset({"C"}),
    cited_variants: frozenset[str] = frozenset({"XP-400"}),
    asked_variant: str | None = "XP-400",
    supporting_chunk_exists: bool = True,
    supporting_chunk_retrieved: bool = True,
    hard_blocked: bool = False,
    signals: GateSignals | None = None,
    language: Language = Language.EN,
) -> AnswerOutcome:
    """A correct, answered outcome by default; every test names only what it changes."""
    return AnswerOutcome(
        question_id=question_id,
        language=language,
        outcome=decided,
        gate_score=gate_score,
        cited_revisions=cited_revisions,
        in_force_revisions=in_force_revisions,
        cited_variants=cited_variants,
        asked_variant=asked_variant,
        supporting_chunk_exists=supporting_chunk_exists,
        supporting_chunk_retrieved=supporting_chunk_retrieved,
        hard_blocked=hard_blocked,
        signals=signals,
    )


def signals(
    *,
    superseded_present: bool = False,
    variant_agreement: bool = True,
    conflicting_evidence: bool = False,
) -> GateSignals:
    return GateSignals(
        top_fused_score=0.9,
        supporting_chunks=3,
        term_coverage=0.8,
        variant_agreement=variant_agreement,
        superseded_present=superseded_present,
        conflicting_evidence=conflicting_evidence,
        exact_identifier_hit=False,
    )


# ------------------------------------------------------------------------------- recall_at_k


def test_recall_counts_the_supporting_passages_inside_the_cut_off() -> None:
    assert recall_at_k(("a", "b", "c"), frozenset({"b"}), 3) == 1.0


def test_recall_is_partial_when_a_question_has_several_supporting_passages() -> None:
    """Two of four supporting chunks retrieved is a half, not a hit."""
    retrieved = ("a", "x", "c", "y")
    assert recall_at_k(retrieved, frozenset({"a", "c", "m", "n"}), 10) == 0.5


def test_recall_ignores_a_supporting_passage_below_the_cut_off() -> None:
    assert recall_at_k(("x", "y", "z", "a"), frozenset({"a"}), 3) == 0.0
    assert recall_at_k(("x", "y", "z", "a"), frozenset({"a"}), 4) == 1.0


def test_recall_accepts_a_k_larger_than_the_retrieved_list() -> None:
    """Asking for ten and receiving three is normal, not an error."""
    assert recall_at_k(("a", "b", "c"), frozenset({"a", "b"}), 10) == 1.0


def test_recall_of_an_empty_retrieved_list_is_zero() -> None:
    """The retriever ran and found nothing. That is a result, and its value is zero."""
    assert recall_at_k((), frozenset({"a"}), 10) == 0.0


def test_recall_credits_a_duplicate_chunk_once() -> None:
    """A retriever returning the same chunk twice must not buy recall with the repeat."""
    assert recall_at_k(("a", "a", "a"), frozenset({"a", "b"}), 10) == 0.5


def test_recall_refuses_a_question_with_no_supporting_passage() -> None:
    """The ninety unanswerable questions must not be scored, in either direction.

    A vacuous 1.0 would lift hold-out recall over the kill-condition floor without retrieving
    anything; a 0.0 would punish the system for correctly having nothing to find.
    """
    with pytest.raises(ValueError, match="no supporting passage"):
        recall_at_k(("a",), frozenset(), 10)


def test_recall_refuses_a_non_positive_cut_off() -> None:
    with pytest.raises(ValueError, match="not a cut-off"):
        recall_at_k(("a",), frozenset({"a"}), 0)


# --------------------------------------------------------------------------------------- mrr


def test_reciprocal_rank_is_one_when_the_supporting_passage_is_first() -> None:
    assert mrr(("a", "b"), frozenset({"a"})) == 1.0


def test_reciprocal_rank_is_one_over_the_first_hit() -> None:
    assert mrr(("x", "y", "a", "b"), frozenset({"a", "b"})) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_was_retrieved() -> None:
    assert mrr(("x", "y"), frozenset({"a"})) == 0.0
    assert mrr((), frozenset({"a"})) == 0.0


def test_reciprocal_rank_refuses_a_question_with_no_supporting_passage() -> None:
    with pytest.raises(ValueError, match="no supporting passage"):
        mrr(("a",), frozenset())


# ----------------------------------------------------------------------------------- ndcg_at_k


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k(("a", "b", "x"), frozenset({"a", "b"}), 10) == 1.0


def test_ndcg_penalises_a_supporting_passage_pushed_down_the_list() -> None:
    """Rank three instead of rank one: gain 1/log2(4) against an ideal of 1/log2(2)."""
    assert ndcg_at_k(("x", "y", "a"), frozenset({"a"}), 10) == pytest.approx(0.5)


def test_ndcg_caps_the_ideal_at_the_cut_off() -> None:
    """Twelve supporting passages and a cut-off of two still reaches 1.0 with two at the top.

    Normalising against an ideal the cut-off makes unreachable would turn nDCG into a function of
    how many passages happen to support a question.
    """
    relevant = frozenset(f"r{index}" for index in range(12))
    assert ndcg_at_k(("r0", "r1", "r2"), relevant, 2) == 1.0


def test_ndcg_of_an_empty_retrieved_list_is_zero() -> None:
    assert ndcg_at_k((), frozenset({"a"}), 10) == 0.0


def test_ndcg_credits_a_duplicate_chunk_once() -> None:
    repeated = ndcg_at_k(("a", "a"), frozenset({"a", "b"}), 10)
    single = ndcg_at_k(("a", "z"), frozenset({"a", "b"}), 10)
    assert repeated == single


def test_ndcg_refuses_a_question_with_no_supporting_passage() -> None:
    with pytest.raises(ValueError, match="no supporting passage"):
        ndcg_at_k(("a",), frozenset(), 10)


def test_ndcg_refuses_a_non_positive_cut_off() -> None:
    with pytest.raises(ValueError, match="not a cut-off"):
        ndcg_at_k(("a",), frozenset({"a"}), -1)


# ------------------------------------------------------------------------- summarise_retrieval


def test_the_summary_excludes_unanswerable_questions_and_says_how_many() -> None:
    """The exclusion is the point; an unreported exclusion is how a recall figure stops meaning
    anything."""
    summary = summarise_retrieval(
        [
            case("q1", ("a", "b"), frozenset({"a"})),
            case("q2", ("x", "y"), frozenset({"z"})),
            case("q3", ("m",), frozenset()),
        ],
        k=10,
    )
    assert summary.scored == 2
    assert summary.skipped_without_relevant == 1
    assert summary.recall_at_k == 0.5  # one of the two scorable questions found its passage
    assert summary.mrr == 0.5  # rank 1 and nothing, averaged
    assert summary.k == 10


def test_the_summary_refuses_a_set_with_nothing_scorable() -> None:
    with pytest.raises(ValueError, match="nothing to"):
        summarise_retrieval([case("q1", ("a",), frozenset())])


def test_the_summary_consumes_a_generator_exactly_once() -> None:
    """A second pass over an exhausted generator would report an empty question set as a score."""
    cases = (case(f"q{index}", ("a",), frozenset({"a"})) for index in range(3))
    summary = summarise_retrieval(cases)
    assert summary.scored == 3
    assert summary.recall_at_k == 1.0


# -------------------------------------------------------------------- coverage and abstention


def test_coverage_counts_answers_only() -> None:
    outcomes = [
        outcome("q1"),
        outcome("q2", decided=GateOutcome.ABSTAIN),
        outcome("q3", decided=GateOutcome.REVIEW),
        outcome("q4"),
    ]
    assert coverage(outcomes) == 0.5


def test_a_review_is_neither_coverage_nor_abstention() -> None:
    """Three outcomes, so abstention is not `1 - coverage`; a review queue is a third thing."""
    outcomes = [
        outcome("q1"),
        outcome("q2", decided=GateOutcome.ABSTAIN),
        outcome("q3", decided=GateOutcome.REVIEW),
        outcome("q4", decided=GateOutcome.REVIEW),
    ]
    assert coverage(outcomes) == 0.25
    assert abstention_rate(outcomes) == 0.25
    assert review_rate(outcomes) == 0.5
    assert non_answer_rate(outcomes) == 0.75
    assert coverage(outcomes) + non_answer_rate(outcomes) == 1.0


def test_every_rate_refuses_an_empty_question_set() -> None:
    for metric in (coverage, abstention_rate, review_rate, non_answer_rate, wrong_answer_rate):
        with pytest.raises(ValueError, match="not a measurement"):
            metric([])


# ------------------------------------------------------------------------- wrong_answer_rate


def test_an_answer_from_a_withdrawn_revision_is_wrong() -> None:
    wrong = outcome(cited_revisions=frozenset({"B"}), in_force_revisions=frozenset({"C"}))
    assert wrong.revision_incorrect is True
    assert wrong.is_wrong is True
    assert wrong_answer_rate([wrong]) == 1.0


def test_an_answer_from_another_variant_is_wrong() -> None:
    wrong = outcome(cited_variants=frozenset({"XP-400S"}), asked_variant="XP-400")
    assert wrong.variant_incorrect is True
    assert wrong_answer_rate([wrong]) == 1.0


def test_an_answer_mixing_two_variants_is_wrong_even_when_none_was_asked_for() -> None:
    """Every chunk carries an effectivity, so an answer drawn from two variants describes no
    single machine."""
    wrong = outcome(cited_variants=frozenset({"XP-400", "XP-400S"}), asked_variant=None)
    assert wrong.variant_incorrect is True


def test_a_single_variant_is_not_wrong_when_the_question_named_none() -> None:
    fine = outcome(cited_variants=frozenset({"XP-400S"}), asked_variant=None)
    assert fine.variant_incorrect is False
    assert fine.is_wrong is False


def test_an_answer_citing_nothing_is_counted_wrong() -> None:
    """The conservative direction. A curve that assumed an uncited answer was right would slope
    upward for free."""
    assert outcome(cited_revisions=frozenset()).revision_incorrect is True


def test_a_correct_answer_is_not_wrong_merely_for_differing_from_a_gold_string() -> None:
    """ADR-001 defines wrongness as revision- or variant-incorrect, and nothing else."""
    assert outcome().is_wrong is False
    assert wrong_answer_rate([outcome()]) == 0.0


def test_the_wrong_answer_denominator_is_answers_given_not_questions_asked() -> None:
    """The measurement choice that decides whether abstaining can be mistaken for being right.

    One wrong answer, one right answer, and eight abstentions on questions whose candidates were
    also wrong. Per answer given the rate is 0.5. Per question asked it would be 0.1, and a system
    could reach the kill-condition ceiling by refusing more rather than by being right more.
    """
    outcomes = [
        outcome("wrong", cited_revisions=frozenset({"B"}), in_force_revisions=frozenset({"C"})),
        outcome("right"),
        *[
            outcome(
                f"abstained{index}",
                decided=GateOutcome.ABSTAIN,
                cited_revisions=frozenset({"B"}),
                in_force_revisions=frozenset({"C"}),
            )
            for index in range(8)
        ],
    ]
    assert wrong_answer_rate(outcomes) == 0.5
    assert coverage(outcomes) == 0.2


def test_a_system_that_answered_nothing_has_no_wrong_answers() -> None:
    outcomes = [outcome("q1", decided=GateOutcome.ABSTAIN, cited_revisions=frozenset())]
    assert wrong_answer_rate(outcomes) == 0.0
    assert coverage(outcomes) == 0.0


def test_the_ungated_rate_divides_by_every_question() -> None:
    """The gate removed: the ungated baseline answers everything, so everything is the
    denominator."""
    outcomes = [
        outcome("wrong", cited_revisions=frozenset({"B"}), in_force_revisions=frozenset({"C"})),
        outcome("right"),
        outcome(
            "refused",
            decided=GateOutcome.ABSTAIN,
            cited_revisions=frozenset({"B"}),
            in_force_revisions=frozenset({"C"}),
        ),
        outcome("right2"),
    ]
    assert ungated_wrong_answer_rate(outcomes) == 0.5
    assert wrong_answer_rate(outcomes) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------- coverage_curve


def _sweepable() -> list[AnswerOutcome]:
    """Ten questions whose gate scores separate the wrong ones from the right ones."""
    right = [outcome(f"right{index}", gate_score=0.6 + index / 100) for index in range(6)]
    wrong = [
        outcome(
            f"wrong{index}",
            gate_score=0.1 + index / 100,
            cited_revisions=frozenset({"B"}),
            in_force_revisions=frozenset({"C"}),
        )
        for index in range(4)
    ]
    return right + wrong


def test_the_curve_has_the_shape_the_kill_test_reads() -> None:
    """Asserted here in the same terms `tests/test_kill_criteria.py` uses, so a shape change is
    caught in a unit test rather than at the end of a full pipeline run."""
    curve = coverage_curve(_sweepable())

    assert len(curve) >= 5
    for point in curve:
        assert {"threshold", "coverage", "wrong_answer_rate"} <= set(point)
    coverages = [point["coverage"] for point in curve]
    assert coverages == sorted(coverages, reverse=True) or coverages == sorted(coverages)


def test_coverage_falls_as_the_threshold_rises() -> None:
    curve = coverage_curve(_sweepable())
    assert curve[0]["coverage"] == 1.0
    assert curve[-1]["coverage"] < curve[0]["coverage"]
    assert curve[0]["threshold"] < curve[-1]["threshold"]


def test_raising_the_threshold_removes_the_wrong_answers_first() -> None:
    """The reason the curve is published: the trade it shows is real."""
    curve = coverage_curve(_sweepable(), thresholds=[0.0, 0.5])
    assert curve[0] == {
        "threshold": 0.0,
        "coverage": 1.0,
        "wrong_answer_rate": 0.4,
        "answered": 10,
        "wrong": 4,
    }
    assert curve[1] == {
        "threshold": 0.5,
        "coverage": 0.6,
        "wrong_answer_rate": 0.0,
        "answered": 6,
        "wrong": 0,
    }


def test_a_hard_blocked_question_is_never_answered_at_any_threshold() -> None:
    """Hard blocks are kept out of the score so that coverage stays monotonic in the threshold."""
    outcomes = [
        outcome("blocked", gate_score=99.0, hard_blocked=True),
        outcome("low", gate_score=0.1),
        outcome("high", gate_score=0.9),
    ]
    curve = coverage_curve(outcomes, thresholds=[0.0, 0.5, 1.0])
    assert [point["answered"] for point in curve] == [2, 1, 0]


def test_the_curve_is_returned_in_ascending_threshold_order_whatever_the_caller_passed() -> None:
    curve = coverage_curve(_sweepable(), thresholds=[0.9, 0.0, 0.5])
    assert [point["threshold"] for point in curve] == [0.0, 0.5, 0.9]


def test_the_curve_refuses_an_empty_question_set() -> None:
    with pytest.raises(ValueError, match="not a measurement"):
        coverage_curve([])


def test_the_default_thresholds_span_the_observed_scores() -> None:
    steps = default_thresholds(_sweepable())
    assert len(steps) == 11
    assert steps[0] == 0.1
    assert steps[-1] == pytest.approx(0.65)
    assert list(steps) == sorted(steps)


def test_the_default_thresholds_refuse_a_constant_gate_score() -> None:
    """A sweep over a constant is five copies of one point dressed as a curve."""
    with pytest.raises(ValueError, match="not a sweep"):
        default_thresholds([outcome("q1", gate_score=0.5), outcome("q2", gate_score=0.5)])


def test_the_default_thresholds_refuse_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least five"):
        default_thresholds(_sweepable(), points=4)


# ------------------------------------------------------------------------------- attribution


def test_a_correct_answer_is_attributed_to_nothing() -> None:
    assert attribute_failure(outcome()) == "none"


def test_a_refused_question_is_attributed_to_nothing() -> None:
    """A refusal costs coverage, not correctness, and the curve is where that shows up."""
    refused = outcome(
        decided=GateOutcome.ABSTAIN,
        cited_revisions=frozenset({"B"}),
        in_force_revisions=frozenset({"C"}),
    )
    assert attribute_failure(refused) == "none"


def test_a_wrong_answer_whose_passage_was_never_retrieved_is_a_retrieval_failure() -> None:
    missed = outcome(
        cited_revisions=frozenset({"B"}),
        in_force_revisions=frozenset({"C"}),
        supporting_chunk_exists=True,
        supporting_chunk_retrieved=False,
    )
    assert attribute_failure(missed) == "retrieval"


def test_a_wrong_answer_from_retrieved_evidence_is_a_generation_failure() -> None:
    fumbled = outcome(
        cited_revisions=frozenset({"B"}),
        in_force_revisions=frozenset({"C"}),
        supporting_chunk_exists=True,
        supporting_chunk_retrieved=True,
    )
    assert attribute_failure(fumbled) == "generation"


def test_an_answer_to_an_unanswerable_question_is_a_gate_failure_not_a_retrieval_failure() -> None:
    """The precedence that matters. Attributing this to retrieval sends somebody hunting for a
    passage the corpus never contained."""
    unanswerable = outcome(
        cited_revisions=frozenset({"B"}),
        in_force_revisions=frozenset(),
        supporting_chunk_exists=False,
        supporting_chunk_retrieved=False,
    )
    assert attribute_failure(unanswerable) == "gate"


def test_answering_over_superseded_evidence_is_a_gate_failure() -> None:
    breached = outcome(
        cited_revisions=frozenset({"B"}),
        in_force_revisions=frozenset({"C"}),
        signals=signals(superseded_present=True),
    )
    assert attribute_failure(breached) == "gate"


def test_answering_over_disagreeing_variants_is_a_gate_failure() -> None:
    breached = outcome(
        cited_variants=frozenset({"XP-400S"}),
        signals=signals(variant_agreement=False),
    )
    assert attribute_failure(breached) == "gate"


def test_attribution_still_works_without_recorded_signals() -> None:
    """Less precise, not wrong: the corpus-level gate failure is still caught."""
    fumbled = outcome(
        cited_revisions=frozenset({"B"}),
        in_force_revisions=frozenset({"C"}),
        signals=None,
    )
    assert attribute_failure(fumbled) == "generation"


def test_the_attribution_counts_carry_every_bucket_and_sum_to_the_question_count() -> None:
    outcomes = [
        outcome("ok"),
        outcome(
            "missed",
            cited_revisions=frozenset({"B"}),
            in_force_revisions=frozenset({"C"}),
            supporting_chunk_retrieved=False,
        ),
        outcome(
            "unanswerable",
            cited_revisions=frozenset({"B"}),
            in_force_revisions=frozenset(),
            supporting_chunk_exists=False,
            supporting_chunk_retrieved=False,
        ),
    ]
    counts = failure_attribution(outcomes)
    assert counts == {"retrieval": 1, "generation": 0, "gate": 1, "none": 1}
    # Summed key by key rather than over `.values()`: a TypedDict's values are typed `object`, and
    # `sum` over them only type-checks by giving up the thing the TypedDict was declared for.
    total = counts["retrieval"] + counts["generation"] + counts["gate"] + counts["none"]
    assert total == len(outcomes)


# ------------------------------------------------------------------------------ by_language


def _trilingual_cases() -> list[RetrievalCase]:
    return [
        case("en1", ("a", "b"), frozenset({"a"}), Language.EN),
        case("en2", ("c",), frozenset({"c"}), Language.EN),
        case("tr1", ("x", "b"), frozenset({"b"}), Language.TR),
        case("tr2", ("y",), frozenset({"z"}), Language.TR),
        case("ru1", ("m",), frozenset({"n"}), Language.RU),
        case("ru2", ("p",), frozenset({"q"}), Language.RU),
    ]


def _trilingual_outcomes() -> list[AnswerOutcome]:
    return [
        outcome("en1", language=Language.EN),
        outcome("en2", language=Language.EN),
        outcome(
            "tr1",
            language=Language.TR,
            cited_revisions=frozenset({"B"}),
            in_force_revisions=frozenset({"C"}),
        ),
        outcome("tr2", language=Language.TR),
        outcome("ru1", language=Language.RU, decided=GateOutcome.ABSTAIN),
        outcome(
            "ru2",
            language=Language.RU,
            cited_revisions=frozenset({"B"}),
            in_force_revisions=frozenset({"C"}),
        ),
    ]


def test_by_language_reports_retrieval_and_answering_separately_for_each_language() -> None:
    """The blueprint asks for the degradation split, not one multilingual number."""
    scores = by_language(_trilingual_cases(), _trilingual_outcomes())

    assert set(scores) == {Language.EN, Language.TR, Language.RU}
    for language in (Language.EN, Language.TR, Language.RU):
        assert "retrieval_recall_at_10" in scores[language]
        assert "answer_correctness" in scores[language]

    assert scores[Language.EN]["retrieval_recall_at_10"] == 1.0
    assert scores[Language.TR]["retrieval_recall_at_10"] == 0.5
    assert scores[Language.RU]["retrieval_recall_at_10"] == 0.0


def test_answer_correctness_is_conditional_on_having_answered() -> None:
    """Russian answered one of two questions and got it wrong: correctness 0.0 over one answer,
    end-to-end 0.0 over two questions, coverage 0.5."""
    russian = by_language(_trilingual_cases(), _trilingual_outcomes())[Language.RU]
    assert russian["answered"] == 1
    assert russian["answer_correctness"] == 0.0
    assert russian["end_to_end_correctness"] == 0.0
    assert russian["coverage"] == 0.5
    assert russian["abstention_rate"] == 0.5


def test_answer_correctness_and_the_wrong_answer_rate_are_complements() -> None:
    turkish = by_language(_trilingual_cases(), _trilingual_outcomes())[Language.TR]
    turkish_outcomes = [item for item in _trilingual_outcomes() if item.language is Language.TR]
    assert turkish["answer_correctness"] == 1.0 - wrong_answer_rate(turkish_outcomes)


def test_the_artifact_keys_are_the_ones_the_kill_test_reads() -> None:
    scores = by_language(_trilingual_cases(), _trilingual_outcomes())
    serialisable = {language.value: block for language, block in scores.items()}
    assert set(serialisable) >= {"en", "tr", "ru"}


def test_by_language_refuses_a_run_that_lost_a_language() -> None:
    """Defaulting to the observed languages would publish a tidy two-language result instead."""
    english_only_cases = [item for item in _trilingual_cases() if item.language is Language.EN]
    english_only = [item for item in _trilingual_outcomes() if item.language is Language.EN]
    with pytest.raises(ValueError, match="no answer outcomes in ru"):
        by_language(english_only_cases, english_only)


def test_by_language_measures_a_subset_when_one_is_asked_for_explicitly() -> None:
    english_only_cases = [item for item in _trilingual_cases() if item.language is Language.EN]
    english_only = [item for item in _trilingual_outcomes() if item.language is Language.EN]
    scores = by_language(english_only_cases, english_only, languages=[Language.EN])
    assert set(scores) == {Language.EN}


def test_by_language_refuses_a_language_with_nothing_scorable_to_retrieve() -> None:
    cases = [
        case("en1", ("a",), frozenset({"a"}), Language.EN),
        case("tr1", ("x",), frozenset(), Language.TR),
    ]
    outcomes = [outcome("en1", language=Language.EN), outcome("tr1", language=Language.TR)]
    with pytest.raises(ValueError, match="no scorable retrieval cases in tr"):
        by_language(cases, outcomes, languages=[Language.EN, Language.TR])
