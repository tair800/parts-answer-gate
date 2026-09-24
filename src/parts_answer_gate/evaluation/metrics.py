"""Scoring, and nothing else.

This module imports `domain` and the standard library. It does not import a retriever, a store, a
database driver or a corpus loader, and it never reads a file. That is deliberate: the numbers that
decide whether this project ships are produced by functions that can be unit-tested against inputs
written by hand in a test file, with no service running and no embedding model downloaded. A metric
that can only be exercised by running the whole pipeline is a metric nobody checks.

Two measurement choices are argued at length in the docstrings below because they are the two places
where a flattering number is easiest to publish by accident:

- **the denominator of the wrong-answer rate** — wrong answers per *answer given*, not per question
  asked, so that abstaining cannot be mistaken for being right (`wrong_answer_rate`);
- **what happens when a question has no relevant passage** — an error rather than a 1.0, so that the
  ninety deliberately unanswerable questions cannot inflate hold-out recall (`recall_at_k`).

`RETRIEVAL_K` mirrors the constant of the same meaning in `tests/test_kill_criteria.py`. That file
is authoritative; this one follows it. They are separate on purpose — the code that computes a
number and the code that grades it should not share a definition that could be edited once to move
both.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

from parts_answer_gate.domain import GateOutcome, GateSignals, Language

__all__ = [
    "RETRIEVAL_K",
    "AnswerOutcome",
    "AttributionCounts",
    "CurvePoint",
    "FailureStage",
    "LanguageScores",
    "RetrievalCase",
    "RetrievalSummary",
    "abstention_rate",
    "attribute_failure",
    "by_language",
    "coverage",
    "coverage_curve",
    "default_thresholds",
    "failure_attribution",
    "mrr",
    "ndcg_at_k",
    "non_answer_rate",
    "recall_at_k",
    "review_rate",
    "summarise_retrieval",
    "ungated_wrong_answer_rate",
    "wrong_answer_rate",
]

#: The cut-off every published retrieval figure in this project uses. ADR-001 kill condition F is
#: stated at ten, so ten is what the artifact reports.
RETRIEVAL_K = 10

#: Which stage a wrong answer is charged to. `none` is not "unknown" — it is "this answer was not
#: wrong", and keeping it in the same alphabet as the failures means the four counts always sum to
#: the number of answers examined.
FailureStage = Literal["retrieval", "generation", "gate", "none"]


# --------------------------------------------------------------------------------- the records


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    """One question's retrieval result, judged against the passages that actually support it.

    `relevant_ids` comes from the generator's construction metadata, not from a judgement about the
    output: the corpus knows which chunk supports which question because it built them together.

    `language` has no default. A default would file every case whose language the caller forgot to
    set under one language, and the multilingual artifact would then report a degradation that is
    really a plumbing bug.
    """

    question_id: str
    retrieved_ids: tuple[str, ...]
    relevant_ids: frozenset[str]
    language: Language


@dataclass(frozen=True, slots=True)
class AnswerOutcome:
    """One question's answering result, carrying enough to decide wrongness without the corpus.

    ADR-001 defines a wrong answer as **revision-incorrect or variant-incorrect** — not "differs
    from the gold string". So the record carries the four facts that decision needs (what the answer
    cited, what was in force, which variant the evidence belonged to, which variant was asked about)
    and this module derives the verdict. A caller that supplied a ready-made `is_wrong` boolean
    would be free to compute it by a looser rule, and the definition would quietly stop being the
    one in the ADR.

    `cited_revisions` and `cited_variants` must be filled for **every** question, including the ones
    the shipped gate refused. The coverage curve re-decides each question at every threshold, so a
    question with no candidate recorded is scored as wrong at the thresholds that would have
    answered it. That direction is conservative by choice: a curve that assumed an uncited answer
    was correct would slope upward for free.

    `hard_blocked` marks a refusal no threshold can lift — superseded evidence, disagreeing
    variants, nothing retrieved at all. Keeping it separate from the score is what makes coverage
    monotonic in the threshold, which is the property the published curve is checked for.
    """

    question_id: str
    language: Language
    outcome: GateOutcome
    #: The gate's deterministic confidence, on whatever scale the gate uses. Never a model output.
    gate_score: float
    cited_revisions: frozenset[str]
    #: The revisions in force at the query's as-of date that genuinely support this question. Empty
    #: for a question the corpus cannot answer, which is what makes citing anything at all wrong.
    in_force_revisions: frozenset[str]
    cited_variants: frozenset[str]
    asked_variant: str | None
    #: Whether the corpus contains a supporting passage at all. False for the unanswerable set.
    supporting_chunk_exists: bool
    #: Whether a supporting passage reached the candidate set. Meaningless when none exists.
    supporting_chunk_retrieved: bool
    hard_blocked: bool = False
    #: The gate's recorded signals, when the runner kept them. Used only to separate a gate failure
    #: from a generation failure; attribution still works without them, less precisely.
    signals: GateSignals | None = None

    @property
    def answered(self) -> bool:
        """Whether the shipped system presented this as an answer.

        `REVIEW` is not an answer. ADR-001 is explicit that nothing is presented while a question
        sits in review, so counting it as coverage would credit the system for output a technician
        never received.
        """
        return self.outcome is GateOutcome.ANSWER

    @property
    def revision_incorrect(self) -> bool:
        """Whether the answer leaned on a revision that was not in force at the as-of date.

        An answer citing nothing counts as revision-incorrect. `domain.Answer` already refuses an
        `ANSWER` without citations, so this only bites on counterfactual points of the coverage
        curve — and there the conservative reading is the honest one.
        """
        if not self.cited_revisions:
            return True
        return bool(self.cited_revisions - self.in_force_revisions)

    @property
    def variant_incorrect(self) -> bool:
        """Whether the evidence belonged to a machine the question was not about.

        The second clause — several variants cited when the technician named none — is deliberate.
        Every chunk in this corpus carries an `Effectivity`, so an answer assembled from two
        variants describes no single machine, and "they did not say which machine" is not a licence
        to mix two.
        """
        if self.asked_variant is not None:
            return bool(self.cited_variants - {self.asked_variant})
        return len(self.cited_variants) > 1

    @property
    def would_be_wrong(self) -> bool:
        """Wrongness of the candidate answer, independent of whether the gate let it through.

        The coverage curve needs this counterfactual: at a lower threshold a refused question
        becomes an answer, and its wrongness has to be known before the sweep, not after.
        """
        return self.revision_incorrect or self.variant_incorrect

    @property
    def is_wrong(self) -> bool:
        """A wrong answer, in the sense ADR-001 kill condition G counts: wrong *and* delivered."""
        return self.answered and self.would_be_wrong

    def answered_at(self, threshold: float) -> bool:
        """Whether a gate set to `threshold` would have answered this question."""
        return not self.hard_blocked and self.gate_score >= threshold


@dataclass(frozen=True, slots=True)
class RetrievalSummary:
    """Retrieval over a question set, with the questions it could not score counted rather than
    dropped silently.

    `skipped_without_relevant` exists so the artifact can state how many questions were excluded.
    Averaging recall over "all questions" while quietly having discarded ninety of them is the
    single easiest way to publish a recall figure that means nothing.
    """

    k: int
    scored: int
    skipped_without_relevant: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


class CurvePoint(TypedDict):
    """One point of the published coverage-versus-wrong-answer curve.

    The three keys `tests/test_kill_criteria.py` requires are the first three. `answered` and
    `wrong` are carried as well because a wrong-answer rate over a handful of answers is not the
    same evidence as the same rate over three hundred, and a curve that hides its denominators
    invites exactly that misreading.
    """

    threshold: float
    coverage: float
    wrong_answer_rate: float
    answered: int
    wrong: int


class AttributionCounts(TypedDict):
    """How many wrong answers each stage owns, plus the answers that were not wrong."""

    retrieval: int
    generation: int
    gate: int
    none: int


class LanguageScores(TypedDict):
    """One language's block of the multilingual artifact.

    `retrieval_recall_at_10` and `answer_correctness` are the two keys the kill test requires: the
    blueprint asks for the degradation *split* into retrieval and generation, and a single
    end-to-end number per language cannot be split after the fact.

    Both readings of correctness are published rather than one, because they answer different
    questions and the choice between them is exactly where a multilingual claim gets flattered:
    `answer_correctness` is conditional on the system having answered (the generation half, holding
    coverage aside), `end_to_end_correctness` divides by every question asked (what a technician
    working in that language actually experiences).
    """

    retrieval_recall_at_10: float
    retrieval_mrr: float
    retrieval_ndcg_at_10: float
    answer_correctness: float
    end_to_end_correctness: float
    coverage: float
    abstention_rate: float
    questions: int
    answered: int
    retrieval_cases_scored: int


# ------------------------------------------------------------------------------------ retrieval


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean over a non-empty sequence, in the order given.

    Summation order is the caller's order and is not sorted, so two runs over the same input produce
    bit-identical floats. ADR-001 kill condition J asks for byte-identical runs, and a metric that
    reorders its own addends can break that on its own.
    """
    if not values:
        raise ValueError("cannot average an empty sequence; there is no measurement here")
    return math.fsum(values) / len(values)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float:
    """Share of a question's supporting passages that appear in the top `k`.

    **Recall is the primary retrieval metric for this task.** A technician needs the supporting
    passage to be *in front of them*; whether it arrived third or seventh changes almost nothing,
    because the answer is read out of the passage either way. Rank-weighted metrics optimise for a
    click-through behaviour this interface does not have, and a system tuned to lift a passage from
    rank seven to rank two while losing a different passage entirely would score better on nDCG and
    worse at the job. `mrr` and `ndcg_at_k` are published alongside as diagnostics — they are what
    distinguishes "found it, ranked it badly" from "did not find it" — but kill condition F is
    stated on recall.

    An **empty `relevant_ids` raises** rather than returning 1.0 or 0.0. The hold-out contains
    ninety questions the corpus deliberately cannot answer; scoring them as vacuously perfect would
    add ninety 1.0s to the mean and push recall over the floor without retrieving anything, and
    scoring them as 0.0 would punish the system for correctly having nothing to find. Neither is a
    measurement, so the caller is made to exclude them — `summarise_retrieval` does, and counts how
    many it excluded.

    An empty `retrieved_ids` is a real result, not an error: the retriever ran and found nothing.
    Duplicates in `retrieved_ids` are credited once, so a retriever that returns the same chunk
    twice cannot buy recall with it.
    """
    if k <= 0:
        raise ValueError(f"k={k} is not a cut-off; recall@0 measures nothing")
    if not relevant_ids:
        raise ValueError(
            "recall is undefined for a question with no supporting passage; exclude the "
            "unanswerable set from retrieval scoring instead of giving it a score"
        )
    found = {chunk_id for chunk_id in retrieved_ids[:k] if chunk_id in relevant_ids}
    return len(found) / len(relevant_ids)


def mrr(retrieved_ids: Sequence[str], relevant_ids: frozenset[str]) -> float:
    """Reciprocal rank of the first supporting passage, over the whole retrieved list.

    This is the reciprocal rank of **one** question; the mean that puts the M in MRR is taken over
    questions by `summarise_retrieval`. Kept per-question so a failure can be pointed at.

    Not truncated at `k`: the diagnostic value here is precisely "how far down was it", and a
    truncated version answers that with 0.0 for every question that missed the cut, which is the
    information recall already carries.

    Returns 0.0 when nothing relevant was retrieved — a defined result, unlike an empty relevant
    set, which raises for the reason given in `recall_at_k`.
    """
    if not relevant_ids:
        raise ValueError("reciprocal rank is undefined for a question with no supporting passage")
    for position, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: frozenset[str], k: int) -> float:
    """Normalised discounted cumulative gain at `k`, with binary relevance.

    Relevance is binary because the ground truth is construction metadata: the generator knows a
    chunk either supports a question or does not. Inventing graded relevance levels over synthetic
    data would produce a more sophisticated-looking metric measuring a distinction nobody recorded.

    The ideal ranking is capped at `min(len(relevant_ids), k)`, so a question with twelve supporting
    passages and a cut-off of ten can still reach 1.0 by filling the top ten with supporting
    passages. Normalising against an unreachable ideal would make nDCG a function of how many
    passages happen to support a question.

    A repeated chunk id is credited once: gain is per distinct supporting passage.
    """
    if k <= 0:
        raise ValueError(f"k={k} is not a cut-off; nDCG@0 measures nothing")
    if not relevant_ids:
        raise ValueError("nDCG is undefined for a question with no supporting passage")

    credited: set[str] = set()
    gains: list[float] = []
    for position, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids and chunk_id not in credited:
            credited.add(chunk_id)
            gains.append(1.0 / math.log2(position + 1))
    ideal = [1.0 / math.log2(position + 1) for position in range(1, min(len(relevant_ids), k) + 1)]
    return math.fsum(gains) / math.fsum(ideal)


def summarise_retrieval(cases: Iterable[RetrievalCase], k: int = RETRIEVAL_K) -> RetrievalSummary:
    """Mean recall, reciprocal rank and nDCG over a question set.

    Questions with no supporting passage are excluded and counted, for the reason argued in
    `recall_at_k`. Raises if nothing is left to score, because a summary over zero questions is a
    number with no measurement behind it.
    """
    # Materialised once: the argument is an iterable, and scoring it after having already counted it
    # would silently summarise an exhausted generator as an empty question set.
    all_cases = list(cases)
    scorable = [case for case in all_cases if case.relevant_ids]
    skipped = len(all_cases) - len(scorable)
    if not scorable:
        raise ValueError(
            f"none of the {len(all_cases)} cases has a supporting passage, so there is nothing to "
            "score; a retrieval summary over only unanswerable questions is not a result"
        )
    recalls = [recall_at_k(case.retrieved_ids, case.relevant_ids, k) for case in scorable]
    ranks = [mrr(case.retrieved_ids, case.relevant_ids) for case in scorable]
    gains = [ndcg_at_k(case.retrieved_ids, case.relevant_ids, k) for case in scorable]
    return RetrievalSummary(
        k=k,
        scored=len(scorable),
        skipped_without_relevant=skipped,
        recall_at_k=_mean(recalls),
        mrr=_mean(ranks),
        ndcg_at_k=_mean(gains),
    )


# ------------------------------------------------------------------------------------ answering


def _materialise(outcomes: Iterable[AnswerOutcome]) -> tuple[AnswerOutcome, ...]:
    """Consume an iterable once, and refuse an empty one.

    Every rate below divides by a count taken from this tuple. An empty input would make each of
    them either a `ZeroDivisionError` or, worse, a zero that reads like a measured result.
    """
    items = tuple(outcomes)
    if not items:
        raise ValueError("no answer outcomes; a rate over zero questions is not a measurement")
    return items


def coverage(outcomes: Iterable[AnswerOutcome]) -> float:
    """Share of questions the system answered at all.

    Counts `ANSWER` only. `REVIEW` is excluded because ADR-001 says nothing is presented as an
    answer while a question sits in review: a system that routed everything to a human queue has
    zero coverage, which is the truth about what it delivered.
    """
    items = _materialise(outcomes)
    return sum(1 for item in items if item.answered) / len(items)


def abstention_rate(outcomes: Iterable[AnswerOutcome]) -> float:
    """Share of questions the system explicitly refused.

    Strict: `ABSTAIN` only, never `ABSTAIN + REVIEW`. This is the figure ADR-001 kill condition I is
    graded on, and the strict reading is the right one there. `REVIEW` asserts something specific —
    that the corpus holds conflicting evidence a person must weigh — and for a question the corpus
    cannot support at all that assertion is false and costs somebody an hour. Routing the
    unanswerable set into a review queue is not the same as refusing it, and the metric should not
    let it look the same.

    Not `1 - coverage`: with three outcomes, the two do not complement each other. `review_rate`
    carries the remainder and `non_answer_rate` carries the union.
    """
    items = _materialise(outcomes)
    return sum(1 for item in items if item.outcome is GateOutcome.ABSTAIN) / len(items)


def review_rate(outcomes: Iterable[AnswerOutcome]) -> float:
    """Share of questions escalated to a person. Reported because it is a staffing cost."""
    items = _materialise(outcomes)
    return sum(1 for item in items if item.outcome is GateOutcome.REVIEW) / len(items)


def non_answer_rate(outcomes: Iterable[AnswerOutcome]) -> float:
    """Share of questions that produced no answer, by either route. `1 - coverage`, named."""
    items = _materialise(outcomes)
    return sum(1 for item in items if not item.answered) / len(items)


def wrong_answer_rate(outcomes: Iterable[AnswerOutcome]) -> float:
    """Wrong answers per answer given, where wrong means revision-incorrect or variant-incorrect.

    **The denominator is answers given, not questions asked.** This is the risk half of a
    risk-coverage pair and the only denominator under which the pair is informative. Dividing by
    every question asked would let a system drive the headline number toward zero by abstaining more
    — the abstentions would enter the denominator and leave the numerator — and the figure ADR-001
    kill condition G caps at 0.02 would then be measuring reticence rather than accuracy. With this
    denominator, refusing a question changes neither side, and the cost of refusing shows up where
    it belongs: in `coverage`.

    Wrongness is derived here from the four recorded facts rather than taken from the caller, so the
    definition stays the one in ADR-001. A near-miss on wording is not wrong; the right procedure
    from a withdrawn revision is.

    Returns 0.0 when nothing was answered. A system that refused everything made no wrong answers,
    which is true and useless — every curve point carries `answered` beside the rate so the reader
    can see that denominator rather than infer it.
    """
    items = _materialise(outcomes)
    answered = [item for item in items if item.answered]
    if not answered:
        return 0.0
    return sum(1 for item in answered if item.would_be_wrong) / len(answered)


def ungated_wrong_answer_rate(outcomes: Iterable[AnswerOutcome]) -> float:
    """The same numerator rule with the gate removed: every question is answered.

    This is the `ungated_rag` baseline of ADR-001 and the comparison behind kill condition H. It is
    computed here rather than by the runner so that both sides of that ratio use one definition of
    wrongness — a gated rate and an ungated rate computed by different code is how a five-times
    improvement gets manufactured.
    """
    items = _materialise(outcomes)
    return sum(1 for item in items if item.would_be_wrong) / len(items)


def default_thresholds(outcomes: Iterable[AnswerOutcome], points: int = 11) -> tuple[float, ...]:
    """Evenly spaced gate thresholds spanning the observed score range, ascending.

    Spans the scores actually observed rather than an assumed `0..1`, because the gate's score is a
    fusion of six signals on no particular scale and a sweep over a range the scores never enter
    would publish a curve of two flat ends.

    Raises when every score is identical: there is no threshold that separates anything, so a
    "sweep" over it would be five copies of one point dressed as a curve. That is a broken gate, and
    the honest response is to fail rather than to emit a flat line.
    """
    if points < 5:
        raise ValueError(
            f"{points} points is not a curve; the published artifact needs at least five"
        )
    scores = sorted({item.gate_score for item in outcomes})
    if len(scores) < 2:
        raise ValueError(
            "every gate score is identical, so no threshold separates any question from another; "
            "a sweep over a constant is not a sweep"
        )
    low, high = scores[0], scores[-1]
    span = high - low
    steps = [low + span * index / (points - 1) for index in range(points)]
    # The last step is pinned to the observed maximum rather than left to floating-point
    # accumulation, so the top of the curve is a threshold some question actually reaches.
    steps[-1] = high
    return tuple(steps)


def coverage_curve(
    outcomes: Iterable[AnswerOutcome], thresholds: Sequence[float] | None = None
) -> list[CurvePoint]:
    """The headline artifact: coverage and wrong-answer rate together, over a threshold sweep.

    ADR-001 refuses to publish a single point, because a single point is a choice of threshold
    presented as a result. Every point re-decides every question — answered at `threshold` when the
    gate did not hard-block it and its score clears the line — so the curve is a property of the
    scored question set rather than of the shipped configuration.

    Coverage is non-increasing across the returned points, which are sorted ascending by threshold.
    That is not cosmetic: `tests/test_kill_criteria.py` checks the monotonicity as evidence the
    sweep is a sweep, and the property holds only because a hard block is kept out of the score
    rather than encoded as a very low one.

    The wrong-answer rate is *not* monotonic and is not expected to be. Raising the threshold
    removes low-confidence answers, most of which were wrong and some of which were not.

    Each point carries its own denominators. Requires every outcome to have a candidate answer
    recorded, including the refused ones — see `AnswerOutcome`.
    """
    items = _materialise(outcomes)
    steps = tuple(thresholds) if thresholds is not None else default_thresholds(items)
    if not steps:
        raise ValueError("no thresholds to sweep")

    curve: list[CurvePoint] = []
    for threshold in sorted(steps):
        answered = [item for item in items if item.answered_at(threshold)]
        wrong = [item for item in answered if item.would_be_wrong]
        curve.append(
            CurvePoint(
                threshold=threshold,
                coverage=len(answered) / len(items),
                wrong_answer_rate=(len(wrong) / len(answered) if answered else 0.0),
                answered=len(answered),
                wrong=len(wrong),
            )
        )
    return curve


# ---------------------------------------------------------------------------------- attribution


def attribute_failure(outcome: AnswerOutcome) -> FailureStage:
    """Charge a wrong answer to the stage that could have prevented it.

    The split the blueprint asks for, and the order of these checks is the whole content of it:

    - **`none`** — the answer was not wrong, or was never delivered. Only delivered answers are
      attributed; a question the gate refused cost coverage, not correctness, and the coverage curve
      is where that shows up.
    - **`gate`** — the answer should never have been produced. Either the corpus contains no
      supporting passage at all (ADR-001 kill condition E), or the gate's own recorded signals said
      refuse — superseded evidence present, or the supporting chunks disagreeing about which variant
      they apply to — and it answered anyway. This is checked **before** retrieval on purpose: an
      unanswerable question has no supporting chunk to retrieve, so the retrieval-first ordering
      would file every kill-E violation as a retrieval miss and send somebody hunting for a passage
      that was never written. No amount of better retrieval or better generation fixes these; the
      only correct action was to refuse.
    - **`retrieval`** — a supporting passage exists in the corpus and never reached the candidate
      set. The answerer was asked to work from evidence it did not have.
    - **`generation`** — the supporting passage was retrieved and the answer is still wrong. The
      evidence was in front of the answerer and it produced a revision- or variant-incorrect answer
      from it. For the extractive arm this should be close to unreachable, which is the point: if
      this bucket fills up, the extractive guarantee is not holding.

    Attribution degrades gracefully without `signals`: the corpus-level gate failure is still
    caught, and a signal-level one is charged to retrieval or generation instead. That is a loss of
    precision, not a wrong answer about a wrong answer, and the artifact should say when signals
    were unavailable.
    """
    if not outcome.is_wrong:
        return "none"
    if not outcome.supporting_chunk_exists:
        return "gate"
    if outcome.signals is not None and (
        outcome.signals.superseded_present or not outcome.signals.variant_agreement
    ):
        return "gate"
    if not outcome.supporting_chunk_retrieved:
        return "retrieval"
    return "generation"


def failure_attribution(outcomes: Iterable[AnswerOutcome]) -> AttributionCounts:
    """Counts per stage over a question set, with every bucket present even when it is zero.

    An absent key and a zero read the same way to a careless reader and differently to a parser. The
    four counts sum to the number of outcomes examined.
    """
    counts: AttributionCounts = {"retrieval": 0, "generation": 0, "gate": 0, "none": 0}
    for outcome in outcomes:
        counts[attribute_failure(outcome)] += 1
    return counts


# --------------------------------------------------------------------------------- multilingual


def by_language(
    retrieval_cases: Iterable[RetrievalCase],
    answer_outcomes: Iterable[AnswerOutcome],
    languages: Iterable[Language] | None = None,
) -> dict[Language, LanguageScores]:
    """Per-language retrieval and answering scores, for the multilingual artifact.

    `k` is not a parameter. The artifact's key is spelled `retrieval_recall_at_10`, and a function
    that accepted `k=5` would write that number under a name claiming ten — a lie no test in this
    repository could catch, because every test would be reading the same mislabelled key.

    `languages` defaults to **every member of `Language`**, not to the languages observed in the
    input. `Language` is closed at exactly the three the blueprint scopes, so a run that lost
    Russian somewhere in the pipeline would otherwise publish a tidy two-language result and a
    reader would have to notice the absence. Instead it raises. Pass an explicit set only when
    measuring a subset on purpose.
    """
    cases = list(retrieval_cases)
    outcomes = list(answer_outcomes)
    wanted = tuple(languages) if languages is not None else tuple(Language)

    scores: dict[Language, LanguageScores] = {}
    for language in sorted(wanted, key=lambda member: member.value):
        language_cases = [case for case in cases if case.language is language]
        language_outcomes = [item for item in outcomes if item.language is language]
        if not language_outcomes:
            raise ValueError(
                f"no answer outcomes in {language.value}; a trilingual claim with one language "
                "unmeasured is the failure this artifact exists to expose"
            )
        if not any(case.relevant_ids for case in language_cases):
            raise ValueError(
                f"no scorable retrieval cases in {language.value}; every case either is missing or "
                "has no supporting passage, so there is no retrieval number to publish"
            )

        retrieval = summarise_retrieval(language_cases, RETRIEVAL_K)
        answered = [item for item in language_outcomes if item.answered]
        correct = sum(1 for item in answered if not item.would_be_wrong)
        scores[language] = LanguageScores(
            retrieval_recall_at_10=retrieval.recall_at_k,
            retrieval_mrr=retrieval.mrr,
            retrieval_ndcg_at_10=retrieval.ndcg_at_k,
            answer_correctness=(correct / len(answered) if answered else 0.0),
            end_to_end_correctness=correct / len(language_outcomes),
            coverage=coverage(language_outcomes),
            abstention_rate=abstention_rate(language_outcomes),
            questions=len(language_outcomes),
            answered=len(answered),
            retrieval_cases_scored=retrieval.scored,
        )
    return scores
