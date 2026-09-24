"""Running every question through the real pipeline, and the four baselines beside it.

One rule shapes this module: **the system and the baselines differ in exactly one thing each.** A
baseline built from a different retriever, a different gate or a different corpus would produce a
comparison that measures the differences nobody meant to make. So all five arms share the candidate
fetch, the corpus and the scoring, and each baseline removes precisely one component:

| arm | what is removed |
|---|---|
| `bm25_only` | the dense signal |
| `dense_only` | the lexical signal |
| `hybrid_without_effectivity` | the temporal and variant predicate — the sole-home skill |
| `ungated_rag` | the gate; every question is answered |

`hybrid_without_effectivity` is the one that matters. It isolates what the effectivity filter buys,
which is the whole argument for this project existing rather than a hybrid-search demo existing.

The runner never decides whether a number passes. It writes what it measured;
`tests/test_kill_criteria.py` — committed before any of this — decides.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from parts_answer_gate import gate as answer_gate
from parts_answer_gate.answerer import ExtractiveAnswerer, PostValidatedAnswerer
from parts_answer_gate.domain import (
    Answer,
    GateDecision,
    GateOutcome,
    Language,
    Query,
    RetrievedChunk,
)
from parts_answer_gate.evaluation.metrics import AnswerOutcome, RetrievalCase
from parts_answer_gate.gate import GateThresholds
from parts_answer_gate.retrieval.pipeline import RetrievalResult, Retriever
from parts_answer_gate.store.schema import DocumentRow

__all__ = [
    "ARMS",
    "ArmName",
    "QuestionRun",
    "RunSet",
    "force_answer",
    "is_hard_block",
    "revision_key",
    "run_arm",
    "run_question",
]

ArmName = str

#: The system plus the four baselines ADR-001 named before any of them was implemented. Iterating a
#: constant rather than calling four functions means a baseline cannot be quietly skipped: the
#: artifact records one block per entry here, and the kill test asserts the set.
ARMS: tuple[ArmName, ...] = (
    "system",
    "bm25_only",
    "dense_only",
    "hybrid_without_effectivity",
    "ungated_rag",
)

#: Wrapped even for the extractive arm. The guarantee has to belong to the system rather than to
#: whichever answerer happens to be plugged in today.
_ANSWERER = PostValidatedAnswerer(ExtractiveAnswerer())


@dataclass(frozen=True)
class QuestionRun:
    """Everything one question produced under one arm, kept rather than reduced.

    The retrieved chunks and the gate signals are carried because the artifacts need them for
    different things — error attribution, the effectivity replay, the groundedness check — and
    re-running the pipeline once per artifact would be both slow and a chance for two artifacts to
    describe different runs.
    """

    question_id: str
    language: Language
    query: Query
    retrieved: tuple[RetrievedChunk, ...]
    answer: Answer
    result: RetrievalResult
    latency_ms: float
    #: `family/revision` for every document this answer cited. Carried rather than re-derived
    #: because the family is only in hand while the document rows are loaded, and a second lookup
    #: later is a second chance to look up the wrong thing.
    cited_revision_keys: frozenset[str] = frozenset()

    @property
    def outcome(self) -> GateOutcome:
        return self.answer.decision.outcome

    @property
    def retrieved_ids(self) -> tuple[str, ...]:
        return tuple(item.chunk.chunk_id for item in self.retrieved)

    @property
    def cited_chunk_ids(self) -> frozenset[str]:
        return frozenset(citation.chunk_id for citation in self.answer.citations)


@dataclass(frozen=True)
class RunSet:
    """One arm over one question set."""

    arm: ArmName
    runs: tuple[QuestionRun, ...]
    retrieval_cases: tuple[RetrievalCase, ...]
    answer_outcomes: tuple[AnswerOutcome, ...]
    timings_ms: tuple[float, ...] = field(default_factory=tuple)

    def digest(self) -> str:
        """A hash over retrieval order and gate decisions, for kill condition J.

        Retrieval identifiers **in order** rather than as a set: a retriever whose ranking wobbled
        between runs while returning the same ten chunks would pass a set comparison and fail every
        reasonable reading of "two runs agree".
        """
        payload = [
            {
                "question_id": run.question_id,
                "retrieved": list(run.retrieved_ids),
                "outcome": str(run.outcome),
                "citations": sorted(run.cited_chunk_ids),
            }
            for run in sorted(self.runs, key=lambda r: r.question_id)
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()


def is_hard_block(decision: GateDecision, thresholds: GateThresholds | None = None) -> bool:
    """Whether this refusal is one the swept threshold could not lift.

    The swept threshold is term coverage. Every *other* abstain or review rule in `gate._RULES` is
    independent of it, so a question those rules refused must be refused at every point on the
    curve — relaxing coverage would not turn any of them into an answer.

    The distinction is what makes the published curve monotonic in coverage. Encoding a hard block
    as a very low score instead would let it drift back above the line at a low threshold, and the
    curve would show coverage rising and falling for reasons nobody could read off the data.

    **This has to enumerate the shipped gate's rules exactly, and for the first iteration it did
    not.** It covered four of the six non-coverage rules and missed two: the top-fused-score floor
    and the support floor. A question refused by either of those kept `hard_blocked=False` and a
    high term coverage, so the curve answered it at every threshold below that coverage — and,
    because an abstention approves no chunks and therefore cites nothing, `revision_incorrect`
    scored it as a wrong answer that was never given. The published curve was neither the shipped
    gate's curve nor a curve of anything else. ADR-002.

    `thresholds` defaults to the shipped configuration, which is what the evaluation runs.
    """
    limits = thresholds or GateThresholds()
    signals = decision.signals

    required_support = (
        limits.min_supporting_chunks_with_exact_hit
        if signals.exact_identifier_hit
        else limits.min_supporting_chunks
    )
    return (
        signals.supporting_chunks == 0
        or signals.superseded_present
        or not signals.variant_agreement
        or signals.conflicting_evidence
        # The two the first version missed. Both are `gate._RULES` entries, both are ABSTAIN, and
        # neither is reachable by moving term coverage.
        or (
            signals.top_fused_score <= limits.min_top_fused_score
            and not signals.exact_identifier_hit
        )
        or signals.supporting_chunks < required_support
    )


def force_answer(
    decision: GateDecision, retrieved: Sequence[RetrievedChunk], limit: int = 5
) -> GateDecision:
    """The `ungated_rag` baseline: answer regardless of what the gate concluded.

    Built here, by replacing the decision, rather than by adding a flag inside `gate.decide`. A flag
    would be a branch the shipped system carries for the benefit of a baseline, and the gate under
    test has to be the gate that ships — otherwise kill condition H compares the real system against
    a variant of itself that only exists during evaluation.

    The signals are carried over unchanged, so the recorded reason still shows what the real gate
    thought. That matters when reading the failure cases: an ungated wrong answer is more damning
    when the gate is on record as having wanted to refuse it.
    """
    return GateDecision(
        outcome=GateOutcome.ANSWER,
        reason="ungated baseline: answered regardless of the gate's decision",
        signals=decision.signals,
        approved_chunks=tuple(retrieved[:limit]),
    )


def _query_for(question: Mapping[str, Any]) -> Query:
    """A question record as a `Query`.

    `as_of` comes from the question, never from the clock. A benchmark whose as-of date drifted with
    the wall clock would return different passages next Tuesday and the hold-out score would quietly
    stop being reproducible — which is exactly the failure this project's own corpus exists to
    surface in other systems.
    """
    return Query(
        text=str(question["text"]),
        language=Language(question["language"]),
        as_of=date.fromisoformat(str(question["as_of"])),
        variant_id=question.get("variant_id") or None,
        serial=question.get("serial"),
        top_k=int(question.get("top_k", 10)),
    )


def _document_facts(
    session: Session, chunks: Sequence[RetrievedChunk]
) -> dict[str, tuple[str, str]]:
    """document id -> (revision, family id), for the answerer and for the scoring key.

    Both come from one query because they describe one row. Fetching the revision here and the
    family somewhere else is how the two drift into describing different documents.
    """
    document_ids = {item.chunk.document_id for item in chunks}
    if not document_ids:
        return {}
    rows = session.execute(
        select(DocumentRow.document_id, DocumentRow.revision, DocumentRow.family_id).where(
            DocumentRow.document_id.in_(document_ids)
        )
    ).all()
    return {str(row[0]): (str(row[1]), str(row[2])) for row in rows}


def revision_key(family_id: str, revision: str) -> str:
    """The identity the wrong-answer rule compares, qualified by family.

    Revision labels are **not unique**: this corpus uses five of them (`A`, `B`, `C`, `D`, `FB1`)
    across 108 documents in 9 families, because a revision letter identifies a document's place in
    its own family's history and nothing more. Comparing the bare labels asks "did the answer cite
    something called C?" when the question is "did the answer cite *this family's* C, and was that
    in force?" — and any citation of another family's C would have read as correct.

    Qualifying by family makes the comparison an identity again. It can only ever move a measured
    rate upward, never downward, which is the property that makes it safe to apply to a hold-out
    that has already been scored: it cannot flatter the result.
    """
    return f"{family_id}/{revision}"


def _retrieve_for_arm(
    retriever: Retriever, session: Session, query: Query, arm: ArmName
) -> RetrievalResult:
    """The one component each baseline removes, and nothing else.

    `hybrid_without_effectivity` **removes the predicate**, through the same retriever, and that is
    the only honest way to express it.

    An earlier version simulated removal by copying the query with `as_of=2099-12-31` and leaving
    the predicate in place. That is strictly *more* filtered, not less: the half-open temporal test
    `valid_from <= as_of < valid_to` admits only rows with no end date once the date is past every
    withdrawal, so the arm deleted the gold passage for every question whose answer had since been
    superseded, and the 0.1714 recall it reported measured a query issued at the wrong date rather
    than a retriever working without effectivity. ADR-002 records it.
    """
    if arm == "bm25_only":
        return retriever.retrieve(session, query, weights={"dense": 0.0})
    if arm == "dense_only":
        return retriever.retrieve(session, query, weights={"lexical": 0.0})
    if arm == "hybrid_without_effectivity":
        return retriever.retrieve(session, query, apply_effectivity=False)
    return retriever.retrieve(session, query)


def run_question(
    retriever: Retriever,
    session: Session,
    question: Mapping[str, Any],
    arm: ArmName = "system",
) -> QuestionRun:
    """One question, end to end, under one arm."""
    query = _query_for(question)
    started = time.perf_counter()

    result = _retrieve_for_arm(retriever, session, query, arm)
    decision = answer_gate.decide(query, result.chunks)

    if arm == "ungated_rag":
        # The baseline that answers regardless. Built by replacing the *decision*, never by editing
        # the gate: the gate under test must be the one that ships, and a flag inside it would be a
        # branch the shipped system carries for the benefit of a baseline.
        decision = force_answer(decision, result.chunks)

    facts = _document_facts(session, decision.approved_chunks)
    revisions = {document_id: revision for document_id, (revision, _) in facts.items()}
    answer = (
        Answer(query=query, decision=decision)
        if decision.outcome is GateOutcome.ABSTAIN
        else _ANSWERER.answer(query, decision, revisions)
    )

    # The citation itself keeps the bare revision, because that is what a technician reads on the
    # document in front of them. The scoring key is separate and qualified.
    cited = frozenset(citation.chunk_id for citation in answer.citations)
    keys = frozenset(
        revision_key(facts[item.chunk.document_id][1], facts[item.chunk.document_id][0])
        for item in decision.approved_chunks
        if item.chunk.chunk_id in cited and item.chunk.document_id in facts
    )

    return QuestionRun(
        question_id=str(question["question_id"]),
        language=query.language,
        query=query,
        retrieved=result.chunks,
        answer=answer,
        result=result,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        cited_revision_keys=keys,
    )


def _retrieval_case(question: Mapping[str, Any], run: QuestionRun) -> RetrievalCase:
    supporting = frozenset(str(c) for c in question.get("supporting_chunk_ids", []) or [])
    return RetrievalCase(
        question_id=run.question_id,
        retrieved_ids=run.retrieved_ids,
        relevant_ids=supporting,
        language=run.language,
    )


def _answer_outcome(
    question: Mapping[str, Any],
    run: QuestionRun,
    in_force_revisions: frozenset[str],
) -> AnswerOutcome:
    """The record the metrics module scores, with wrongness left for it to derive.

    Every field here is a fact about what happened. Not one of them is a verdict — `metrics` decides
    what counts as wrong, from ADR-001's definition, so the definition cannot drift by a caller
    computing it slightly differently.
    """
    supporting = frozenset(str(c) for c in question.get("supporting_chunk_ids", []) or [])
    decision = run.answer.decision

    return AnswerOutcome(
        question_id=run.question_id,
        language=run.language,
        outcome=decision.outcome,
        gate_score=decision.signals.term_coverage,
        cited_revisions=run.cited_revision_keys,
        in_force_revisions=in_force_revisions,
        cited_variants=frozenset(
            item.chunk.effectivity.variant_id
            for item in decision.approved_chunks
            if item.chunk.chunk_id in run.cited_chunk_ids
        ),
        asked_variant=run.query.variant_id,
        supporting_chunk_exists=bool(supporting),
        supporting_chunk_retrieved=bool(supporting & set(run.retrieved_ids)),
        hard_blocked=is_hard_block(decision),
        signals=decision.signals,
    )


def run_arm(
    retriever: Retriever,
    session: Session,
    questions: Iterable[Mapping[str, Any]],
    in_force: Mapping[str, frozenset[str]],
    arm: ArmName = "system",
) -> RunSet:
    """Every question under one arm, with the records the metrics module needs.

    `in_force` maps a question id to the revisions that genuinely support it at its as-of date. It
    comes from the generator's construction metadata rather than from anything this pipeline
    produced — a ground truth derived from the system under test is not a ground truth.
    """
    runs: list[QuestionRun] = []
    cases: list[RetrievalCase] = []
    outcomes: list[AnswerOutcome] = []

    for question in questions:
        run = run_question(retriever, session, question, arm)
        runs.append(run)
        cases.append(_retrieval_case(question, run))
        outcomes.append(_answer_outcome(question, run, in_force.get(run.question_id, frozenset())))

    return RunSet(
        arm=arm,
        runs=tuple(runs),
        retrieval_cases=tuple(cases),
        answer_outcomes=tuple(outcomes),
        timings_ms=tuple(run.latency_ms for run in runs),
    )
