"""The evidence files `tests/test_kill_criteria.py` grades, and how each one is earned.

That test was committed before any source file existed and has not been edited since. Nothing here
decides what counts as passing — the thresholds live in the test, and this module's only job is to
produce honest inputs to them.

Each artifact records not only its result but **how the result was obtained**, including where the
method is weaker than it looks. The multilingual one is the clearest case: it can measure retrieval
and answering over three languages, and it cannot tell you whether the Turkish reads like Turkish,
because no native speaker saw it. The artifact says that in its own body rather than in a footnote
somebody may not reach.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

from sqlalchemy.orm import Session

from parts_answer_gate.citations import verify_citation
from parts_answer_gate.domain import GateOutcome, Language, Query, part_numbers_in
from parts_answer_gate.evaluation import metrics
from parts_answer_gate.evaluation.runner import ARMS, RunSet, run_arm
from parts_answer_gate.gate import PRIMARY_THRESHOLD_SWEEP
from parts_answer_gate.holdout import load_frozen
from parts_answer_gate.retrieval.embeddings import Embedder
from parts_answer_gate.retrieval.pipeline import Retriever
from parts_answer_gate.store.comparison import build_storage_comparison_artifact
from parts_answer_gate.store.diagnostics import build_pgvector_artifact
from parts_answer_gate.store.effectivity import (
    FILTER_STAGE,
    CandidateFilter,
    candidate_filter,
)
from parts_answer_gate.store.lifecycle import measure_index_lifecycle
from parts_answer_gate.store.queries import fetch_candidates

__all__ = ["BuildReport", "build_all"]

#: The shipped configuration. Named here so the curve can mark it and the README can quote it
#: without either of them choosing a different number from the one the service runs.
SHIPPED_THRESHOLD: float = 0.60

#: As-of dates the effectivity replay uses. Three at minimum per the kill test, chosen to straddle
#: the corpus's revision boundaries rather than to be far apart: a replay at dates that all fall
#: after every withdrawal would prove nothing.
#:
#: The comment above said exactly that while the dates underneath it did the opposite. The four
#: previous values — Feb, May, Aug 2026 and Jan 2027 — all sat past 186 of the corpus's 192
#: withdrawals, so all four described one identical temporal state and the replay tested one date
#: four times. ADR-002.
#:
#: These five are the 10th, 30th, 50th, 70th and 90th percentiles of the corpus's distinct
#: withdrawal dates, so each one sees a genuinely different set of revisions in force: 6, 17, 28,
#: 39 and 50 of the 55 distinct withdrawals already past, respectively. A date is only worth
#: replaying at if something was in force then that is not in force now.
REPLAY_DATES: tuple[date, ...] = (
    date(2020, 11, 29),
    date(2022, 3, 8),
    date(2023, 1, 4),
    date(2023, 11, 18),
    date(2024, 10, 11),
)


def _write(directory: Path, name: str, payload: Mapping[str, Any]) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", "utf-8")
    return path


def _provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    # The hold-out digest travels with every artifact. Without it an artifact says which commit
    # produced it and nothing about which question set it scored, so a file left behind by an
    # earlier corpus is indistinguishable from a current one — and `check_artifacts_current.py`
    # would have nothing cheap to compare.
    frozen = load_frozen(Path(__file__).resolve().parents[3] / "artifacts")
    return {
        "commit": commit or "unknown",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "corpus_is_synthetic": True,
        "holdout_digest": frozen.digest if frozen is not None else None,
        "holdout_questions": len(frozen.questions) if frozen is not None else 0,
    }


@dataclass(frozen=True)
class BuildReport:
    artifacts: dict[str, Path]
    holdout_recall_at_10: float
    holdout_wrong_answer_rate: float
    ungated_wrong_answer_rate: float
    abstention_on_unanswerable: float


#: How many questions the A/B replay covers, per as-of date.
_REPLAY_SAMPLE: Final = 120


def _predicate_is_real(filters: CandidateFilter) -> bool:
    """Whether the candidate predicate actually constrains what the ranking statements see.

    Checked against the fragment rather than asserted, and checked on the *terms* rather than on
    the string: a fragment is only doing this job if it names both temporal bounds, the variant and
    the language, and carries the bound values for them.
    """
    sql = filters.sql
    required = ("valid_from", "valid_to", "variant_id", "language")
    knowledge = "known_to" in sql or "known_from" in sql
    return (
        all(term in sql for term in required)
        and knowledge
        and filters.effectivity_applied
        and filters.as_of_constrained
        and filters.knowledge_constrained
    )


def _knowledge_replay(
    session: Session,
    questions: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Ask one corrected specification at three knowledge dates and record what came back.

    The whole of the second bitemporal axis, as a number rather than as a paragraph: the same
    machine and the same as-of date, asked with the knowledge of the time and with today's, must
    return different documents — and the historical answer must be the pre-correction one even
    though the correction is sitting in the same table.
    """
    corrected = sorted(
        (d for d in documents.values() if d.get("corrected_by") and d.get("known_to")),
        key=lambda d: str(d["document_id"]),
    )
    if not corrected:
        return {"corrections_in_corpus": 0, "tested": False}

    original = corrected[0]
    correction_id = str(original["corrected_by"])
    variant = next(
        (
            str(q["variant_id"])
            for q in questions
            if str(q.get("family_id")) == str(original["family_id"]) and q.get("variant_id")
        ),
        None,
    )
    valid_from = date.fromisoformat(str(original["valid_from"]))
    valid_to = date.fromisoformat(str(original["valid_to"]))
    known_to = date.fromisoformat(str(original["known_to"]))
    as_of = valid_from + (valid_to - valid_from) // 2

    def seen(known_as_of: date | None) -> set[str]:
        query = Query(
            text="specification",
            language=Language.EN,
            as_of=as_of,
            known_as_of=known_as_of,
            variant_id=variant,
            top_k=50,
        )
        return {chunk.document_id for chunk in fetch_candidates(session, candidate_filter(query))}

    before = seen(known_to - timedelta(days=1))
    now = seen(None)
    after = seen(known_to + timedelta(days=1))

    return {
        "corrections_in_corpus": len(corrected),
        "tested": True,
        "case": {
            "family_id": str(original["family_id"]),
            "variant_id": variant,
            "as_of": as_of.isoformat(),
            "correction_known_from": known_to.isoformat(),
            "original_document": str(original["document_id"]),
            "correction_document": correction_id,
        },
        "original_returned_at_historical_knowledge": str(original["document_id"]) in before,
        "correction_hidden_at_historical_knowledge": correction_id not in before,
        "correction_returned_under_current_knowledge": correction_id in now,
        "original_hidden_under_current_knowledge": str(original["document_id"]) not in now,
        "correction_returned_after_it_was_issued": correction_id in after,
        "a_later_correction_did_not_rewrite_history": (
            str(original["document_id"]) in before and correction_id not in before
        ),
    }


# ------------------------------------------------------------------ A and B: the headline claim


def _effectivity(
    retriever: Retriever,
    session: Session,
    questions: Sequence[Mapping[str, Any]],
    chunk_index: Mapping[str, Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay a sample of questions at several as-of dates and inspect what came back.

    Checked against the **corpus's own record** of which chunks were in force and which variant they
    belong to, not against anything the retriever reported. A filter that reported its own
    correctness would be marking its own homework.
    """
    superseded: list[dict[str, Any]] = []
    wrong_variant: list[dict[str, Any]] = []
    queries = 0
    probe_filter: CandidateFilter | None = None

    # Round-robin across families rather than the first 120 rows. The corpus is written
    # family by family, so an unshuffled prefix covered two of the hold-out's six families and
    # graded the project's headline claim on a third of the machines it was measured over.
    # Deterministic — sorted, then dealt — because kill condition J compares two runs byte for byte
    # and a shuffled sample would differ between them for a reason unrelated to retrieval.
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for question in sorted(questions, key=lambda q: str(q["question_id"])):
        if question.get("variant_id"):
            by_family.setdefault(str(question["family_id"]), []).append(question)

    sample: list[Mapping[str, Any]] = []
    families_in_sample: set[str] = set()
    for index in range(max((len(v) for v in by_family.values()), default=0)):
        for family in sorted(by_family):
            if index < len(by_family[family]) and len(sample) < _REPLAY_SAMPLE:
                sample.append(by_family[family][index])
                families_in_sample.add(family)

    for as_of in REPLAY_DATES:
        for question in sample:
            query = Query(
                text=str(question["text"]),
                language=Language(question["language"]),
                as_of=as_of,
                variant_id=question.get("variant_id") or None,
                serial=question.get("serial"),
            )
            if probe_filter is None:
                probe_filter = candidate_filter(query)
            result = retriever.retrieve(session, query)
            queries += 1

            for item in result.chunks:
                record = chunk_index.get(item.chunk.chunk_id)
                if record is None:
                    continue
                valid_to = record.get("valid_to")
                if valid_to and date.fromisoformat(str(valid_to)) <= as_of:
                    superseded.append(
                        {
                            "as_of": as_of.isoformat(),
                            "chunk_id": item.chunk.chunk_id,
                            "valid_to": str(valid_to),
                            "question": question["question_id"],
                        }
                    )
                asked = query.variant_id
                effectivity = record.get("effectivity") or {}
                got = str(record.get("variant_id") or effectivity.get("variant_id", ""))
                if asked and got and got != asked:
                    wrong_variant.append(
                        {
                            "as_of": as_of.isoformat(),
                            "chunk_id": item.chunk.chunk_id,
                            "asked_variant": asked,
                            "chunk_variant": got,
                            "question": question["question_id"],
                        }
                    )

    return {
        "kill_conditions": ["A", "B"],
        "as_of_dates_replayed": len(REPLAY_DATES),
        "queries": queries,
        "superseded_chunks_returned": len(superseded),
        "superseded_examples": superseded[:10],
        "wrong_variant_chunks_returned": len(wrong_variant),
        "wrong_variant_examples": wrong_variant[:10],
        "filter_applied": FILTER_STAGE,
        # Read off the predicate the retriever actually built for one of the replayed queries,
        # rather than written as `True`. The kill test asserts this field, and a literal makes that
        # assertion a statement about a literal — it would stay green if the predicate were deleted
        # tomorrow. What is checked here is that the SQL fragment the ranking statements interpolate
        # really carries both temporal bounds, the variant term and the language term.
        "candidate_set_constrained_in_sql": (
            probe_filter is not None and _predicate_is_real(probe_filter)
        ),
        "candidate_predicate_sql": probe_filter.sql if probe_filter else "",
        "candidate_predicate_terms": {
            "as_of": probe_filter.as_of_constrained if probe_filter else False,
            "knowledge": probe_filter.knowledge_constrained if probe_filter else False,
            "variant": probe_filter.variant_constrained if probe_filter else False,
            "serial": probe_filter.serial_constrained if probe_filter else False,
            "language": probe_filter.language_constrained if probe_filter else False,
        },
        # Knowledge time, exercised as evidence rather than only in the test suite. The second axis
        # is a claim this project makes about itself, and a claim whose only witness is a test that
        # a reader has to run is weaker than a number in the artifact beside the first axis.
        "knowledge_time": _knowledge_replay(session, questions, documents),
        "families_in_replay_sample": len(families_in_sample),
        "replay_sample_size": len(sample),
        "method": (
            "each question is replayed at every as-of date and every returned chunk is checked "
            "against the corpus's own validity interval and variant, never against anything the "
            "retriever reported about itself"
        ),
    }


# ------------------------------------------------------------------------- C and D: citations


def _groundedness(runs: RunSet, document_text: Mapping[str, str]) -> dict[str, Any]:
    """Every answer's part numbers against its quotes, and every quote against its document."""
    ungrounded: list[dict[str, Any]] = []
    unfaithful: list[dict[str, Any]] = []
    answers = 0
    citations = 0

    for run in runs.runs:
        answer = run.answer
        if answer.decision.outcome is not GateOutcome.ANSWER or not answer.text:
            continue
        answers += 1

        cited: frozenset[str] = frozenset()
        if answer.citations:
            cited = frozenset().union(*(part_numbers_in(c.quote) for c in answer.citations))
        missing = part_numbers_in(answer.text) - cited
        if missing:
            ungrounded.append(
                {
                    "question_id": run.question_id,
                    "part_numbers": sorted(missing),
                    "answer": answer.text[:200],
                }
            )

        for citation in answer.citations:
            citations += 1
            text = document_text.get(citation.document_id)
            if text is None or not verify_citation(citation, text):
                unfaithful.append(
                    {
                        "question_id": run.question_id,
                        "document_id": citation.document_id,
                        "offset": [citation.start_offset, citation.end_offset],
                        "quote": citation.quote[:120],
                    }
                )

    return {
        "kill_conditions": ["C", "D"],
        "answers_checked": answers,
        "ungrounded_part_numbers": len(ungrounded),
        "ungrounded_examples": ungrounded[:10],
        "citations_checked": citations,
        "unfaithful_citations": len(unfaithful),
        "unfaithful_examples": unfaithful[:10],
        "method": "exact span located in the source document by offset",
        "note": (
            "the shipped answerer is extractive, so it returns spans verbatim and cannot construct "
            "a part number the evidence lacks. The check is run anyway: a guarantee that holds "
            "only "
            "because of which arm is plugged in is a property of today's arm"
        ),
    }


# ---------------------------------------------------------------------------- E and I: the gate


def _gate(runs: RunSet, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Abstention on the unanswerable set, per kind, plus the Article 50 check."""
    unanswerable = [
        run for run in runs.runs if not questions[run.question_id].get("answerable", True)
    ]
    answered = [r for r in unanswerable if r.outcome is GateOutcome.ANSWER]

    by_kind: dict[str, dict[str, Any]] = {}
    for run in unanswerable:
        kind = str(questions[run.question_id].get("unanswerable_kind", "unknown"))
        bucket = by_kind.setdefault(kind, {"total": 0, "refused": 0})
        bucket["total"] += 1
        if run.outcome is not GateOutcome.ANSWER:
            bucket["refused"] += 1
    for bucket in by_kind.values():
        bucket["rate"] = bucket["refused"] / bucket["total"] if bucket["total"] else 0.0

    missing_disclosure = sum(1 for run in runs.runs if not run.answer.ai_disclosure)

    return {
        "kill_conditions": ["E", "I"],
        "unanswerable_questions": len(unanswerable),
        "answered_without_support": len(answered),
        "unsupported_examples": [
            {"question_id": r.question_id, "text": questions[r.question_id]["text"][:120]}
            for r in answered[:10]
        ],
        # ADR-001 condition I: the share of the unanswerable set on which nothing was presented.
        # `REVIEW` counts as withheld here, and that is the reading `gate.withholds_answer` states:
        # nothing reaches the technician while a question sits in review. It is nonetheless the
        # *lenient* reading of the word "abstention", so the strict split is published beside it —
        # a floor of 0.90 leaves ten points of headroom, and a reader is entitled to know how much
        # of the margin is refusal and how much is a review queue.
        "abstention_rate_on_unanswerable": (
            1.0 - len(answered) / len(unanswerable) if unanswerable else 0.0
        ),
        "withheld_breakdown": {
            "definition": "withheld = ABSTAIN or REVIEW; both present nothing to the technician",
            "abstained": sum(1 for r in unanswerable if r.outcome is GateOutcome.ABSTAIN),
            "sent_to_review": sum(1 for r in unanswerable if r.outcome is GateOutcome.REVIEW),
            "answered": len(answered),
            "strict_abstention_rate": (
                sum(1 for r in unanswerable if r.outcome is GateOutcome.ABSTAIN) / len(unanswerable)
                if unanswerable
                else 0.0
            ),
        },
        "unanswerable_kinds": by_kind,
        "decision_precedes_generation": True,
        "model_can_override_gate": False,
        "responses_checked": len(runs.runs),
        "responses_missing_ai_disclosure": missing_disclosure,
        "article_50_note": (
            "the disclosure is a field on the response model with a minimum length, and the model "
            "is frozen, so it cannot be blanked by a caller or stripped by middleware. It is a "
            "transparency obligation and nothing else is asserted"
        ),
    }


# --------------------------------------------- F, G, H: retrieval, the curve, the gate


def _split_block(
    arms: Mapping[str, RunSet], questions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    system = arms["system"]
    retrieval = metrics.summarise_retrieval(system.retrieval_cases)

    baselines: dict[str, Any] = {}
    for name in ARMS:
        if name == "system":
            continue
        summary = metrics.summarise_retrieval(arms[name].retrieval_cases)
        baselines[name] = {
            "recall_at_10": summary.recall_at_k,
            "mrr": summary.mrr,
            "ndcg_at_10": summary.ndcg_at_k,
        }

    outcomes = system.answer_outcomes
    ungated = arms["ungated_rag"].answer_outcomes

    wrong_examples = [
        {
            "question_id": outcome.question_id,
            "asked_variant": outcome.asked_variant,
            "as_of": str(questions[outcome.question_id].get("as_of", "")),
            "cited": sorted(outcome.cited_revisions),
            "reason": (
                "cited a revision not in force at the as-of date"
                if outcome.cited_revisions - outcome.in_force_revisions
                else "cited a variant the question did not ask about"
            ),
        }
        for outcome in outcomes
        if outcome.answered and outcome.would_be_wrong
    ]

    return {
        "retrieval": {
            "system": {
                "recall_at_10": retrieval.recall_at_k,
                "mrr": retrieval.mrr,
                "ndcg_at_10": retrieval.ndcg_at_k,
                "scored": retrieval.scored,
                "skipped_without_relevant": retrieval.skipped_without_relevant,
            },
            "baselines": baselines,
        },
        "answering": {
            "definition": "revision-incorrect or variant-incorrect",
            "wrong_answer_rate": metrics.wrong_answer_rate(outcomes),
            "ungated_wrong_answer_rate": metrics.ungated_wrong_answer_rate(ungated),
            # Disclosed supplementary figures, added after the hold-out was scored. They carry
            # none of the weight of the twelve predeclared conditions and do not replace H, which
            # is reported against ADR-001's own definition whatever it comes to. See
            # `metrics.unsupported_answer_rate` for why H cannot measure what it meant to.
            "supplementary": {
                "declared_after_holdout_scoring": True,
                "replaces_kill_condition": None,
                "gated_unsupported_answer_rate": metrics.unsupported_answer_rate(outcomes),
                "ungated_unsupported_answer_rate": metrics.unsupported_answer_rate(ungated),
                "gated_answers": sum(1 for item in outcomes if item.answered),
                "ungated_answers": sum(1 for item in ungated if item.answered),
            },
            "coverage": metrics.coverage(outcomes),
            "abstention_rate": metrics.abstention_rate(outcomes),
            "review_rate": metrics.review_rate(outcomes),
            "shipped_threshold": SHIPPED_THRESHOLD,
            "coverage_curve": metrics.coverage_curve(outcomes, PRIMARY_THRESHOLD_SWEEP),
            "attribution": metrics.failure_attribution(outcomes),
            "wrong_examples": wrong_examples[:10],
        },
    }


def _evaluation(
    holdout_arms: Mapping[str, RunSet],
    development_arms: Mapping[str, RunSet],
    questions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "kill_conditions": ["F", "G", "H"],
        "holdout": _split_block(holdout_arms, questions),
        "development": _split_block(development_arms, questions),
        "arms": list(ARMS),
        "scored_once": (
            "the hold-out membership was materialised into artifacts/holdout.json and committed "
            "before this file first existed; its position in git history is the evidence"
        ),
    }


# ------------------------------------------------------------------------------- multilingual


def _multilingual(runs: RunSet, ablated: RunSet | None) -> dict[str, Any]:
    """Per-language scores, and the honest caveat about what they do and do not establish."""
    scores = metrics.by_language(runs.retrieval_cases, runs.answer_outcomes)
    by_language = {
        str(language): {
            "retrieval_recall_at_10": block["retrieval_recall_at_10"],
            "answer_correctness": block["answer_correctness"],
            "end_to_end_correctness": block["end_to_end_correctness"],
            "coverage": block["coverage"],
            "abstention_rate": block["abstention_rate"],
            "questions": block["questions"],
        }
        for language, block in scores.items()
    }

    # `ablated` is never None on the shipped build. It was for the whole of the first iteration,
    # which meant `measured_effect` published the literal string "not measured" while the kill test
    # asserted only that the key existed. A key-presence assertion over a placeholder is a test that
    # cannot fail. ADR-002.
    effect = "not measured"
    if ablated is not None:
        before = metrics.by_language(ablated.retrieval_cases, ablated.answer_outcomes)
        deltas = {
            str(language): round(
                by_language[str(language)]["retrieval_recall_at_10"]
                - before[language]["retrieval_recall_at_10"],
                4,
            )
            for language in scores
        }
        effect = f"recall@10 change per language when the fix is enabled: {deltas}"

    return {
        "languages": sorted(str(language) for language in Language),
        "by_language": by_language,
        "ablated_fix": {
            "name": "multilingual encoder shared across all three languages",
            "measured_effect": effect,
            "what_it_ablates": (
                "the alternative is a per-language lexical index with no cross-lingual vector "
                "signal, which is what the ablated arm measures"
            ),
        },
        "judge_calibration": {
            "performed": False,
            "why": (
                "no native speaker of Turkish or Russian reviewed this corpus. The scores above "
                "are computed against construction metadata — the generator knows which passage "
                "supports which question because it wrote both — so they measure retrieval and "
                "gating over synthetic parallel text. They do not establish that the Turkish reads "
                "like Turkish, and no LLM judge was used, so there is no judge to calibrate."
            ),
        },
        "corpus_is_synthetic_parallel_text": True,
    }


# ------------------------------------------------------------------------------- determinism


def _determinism(first: RunSet, second: RunSet) -> dict[str, Any]:
    retrieval_digests = [first.digest(), second.digest()]
    gate_digests = [
        hashlib.sha256(
            json.dumps(
                sorted((r.question_id, str(r.outcome)) for r in run.runs), sort_keys=True
            ).encode()
        ).hexdigest()
        for run in (first, second)
    ]
    return {
        "kill_condition": "J",
        "runs": 2,
        "questions": len(first.runs),
        "retrieval_digests": retrieval_digests,
        "retrieval_digest_stable": len(set(retrieval_digests)) == 1,
        "gate_digests": gate_digests,
        "gate_digest_stable": len(set(gate_digests)) == 1,
        "method": (
            "retrieved chunk identifiers in rank order, not as a set — a ranking that wobbled "
            "while "
            "returning the same ten chunks would pass a set comparison and fail every reasonable "
            "reading of 'two runs agree'"
        ),
    }


def _retrieval_config(embedder: Embedder) -> dict[str, Any]:
    """The configuration as the running system holds it, with the normalisation **measured**.

    The normalisation line used to be the literal string "unit vectors, cosine distance", and it
    was false. `EMBEDDING_POLICY` in the same repository already recorded the truth — the quantised
    ONNX build fastembed serves for this model does not L2-normalise, and the probe vector's norm
    is about 5.4 rather than 1.0 — so the artifact contradicted the module it describes. Retrieval
    is unaffected, because cosine distance normalises internally; the claim was simply wrong, and a
    shipped artifact stating a measured property that the measurement contradicts is the kind of
    thing that makes a reader stop believing the rest of the file. ADR-002.
    """
    from parts_answer_gate.retrieval.embeddings import (  # noqa: PLC0415
        EMBEDDING_DIM,
        EMBEDDING_MODEL_NAME,
        EMBEDDING_POLICY,
    )
    from parts_answer_gate.retrieval.pipeline import STAGE_WEIGHTS  # noqa: PLC0415
    from parts_answer_gate.store.engine import SESSION_SETTINGS  # noqa: PLC0415
    from parts_answer_gate.store.schema import HNSW_BUILD_PARAMETERS  # noqa: PLC0415

    norm = embedder.measured_norm()
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": EMBEDDING_DIM,
        "normalisation": {
            "l2_normalised": bool(abs(norm - 1.0) < 0.01),
            "measured_probe_norm": round(norm, 4),
            "distance": EMBEDDING_POLICY["distance"],
            "note": EMBEDDING_POLICY["normalization"],
        },
        "fusion": "reciprocal rank fusion",
        "stage_weights": STAGE_WEIGHTS,
        "hnsw_build_parameters": HNSW_BUILD_PARAMETERS,
        "session_settings": SESSION_SETTINGS,
        "effectivity_filter_stage": FILTER_STAGE,
        "gate_primary_threshold": SHIPPED_THRESHOLD,
        "gate_threshold_sweep": list(PRIMARY_THRESHOLD_SWEEP),
    }


def _representative_query(questions: Sequence[Mapping[str, Any]]) -> Query:
    """The query `pgvector.json` explains — taken from the hold-out, not invented for the artifact.

    This matters more than it looks. A hand-written probe can be given any shape, and the shape
    decides the plan: a query with no variant and no date scans far more rows than a real one and
    will happily use the HNSW index, while the queries this system actually issues arrive with an
    effectivity predicate that has already cut the candidate set down. Explaining the first and
    publishing it as proof about the second is how kill condition K gets passed without being met.

    So the artifact explains a question the evaluation itself runs: the first hold-out question
    that names a variant, chosen by sorted id so two builds pick the same one.
    """
    named = sorted(
        (q for q in questions if q.get("variant_id")), key=lambda q: str(q["question_id"])
    )
    chosen = named[0] if named else sorted(questions, key=lambda q: str(q["question_id"]))[0]
    return Query(
        text=str(chosen["text"]),
        language=Language(chosen["language"]),
        as_of=date.fromisoformat(str(chosen["as_of"])),
        variant_id=chosen.get("variant_id") or None,
        serial=chosen.get("serial"),
        top_k=int(chosen.get("top_k", 10)),
    )


def _merge(first: RunSet, second: RunSet) -> RunSet:
    """Two splits of one arm as a single run set, for the guarantees that span the whole corpus."""
    if first.arm != second.arm:
        raise ValueError(f"cannot merge different arms: {first.arm} and {second.arm}")
    return RunSet(
        arm=first.arm,
        runs=first.runs + second.runs,
        retrieval_cases=first.retrieval_cases + second.retrieval_cases,
        answer_outcomes=first.answer_outcomes + second.answer_outcomes,
        timings_ms=first.timings_ms + second.timings_ms,
    )


def build_all(
    session: Session,
    retriever: Retriever,
    corpus: Mapping[str, Any],
    artifacts_dir: Path,
) -> BuildReport:
    """Run every arm over both splits and write every artifact.

    Both splits are scored. The hold-out is the one the kill test grades and the one the README
    headlines; development is published beside it because a system that does far better on the data
    it was built against than on the data it was not is saying something a reader deserves to see.
    """
    questions = {str(q["question_id"]): q for q in corpus["questions"]}
    holdout_ids = set(corpus["holdout_question_ids"])
    holdout_questions = [q for qid, q in questions.items() if qid in holdout_ids]
    development_questions = [q for qid, q in questions.items() if qid not in holdout_ids]
    in_force = {str(k): frozenset(v) for k, v in corpus["in_force_revisions"].items()}

    holdout_arms = {
        arm: run_arm(retriever, session, holdout_questions, in_force, arm) for arm in ARMS
    }
    development_arms = {
        arm: run_arm(retriever, session, development_questions, in_force, arm) for arm in ARMS
    }

    # A second pass over the hold-out for kill condition J. Genuinely re-run rather than compared
    # with itself: a determinism check that hashed one result twice would pass unconditionally.
    replay = run_arm(retriever, session, holdout_questions, in_force, "system")

    system = holdout_arms["system"]

    # The absolute-zero guarantees — C, D, E, I and the Article 50 label — are properties of the
    # *system*, not of a split. Scoring them over the hold-out alone would have graded them on 153
    # of 432 questions and, for E, on 48 of the corpus's 132 unanswerable ones. They are graded
    # over everything instead. This can only make them harder to satisfy: it adds questions to a
    # count that must stay at zero and to a rate with a floor.
    #
    # F, G and H stay hold-out-only. Those are comparative scores and grading them on data the
    # system was built against is exactly what the hold-out exists to prevent.
    everything = _merge(system, development_arms["system"])

    provenance = _provenance()
    payloads: dict[str, Mapping[str, Any]] = {
        "effectivity.json": _effectivity(
            retriever,
            session,
            holdout_questions,
            corpus["chunk_index"],
            corpus["document_index"],
        ),
        "groundedness.json": _groundedness(everything, corpus["document_text"]),
        "gate.json": _gate(everything, questions),
        "evaluation.json": _evaluation(holdout_arms, development_arms, questions),
        # The ablated arm is `bm25_only`, which is exactly what this artifact's own
        # `what_it_ablates` describes: a per-language lexical index with no cross-lingual vector
        # signal. The measured effect is the per-language recall difference between the shipped
        # hybrid and that, which is the number the blueprint asks for.
        "multilingual.json": _multilingual(system, holdout_arms["bm25_only"]),
        "determinism.json": _determinism(system, replay),
        "retrieval_config.json": _retrieval_config(retriever.embedder),
        # Kill condition K, and the index-lifecycle measurement, produced by the ordinary build
        # rather than by a script somebody remembers to run.
        #
        # Both of these existed as helper functions for the whole of the first iteration and were
        # called from nowhere. `pgvector.json`, `index_lifecycle.json` and `storage_comparison.json`
        # were simply absent from the artifacts directory, so `test_K_...` failed on a missing file
        # while a hand-run script elsewhere was producing numbers that no committed code path could
        # reproduce. Evidence that only a person can regenerate is not evidence. ADR-002.
        "pgvector.json": build_pgvector_artifact(
            session, _representative_query(holdout_questions), embedder=retriever.embedder
        ),
        "index_lifecycle.json": measure_index_lifecycle(session, retriever.embedder),
        # The managed-vector comparison, against a **local** Qdrant container. The artifact states
        # that in its own body, and `qdrant_cloud_tested` is false: nothing in this repository has
        # ever reached Qdrant Cloud, and the cost column is published-list-price arithmetic rather
        # than a measured bill. It raises if Qdrant is unreachable rather than emitting a
        # one-backend artifact, because a comparison with one side missing that still passes its
        # test is worse than a failing build.
        "storage_comparison.json": build_storage_comparison_artifact(
            session, holdout_questions, embedder=retriever.embedder
        ),
    }

    written: dict[str, Path] = {}
    for name, payload in payloads.items():
        written[name] = _write(artifacts_dir, name, {**payload, "provenance": provenance})

    evaluation = payloads["evaluation.json"]
    gate_report = payloads["gate.json"]
    return BuildReport(
        artifacts=written,
        holdout_recall_at_10=evaluation["holdout"]["retrieval"]["system"]["recall_at_10"],
        holdout_wrong_answer_rate=evaluation["holdout"]["answering"]["wrong_answer_rate"],
        ungated_wrong_answer_rate=evaluation["holdout"]["answering"]["ungated_wrong_answer_rate"],
        abstention_on_unanswerable=gate_report["abstention_rate_on_unanswerable"],
    )
