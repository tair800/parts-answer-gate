"""The twelve conditions from ADR-001, graded against the artifacts a build produces.

**Committed before any source file exists.** Every threshold here is fixed in advance, and the point
of the file is that it can be read now and compared with what the repository later claims. A
criterion invented after seeing a number is not a criterion, it is a description.

Thresholds may be raised. They may never be lowered. If an implementation misses one, the honest
outcomes are to fix the implementation or to record the failure — not to move the line.

Nothing here computes a measurement. It reads `artifacts/*.json` and grades them, so the code that
produces the numbers and the code that decides whether they pass are separate things, written at
different times, by a process that could not see the result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

# --------------------------------------------------------------------------- ADR-001 thresholds

#: A, B, C, D, E and L. Six separate zeros, and none of them is a rate.
MAX_SUPERSEDED_RETURNED = 0
MAX_WRONG_VARIANT_RETURNED = 0
MAX_UNGROUNDED_PART_NUMBERS = 0
MAX_UNFAITHFUL_CITATIONS = 0
MAX_ANSWERED_WITHOUT_SUPPORT = 0
MAX_SPLIT_LEAKS = 0

#: F. Hold-out recall@10, and it must also beat every predeclared baseline.
MIN_HOLDOUT_RECALL_AT_10 = 0.85
RETRIEVAL_K = 10

#: G. A wrong answer is revision-incorrect or variant-incorrect, which is the definition the domain
#: cares about rather than the one that flatters a number.
MAX_HOLDOUT_WRONG_ANSWER_RATE = 0.02

#: H. The gate is not decorative: an ungated system must be at least five times worse.
MIN_UNGATED_WRONG_ANSWER_MULTIPLE = 5.0

#: I. Abstention on questions the corpus genuinely cannot answer.
MIN_ABSTENTION_ON_UNANSWERABLE = 0.90

#: The corpus contract from ADR-001.
MIN_VARIANTS = 6
MIN_DOCUMENTS = 40
MIN_SUPERSESSION_EDGES = 25
MIN_CHUNKS = 1_500
MIN_QUESTIONS = 300
MIN_UNANSWERABLE = 90
REQUIRED_LANGUAGES = {"en", "tr", "ru"}

#: The four baselines named in ADR-001, before any of them was implemented.
PREDECLARED_BASELINES = {
    "bm25_only",
    "dense_only",
    "hybrid_without_effectivity",
    "ungated_rag",
}


def _load(name: str) -> Any:
    path = ARTIFACTS / name
    if not path.is_file():
        pytest.fail(f"{name} is missing; run `make artifacts`")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ A and B: the headline claim


def test_A_an_as_of_query_never_returns_a_superseded_chunk() -> None:
    """Claim 1. The failure this project exists to prevent, replayed at several query dates.

    A withdrawn bulletin returned as current is a wrong part shipped or a voided warranty. It has no
    acceptable rate.
    """
    report = _load("effectivity.json")

    assert report["as_of_dates_replayed"] >= 3, (
        "an as-of guarantee tested at one date is a guarantee about that date"
    )
    assert report["queries"] > 0, "an empty replay cannot fail this"
    assert report["superseded_chunks_returned"] <= MAX_SUPERSEDED_RETURNED, (
        f"ADR-001 kill condition A: {report['superseded_chunks_returned']} superseded chunks were "
        f"returned by an as-of query. Examples: {report['superseded_examples'][:3]}"
    )


def test_B_an_as_of_query_never_returns_another_variants_chunk() -> None:
    """Claim 1's other half. The right revision of the wrong machine is still the wrong answer."""
    report = _load("effectivity.json")

    assert report["wrong_variant_chunks_returned"] <= MAX_WRONG_VARIANT_RETURNED, (
        f"ADR-001 kill condition B: {report['wrong_variant_chunks_returned']} chunks from a "
        f"variant the question did not ask for. "
        f"Examples: {report['wrong_variant_examples'][:3]}"
    )


def test_the_effectivity_filter_runs_before_ranking() -> None:
    """The sole-home skill is *filtering before ranking*, not filtering a ranked list.

    Post-filtering a top-k list silently shrinks it: ask for ten, filter six away, answer from four
    and never notice the recall you lost. The artifact records which it is because the difference is
    invisible in the output.
    """
    report = _load("effectivity.json")
    assert report["filter_applied"] == "before_ranking"
    assert report["candidate_set_constrained_in_sql"] is True


# ------------------------------------------------------------------------- C and D: citations


def test_C_no_answer_contains_a_part_number_its_evidence_does_not() -> None:
    """Claim 3. Structural, and the strongest guarantee in the project."""
    report = _load("groundedness.json")

    assert report["answers_checked"] > 0
    assert report["ungrounded_part_numbers"] <= MAX_UNGROUNDED_PART_NUMBERS, (
        f"ADR-001 kill condition C: {report['ungrounded_part_numbers']} answers contain a part "
        f"number absent from every cited chunk. Examples: {report['ungrounded_examples'][:3]}"
    )


def test_D_every_cited_span_appears_in_the_document_it_is_attributed_to() -> None:
    """A citation that points near the truth is not a citation.

    Checked by locating the quoted span in the source document, not by confirming the document was
    in the retrieved set — "the right document was nearby" is what makes bad RAG look good.
    """
    report = _load("groundedness.json")

    assert report["citations_checked"] > 0
    assert report["unfaithful_citations"] <= MAX_UNFAITHFUL_CITATIONS, (
        f"ADR-001 kill condition D: {report['unfaithful_citations']} cited spans do not occur in "
        f"the document they name. Examples: {report['unfaithful_examples'][:3]}"
    )
    assert report["method"] == "exact span located in the source document by offset"


# --------------------------------------------------------------------------- E and I: the gate


def test_E_the_gate_never_answers_a_question_the_corpus_cannot_support() -> None:
    report = _load("gate.json")

    assert report["unanswerable_questions"] >= MIN_UNANSWERABLE
    assert report["answered_without_support"] <= MAX_ANSWERED_WITHOUT_SUPPORT, (
        f"ADR-001 kill condition E: {report['answered_without_support']} questions were answered "
        f"with no supporting passage in the corpus. Examples: {report['unsupported_examples'][:3]}"
    )


def test_I_the_system_abstains_on_questions_it_cannot_answer() -> None:
    report = _load("gate.json")
    rate = report["abstention_rate_on_unanswerable"]

    assert rate >= MIN_ABSTENTION_ON_UNANSWERABLE, (
        f"ADR-001 kill condition I: abstained on {rate:.4f} of the unanswerable set, floor "
        f"{MIN_ABSTENTION_ON_UNANSWERABLE}"
    )


def test_the_gate_decides_before_the_answerer_is_called() -> None:
    """A gate consulted after generation is a filter, and a filter is not what was claimed."""
    report = _load("gate.json")
    assert report["decision_precedes_generation"] is True
    assert report["model_can_override_gate"] is False


# ---------------------------------------------------------- F, G and H: retrieval and the curve


def test_F_holdout_recall_at_10_clears_the_floor_and_beats_every_baseline() -> None:
    evaluation = _load("evaluation.json")
    system = evaluation["holdout"]["retrieval"]["system"][f"recall_at_{RETRIEVAL_K}"]
    baselines = evaluation["holdout"]["retrieval"]["baselines"]

    assert system >= MIN_HOLDOUT_RECALL_AT_10, (
        f"ADR-001 kill condition F: recall@{RETRIEVAL_K} is {system:.4f}, floor "
        f"{MIN_HOLDOUT_RECALL_AT_10}"
    )
    for name, scores in baselines.items():
        assert system > scores[f"recall_at_{RETRIEVAL_K}"], (
            f"ADR-001 kill condition F: the {name} baseline reaches "
            f"{scores[f'recall_at_{RETRIEVAL_K}']:.4f} and the system only {system:.4f}"
        )


def test_F_all_four_predeclared_baselines_were_scored() -> None:
    """Dropping the strongest baseline is how this criterion would be won dishonestly."""
    evaluation = _load("evaluation.json")
    assert set(evaluation["holdout"]["retrieval"]["baselines"]) == PREDECLARED_BASELINES


def test_G_the_holdout_wrong_answer_rate_is_within_the_declared_ceiling() -> None:
    evaluation = _load("evaluation.json")
    answering = evaluation["holdout"]["answering"]

    assert answering["definition"] == "revision-incorrect or variant-incorrect"
    assert answering["wrong_answer_rate"] <= MAX_HOLDOUT_WRONG_ANSWER_RATE, (
        f"ADR-001 kill condition G: {answering['wrong_answer_rate']:.4f} wrong answers, ceiling "
        f"{MAX_HOLDOUT_WRONG_ANSWER_RATE}. Examples: {answering['wrong_examples'][:3]}"
    )


def test_H_the_gate_is_not_decorative() -> None:
    """An ungated system must be materially worse, or gating bought nothing worth describing."""
    evaluation = _load("evaluation.json")
    gated = evaluation["holdout"]["answering"]["wrong_answer_rate"]
    ungated = evaluation["holdout"]["answering"]["ungated_wrong_answer_rate"]

    assert ungated > 0, "an ungated baseline with no wrong answers means the corpus is too easy"
    if gated == 0:
        return  # Infinitely better; the multiple is undefined rather than unmet.
    assert ungated / gated >= MIN_UNGATED_WRONG_ANSWER_MULTIPLE, (
        f"ADR-001 kill condition H: ungated {ungated:.4f} against gated {gated:.4f} is only "
        f"{ungated / gated:.1f} times, floor {MIN_UNGATED_WRONG_ANSWER_MULTIPLE}"
    )


def test_the_coverage_versus_wrong_answer_curve_is_published() -> None:
    """A single point is a choice of threshold presented as a result."""
    evaluation = _load("evaluation.json")
    curve = evaluation["holdout"]["answering"]["coverage_curve"]

    assert len(curve) >= 5, "a curve needs more than a couple of points"
    for point in curve:
        assert {"threshold", "coverage", "wrong_answer_rate"} <= set(point)
    coverages = [p["coverage"] for p in curve]
    assert coverages == sorted(coverages, reverse=True) or coverages == sorted(coverages), (
        "the curve is not monotonic in coverage; the threshold sweep is not a sweep"
    )


# ------------------------------------------------------------------- J, K, L: the method itself


def test_J_two_runs_agree_exactly() -> None:
    determinism = _load("determinism.json")

    assert determinism["runs"] >= 2
    assert determinism["retrieval_digest_stable"] is True, (
        f"ADR-001 kill condition J: retrieval differed between runs — "
        f"{determinism['retrieval_digests']}"
    )
    assert determinism["gate_digest_stable"] is True, (
        f"ADR-001 kill condition J: gate decisions differed between runs — "
        f"{determinism['gate_digests']}"
    )


def test_K_vector_retrieval_really_executes_through_pgvector() -> None:
    """Importing pgvector is not using it.

    The artifact records the executed SQL operator and the index actually consulted, read back from
    the database's own query plan rather than from the code that built the query.
    """
    report = _load("pgvector.json")

    assert report["extension_installed"] is True
    assert report["column_type"] == "vector"
    assert report["distance_operator"] in {"<=>", "<->", "<#>"}
    assert report["explain_mentions_index"] is True, (
        f"ADR-001 kill condition K: the query plan does not use a vector index — "
        f"{report['explain_plan'][:200]}"
    )
    assert report["breach_caught"] is True, (
        "replacing the pgvector query with an in-memory sort did not fail any test, so nothing "
        "proves the extension participates"
    )


def test_L_no_document_appears_in_both_splits() -> None:
    corpus = _load("corpus.json")

    assert corpus["split_rule"].startswith("blake2b(family_id")
    assert corpus["documents_in_both_splits"] <= MAX_SPLIT_LEAKS, (
        f"ADR-001 kill condition L: {corpus['documents_in_both_splits']} documents appear in both "
        f"the development and hold-out splits: {corpus['leaked_documents'][:5]}"
    )
    assert corpus["split_by"] == "product_family"


# --------------------------------------------------------------------------- the corpus contract


def test_the_corpus_meets_the_size_contract() -> None:
    corpus = _load("corpus.json")

    for key, floor in (
        ("variants", MIN_VARIANTS),
        ("documents", MIN_DOCUMENTS),
        ("supersession_edges", MIN_SUPERSESSION_EDGES),
        ("chunks", MIN_CHUNKS),
        ("questions", MIN_QUESTIONS),
        ("unanswerable_questions", MIN_UNANSWERABLE),
    ):
        assert corpus[key] >= floor, f"only {corpus[key]} {key}, contract {floor}"


def test_the_corpus_is_trilingual_and_says_it_is_synthetic() -> None:
    corpus = _load("corpus.json")

    assert set(corpus["languages"]) >= REQUIRED_LANGUAGES
    assert corpus["is_synthetic"] is True, (
        "ADR-001 forbids describing a generated corpus as real-world data"
    )
    assert corpus["seed"] is not None
    assert corpus["generator_version"]


def test_the_multilingual_split_reports_retrieval_and_generation_separately() -> None:
    """The blueprint asks for the degradation *split*, not a single multilingual number."""
    multilingual = _load("multilingual.json")

    assert set(multilingual["languages"]) >= REQUIRED_LANGUAGES
    for language in REQUIRED_LANGUAGES:
        scores = multilingual["by_language"][language]
        assert "retrieval_recall_at_10" in scores
        assert "answer_correctness" in scores
    assert multilingual["ablated_fix"]["name"]
    assert "measured_effect" in multilingual["ablated_fix"]
    assert multilingual["judge_calibration"]["performed"] is False, (
        "no native speaker reviewed this corpus; claiming calibration would be an invented result"
    )


# ------------------------------------------------------- the managed vector store comparison


def test_the_storage_port_was_compared_against_a_second_backend() -> None:
    comparison = _load("storage_comparison.json")

    assert len(comparison["backends"]) >= 2
    assert "pgvector" in comparison["backends"]
    for name, scores in comparison["backends"].items():
        assert "recall_at_10" in scores, f"{name} has no recall figure"
        assert "p95_ms" in scores, f"{name} has no latency figure"
    assert comparison["cost_basis"] == "published list price arithmetic, not a measured bill"


def test_the_index_lifecycle_was_measured_rather_than_described() -> None:
    lifecycle = _load("index_lifecycle.json")

    assert lifecycle["chunks_changed"] > 0
    assert lifecycle["chunks_re_embedded"] == lifecycle["chunks_changed"], (
        "incremental re-embedding that re-embeds more than what changed is a full rebuild with "
        "extra steps"
    )
    assert lifecycle["chunks_total"] > lifecycle["chunks_re_embedded"]
    assert "staleness_seconds" in lifecycle


# ------------------------------------------------------------------ Article 50 transparency


def test_every_answer_is_labelled_machine_generated() -> None:
    """An in-force obligation, implemented as a property of the response rather than of the UI."""
    report = _load("gate.json")

    assert report["responses_checked"] > 0
    assert report["responses_missing_ai_disclosure"] == 0, (
        f"{report['responses_missing_ai_disclosure']} responses carried no Article 50 disclosure"
    )
