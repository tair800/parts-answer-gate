"""Print the figures the README quotes, read from the artifacts that measured them.

    python scripts/readme_numbers.py            # print them
    python scripts/readme_numbers.py --check    # fail if README.md disagrees with the artifacts

The README is the document most likely to keep a number after the number has changed. ADR-001 says
nothing may claim a figure that was not measured, and a README written by hand three iterations ago
is exactly how that rule gets broken without anybody deciding to break it.

So every figure the README quotes is emitted here from `artifacts/*.json`, wrapped in a marked
block, and `--check` compares the block in the file against a freshly generated one. CI runs the
check. A number that changes in the evidence and not in the README fails the build.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"
README = REPO_ROOT / "README.md"

BEGIN = "<!-- BEGIN MEASURED -->"
END = "<!-- END MEASURED -->"


def _load(name: str) -> dict[str, Any]:
    path = ARTIFACTS / name
    if not path.is_file():
        raise SystemExit(f"{name} is missing; run `make artifacts`")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _pct(value: float) -> str:
    return f"{value:.1%}"


#: The twelve conditions of ADR-001, in order, with the test that grades each. The mapping is here
#: rather than in the test because `tests/test_kill_criteria.py` is predeclared and may not gain a
#: dependency on anything written after it — the arrow points one way.
KILL_CONDITIONS: tuple[tuple[str, str, str], ...] = (
    (
        "A",
        "an as-of query returns a superseded chunk",
        "test_A_an_as_of_query_never_returns_a_superseded_chunk",
    ),
    (
        "B",
        "an as-of query returns another variant's chunk",
        "test_B_an_as_of_query_never_returns_another_variants_chunk",
    ),
    (
        "C",
        "a part number in an answer that its evidence lacks",
        "test_C_no_answer_contains_a_part_number_its_evidence_does_not",
    ),
    (
        "D",
        "a cited span absent from the document it names",
        "test_D_every_cited_span_appears_in_the_document_it_is_attributed_to",
    ),
    (
        "E",
        "the gate answers an unsupportable question",
        "test_E_the_gate_never_answers_a_question_the_corpus_cannot_support",
    ),
    (
        "F",
        "hold-out recall@10 >= 0.85 and above every baseline",
        "test_F_holdout_recall_at_10_clears_the_floor_and_beats_every_baseline",
    ),
    (
        "G",
        "hold-out wrong-answer rate <= 0.02",
        "test_G_the_holdout_wrong_answer_rate_is_within_the_declared_ceiling",
    ),
    ("H", "ungated wrong-answer rate >= 5x the gated rate", "test_H_the_gate_is_not_decorative"),
    (
        "I",
        "abstention on the unanswerable set >= 0.90",
        "test_I_the_system_abstains_on_questions_it_cannot_answer",
    ),
    ("J", "two runs agree byte for byte", "test_J_two_runs_agree_exactly"),
    (
        "K",
        "vector retrieval executes through pgvector",
        "test_K_vector_retrieval_really_executes_through_pgvector",
    ),
    ("L", "a document appears in both splits", "test_L_no_document_appears_in_both_splits"),
)

#: Annotations that no test can produce, because they are judgements about what a test *means*.
#: Each is argued in DECISIONS.md and each makes a result weaker than it looks, never stronger.
CAVEATS: dict[str, str] = {
    "G": "**vacuous** — see below",
}


def _verdicts() -> dict[str, str]:
    """Run the predeclared kill test and read the outcome of each condition off the result.

    The table below is therefore the test's own verdict rather than a summary somebody typed and
    later forgot to update. A README that states a pass the suite does not is the failure mode this
    whole project is about, and it is the one a README is most likely to have.
    """
    import subprocess  # noqa: PLC0415 - only needed on this path

    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            str(REPO_ROOT / "tests" / "test_kill_criteria.py"),
            "-q",
            "--no-header",
            "--tb=no",
            "-rA",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    output = completed.stdout + completed.stderr
    verdicts: dict[str, str] = {}
    for letter, _, test in KILL_CONDITIONS:
        if f"PASSED tests/test_kill_criteria.py::{test}" in output:
            verdicts[letter] = "PASS"
        elif f"FAILED tests/test_kill_criteria.py::{test}" in output:
            verdicts[letter] = "FAIL"
        else:
            verdicts[letter] = "NOT RUN"
    return verdicts


def block() -> str:
    evaluation = _load("evaluation.json")
    corpus = _load("corpus.json")
    gate = _load("gate.json")
    groundedness = _load("groundedness.json")
    effectivity = _load("effectivity.json")
    multilingual = _load("multilingual.json")
    candidates = _load("candidate_profile.json")
    pgvector = _load("pgvector.json")
    comparison = _load("storage_comparison.json")
    lifecycle = _load("index_lifecycle.json")

    holdout = evaluation["holdout"]
    retrieval = holdout["retrieval"]
    answering = holdout["answering"]
    baselines = retrieval["baselines"]

    verdicts = _verdicts()
    failed = [letter for letter, _, _ in KILL_CONDITIONS if verdicts[letter] == "FAIL"]

    lines: list[str] = [BEGIN, ""]

    lines += [
        "### The twelve predeclared kill conditions",
        "",
        f"**{len(failed)} of 12 fail: {', '.join(failed) if failed else 'none'}.** "
        "Their thresholds were fixed in `DECISIONS.md` before any source file existed, none has "
        "been lowered, and this table is generated by running the graded test — not written by "
        "hand.",
        "",
        "| | condition | verdict |",
        "|---|---|---|",
    ]
    for letter, description, _ in KILL_CONDITIONS:
        verdict = verdicts[letter]
        mark = "**FAIL**" if verdict == "FAIL" else verdict.title()
        caveat = CAVEATS.get(letter)
        lines.append(f"| **{letter}** | {description} | {mark}{' · ' + caveat if caveat else ''} |")
    lines += [
        "",
        "**G passes and the pass is worth nothing.** The effectivity predicate runs *before* the "
        "gate and constrains family, variant and validity in SQL; every hold-out question names a "
        "variant; every variant belongs to exactly one family. So no answer this system can give "
        "is capable of being revision- or variant-incorrect, G's numerator is empty by "
        "construction, and **G cannot tell this system apart from a broken one**. It is reported "
        "as PASS because that is what it measured. ADR-003 has the argument.",
        "",
    ]

    lines += [
        "### The corpus, counted as distinct content",
        "",
        "| | distinct | rows (one per language) |",
        "|---|---:|---:|",
        f"| product families | {corpus['families']} | — |",
        f"| variants | {corpus['variants']} | — |",
        f"| documents | {corpus['documents']} | {corpus['document_rows']} |",
        f"| supersession edges | {corpus['supersession_edges']} | "
        f"{corpus['supersession_edge_rows']} |",
        f"| chunks | {corpus['chunks']} | {corpus['chunk_rows']} |",
        f"| questions | {corpus['questions']} | {corpus['question_rows']} |",
        f"| unanswerable | {corpus['unanswerable_questions']} | "
        f"{corpus['unanswerable_question_rows']} |",
        "",
        "A translation of a passage is not a second passage. The left column is what the contract "
        "is graded against.",
        "",
        "### Is the retrieval task non-trivial?",
        "",
        f"After effectivity filtering the hold-out leaves a median of "
        f"**{candidates['holdout']['p50']}** eligible passages "
        f"(p90 {candidates['holdout']['p90']}, p95 {candidates['holdout']['p95']}, "
        f"max {candidates['holdout']['max']}) against `top_k` = {candidates['top_k']}. "
        f"**{_pct(candidates['holdout']['fraction_at_or_below_top_k'])}** of hold-out questions "
        f"leave `top_k` or fewer, so ranking decides the result rather than the filter.",
        "",
        "### Hold-out retrieval",
        "",
        "| arm | recall@10 | MRR | nDCG@10 |",
        "|---|---:|---:|---:|",
        f"| **system** | {retrieval['system']['recall_at_10']:.4f} | "
        f"{retrieval['system']['mrr']:.4f} | {retrieval['system']['ndcg_at_10']:.4f} |",
    ]
    for name in sorted(baselines):
        b = baselines[name]
        lines.append(
            f"| `{name}` | {b['recall_at_10']:.4f} | {b['mrr']:.4f} | {b['ndcg_at_10']:.4f} |"
        )
    lines += [
        "",
        f"Scored over {retrieval['system']['scored']} hold-out questions; "
        f"{retrieval['system']['skipped_without_relevant']} have no supporting passage in the "
        "corpus and are excluded from recall rather than counted as a free 1.0.",
        "",
        "### Hold-out answering",
        "",
        f"- wrong-answer rate (revision- or variant-incorrect, per answer given): "
        f"**{answering['wrong_answer_rate']:.4f}**",
        f"- ungated baseline, same definition: **{answering['ungated_wrong_answer_rate']:.4f}**",
        f"- coverage: {answering['coverage']:.4f} · abstention: {answering['abstention_rate']:.4f}"
        f" · review: {answering['review_rate']:.4f}",
        f"- answers delivered: {answering['supplementary']['gated_answers']} gated against "
        f"{answering['supplementary']['ungated_answers']} ungated",
        f"- of those, with **no supporting passage at all**: "
        f"{_pct(answering['supplementary']['gated_unsupported_answer_rate'])} gated against "
        f"**{_pct(answering['supplementary']['ungated_unsupported_answer_rate'])}** ungated",
        "",
        "### The absolute guarantees, over the whole corpus",
        "",
        f"- superseded passages returned by an as-of query: "
        f"**{effectivity['superseded_chunks_returned']}** "
        f"(over {effectivity['queries']} queries at {effectivity['as_of_dates_replayed']} dates)",
        f"- passages from a variant the question did not ask for: "
        f"**{effectivity['wrong_variant_chunks_returned']}**",
        f"- part numbers in an answer absent from its cited passages: "
        f"**{groundedness['ungrounded_part_numbers']}** "
        f"(over {groundedness['answers_checked']} answers)",
        f"- cited spans not present in the document they name: "
        f"**{groundedness['unfaithful_citations']}** "
        f"(over {groundedness['citations_checked']} citations)",
        f"- questions answered with no supporting passage: "
        f"**{gate['answered_without_support']}** "
        f"(of {gate['unanswerable_questions']} unanswerable)",
        f"- abstention on the unanswerable set: **{gate['abstention_rate_on_unanswerable']:.4f}**",
        f"- responses missing the Article 50 disclosure: "
        f"**{gate['responses_missing_ai_disclosure']}** of {gate['responses_checked']}",
        "",
        "### Per language",
        "",
        "| | recall@10 | answer correctness | end to end | coverage | questions |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for language in ("en", "tr", "ru"):
        s = multilingual["by_language"][language]
        lines.append(
            f"| {language.upper()} | {s['retrieval_recall_at_10']:.4f} | "
            f"{s['answer_correctness']:.4f} | {s['end_to_end_correctness']:.4f} | "
            f"{s['coverage']:.4f} | {s['questions']} |"
        )
    lines += [
        "",
        "Synthetic parallel text. **No native speaker reviewed this corpus and no LLM judge was "
        "used**, so there is no judge to calibrate; the scores measure retrieval and gating over "
        "generated text whose ground truth is construction metadata.",
        "",
        "### pgvector",
        "",
        f"- extension: {pgvector['extension_version']} · column `{pgvector['column']}` of type "
        f"`{pgvector['column_type']}` · operator `{pgvector['distance_operator']}`",
        f"- candidates after effectivity filtering, for the explained query: "
        f"{pgvector['candidates_after_effectivity_filter']}",
        f"- planner's own choice uses a vector index: "
        f"`{pgvector['plan_choice']['planner_default_uses_index']}` "
        f"({pgvector['plan_choice']['planner_default_median_ms']} ms) — forced ANN plan "
        f"{pgvector['plan_choice']['ann_plan_median_ms']} ms",
        f"- the planted in-process substitution is caught: **{pgvector['breach_caught']}**",
        "",
        "### The second backend",
        "",
        f"pgvector against **{comparison['deployment']}** Qdrant "
        f"(`qdrant_cloud_tested: {str(comparison['qdrant_cloud_tested']).lower()}`), same corpus, "
        "same vectors, same queries:",
        "",
        "| | recall@10 | p50 | p95 |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(comparison["backends"]):
        b = comparison["backends"][name]
        lines.append(f"| {name} | {b['recall_at_10']:.4f} | {b['p50_ms']} ms | {b['p95_ms']} ms |")
    lines += [
        "",
        f"The two engines admitted identical candidate sets on "
        f"{comparison['like_for_like']['queries_with_identical_candidate_sets']} of "
        f"{comparison['like_for_like']['queries_compared']} queries, which is what makes this a "
        f"comparison of two stores rather than of two filters. "
        f"Cost basis: {comparison['cost_basis']}.",
        "",
        "### Index lifecycle",
        "",
        f"- chunks in the index: {lifecycle['chunks_total']}",
        f"- changed: {lifecycle['chunks_changed']} · re-embedded: "
        f"{lifecycle['chunks_re_embedded']} · everything else was left alone",
        "",
        END,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    generated = block()
    if not args.check:
        print(generated)
        return 0

    if not README.is_file():
        print("README.md does not exist", file=sys.stderr)
        return 1
    text = README.read_text(encoding="utf-8")
    match = re.search(re.escape(BEGIN) + r".*?" + re.escape(END), text, re.S)
    if match is None:
        print(f"README.md has no {BEGIN} ... {END} block", file=sys.stderr)
        return 1
    if match.group(0).strip() != generated.strip():
        print(
            "README.md quotes figures that the artifacts no longer report. Regenerate with\n"
            "  python scripts/readme_numbers.py",
            file=sys.stderr,
        )
        return 1
    print("every figure the README quotes matches the artifact that measured it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
