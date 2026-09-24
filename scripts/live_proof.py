"""Ask the deployed service six questions and check what comes back.

    python scripts/live_proof.py --base-url https://parts-answer-gate.onrender.com

Writes `artifacts/live_retrieval.json` and exits non-zero if any case fails.

Everything else in `artifacts/` was measured in-process against a local database. This file is the
only evidence that the *deployed* system — a free container in Frankfurt talking to a free Neon
PostgreSQL over TLS — does what the in-process numbers say it does. The two can diverge for dull
reasons: a pooled connection that discards `hnsw.ef_search`, an image built from a different corpus,
an environment variable that turned a filter off. So this asks over HTTP and reads the answers.

**The six cases are the claims worth doubting**, not a smoke test:

- **A** a supported question is answered, with a citation carrying document, revision and section;
- **B** an unsupported question is refused, and the refusal carries no evidence;
- **C** the same question at two validity dates returns two different revisions with two different
  values — valid time;
- **D** the same question at the *same* validity date but an earlier knowledge date returns the
  belief held then — knowledge time, which is the axis most systems do not have;
- **E** asked with current knowledge, the corrected document wins over the belief it replaced;
- **F** a question naming one variant is never answered from another variant's passage, and no
  withdrawn revision appears in the citations.

Every question text is one the corpus contains, because the public instance serves precomputed
query vectors (see `retrieval.precomputed`). The dates are varied freely: a date is a parameter of
the SQL predicate, not of the embedding, so varying them exercises the thing being tested.

This scores nothing. It is a deployment check, not an evaluation, and it does not touch the
hold-out: every question below is from the development split.

No credential is read, printed or written here. The only input is a base URL.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://parts-answer-gate.onrender.com"
ARTIFACT = REPO_ROOT / "artifacts" / "live_retrieval.json"

#: A free instance sleeps. The first request after idle pays for the container starting, the
#: migration and the index load, and a timeout shorter than that would report a cold start as a
#: broken deployment.
TIMEOUT_SECONDS = 180


@dataclass
class Case:
    """One question, the parameters it is asked under, and what must be true of the answer."""

    key: str
    claim: str
    params: dict[str, str]
    expect_outcome: str | None = None
    expect_span: str | None = None
    expect_document: str | None = None
    forbid_documents: tuple[str, ...] = ()
    forbid_text: tuple[str, ...] = ()
    #: Asserted over `decision.approved_chunks`, which is the evidence the gate let through, rather
    #: than over the quoted span. The two are not the same thing: quote selection uses the gate's
    #: coverage rule and on these passages it often picks the section heading, so asserting a
    #: measured value against the quote would be testing the extractive answerer's span choice when
    #: the claim under test is which *document* the bitemporal predicate admitted.
    expect_evidence: tuple[str, ...] = ()
    forbid_evidence: tuple[str, ...] = ()
    expect_status: int = 200
    observed: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def ask(base_url: str, params: dict[str, str]) -> tuple[int, dict[str, Any], float]:
    url = f"{base_url.rstrip('/')}/api/ask?" + urllib.parse.urlencode(params)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return int(response.status), body, (time.perf_counter() - started) * 1000.0
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"detail": raw[:400]}
        return int(error.code), body, (time.perf_counter() - started) * 1000.0


def evidence_texts(body: dict[str, Any]) -> list[str]:
    """The text of every chunk the gate approved, which is what the answer was drawn from."""
    decision = body.get("decision") or {}
    approved = decision.get("approved_chunks") or []
    return [str((entry.get("chunk") or {}).get("text", "")) for entry in approved]


def summarise(body: dict[str, Any]) -> dict[str, Any]:
    """The parts of an answer worth recording: the decision, and where the words came from."""
    decision = body.get("decision") or {}
    citations = body.get("citations") or []
    return {
        "outcome": decision.get("outcome"),
        "reason": decision.get("reason"),
        "signals": decision.get("signals"),
        "text": body.get("text"),
        "answerer": body.get("answerer"),
        "ai_disclosure_present": bool(body.get("ai_disclosure")),
        "citations": [
            {
                "document_id": citation.get("document_id"),
                "revision": citation.get("revision"),
                "section": citation.get("section"),
                "quote": citation.get("quote"),
            }
            for citation in citations
        ],
        "approved_evidence": [text[:400] for text in evidence_texts(body)],
    }


def check(case: Case, status: int, body: dict[str, Any]) -> None:
    """Compare what came back against what the case asserts, accumulating every disagreement.

    Every failure is collected rather than raised on the first one, because a case that failed for
    two reasons and reported one would be fixed once and rerun expecting to pass.
    """
    if status != case.expect_status:
        case.failures.append(f"HTTP {status}, expected {case.expect_status}: {body.get('detail')}")
        return
    if case.expect_status != 200:
        return

    seen = summarise(body)
    if case.expect_outcome and seen["outcome"] != case.expect_outcome:
        case.failures.append(f"outcome {seen['outcome']}, expected {case.expect_outcome}")

    quotes = [citation["quote"] or "" for citation in seen["citations"]]
    haystack = " ".join([seen["text"] or "", *quotes])
    if case.expect_span and case.expect_span not in haystack:
        case.failures.append(f"{case.expect_span!r} absent from the answer and its quotes")

    documents = {citation["document_id"] for citation in seen["citations"]}
    if case.expect_document and case.expect_document not in documents:
        case.failures.append(f"{case.expect_document} not cited; cited {sorted(documents)}")
    for forbidden in case.forbid_documents:
        if forbidden in documents:
            case.failures.append(f"{forbidden} was cited and must not have been")
    for forbidden in case.forbid_text:
        if forbidden in haystack:
            case.failures.append(f"{forbidden!r} appears in an answer that must not contain it")

    check_evidence(case, body)

    # An abstention carrying evidence would mean an answerer could have been asked to use it.
    if seen["outcome"] == "abstain" and seen["citations"]:
        case.failures.append("an abstention carried citations")


def check_evidence(case: Case, body: dict[str, Any]) -> None:
    """Assert over what the gate approved, which is what the answer could have been drawn from."""
    evidence = " ".join(evidence_texts(body))
    for wanted in case.expect_evidence:
        if wanted not in evidence:
            case.failures.append(f"{wanted!r} absent from the evidence the gate approved")
    for forbidden in case.forbid_evidence:
        if forbidden in evidence:
            case.failures.append(f"{forbidden!r} reached the gate and must not have")


# The family is `fam-ax7-accumulator-station`, whose English manual has five revisions, a
# supersession chain, and one knowledge-time correction: revision B's AX7-165 case drain limit was
# believed to be 9 bar until 2022-03-13 and has been 8 bar since. That single correction is what
# makes cases D and E distinguishable at all — same question, same validity date, two answers.
QUESTIONS = {
    "foot_torque": "To what torque are the baseplate mounting foot bolts of the AX7-160 tightened?",
    "case_drain": "What is the maximum permissible case drain pressure for the AX7-165?",
    "paint_batch": "What is the paint batch number of the housing for the AX7-160?",
    "absent_family": "What is the Drive coupling bolt torque for the ZM-800 booster pump?",
    "reservoir": "How much oil does the hydraulic reservoir of the AX7-160 hold?",
}

CASES: tuple[Case, ...] = (
    Case(
        key="A_supported_answered",
        claim="a supported question is answered from a cited passage",
        params={
            "q": QUESTIONS["foot_torque"],
            "lang": "en",
            "variant": "AX7-160",
            "as_of": "2023-01-01",
        },
        expect_outcome="answer",
        expect_span="52 Nm",
        expect_document="doc-ax7-accumulator-station-man-d-en",
    ),
    Case(
        key="B_unsupported_refused",
        claim="a question about a product family the corpus does not contain is refused, "
        "with no evidence attached",
        params={"q": QUESTIONS["absent_family"], "lang": "en", "as_of": "2025-09-11"},
        expect_outcome="abstain",
    ),
    Case(
        key="C_valid_time_earlier_revision",
        claim="asked at a 2021 validity date, the revision in force in 2021 answers",
        params={
            "q": QUESTIONS["foot_torque"],
            "lang": "en",
            "variant": "AX7-160",
            "as_of": "2021-06-17",
        },
        expect_outcome="answer",
        expect_span="44 Nm",
        forbid_text=("52 Nm",),
        forbid_documents=(
            "doc-ax7-accumulator-station-man-a-en",
            "doc-ax7-accumulator-station-man-d-en",
            "doc-ax7-accumulator-station-man-e-en",
        ),
    ),
    Case(
        key="D_knowledge_time_as_believed_then",
        claim="pinned to what was known in 2021, the belief held then answers",
        params={
            "q": QUESTIONS["case_drain"],
            "lang": "en",
            "variant": "AX7-165",
            "as_of": "2021-06-17",
            "known_as_of": "2021-06-17",
        },
        expect_outcome="answer",
        expect_evidence=("must not exceed 9 bar",),
        forbid_evidence=("must not exceed 8 bar",),
        expect_document="doc-ax7-accumulator-station-man-b-en",
        forbid_documents=("doc-ax7-accumulator-station-manc-b-en",),
    ),
    Case(
        key="E_current_knowledge_prefers_the_correction",
        claim="the same validity date under current knowledge answers from the correction",
        params={
            "q": QUESTIONS["case_drain"],
            "lang": "en",
            "variant": "AX7-165",
            "as_of": "2021-06-17",
        },
        expect_outcome="answer",
        expect_evidence=("must not exceed 8 bar",),
        forbid_evidence=("must not exceed 9 bar",),
        expect_document="doc-ax7-accumulator-station-manc-b-en",
        forbid_documents=("doc-ax7-accumulator-station-man-b-en",),
    ),
    Case(
        key="F_wrong_variant_and_withdrawn_revision_excluded",
        claim="a question naming AX7-160 is never answered from AX7-160P, AX7-165 "
        "or a withdrawn revision",
        params={
            "q": QUESTIONS["foot_torque"],
            "lang": "en",
            "variant": "AX7-160",
            "as_of": "2023-01-01",
        },
        expect_outcome="answer",
        forbid_documents=(
            "doc-ax7-accumulator-station-man-a-en",
            "doc-ax7-accumulator-station-man-b-en",
            "doc-ax7-accumulator-station-manc-b-en",
            "doc-ax7-accumulator-station-man-c-en",
            "doc-ax7-accumulator-station-man-e-en",
        ),
        # The other two variants' figures for the same section of the same revision, and their
        # names. Either appearing would mean the variant predicate did not run.
        forbid_text=("108 Nm", "AX7-165", "AX7-160P"),
    ),
    Case(
        key="extra_ambiguous_variant_escalates_to_review",
        claim="the same question with no variant named is escalated to a person, not answered "
        "from whichever variant ranked first",
        # Five machines have a hydraulic reservoir and five different capacities. Without a variant
        # there is nothing for the effectivity predicate to filter on, all five survive, and the
        # gate's variant-agreement signal is false. REVIEW is the correct outcome and is the reason
        # the gate has three outcomes rather than two: answering "55 L" here would be a coin toss
        # presented as a fact.
        params={"q": QUESTIONS["reservoir"], "lang": "en", "as_of": "2022-04-08"},
        expect_outcome="review",
    ),
)

#: One question per language. The Turkish and Russian texts are the corpus's own renderings of the
#: English one, which is what makes the three comparable at all — and they are copied byte for byte,
#: because the public instance looks a question up by the hash of its text. The dotless i is the
#: correct Turkish letter here, so RUF001 is suppressed rather than the word spelled wrongly.
SMOKE: tuple[tuple[str, str, str, str], ...] = (
    (
        "en",
        "How much oil does the hydraulic reservoir of the AX7-160 hold?",
        "AX7-160",
        "2022-04-08",
    ),
    ("tr", "AX7-160 hidrolik deposu ne kadar yağ alır?", "AX7-160", "2022-04-08"),  # noqa: RUF001
    ("ru", "Сколько масла вмещает гидравлический бак AX7-160?", "AX7-160", "2022-04-08"),
)


#: The other half of the truth about abstention, asked deliberately.
#:
#: Kill condition E fails, and `release_gate.json` says why: the gate refuses 44 of 45 questions
#: about a product family that does not exist, and only 17 of 45 where the product exists and the
#: attribute does not. Case B is one of the first kind. This is one of the second, and it is here so
#: the live evidence shows the published weakness instead of only the part that works.
#:
#: It is recorded, never asserted. Pinning it to "answers" would make the failure look intended, and
#: pinning it to "abstains" would make this file fail for a reason the project has already reported
#: and deliberately not fixed. Nothing here is tuned: the question, the gate, the thresholds and the
#: prefix-matching rule behind the weakness are all exactly as they were when scored.
KNOWN_FAILURE = {
    "kill_condition": "E",
    "question": "What is the paint batch number of the housing for the AX7-160?",
    "params": {
        "q": "What is the paint batch number of the housing for the AX7-160?",
        "lang": "en",
        "variant": "AX7-160",
        "as_of": "2023-07-03",
    },
    "unanswerable_kind": "absent_specification",
    "kill_condition_E_requires": "abstain",
}


def probe_known_failure(base_url: str) -> dict[str, Any]:
    """Ask the question kill condition E fails on, and record what the live system does with it."""
    status, body, elapsed = ask(base_url, dict(KNOWN_FAILURE["params"]))  # type: ignore[arg-type]
    seen = summarise(body) if status == 200 else {}
    observed = seen.get("outcome")
    return {
        **{key: value for key, value in KNOWN_FAILURE.items() if key != "params"},
        "parameters": KNOWN_FAILURE["params"],
        "http_status": status,
        "observed_outcome": observed,
        "reason_given": seen.get("reason"),
        "term_coverage": (seen.get("signals") or {}).get("term_coverage"),
        "elapsed_ms": round(elapsed, 1),
        "reproduces_the_published_failure": observed != "abstain",
        "note": (
            "kill condition E requires an abstention here and the live system does not abstain, "
            "reproducing the published FAIL"
            if observed != "abstain"
            else "kill condition E requires an abstention here and the live system abstained on "
            "this question, which does not change the rate measured over the whole corpus in "
            "gate.json"
        ),
    }


def run_smoke(base_url: str, smoke: tuple[tuple[str, str, str, str], ...]) -> list[dict[str, Any]]:
    """One question per language, to show the deployed index serves all three.

    **Three queries are not an evaluation.** The per-language scores live in
    `artifacts/multilingual.json`, measured over the whole corpus under a recorded configuration.
    This shows only that the deployment answers in three languages, which is a far smaller claim
    and must not be quoted as the other one.
    """
    rows: list[dict[str, Any]] = []
    for language, text, variant, as_of in smoke:
        status, body, elapsed = ask(
            base_url, {"q": text, "lang": language, "variant": variant, "as_of": as_of}
        )
        seen = summarise(body) if status == 200 else {}
        rows.append(
            {
                "language": language,
                "question": text,
                "variant": variant,
                "http_status": status,
                "outcome": seen.get("outcome"),
                "cited_documents": sorted(
                    {citation["document_id"] for citation in seen.get("citations", [])}
                ),
                "elapsed_ms": round(elapsed, 1),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--smoke-file",
        type=Path,
        default=None,
        help="JSON list of [language, question, variant, as_of] replacing the built-in smoke set",
    )
    args = parser.parse_args(argv)

    smoke_set = SMOKE
    if args.smoke_file is not None:
        raw = json.loads(args.smoke_file.read_text(encoding="utf-8"))
        smoke_set = tuple((str(a), str(b), str(c), str(d)) for a, b, c, d in raw)

    print(f"asking {args.base_url}\n")
    for case in CASES:
        status, body, elapsed = ask(args.base_url, case.params)
        case.observed = {"http_status": status, "elapsed_ms": round(elapsed, 1)}
        if status == 200:
            case.observed.update(summarise(body))
        else:
            case.observed["detail"] = body.get("detail")
        check(case, status, body)

        mark = "ok  " if not case.failures else "FAIL"
        print(f"{mark} {case.key}")
        print(f"       {case.claim}")
        for failure in case.failures:
            print(f"       ! {failure}")
        for citation in case.observed.get("citations", []):
            print(f"       cited {citation['document_id']} (revision {citation['revision']})")

    known_failure = probe_known_failure(args.base_url)
    print("")
    print("kill condition E, reproduced live (recorded, not asserted):")
    print(f"  required: abstain    observed: {known_failure['observed_outcome']}")
    print(f"  {known_failure['reason_given']}")

    smoke = run_smoke(args.base_url, smoke_set)
    print("\nmultilingual smoke -- three queries, NOT the evaluation:")
    for row in smoke:
        print(
            f"  {row['language']}: HTTP {row['http_status']} -> {row['outcome']} "
            f"from {', '.join(row['cited_documents']) or 'no citation'}"
        )

    failed = [case.key for case in CASES if case.failures]
    payload = {
        "is_synthetic_corpus": True,
        "base_url": args.base_url,
        "what_this_is": (
            "six questions asked over HTTP against the deployed service and its Neon PostgreSQL "
            "pgvector index, to show the deployed system behaves as the in-process artifacts "
            "describe"
        ),
        "what_this_is_not": (
            "an evaluation. It scores nothing and changes no published metric. The multilingual "
            "smoke below is three queries, not the per-language measurement in multilingual.json, "
            "and none of it bears on the twelve predeclared kill conditions -- in particular a "
            "working pgvector deployment does not make kill condition K pass, because K asks for "
            "an index scan in the query plan and the planner prefers a sequential scan over the "
            "small filtered candidate set"
        ),
        "split_used": "development",
        "all_cases_passed": not failed,
        "cases": [
            {
                "key": case.key,
                "claim": case.claim,
                "parameters": case.params,
                "passed": not case.failures,
                "failures": case.failures,
                "observed": case.observed,
            }
            for case in CASES
        ],
        "known_failure_reproduced": known_failure,
        "multilingual_smoke": smoke,
        "observed_limitation": (
            "on these passages the extractive answerer's quote selection often returns the section "
            "heading rather than the sentence carrying the measured value, so the assertions on "
            "measured values above read decision.approved_chunks rather than the quoted span. The "
            "citation still names the right document, revision and section. Quote selection is "
            "frozen along with everything else and was not adjusted for this file"
        ),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {ARTIFACT.relative_to(REPO_ROOT)}")

    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"all {len(CASES)} live retrieval cases hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
