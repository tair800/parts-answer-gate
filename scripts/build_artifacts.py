"""Run every arm over both splits and write the evidence the kill test grades.

    python scripts/build_artifacts.py

Needs a PostgreSQL with pgvector — `make db` provides one — and an indexed corpus from
`make corpus && make index`.

**Refuses to run if the hold-out freeze is missing or stale.** The membership must have been
materialised and committed before any score over it existed, and a build that quietly re-drew it
would produce numbers whose meaning nobody could reconstruct afterwards.

This prints the four numbers that decide whether the project stands: hold-out recall, the
wrong-answer rate, the ungated rate it must beat five times over, and abstention on the unanswerable
set. It does not decide whether they pass. `tests/test_kill_criteria.py` does, and it was committed
before any of this code existed.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from parts_answer_gate.evaluation.artifacts import build_all  # noqa: E402
from parts_answer_gate.evaluation.runner import revision_key  # noqa: E402
from parts_answer_gate.holdout import compute, load_frozen  # noqa: E402
from parts_answer_gate.retrieval.pipeline import Retriever  # noqa: E402
from parts_answer_gate.store.engine import (  # noqa: E402
    build_engine,
    database_url,
    session_scope,
)

CORPUS = REPO_ROOT / "data" / "generated"
ARTIFACTS = REPO_ROOT / "artifacts"


def _load_corpus() -> dict[str, Any]:
    """Everything the evaluation needs, read from the generated corpus rather than the database.

    The ground truth deliberately does not come from the indexed rows. A benchmark that read its
    answers back out of the system under test would agree with that system by construction.
    """
    documents = json.loads((CORPUS / "documents.json").read_text(encoding="utf-8"))
    chunks = json.loads((CORPUS / "chunks.json").read_text(encoding="utf-8"))
    questions = json.loads((CORPUS / "questions.json").read_text(encoding="utf-8"))

    documents = documents["documents"] if isinstance(documents, dict) else documents
    chunks = chunks["chunks"] if isinstance(chunks, dict) else chunks
    questions = questions["questions"] if isinstance(questions, dict) else questions

    frozen = load_frozen(ARTIFACTS)
    if frozen is None:
        raise SystemExit(
            "artifacts/holdout.json does not exist. Run `python scripts/freeze_holdout.py` and "
            "commit it before scoring anything — its position in git history is what makes the "
            "hold-out result mean anything."
        )

    computed = compute(documents, questions)
    if computed.digest != frozen.digest:
        raise SystemExit(
            "the corpus no longer produces the frozen hold-out membership.\n"
            f"  frozen   {frozen.digest}\n"
            f"  computed {computed.digest}\n"
            "Scoring now would measure a different set from the one that was committed."
        )

    document_of_chunk = {str(c["chunk_id"]): str(c["document_id"]) for c in chunks}

    # The revisions that genuinely support each question, resolved chunk -> document -> revision.
    #
    # This read `supporting_document_ids`, which the corpus does not record — it records
    # `supporting_chunk_ids`. `.get()` returned an empty list for every question, so every answer
    # cited a revision that was not in the (empty) in-force set and the wrong-answer rate came out
    # at exactly 1.0000. A measurement that reports a perfectly round number for every arm is
    # reporting a bug, and the four identical 1.0000s were what gave it away.
    in_force: dict[str, list[str]] = {}
    for question in questions:
        as_of = date.fromisoformat(str(question["as_of"]))
        family = str(question["family_id"])

        # Every revision of this family that was in force on the as-of date — not merely the
        # revisions of the chunks the corpus names as supporting.
        #
        # The narrow version scored three correct answers as wrong. Each cited manual revision D
        # *and* field bulletin FB1; FB1 is valid from 2023-08-25 with no end date, so it was in
        # force at the question's date and citing it is not revision-incorrect. A bulletin that
        # amends a procedure is exactly the kind of extra evidence a technician wants, and a ground
        # truth that punished it was measuring something nobody asked for.
        #
        # ADR-001 defines a wrong answer as revision-**incorrect**: citing a withdrawn revision.
        # This is the set that definition needs.
        #
        # Keyed `family/revision` rather than by the bare label. There are five labels (`A`, `B`,
        # `C`, `D`, `FB1`) across 108 documents in 9 families, so the bare label is not an
        # identity: any citation of *another* family's `C` would have compared equal to this
        # family's in-force `C` and scored as correct. `runner.revision_key` builds the other side
        # of this comparison and the two must agree, which is why it is imported rather than
        # re-spelled here.
        revisions = {
            revision_key(family, str(document["revision"]))
            for document in documents
            if str(document["family_id"]) == family
            and date.fromisoformat(str(document["valid_from"])) <= as_of
            and (
                document.get("valid_to") is None
                or as_of < date.fromisoformat(str(document["valid_to"]))
            )
        }
        in_force[str(question["question_id"])] = sorted(revisions)

    return {
        "chunk_to_document": document_of_chunk,
        "questions": questions,
        "holdout_question_ids": list(frozen.questions),
        "chunk_index": {str(c["chunk_id"]): c for c in chunks},
        "document_text": {str(d["document_id"]): str(d.get("text", "")) for d in documents},
        "in_force_revisions": in_force,
    }


def main() -> int:
    if not (CORPUS / "questions.json").is_file():
        print(
            "no corpus. Run `python scripts/generate_corpus.py` then "
            "`python scripts/seed_index.py`.",
            file=sys.stderr,
        )
        return 1

    corpus = _load_corpus()
    engine = build_engine(database_url())
    retriever = Retriever()

    print(
        f"scoring {len(corpus['holdout_question_ids'])} hold-out questions and the rest as "
        f"development, across 5 arms"
    )

    with session_scope(engine) as session:
        report = build_all(session, retriever, corpus, ARTIFACTS)

    print(f"\nwrote {len(report.artifacts)} artifacts to artifacts/")
    for name in sorted(report.artifacts):
        print(f"  {name}")

    print()
    print(f"  hold-out recall@10              {report.holdout_recall_at_10:.4f}")
    print(f"  hold-out wrong-answer rate      {report.holdout_wrong_answer_rate:.4f}")
    print(f"  ungated wrong-answer rate       {report.ungated_wrong_answer_rate:.4f}")
    print(f"  abstention on the unanswerable  {report.abstention_on_unanswerable:.4f}")
    print()
    print(
        "run `pytest tests/test_kill_criteria.py` to grade these against the committed thresholds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
