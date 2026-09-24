"""Break each guarantee on purpose, and check that something notices.

    python scripts/plant_breaches.py

ADR-001's falsifiability clause is the reason this file exists: *a passing suite is evidence the
tests pass, not that they would fail*. Every guarantee in this project is therefore broken here,
one at a time, and a detector has to report the breakage. A guarantee whose breach nobody notices
is a sentence in a README.

**Guards that only read source text do not count.** None of the breaches below is a string
substitution checked by a grep. Each one replaces a function the running system calls — the
candidate predicate, the gate's decision, the citation builder, the content hash — and each
detector observes a *behaviour*: rows that should not have been returned, an answer that should not
have been given, a re-embed that should have happened and did not. The breach changes what the
system does, and the detector sees it doing that.

The output is `artifacts/breaches.json`, and `tests/test_falsifiability.py` grades it.

Needs the database and an indexed corpus: `make db && make corpus && make index`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sqlalchemy import text as sql_text  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from parts_answer_gate import citations as citations_module  # noqa: E402
from parts_answer_gate import gate as gate_module  # noqa: E402
from parts_answer_gate.corpus.generate import _leaked_documents  # noqa: E402
from parts_answer_gate.corpus.splits import DEVELOPMENT, HOLDOUT  # noqa: E402
from parts_answer_gate.domain import (  # noqa: E402
    Answer,
    Chunk,
    GateDecision,
    GateOutcome,
    GateSignals,
    Language,
    Query,
)
from parts_answer_gate.evaluation.runner import force_answer  # noqa: E402
from parts_answer_gate.retrieval.dense import explain_dense  # noqa: E402
from parts_answer_gate.retrieval.embeddings import Embedder  # noqa: E402
from parts_answer_gate.retrieval.pipeline import Retriever  # noqa: E402
from parts_answer_gate.store import effectivity as effectivity_module  # noqa: E402
from parts_answer_gate.store import loader as loader_module  # noqa: E402
from parts_answer_gate.store.diagnostics import (  # noqa: E402
    executed_through_pgvector,
    in_memory_dense_search_breach,
)
from parts_answer_gate.store.effectivity import candidate_filter  # noqa: E402
from parts_answer_gate.store.engine import (  # noqa: E402
    build_engine,
    database_url,
    session_scope,
)
from parts_answer_gate.store.lifecycle import measure_index_lifecycle  # noqa: E402
from parts_answer_gate.store.queries import fetch_candidates  # noqa: E402

CORPUS = REPO_ROOT / "data" / "generated"
ARTIFACTS = REPO_ROOT / "artifacts"


@dataclass(frozen=True)
class BreachResult:
    """One planted breach, and whether anything noticed.

    `mechanism` records what was replaced, so a reader can check that the breach was a change to
    behaviour rather than to a comment. `detector` records what observed it.
    """

    name: str
    guarantee: str
    mechanism: str
    detector: str
    #: What the detector saw with the system intact. The breach is only meaningful if this is clean.
    baseline: str
    #: What it saw with the breach in place.
    breached: str
    #: The same two observations as **comparable values**, which is what lets the clean-baseline
    #: check fail. `baseline` and `breached` are prose, and two sentences written by different
    #: f-strings differ whatever the detector saw -- so a test comparing them asserts only that two
    #: strings are not identical. These are the numbers it compares instead.
    baseline_value: int | float | bool
    breached_value: int | float | bool
    caught: bool


@contextmanager
def _patched(module: Any, name: str, replacement: Any) -> Iterator[None]:
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, original)


def _records(name: str, key: str) -> list[dict[str, Any]]:
    raw = json.loads((CORPUS / name).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = raw[key] if isinstance(raw, dict) else raw
    return records


# --------------------------------------------------------------------------------- the fixtures


def _superseded_case(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    withdrawn = sorted(
        (
            c
            for c in chunks
            if c.get("superseded_by") and c.get("valid_to") and c["language"] == Language.EN.value
        ),
        key=lambda c: str(c["chunk_id"]),
    )
    chunk = withdrawn[0]
    return {"chunk": chunk, "after": date.fromisoformat(str(chunk["valid_to"])) + timedelta(days=1)}


def _query_for(chunk: dict[str, Any], as_of: date, **kwargs: Any) -> Query:
    return Query(
        text=str(chunk["text"])[:2000],
        language=Language(chunk["language"]),
        as_of=as_of,
        variant_id=str(chunk["effectivity"]["variant_id"]),
        top_k=10,
        **kwargs,
    )


# ------------------------------------------------------------------------------- the breaches


def breach_effectivity_bypass(session: Session, chunks: list[dict[str, Any]]) -> BreachResult:
    """Drop the whole effectivity predicate from the candidate query."""
    case = _superseded_case(chunks)
    query = _query_for(case["chunk"], case["after"])
    target = str(case["chunk"]["chunk_id"])

    def admitted() -> set[str]:
        return {c.chunk_id for c in fetch_candidates(session, candidate_filter(query))}

    before = target in admitted()

    # Calls the real builder with the switch flipped, so the breach is the predicate going
    # missing rather than a second implementation of the predicate.
    def patched(q: Query, **kwargs: Any) -> Any:
        kwargs["apply_effectivity"] = False
        return _REAL_CANDIDATE_FILTER(q, **kwargs)

    after = target in {c.chunk_id for c in fetch_candidates(session, patched(query))}

    return BreachResult(
        name="effectivity_bypass",
        guarantee="ADR-001 A: an as-of query never returns a superseded chunk",
        mechanism="candidate_filter replaced by one that builds the predicate with the "
        "effectivity clauses omitted",
        detector="the withdrawn chunk's membership of the candidate set",
        baseline=f"withdrawn chunk admitted: {before}",
        breached=f"withdrawn chunk admitted: {after}",
        baseline_value=before,
        breached_value=after,
        caught=(not before) and after,
    )


def breach_wrong_variant(session: Session, chunks: list[dict[str, Any]]) -> BreachResult:
    """Keep the dates, drop the variant equality."""
    case = _superseded_case(chunks)
    asked = str(case["chunk"]["effectivity"]["variant_id"])
    query = _query_for(case["chunk"], date.fromisoformat(str(case["chunk"]["valid_from"])))

    def foreign(rows: list[Chunk]) -> int:
        return sum(1 for c in rows if c.effectivity.variant_id != asked)

    before = foreign(fetch_candidates(session, candidate_filter(query)))

    def patched(q: Query, **kwargs: Any) -> Any:
        stripped = q.model_copy(update={"variant_id": None, "serial": None})
        return _REAL_CANDIDATE_FILTER(stripped, **kwargs)

    after = foreign(fetch_candidates(session, patched(query)))

    return BreachResult(
        name="wrong_variant",
        guarantee="ADR-001 B: an as-of query never returns another variant's chunk",
        mechanism="the variant and serial terms removed from the candidate predicate",
        detector="count of returned chunks belonging to a variant other than the one asked",
        baseline=f"{before} foreign-variant chunks",
        breached=f"{after} foreign-variant chunks",
        baseline_value=before,
        breached_value=after,
        caught=before == 0 and after > 0,
    )


def breach_stale_revision_off_by_one(
    session: Session, chunks: list[dict[str, Any]]
) -> BreachResult:
    """Make the validity upper bound inclusive — the classic off-by-one.

    On the single day a revision is withdrawn, an inclusive bound returns both it and its
    successor. One day a year, two revisions of one procedure, and nothing else looks wrong.
    """
    case = _superseded_case(chunks)
    boundary = date.fromisoformat(str(case["chunk"]["valid_to"]))
    query = _query_for(case["chunk"], boundary)

    def revisions_returned(filters: Any) -> int:
        rows = fetch_candidates(session, filters)
        return len({(c.family_id, c.valid_from) for c in rows})

    before = revisions_returned(candidate_filter(query))

    inclusive = effectivity_module.AS_OF_PREDICATE_SQL.replace(
        ":as_of < {a}.valid_to", ":as_of <= {a}.valid_to"
    )
    with _patched(effectivity_module, "AS_OF_PREDICATE_SQL", inclusive):
        after = revisions_returned(_REAL_CANDIDATE_FILTER(query))

    return BreachResult(
        name="stale_revision_off_by_one",
        guarantee=(
            "the validity interval is half-open, so a withdrawal date belongs to one revision"
        ),
        mechanism="`:as_of < valid_to` changed to `:as_of <= valid_to`",
        detector="distinct revisions admitted on the exact day of a withdrawal",
        baseline=f"{before} revision(s) on the boundary date",
        breached=f"{after} revision(s) on the boundary date",
        baseline_value=before,
        breached_value=after,
        caught=before == 1 and after > 1,
    )


def breach_knowledge_time(session: Session, documents: list[dict[str, Any]]) -> BreachResult:
    """Drop the knowledge predicate, so a correction leaks into a historical answer."""
    corrected = sorted(
        (d for d in documents if d.get("corrected_by") and d["language"] == Language.EN.value),
        key=lambda d: str(d["document_id"]),
    )
    if not corrected:
        raise RuntimeError("no corrections in this corpus; the knowledge breach cannot be planted")
    original = corrected[0]
    correction_id = str(original["corrected_by"])
    chunks = _records("chunks.json", "chunks")
    variant = next(
        str(c["effectivity"]["variant_id"])
        for c in chunks
        if str(c["document_id"]) == str(original["document_id"])
    )
    valid_from = date.fromisoformat(str(original["valid_from"]))
    valid_to = date.fromisoformat(str(original["valid_to"]))
    known_to = date.fromisoformat(str(original["known_to"]))
    query = Query(
        text="specification",
        language=Language.EN,
        as_of=valid_from + (valid_to - valid_from) // 2,
        known_as_of=known_to - timedelta(days=1),
        variant_id=variant,
        top_k=50,
    )

    def correction_visible(filters: Any) -> bool:
        return any(c.document_id == correction_id for c in fetch_candidates(session, filters))

    before = correction_visible(candidate_filter(query))

    with _patched(effectivity_module, "KNOWN_AS_OF_PREDICATE_SQL", "TRUE"):
        after = correction_visible(_REAL_CANDIDATE_FILTER(query))

    return BreachResult(
        name="bitemporal_knowledge_time",
        guarantee="a later correction does not rewrite the answer to a historical knowledge query",
        mechanism="the knowledge-time predicate replaced by TRUE",
        detector="whether a correction issued after the asked knowledge date became a candidate",
        baseline=f"correction visible at an earlier knowledge date: {before}",
        breached=f"correction visible at an earlier knowledge date: {after}",
        baseline_value=before,
        breached_value=after,
        caught=(not before) and after,
    )


def breach_pgvector_bypass(session: Session, chunks: list[dict[str, Any]]) -> BreachResult:
    """Replace the pgvector ranking with an in-process sort over the same candidates."""
    case = _superseded_case(chunks)
    query = _query_for(case["chunk"], date.fromisoformat(str(case["chunk"]["valid_from"])))
    filters = candidate_filter(query)
    vector = Retriever().embedder.embed_query(query.text)

    real = explain_dense(session, filters, vector, limit=10)
    _, breach_plan = in_memory_dense_search_breach(session, filters, vector, limit=10)

    accepted = executed_through_pgvector(real.plan)
    rejected = not executed_through_pgvector(breach_plan)

    return BreachResult(
        name="pgvector_bypass",
        guarantee="ADR-001 K: vector retrieval executes inside PostgreSQL",
        mechanism="the pgvector distance ranking replaced by an in-process sort over the same "
        "filtered candidates, which returns a near-identical ranking",
        detector="the executed query plan, read back from PostgreSQL",
        baseline=f"real query plan accepted: {accepted}",
        breached=f"breach plan accepted: {not rejected}",
        baseline_value=accepted,
        breached_value=not rejected,
        caught=accepted and rejected,
    )


def breach_unsupported_answer(session: Session) -> BreachResult:
    """Make the gate answer regardless — the ungated arm, applied to a question with no support."""
    query = Query(
        text="What is the torque for the ZZ-999 assembly that does not exist?",
        language=Language.EN,
        as_of=date(2024, 6, 1),
        variant_id=None,
        top_k=10,
    )
    result = Retriever().retrieve(session, query)
    real = gate_module.decide(query, result.chunks)
    before = real.outcome is GateOutcome.ANSWER

    def always_answer(q: Query, retrieved: Any, *args: Any, **kwargs: Any) -> Any:
        return force_answer(_REAL_DECIDE(q, retrieved, *args, **kwargs), retrieved)

    with _patched(gate_module, "decide", always_answer):
        after = gate_module.decide(query, result.chunks).outcome is GateOutcome.ANSWER

    return BreachResult(
        name="unsupported_answer",
        guarantee="ADR-001 E: the gate never answers a question the corpus cannot support",
        mechanism="gate.decide replaced by one that returns ANSWER whatever the rules concluded",
        detector="the outcome for a question naming a product that is not in the corpus",
        baseline=f"answered: {before}",
        breached=f"answered: {after}",
        baseline_value=before,
        breached_value=after,
        caught=(not before) and after,
    )


def breach_wrong_citation(chunks: list[dict[str, Any]]) -> BreachResult:
    """Point a citation's span at text the document does not contain."""
    case = _superseded_case(chunks)
    chunk_record = case["chunk"]
    records = _records("documents.json", "documents")
    documents = {str(d["document_id"]): str(d["text"]) for d in records}
    document_text = documents[str(chunk_record["document_id"])]

    fields = set(Chunk.model_fields)
    chunk = Chunk.model_validate({k: v for k, v in chunk_record.items() if k in fields})

    honest = citations_module.build_citation(chunk, "A")
    before = citations_module.verify_citation(honest, document_text)

    tampered = honest.model_copy(update={"quote": "a span this document does not contain at all"})
    after = citations_module.verify_citation(tampered, document_text)

    return BreachResult(
        name="wrong_citation",
        guarantee="ADR-001 D: every cited span appears in the document it is attributed to",
        mechanism="the citation's quote replaced by text absent from the source document, with "
        "its offsets left untouched",
        detector="verify_citation, which locates the span in the source by offset",
        baseline=f"honest citation verified: {before}",
        breached=f"tampered citation verified: {after}",
        baseline_value=before,
        breached_value=after,
        caught=before and not after,
    )


def breach_missing_disclosure() -> BreachResult:
    """Strip the Article 50 label from a response."""
    query = Query(text="anything", language=Language.EN, as_of=date(2024, 1, 1))
    decision = GateDecision(
        outcome=GateOutcome.ABSTAIN,
        reason="nothing was retrieved for this probe",
        signals=GateSignals(
            top_fused_score=0.0,
            supporting_chunks=0,
            term_coverage=0.0,
            variant_agreement=True,
            superseded_present=False,
            conflicting_evidence=False,
            exact_identifier_hit=False,
        ),
    )
    honest = Answer(query=query, decision=decision)
    before = bool(honest.ai_disclosure)

    caught = False
    detail = ""
    try:
        Answer(query=query, decision=decision, ai_disclosure="")
    except Exception as exc:  # the model must refuse it
        caught = True
        detail = type(exc).__name__

    return BreachResult(
        name="missing_disclosure",
        guarantee="Article 50: every response carries a machine-generated disclosure",
        mechanism="an Answer constructed with an empty ai_disclosure",
        detector="the Answer model's own validation",
        baseline=f"disclosure present by default: {before}",
        breached=f"construction refused with {detail}" if caught else "constructed successfully",
        baseline_value=before,
        breached_value=not caught,
        caught=caught,
    )


def breach_incremental_index_skip(session: Session, embedder: Embedder) -> BreachResult:
    """Freeze the content hash, so changed text never looks changed and is never re-embedded.

    Measured through `measure_index_lifecycle`, the function that writes `index_lifecycle.json`,
    rather than by calling `content_hash` twice and comparing the results. The first version of
    this breach did exactly that: it patched `content_hash`, then asserted that `content_hash` now
    returned a constant. That is a tautology dressed as a detector — it would have passed against a
    loader that ignored hashing entirely. What has to be observed is the *loader* failing to
    re-embed text that changed, and that is what this observes.
    """
    real = measure_index_lifecycle(session, embedder, sample_size=8)
    before = int(real["chunks_re_embedded"])

    with _patched(loader_module, "content_hash", lambda _text, **_kwargs: "constant"):
        frozen = measure_index_lifecycle(session, embedder, sample_size=8)
    after = int(frozen["chunks_re_embedded"])

    return BreachResult(
        name="incremental_index_skip",
        guarantee="changed chunks are re-embedded; unchanged chunks are not",
        mechanism="content_hash replaced by a constant, so no edit ever registers as a change",
        detector="measure_index_lifecycle's count of chunks the loader actually re-embedded "
        "after their text was changed",
        baseline=f"{before} of 8 changed chunks re-embedded",
        breached=f"{after} of 8 changed chunks re-embedded",
        baseline_value=before,
        breached_value=after,
        caught=before == 8 and after == 0,
    )


def breach_split_leak() -> BreachResult:
    """Move one question across the split, so a hold-out document is reachable from development."""
    documents = _records("documents.json", "documents")
    chunk_records = _records("chunks.json", "chunks")
    questions = _records("questions.json", "questions")
    fields = set(Chunk.model_fields)
    by_id = {
        str(c["chunk_id"]): Chunk.model_validate({k: v for k, v in c.items() if k in fields})
        for c in chunk_records
    }

    before = len(_leaked_documents(questions, by_id, documents))

    holdout_question = next(
        q for q in questions if q["split"] == HOLDOUT and q["supporting_chunk_ids"]
    )
    planted = [
        {**q, "split": DEVELOPMENT} if q["question_id"] == holdout_question["question_id"] else q
        for q in questions
    ]
    after = len(_leaked_documents(planted, by_id, documents))

    return BreachResult(
        name="split_leak",
        guarantee="ADR-001 L: no document appears in both splits",
        mechanism="one hold-out question relabelled as development while still citing its "
        "hold-out evidence",
        detector="the partition check, resolving question -> chunk -> document -> family",
        baseline=f"{before} leaked documents",
        breached=f"{after} leaked documents",
        baseline_value=before,
        breached_value=after,
        caught=before == 0 and after > 0,
    )


_REAL_CANDIDATE_FILTER = effectivity_module.candidate_filter
_REAL_DECIDE = gate_module.decide


def main() -> int:
    if not (CORPUS / "chunks.json").is_file():
        print("no corpus; run `python scripts/generate_corpus.py`", file=sys.stderr)
        return 1

    chunks = _records("chunks.json", "chunks")
    documents = _records("documents.json", "documents")

    engine = build_engine(database_url())
    embedder = Retriever().embedder
    results: list[BreachResult] = []

    with session_scope(engine) as session:
        if not session.execute(sql_text("SELECT count(*) FROM chunk")).scalar_one():
            print("the chunk table is empty; run `python scripts/seed_index.py`", file=sys.stderr)
            return 1

        planned: list[Callable[[], BreachResult]] = [
            lambda: breach_effectivity_bypass(session, chunks),
            lambda: breach_wrong_variant(session, chunks),
            lambda: breach_stale_revision_off_by_one(session, chunks),
            lambda: breach_knowledge_time(session, documents),
            lambda: breach_pgvector_bypass(session, chunks),
            lambda: breach_unsupported_answer(session),
            lambda: breach_wrong_citation(chunks),
            breach_missing_disclosure,
            lambda: breach_incremental_index_skip(session, embedder),
            breach_split_leak,
        ]
        for plant in planned:
            results.append(plant())

    payload = {
        "planted": len(results),
        "caught": sum(1 for r in results if r.caught),
        "uncaught": sorted(r.name for r in results if not r.caught),
        "method": (
            "each breach replaces a function the running system calls and each detector observes a "
            "behaviour, never source text"
        ),
        "breaches": [asdict(r) for r in results],
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "breaches.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    width = max(len(r.name) for r in results)
    for r in results:
        print(f"  {'CAUGHT ' if r.caught else 'MISSED '} {r.name:<{width}}  {r.breached}")
    print()
    print(f"{payload['caught']} of {payload['planted']} planted breaches were caught")
    return 0 if payload["caught"] == payload["planted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
