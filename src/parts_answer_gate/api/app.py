"""The Answer Gate Lab: four screens, a JSON endpoint, and no way to write anything.

The service is read-only by construction rather than by configuration. There is no route that
ingests a document, rebuilds an index or edits a chunk — the corpus is loaded by a script an
operator runs, and the only thing a visitor can do is ask a question. `PAG_READ_ONLY` is therefore a
label on the header rather than a gate in front of a write, and the honest thing is to say so here
rather than to imply a permission system that has nothing to protect.

The pipeline is fixed and the order matters:

    question -> effectivity-filtered candidates (in SQL) -> hybrid retrieval -> the gate -> maybe
    an answerer -> citations

`gate.decide` runs **before** any answerer is constructed, and the answerer receives only
`decision.approved_chunks`. There is no code path in this module where text is generated for a
question the gate refused, which is the property kill condition E turns on.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi import Query as QueryParam
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from parts_answer_gate import gate as answer_gate
from parts_answer_gate.answerer import ExtractiveAnswerer, PostValidatedAnswerer
from parts_answer_gate.config import Settings, get_settings
from parts_answer_gate.domain import Answer, GateOutcome, Language, Query
from parts_answer_gate.retrieval.pipeline import Retriever
from parts_answer_gate.store.engine import build_engine, session_scope
from parts_answer_gate.store.schema import ChunkRow, DocumentRow

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI(
    title="Parts Answer Gate",
    description=(
        "Answers a technician's parts question with a variant-correct, revision-correct citation, "
        "or abstains. Retrieval is filtered by effectivity and as-of date before ranking; the gate "
        "decides before any text is generated."
    ),
    version="0.1.0",
)

#: Built once. The encoder loads ~220MB of ONNX and holding it per request would make every question
#: pay for it.
_RETRIEVER = Retriever()

#: The extractive arm, wrapped in the post-validator. Wrapped even though the extractive answerer
#: structurally cannot emit an ungrounded part number: the wrapper is what makes the guarantee
#: independent of which arm is plugged in, and a guarantee that depends on the implementation is a
#: property of today's implementation.
_ANSWERER = PostValidatedAnswerer(ExtractiveAnswerer())

#: The eight artifacts the evidence screen reads. Missing is normal in a fresh checkout.
_ARTIFACT_NAMES = (
    "evaluation",
    "gate",
    "groundedness",
    "effectivity",
    "corpus",
    "multilingual",
    "pgvector",
    "index_lifecycle",
    "storage_comparison",
    "retrieval_config",
    "determinism",
)


def settings() -> Settings:
    return get_settings()


def db(config: Annotated[Settings, Depends(settings)]) -> Iterator[Session]:
    with session_scope(build_engine(config.database_url)) as session:
        yield session


def _artifacts(config: Settings) -> dict[str, Any]:
    directory = Path(config.artifacts_dir)
    loaded: dict[str, Any] = {}
    for name in _ARTIFACT_NAMES:
        path = directory / f"{name}.json"
        if path.is_file():
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _release_gate(config: Settings) -> dict[str, Any]:
    """The release-gate verdict, read from `artifacts/release_gate.json`.

    On every screen and above the fold, because the honest headline for this project is that its
    **original release gate failed** — three of twelve predeclared kill conditions do not hold and a
    fourth passes vacuously. A console that showed only the working parts would be the same
    selective reporting the project exists to argue against, and it would be doing it on the page a
    visitor actually looks at.
    """
    path = Path(config.artifacts_dir) / "release_gate.json"
    if not path.is_file():
        return {"release_gate": "NOT RUN", "failing": [], "vacuous_passes": []}
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"release_gate": "UNREADABLE", "failing": [], "vacuous_passes": []}
    return payload


def _context(config: Settings, **extra: Any) -> dict[str, Any]:
    return {
        "settings": config,
        "environment": config.environment,
        "read_only": config.read_only,
        "gate_report": _release_gate(config),
        **extra,
    }


def _variants(session: Session) -> list[str]:
    rows = session.execute(
        select(ChunkRow.variant_id).distinct().order_by(ChunkRow.variant_id)
    ).scalars()
    return [row for row in rows if row]


def _with_verdicts(rows: list[dict[str, str]], gate_report: dict[str, Any]) -> list[dict[str, str]]:
    """Attach each condition's verdict, taken from `artifacts/release_gate.json`.

    From the artifact rather than recomputed here, because the verdict is the graded test's own
    result and a second opinion rendered in a template is how a console comes to disagree with the
    suite. A condition the report does not mention renders as `NOT RUN`, never as blank — a blank
    cell in a verdict column reads as a pass.
    """
    conditions = gate_report.get("conditions") or {}
    for row in rows:
        entry = conditions.get(row["letter"]) or {}
        verdict = str(entry.get("verdict") or "NOT RUN")
        if entry.get("vacuous"):
            verdict = "PASS BUT VACUOUS"
        row["verdict"] = verdict
        row["why"] = str(entry.get("why") or "")
    return rows


def _kill_rows(artifacts: dict[str, Any]) -> list[dict[str, str]]:
    """The twelve conditions, rendered from what the artifacts actually say.

    Built here rather than in the template so a missing artifact produces a row saying so, instead
    of a blank cell a reader would read as a pass.
    """
    evaluation = artifacts.get("evaluation")
    effectivity = artifacts.get("effectivity")
    groundedness = artifacts.get("groundedness")
    gate_report = artifacts.get("gate")
    corpus = artifacts.get("corpus")
    determinism = artifacts.get("determinism")
    pgvector = artifacts.get("pgvector")

    def measured(source: Any, render: Any) -> str:
        return render(source) if source else "not measured"

    return [
        {
            "letter": "A",
            "name": "no superseded chunk from an as-of query",
            "measured": measured(
                effectivity,
                lambda r: (
                    f"{r['superseded_chunks_returned']} over {r['queries']} queries at "
                    f"{r['as_of_dates_replayed']} dates"
                ),
            ),
            "artifact": "effectivity.json",
        },
        {
            "letter": "B",
            "name": "no wrong-variant chunk",
            "measured": measured(
                effectivity, lambda r: f"{r['wrong_variant_chunks_returned']} returned"
            ),
            "artifact": "effectivity.json",
        },
        {
            "letter": "C",
            "name": "no part number its evidence lacks",
            "measured": measured(
                groundedness,
                lambda r: f"{r['ungrounded_part_numbers']} over {r['answers_checked']} answers",
            ),
            "artifact": "groundedness.json",
        },
        {
            "letter": "D",
            "name": "every cited span is in its document",
            "measured": measured(
                groundedness,
                lambda r: f"{r['unfaithful_citations']} over {r['citations_checked']} citations",
            ),
            "artifact": "groundedness.json",
        },
        {
            "letter": "E",
            "name": "never answers without support",
            "measured": measured(
                gate_report,
                lambda r: (
                    f"{r['answered_without_support']} over "
                    f"{r['unanswerable_questions']} unanswerable questions"
                ),
            ),
            "artifact": "gate.json",
        },
        {
            "letter": "F",
            "name": "recall@10 beats every baseline",
            "measured": measured(
                evaluation,
                lambda r: (
                    f"{r['holdout']['retrieval']['system']['recall_at_10']:.4f} against a "
                    f"0.85 floor and the best baseline"
                ),
            ),
            "artifact": "evaluation.json",
        },
        {
            "letter": "G",
            "name": "wrong-answer rate within ceiling",
            "measured": measured(
                evaluation,
                lambda r: f"{r['holdout']['answering']['wrong_answer_rate']:.4f}, ceiling 0.02",
            ),
            "artifact": "evaluation.json",
        },
        {
            "letter": "H",
            "name": "the gate is not decorative",
            "measured": measured(
                evaluation,
                lambda r: (
                    f"ungated {r['holdout']['answering']['ungated_wrong_answer_rate']:.4f} "
                    f"against gated {r['holdout']['answering']['wrong_answer_rate']:.4f}"
                ),
            ),
            "artifact": "evaluation.json",
        },
        {
            "letter": "I",
            "name": "abstains on the unanswerable",
            "measured": measured(
                gate_report, lambda r: f"{r['abstention_rate_on_unanswerable']:.4f}, floor 0.90"
            ),
            "artifact": "gate.json",
        },
        {
            "letter": "J",
            "name": "two runs agree exactly",
            "measured": measured(
                determinism,
                lambda r: (
                    "retrieval and gate digests stable"
                    if r["retrieval_digest_stable"] and r["gate_digest_stable"]
                    else "DIFFERED"
                ),
            ),
            "artifact": "determinism.json",
        },
        {
            "letter": "K",
            "name": "vector search runs in pgvector",
            "measured": measured(
                pgvector,
                lambda r: (
                    f"operator {r['distance_operator']}, index used "
                    f"{'yes' if r['explain_mentions_index'] else 'NO'}"
                ),
            ),
            "artifact": "pgvector.json",
        },
        {
            "letter": "L",
            "name": "no document in both splits",
            "measured": measured(corpus, lambda r: f"{r['documents_in_both_splits']} leaked"),
            "artifact": "corpus.json",
        },
    ]


@app.get("/healthz", include_in_schema=False)
def healthz(
    session: Annotated[Session, Depends(db)],
    config: Annotated[Settings, Depends(settings)],
) -> JSONResponse:
    """Liveness, plus the two facts an operator needs: is the index there, and can it be written to.

    A health check that returned `{"status": "ok"}` alone would be green on an instance whose chunk
    table is empty, which is the state in which every question abstains and the demo looks broken
    for a reason nothing reports.
    """
    try:
        chunks = session.execute(select(func.count()).select_from(ChunkRow)).scalar_one()
        documents = session.execute(select(func.count()).select_from(DocumentRow)).scalar_one()
    except Exception as exc:  # a health check reports failures, it does not raise them
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "database": type(exc).__name__},
        )
    return JSONResponse(
        {
            "status": "ok" if chunks else "degraded",
            "database": "ok",
            "chunks": chunks,
            "documents": documents,
            "read_only": config.read_only,
            "live_model_configured": config.live_model_available,
            "environment": config.environment,
        },
        status_code=200 if chunks else 503,
    )


def _answer(session: Session, query: Query) -> tuple[Answer, Any]:
    """The whole pipeline, in the order ADR-001 fixes.

    Written as one function rather than spread across handlers so the ordering is readable in one
    place: the gate decides, and only then is an answerer asked for anything.
    """
    result = _RETRIEVER.retrieve(session, query)
    decision = answer_gate.decide(query, result.chunks)

    if decision.outcome is GateOutcome.ABSTAIN:
        return Answer(query=query, decision=decision), result

    # The revision label per cited document, fetched here rather than carried on the chunk.
    # `answerer` raises if a document it was asked to cite has no label, so a passage can never be
    # described by a revision nobody supplied.
    document_ids = {item.chunk.document_id for item in decision.approved_chunks}
    revisions: dict[str, str] = {
        str(row[0]): str(row[1])
        for row in session.execute(
            select(DocumentRow.document_id, DocumentRow.revision).where(
                DocumentRow.document_id.in_(document_ids)
            )
        ).all()
    }
    answer = _ANSWERER.answer(query, decision, revisions)
    return answer, result


@app.get("/", response_class=HTMLResponse)
def ask(  # noqa: PLR0917 - a FastAPI handler's parameters are injected, never passed positionally
    request: Request,
    session: Annotated[Session, Depends(db)],
    config: Annotated[Settings, Depends(settings)],
    q: Annotated[str | None, QueryParam()] = None,
    variant: Annotated[str | None, QueryParam()] = None,
    serial: Annotated[int | None, QueryParam()] = None,
    as_of: Annotated[str | None, QueryParam()] = None,
    known_as_of: Annotated[str | None, QueryParam()] = None,
    lang: Annotated[str, QueryParam()] = "en",
) -> HTMLResponse:
    """Screen 1. The question, the evidence, the gate's reasoning, and the answer or the refusal.

    `as_of` and `known_as_of` are separate inputs because they are separate questions. Leaving
    `known_as_of` blank asks what we believe *now* about the date in `as_of`, which is what a
    technician wants. Filling it asks what we believed *then*, which is what an auditor wants, and
    the two have different right answers wherever a correction has landed.
    """
    today = datetime.now(tz=UTC).date()
    asked: Query | None = None
    answer: Answer | None = None
    retrieved: Any = ()

    if q:
        asked = Query(
            text=q,
            language=Language(lang) if lang in {"en", "tr", "ru"} else Language.EN,
            as_of=date.fromisoformat(as_of) if as_of else today,
            known_as_of=date.fromisoformat(known_as_of) if known_as_of else None,
            variant_id=variant or None,
            serial=serial,
        )
        answer, result = _answer(session, asked)
        retrieved = result.chunks

    return TEMPLATES.TemplateResponse(
        request,
        "ask.html",
        _context(
            config,
            asked=asked,
            answer=answer,
            retrieved=retrieved,
            variants=_variants(session),
            default_as_of=today.isoformat(),
            examples=_EXAMPLES,
        ),
    )


@app.get("/api/ask")
def ask_json(  # noqa: PLR0917 - injected by FastAPI, not called positionally
    session: Annotated[Session, Depends(db)],
    q: Annotated[str, QueryParam()],
    variant: Annotated[str | None, QueryParam()] = None,
    serial: Annotated[int | None, QueryParam()] = None,
    as_of: Annotated[str | None, QueryParam()] = None,
    known_as_of: Annotated[str | None, QueryParam()] = None,
    lang: Annotated[str, QueryParam()] = "en",
) -> Answer:
    """The same pipeline as JSON.

    Returns the `Answer` model itself, which is what carries the Article 50 disclosure — a consumer
    of this endpoint gets the obligation for the same reason a reader of the page does.
    """
    query = Query(
        text=q,
        language=Language(lang) if lang in {"en", "tr", "ru"} else Language.EN,
        as_of=date.fromisoformat(as_of) if as_of else datetime.now(tz=UTC).date(),
        known_as_of=date.fromisoformat(known_as_of) if known_as_of else None,
        variant_id=variant or None,
        serial=serial,
    )
    answer, _ = _answer(session, query)
    return answer


@app.get("/evidence", response_class=HTMLResponse)
def evidence(request: Request, config: Annotated[Settings, Depends(settings)]) -> HTMLResponse:
    """Screen 2. Every measured number, traced to the artifact that produced it."""
    artifacts = _artifacts(config)
    return TEMPLATES.TemplateResponse(
        request,
        "evidence.html",
        _context(
            config,
            artifacts=artifacts,
            kill_rows=_with_verdicts(_kill_rows(artifacts), _release_gate(config)),
        ),
    )


@app.get("/failures", response_class=HTMLResponse)
def failures(request: Request, config: Annotated[Settings, Depends(settings)]) -> HTMLResponse:
    """Screen 3. What the system refuses, and what it still gets wrong."""
    artifacts = _artifacts(config)
    return TEMPLATES.TemplateResponse(
        request, "failures.html", _context(config, artifacts=artifacts, reviews=())
    )


@app.get("/provenance", response_class=HTMLResponse)
def provenance(request: Request, config: Annotated[Settings, Depends(settings)]) -> HTMLResponse:
    """Screen 4. Where the passages came from, how they were cut, and what embedded them."""
    return TEMPLATES.TemplateResponse(
        request, "provenance.html", _context(config, artifacts=_artifacts(config))
    )


#: Questions worth arriving at the demo already asked, chosen to show the three outcomes rather than
#: the flattering one. The second is the point of the whole project: the same question at two dates.
_EXAMPLES: tuple[dict[str, str], ...] = (
    {
        "label": "a torque figure, answered",
        "href": "/?q=What+is+the+torque+for+the+impeller+retaining+bolt%3F&lang=en",
    },
    {
        "label": "the same question, dated before a bulletin was withdrawn",
        "href": (
            "/?q=What+is+the+torque+for+the+impeller+retaining+bolt%3F&as_of=2026-03-01&lang=en"
        ),
    },
    {
        "label": "a specification the corpus does not contain",
        "href": "/?q=What+is+the+maximum+ambient+humidity+rating%3F&lang=en",
    },
    {
        "label": "the same question in Turkish",
        "href": "/?q=Carki+tutma+civatasinin+sikma+torku+nedir%3F&lang=tr",
    },
)
