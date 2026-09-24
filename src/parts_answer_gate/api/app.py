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
import os
from collections.abc import Iterator
from datetime import UTC, date, datetime
from functools import cache
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, HTTPException, Request
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
from parts_answer_gate.retrieval.precomputed import (
    CachedEmbedder,
    QueryNotPrecomputedError,
    load_cache,
)
from parts_answer_gate.store.engine import DATABASE_URL_ENV, build_engine, session_scope
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
    """A session, or an error. Used where a database is genuinely required."""
    with session_scope(build_engine(config.database_url)) as session:
        yield session


#: Set on a deployment whose instance cannot hold the encoder. The model is multilingual with a
#: 250,000-token vocabulary, so its loaded session measures about 671MB resident and a 512MB
#: instance is killed the moment a query touches it.
#:
#: With this set the service serves query vectors from the same build-time cache the index was
#: loaded from. `Embedder.embed_query` is `embed_documents([text])[0]` — no query or passage prefix
#: — so a cached vector is bit-identical to a freshly computed one, and **retrieval is unchanged**:
#: the same hybrid search over the same pgvector index, the same fusion, the same gate. What
#: changes is only where the arithmetic happened, exactly as it already has for every passage.
#:
#: The cost is real and is stated on the page rather than hidden: a question whose text is not in
#: the corpus has no precomputed vector, and such a question is refused with an explanation instead
#: of being answered from a partial signal.
QUERY_CACHE_ONLY_ENV = "PAG_QUERY_CACHE_ONLY"


@cache
def _retriever(config: Settings) -> Retriever:
    """One retriever for the process, holding one encoder.

    Cached because `Embedder` loads an ONNX session on first use and a per-request instance would
    load it per request. On the constrained deployment it holds a `CachedEmbedder` that never opens
    a session at all.
    """
    if os.environ.get(QUERY_CACHE_ONLY_ENV, "").lower() in {"1", "true", "yes"}:
        vectors = load_cache(Path(config.corpus_dir))
        if vectors:
            return Retriever(CachedEmbedder(vectors, encoder_allowed=False))
    return Retriever()


def has_configured_database() -> bool:
    """Whether this instance was given a database, decided from configuration not from a probe.

    Configuration rather than a connection attempt, and the reason is unglamorous: a TCP connection
    to an address that drops packets rather than refusing them does not fail fast. It waits out the
    operating system, which is long enough that the page looks hung rather than degraded, and
    `connect_timeout` is not reliably honoured across every path into the driver. Meanwhile the
    question the page is actually asking — *was this instance given an index* — is answered exactly
    by whether `PAG_DATABASE_URL` was set.

    `store.engine.database_url()` falls back to the local compose URL so that a developer needs no
    environment at all, which is right for a developer and wrong here: on a public instance that
    fallback points at a port with nothing behind it, and treating it as a database is what made
    the page wait.
    """
    return bool(os.environ.get(DATABASE_URL_ENV))


def optional_db(config: Annotated[Settings, Depends(settings)]) -> Iterator[Session | None]:
    """A session when this instance has a database, `None` when it does not.

    The public deployment runs without one: Render allows a single free PostgreSQL per workspace and
    another project in this portfolio holds it. That is worth deploying around rather than not
    deploying, because the screens carrying this project's result — the twelve verdicts, the
    release-gate banner, the failure cases, the provenance — read `artifacts/*.json` and never touch
    a database.

    `None` rather than an exception so the Ask screen can render and **say what is missing**. A 500
    tells a visitor the site is broken; an empty result list tells them the retriever is broken; the
    truth is that this instance has no index, and only the page can say so.
    """
    if not has_configured_database():
        yield None
        return
    try:
        with session_scope(build_engine(config.database_url)) as session:
            yield session
    except Exception:
        yield None


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
    **original release gate failed** — four of twelve predeclared kill conditions do not hold and a
    fifth passes near-vacuously. The count is not written into the page: `_with_verdicts` reads it
    out of `release_gate.json`, so a verdict that changed in the evidence and not here would be
    visible rather than quietly contradicted by prose.

    A console that showed only the working parts would be the same selective reporting the project
    exists to argue against, and it would be doing it on the page a visitor actually looks at.
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
    """The template context, carrying the two settings a page renders and **not the object**.

    `Settings` holds `database_url` and `llm_api_key`. Passing it whole put a live DSN and a model
    key one `{{ settings.database_url }}` away from being rendered into a public page by anybody
    adding a debug line, and no template has ever used it. The two fields the header actually shows
    are passed by value instead, which makes the mistake impossible rather than merely absent.

    `tests/test_no_credentials_in_responses.py` renders every screen with a sentinel DSN and a
    sentinel key in the environment and asserts neither string comes back, so the guard observes
    responses rather than inspecting template source.
    """
    return {
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


@app.get("/livez", include_in_schema=False)
def livez() -> JSONResponse:
    """Is the process serving? Nothing more.

    Separate from `/healthz` on purpose, and the separation is the standard liveness/readiness
    split rather than a convenience. `/healthz` answers 503 without a database, which is the
    truthful answer and must stay that way: an operator asking whether this instance can answer a
    question needs to be told no. But a platform health check wired to that endpoint would take a
    console whose evidence screens work perfectly without a database and refuse to route traffic to
    it at all.

    So the platform is asked the question it is actually asking — is this process alive — and
    `/healthz` keeps reporting the state of the index to whoever wants to know it.
    """
    return JSONResponse({"status": "ok"})


@app.get("/healthz", include_in_schema=False)
def healthz(
    session: Annotated[Session | None, Depends(optional_db)],
    config: Annotated[Settings, Depends(settings)],
) -> JSONResponse:
    """Liveness, plus the two facts an operator needs: is the index there, and can it be written to.

    A health check that returned `{"status": "ok"}` alone would be green on an instance whose chunk
    table is empty, which is the state in which every question abstains and the demo looks broken
    for a reason nothing reports.

    Takes the **tolerant** session, and that is not a detail. With `Depends(db)` a database that
    cannot be reached raises while FastAPI is still resolving dependencies — before this function
    body exists — so the route answered **500** where its own docstring promised 503. A health check
    that crashes instead of reporting is the one endpoint that must never do that: 500 says the
    service is broken, and the truth was that its database was.
    """
    if session is None:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "database": "unreachable or not configured"},
        )
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


def _answer(session: Session, query: Query, config: Settings) -> tuple[Answer, Any]:
    """The whole pipeline, in the order ADR-001 fixes.

    Written as one function rather than spread across handlers so the ordering is readable in one
    place: the gate decides, and only then is an answerer asked for anything.
    """
    result = _retriever(config).retrieve(session, query)
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
    session: Annotated[Session | None, Depends(optional_db)],
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
    uncached_question: str | None = None

    if q and session is not None:
        asked = Query(
            text=q,
            language=Language(lang) if lang in {"en", "tr", "ru"} else Language.EN,
            as_of=date.fromisoformat(as_of) if as_of else today,
            known_as_of=date.fromisoformat(known_as_of) if known_as_of else None,
            variant_id=variant or None,
            serial=serial,
        )
        try:
            answer, result = _answer(session, asked, config)
            retrieved = result.chunks
        except QueryNotPrecomputedError:
            # This instance serves precomputed query vectors only. Rather than answer from a
            # partial signal -- which would be a different system from the measured one -- it says
            # so. See `QUERY_CACHE_ONLY_ENV`.
            asked = None
            uncached_question = q

    return TEMPLATES.TemplateResponse(
        request,
        "ask.html",
        _context(
            config,
            asked=asked,
            answer=answer,
            retrieved=retrieved,
            variants=_variants(session) if session is not None else [],
            default_as_of=today.isoformat(),
            examples=_EXAMPLES,
            index_available=session is not None,
            uncached_question=uncached_question,
        ),
    )


@app.get("/api/ask")
def ask_json(  # noqa: PLR0917 - injected by FastAPI, not called positionally
    session: Annotated[Session, Depends(db)],
    config: Annotated[Settings, Depends(settings)],
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
    try:
        answer, _ = _answer(session, query, config)
    except QueryNotPrecomputedError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "This public instance serves precomputed query vectors only, because the "
                "multilingual encoder needs about 671MB and the free tier provides 512. Every "
                "question in the corpus is answerable here against the real pgvector index; free "
                "text outside it is not. Clone the repository and run `make console` for that."
            ),
        ) from exc
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


#: Questions worth arriving at the demo already asked, chosen to show the three outcomes rather
#: than the flattering one, and to put the bitemporal axis in front of a visitor who will not read
#: this far in the README.
#:
#: **Every one of these is a question the corpus contains.** That is not a nicety: the public
#: instance serves precomputed query vectors, so an example written by hand would show a visitor the
#: "cannot encode that question" panel instead of the system working, and the first thing anyone
#: clicks would be the one thing that fails. `scripts/check_ask_examples.py` runs in the evidence
#: lane and refuses a build whose demo links are not corpus questions, because this went wrong once
#: already, in exactly that way, the day query-vector caching was deployed.
#:
#: The parameters are varied freely — a date is an argument to the SQL predicate, not to the
#: embedding — so the pair of dates below is genuinely the same question asked twice.
_FOOT_TORQUE_EN = "To what torque are the baseplate mounting foot bolts of the AX7-160 tightened?"
_FOOT_TORQUE_TR = "AX7-160 taban plakası ayak cıvataları hangi torkla sıkılır?"  # noqa: RUF001
_ABSENT_FAMILY_EN = "What is the Drive coupling bolt torque for the ZM-800 booster pump?"

_EXAMPLE_QUERIES: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "a torque figure, answered",
        {"q": _FOOT_TORQUE_EN, "variant": "AX7-160", "as_of": "2023-01-01", "lang": "en"},
    ),
    (
        "the same question, asked as of 2021 — a different revision, a different figure",
        {"q": _FOOT_TORQUE_EN, "variant": "AX7-160", "as_of": "2021-06-17", "lang": "en"},
    ),
    (
        "a machine the corpus has never heard of, refused",
        {"q": _ABSENT_FAMILY_EN, "as_of": "2025-09-11", "lang": "en"},
    ),
    (
        "the same question, with no machine named — escalated, not guessed",
        {"q": _FOOT_TORQUE_EN, "as_of": "2023-01-01", "lang": "en"},
    ),
    (
        "the same question in Turkish",
        {"q": _FOOT_TORQUE_TR, "variant": "AX7-160", "as_of": "2023-01-01", "lang": "tr"},
    ),
)

#: What the template renders. Built once, so the query string and the question it came from cannot
#: drift apart the way a hand-written href does.
_EXAMPLES: tuple[dict[str, str], ...] = tuple(
    {"label": label, "href": "/?" + urlencode(params)} for label, params in _EXAMPLE_QUERIES
)
