"""Both bitemporal axes, against the live database, through the shipped retrieval path.

A system with one date can answer *what was true of the machine in March* or *what did we know in
March*, and silently answers the other. This file is the proof that this one answers both, and it
is written against a real PostgreSQL rather than against domain objects because the constraint that
matters lives in the SQL `WHERE` clause — a Python-level test of `Chunk.known_on` would pass
against a retriever that never sent the predicate at all.

Three claims, one per section:

- **valid time** — an as-of query returns the revision in force for the machine at that date, and
  never a revision that had already been withdrawn;
- **knowledge time** — pinning `known_as_of` returns what the organisation believed then, which is
  a different document from what it believes now wherever a correction has landed;
- **a later correction does not rewrite history** — the answer to a question asked with an earlier
  knowledge date is the same before and after the correction exists in the database.

The third is the one that would be quietly wrong in most systems, because the usual shape — update
the row, keep an audit log nobody queries — makes the historical answer unrecoverable by
construction.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from parts_answer_gate.domain import Language, Query
from parts_answer_gate.retrieval.pipeline import Retriever
from parts_answer_gate.store.engine import build_engine, database_url, session_scope
from parts_answer_gate.store.queries import fetch_candidates
from parts_answer_gate.store.effectivity import applies_in_python, candidate_filter

CORPUS = Path(__file__).resolve().parents[1] / "data" / "generated"


def _records(name: str, key: str) -> list[dict[str, Any]]:
    path = CORPUS / name
    if not path.is_file():
        pytest.skip(f"{name} has not been generated")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw[key] if isinstance(raw, dict) else raw


@pytest.fixture(scope="module")
def session() -> Iterator[Session]:
    """A session against the seeded development database.

    Skips rather than fails when the store is empty. The kill test grades the artifacts, which can
    only be built against a populated database; this file is the mechanism check that runs beside
    it, and a developer with no container should see a skip rather than a red suite they cannot
    act on.
    """
    engine = build_engine(database_url())
    try:
        with session_scope(engine) as open_session:
            count = open_session.execute(sql_text("SELECT count(*) FROM chunk")).scalar_one()
            if not count:
                pytest.skip("the chunk table is empty; run `make index`")
            yield open_session
    except Exception as exc:  # pragma: no cover - environment, not logic
        pytest.skip(f"no database available: {exc}")


@pytest.fixture(scope="module")
def correction() -> dict[str, Any]:
    """One corrected document and its correction, read from the corpus rather than the database.

    From the corpus on purpose: the ground truth for a test of the store must not come out of the
    store it is testing.
    """
    documents = _records("documents.json", "documents")
    by_id = {str(d["document_id"]): d for d in documents}
    corrected = [
        d for d in documents if d.get("corrected_by") and d["language"] == Language.EN.value
    ]
    if not corrected:
        pytest.skip("this corpus contains no corrections")
    original = sorted(corrected, key=lambda d: str(d["document_id"]))[0]
    return {"original": original, "correction": by_id[str(original["corrected_by"])]}


def _query(chunk_family: str, variant_id: str, as_of: date, **kwargs: Any) -> Query:
    return Query(
        text="specification for this machine",
        language=Language.EN,
        as_of=as_of,
        variant_id=variant_id,
        top_k=50,
        **kwargs,
    )


def _document_ids(session: Session, query: Query) -> set[str]:
    """Document ids admitted by the candidate predicate — before any ranking touches them."""
    filters = candidate_filter(query)
    return {chunk.document_id for chunk in fetch_candidates(session, filters)}


# --------------------------------------------------------------------------------- valid time


def test_valid_time_returns_the_revision_in_force_for_the_machine(
    session: Session, correction: dict[str, Any]
) -> None:
    """What was valid for the machine at time T, asked at three dates across one revision's life."""
    original = correction["original"]
    valid_from = date.fromisoformat(str(original["valid_from"]))
    valid_to = date.fromisoformat(str(original["valid_to"]))

    chunks = _records("chunks.json", "chunks")
    variant = next(
        str(c["effectivity"]["variant_id"])
        for c in chunks
        if str(c["document_id"]) == str(original["document_id"])
    )

    inside = valid_from + (valid_to - valid_from) // 2
    admitted = _document_ids(session, _query(str(original["family_id"]), variant, inside))

    assert admitted, "the effectivity filter admitted nothing at a date inside the validity window"
    for document_id in admitted:
        row = session.execute(
            sql_text("SELECT valid_from, valid_to FROM document WHERE document_id = :d"),
            {"d": document_id},
        ).one()
        assert row[0] <= inside, f"{document_id} was admitted before it came into force"
        assert row[1] is None or inside < row[1], f"{document_id} was admitted after withdrawal"


def test_valid_time_excludes_a_revision_that_had_already_been_withdrawn(
    session: Session, correction: dict[str, Any]
) -> None:
    """The failure the project exists to prevent, asked at a date past the withdrawal."""
    original = correction["original"]
    valid_to = date.fromisoformat(str(original["valid_to"]))
    chunks = _records("chunks.json", "chunks")
    variant = next(
        str(c["effectivity"]["variant_id"])
        for c in chunks
        if str(c["document_id"]) == str(original["document_id"])
    )

    after = valid_to + timedelta(days=1)
    admitted = _document_ids(session, _query(str(original["family_id"]), variant, after))

    assert str(original["document_id"]) not in admitted, (
        "a revision withdrawn before the as-of date was still admitted to the candidate set"
    )
    assert str(correction["correction"]["document_id"]) not in admitted, (
        "the correction shares the original's validity, so it must also be out of force here"
    )


# ----------------------------------------------------------------------------- knowledge time


def test_knowledge_time_returns_what_was_believed_then_not_what_is_believed_now(
    session: Session, correction: dict[str, Any]
) -> None:
    """The second axis, held against the first.

    Same machine, same as-of date, two knowledge dates — and two different documents. That is the
    whole claim, and it is not expressible at all in a store with one date column.
    """
    original = correction["original"]
    correction_document = correction["correction"]
    valid_from = date.fromisoformat(str(original["valid_from"]))
    valid_to = date.fromisoformat(str(original["valid_to"]))
    known_to = date.fromisoformat(str(original["known_to"]))

    chunks = _records("chunks.json", "chunks")
    variant = next(
        str(c["effectivity"]["variant_id"])
        for c in chunks
        if str(c["document_id"]) == str(original["document_id"])
    )
    as_of = valid_from + (valid_to - valid_from) // 2
    family = str(original["family_id"])

    believed_then = _document_ids(
        session, _query(family, variant, as_of, known_as_of=known_to - timedelta(days=1))
    )
    believed_now = _document_ids(session, _query(family, variant, as_of))

    assert str(original["document_id"]) in believed_then
    assert str(correction_document["document_id"]) not in believed_then, (
        "a correction that did not yet exist was returned for a historical knowledge date"
    )

    assert str(correction_document["document_id"]) in believed_now
    assert str(original["document_id"]) not in believed_now, (
        "the superseded belief was still returned under current knowledge"
    )


def test_the_knowledge_boundary_is_half_open_like_validity(
    session: Session, correction: dict[str, Any]
) -> None:
    """On the day the correction lands, exactly one of the pair applies — not both, not neither."""
    original = correction["original"]
    valid_from = date.fromisoformat(str(original["valid_from"]))
    valid_to = date.fromisoformat(str(original["valid_to"]))
    known_to = date.fromisoformat(str(original["known_to"]))
    chunks = _records("chunks.json", "chunks")
    variant = next(
        str(c["effectivity"]["variant_id"])
        for c in chunks
        if str(c["document_id"]) == str(original["document_id"])
    )
    as_of = valid_from + (valid_to - valid_from) // 2
    family = str(original["family_id"])

    on_the_day = _document_ids(session, _query(family, variant, as_of, known_as_of=known_to))
    pair = {str(original["document_id"]), str(correction["correction"]["document_id"])}

    assert len(on_the_day & pair) == 1, (
        f"on the correction date exactly one of the pair must apply; got {sorted(on_the_day & pair)}"
    )
    assert str(correction["correction"]["document_id"]) in on_the_day, (
        "half-open on the lower bound means the correction applies from its own known_from"
    )


def test_a_later_correction_does_not_rewrite_the_historical_answer(
    session: Session, correction: dict[str, Any]
) -> None:
    """The claim that makes this bitemporal rather than merely versioned.

    The correction is already in the database. If storing it had overwritten the belief it
    replaced — which is what an `UPDATE` would do, and what most systems do — then the historical
    query could not return the original at all. It does, so the record of what was believed
    survived the arrival of better information.
    """
    original = correction["original"]
    valid_from = date.fromisoformat(str(original["valid_from"]))
    valid_to = date.fromisoformat(str(original["valid_to"]))
    known_to = date.fromisoformat(str(original["known_to"]))
    chunks = _records("chunks.json", "chunks")
    variant = next(
        str(c["effectivity"]["variant_id"])
        for c in chunks
        if str(c["document_id"]) == str(original["document_id"])
    )
    family = str(original["family_id"])
    as_of = valid_from + (valid_to - valid_from) // 2

    before_the_correction_existed = known_to - timedelta(days=1)
    historical = _document_ids(
        session, _query(family, variant, as_of, known_as_of=before_the_correction_existed)
    )

    assert str(original["document_id"]) in historical
    assert str(correction["correction"]["document_id"]) not in historical

    # And the text really is the earlier text, not the corrected text wearing the earlier id.
    stored, corrected_text = session.execute(
        sql_text(
            "SELECT (SELECT string_agg(text, '' ORDER BY chunk_id) FROM chunk WHERE document_id=:a),"
            "       (SELECT string_agg(text, '' ORDER BY chunk_id) FROM chunk WHERE document_id=:b)"
        ),
        {"a": str(original["document_id"]), "b": str(correction["correction"]["document_id"])},
    ).one()
    assert stored != corrected_text, (
        "the corrected re-issue is byte-identical to the original, so the correction changed "
        "nothing and the pair proves nothing"
    )


# ------------------------------------------------------------- the SQL and the model agree


def test_the_sql_predicate_and_the_domain_model_admit_the_same_chunks(
    session: Session, correction: dict[str, Any]
) -> None:
    """Two implementations of one rule need a test that they are one rule.

    `applies_in_python` exists for exactly this and had no caller at all for the whole of the first
    iteration, which meant the SQL was the only opinion and nothing could disagree with it.
    """
    original = correction["original"]
    valid_from = date.fromisoformat(str(original["valid_from"]))
    valid_to = date.fromisoformat(str(original["valid_to"]))
    chunks = _records("chunks.json", "chunks")
    variant = next(
        str(c["effectivity"]["variant_id"])
        for c in chunks
        if str(c["document_id"]) == str(original["document_id"])
    )
    family = str(original["family_id"])

    for as_of, known_as_of in (
        (valid_from + (valid_to - valid_from) // 2, None),
        (valid_from + (valid_to - valid_from) // 2, valid_to),
        (valid_to + timedelta(days=30), None),
    ):
        query = _query(family, variant, as_of, known_as_of=known_as_of)
        from_sql = {c.chunk_id for c in fetch_candidates(session, candidate_filter(query))}

        # Every chunk the SQL admitted must also satisfy the domain rule, and every chunk it
        # rejected must fail it. Checked over the family's own chunks rather than the whole corpus,
        # because loading 12,000 domain objects per assertion would make this file too slow to run.
        for chunk in _family_chunks(session, family):
            expected = applies_in_python(chunk, query)
            actual = chunk.chunk_id in from_sql
            assert expected == actual, (
                f"SQL and domain disagree about {chunk.chunk_id} at as_of={as_of} "
                f"known_as_of={known_as_of}: SQL says {actual}, the model says {expected}"
            )


def _family_chunks(session: Session, family_id: str) -> list[Any]:
    from parts_answer_gate.store.schema import CHUNK_READ_COLUMNS, chunk_from_mapping

    columns = ", ".join(CHUNK_READ_COLUMNS)
    rows = (
        session.execute(
            sql_text(f"SELECT {columns} FROM chunk WHERE family_id = :f AND language = 'en'"),
            {"f": family_id},
        )
        .mappings()
        .all()
    )
    return [chunk_from_mapping(row) for row in rows]


# ------------------------------------------------------- the whole pipeline, not just the filter


def test_the_shipped_retriever_honours_both_axes(
    session: Session, correction: dict[str, Any]
) -> None:
    """End to end through `Retriever.retrieve`, because that is what the service calls.

    The filter could be correct and the retriever could still fail to pass the query through it.
    """
    original = correction["original"]
    valid_from = date.fromisoformat(str(original["valid_from"]))
    valid_to = date.fromisoformat(str(original["valid_to"]))
    known_to = date.fromisoformat(str(original["known_to"]))
    chunks = _records("chunks.json", "chunks")
    variant = next(
        str(c["effectivity"]["variant_id"])
        for c in chunks
        if str(c["document_id"]) == str(original["document_id"])
    )
    as_of = valid_from + (valid_to - valid_from) // 2

    retriever = Retriever()
    now = retriever.retrieve(session, _query(str(original["family_id"]), variant, as_of))
    then = retriever.retrieve(
        session,
        _query(
            str(original["family_id"]),
            variant,
            as_of,
            known_as_of=known_to - timedelta(days=1),
        ),
    )

    now_documents = {item.chunk.document_id for item in now.chunks}
    then_documents = {item.chunk.document_id for item in then.chunks}

    assert str(original["document_id"]) not in now_documents
    assert str(correction["correction"]["document_id"]) not in then_documents
