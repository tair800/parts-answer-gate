"""The adversarial case: a withdrawn passage that *wants* to win, and is never ranked at all.

Every claim about filtering before ranking is easy to make and easy to satisfy accidentally. A
system that ranks first and discards afterwards produces the same visible answer in almost every
case, and the difference only shows when the withdrawn passage is the one the ranker would have
picked. So this file does not construct a gentle case. It takes a chunk that **is** superseded and
uses that chunk's own text as the query.

Against an unfiltered retriever that is the best possible match — the query is the document. BM25
scores it top because every term matches, and the dense signal scores it top because the query
vector and the chunk vector are the same vector. If effectivity filtering happened after ranking,
the withdrawn passage would arrive at rank 1 and then be dropped, and the evaluation would record
that ten chunks were retrieved while the answer came from nine.

What must actually happen is that the passage never enters the candidate set, and therefore never
receives a score at all. The tests below check exactly that, and they check it through the count of
rows the predicate admits rather than through the output, because the output of the two designs is
what looks the same.
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
from parts_answer_gate.retrieval.dense import dense_sql
from parts_answer_gate.retrieval.pipeline import Retriever
from parts_answer_gate.store.effectivity import FILTER_STAGE, candidate_filter
from parts_answer_gate.store.engine import build_engine, database_url, session_scope
from parts_answer_gate.store.queries import candidate_count, fetch_candidates

CORPUS = Path(__file__).resolve().parents[1] / "data" / "generated"


@pytest.fixture(scope="module")
def session() -> Iterator[Session]:
    engine = build_engine(database_url())
    try:
        with session_scope(engine) as open_session:
            if not open_session.execute(sql_text("SELECT count(*) FROM chunk")).scalar_one():
                pytest.skip("the chunk table is empty; run `make index`")
            yield open_session
    except Exception as exc:  # pragma: no cover - environment, not logic
        pytest.skip(f"no database available: {exc}")


@pytest.fixture(scope="module")
def superseded() -> dict[str, Any]:
    """A withdrawn chunk, and a date at which it is unambiguously out of force.

    Chosen from the corpus by sorted id so the case is the same on every run, and taken from a
    document that has a successor — a chunk with no successor was never superseded and would make
    the test vacuous in the quietest possible way.
    """
    path = CORPUS / "chunks.json"
    if not path.is_file():
        pytest.skip("the corpus has not been generated")
    raw = json.loads(path.read_text(encoding="utf-8"))
    chunks = raw["chunks"] if isinstance(raw, dict) else raw

    withdrawn = sorted(
        (
            c
            for c in chunks
            if c.get("superseded_by") and c.get("valid_to") and c["language"] == Language.EN.value
        ),
        key=lambda c: str(c["chunk_id"]),
    )
    if not withdrawn:
        pytest.skip("this corpus contains no superseded chunks")
    chunk = withdrawn[0]
    return {
        "chunk": chunk,
        "after_withdrawal": date.fromisoformat(str(chunk["valid_to"])) + timedelta(days=1),
    }


def _query(chunk: dict[str, Any], as_of: date) -> Query:
    """The withdrawn passage's own text, asked about its own machine, after it was withdrawn."""
    return Query(
        text=str(chunk["text"])[:2000],
        language=Language(chunk["language"]),
        as_of=as_of,
        variant_id=str(chunk["effectivity"]["variant_id"]),
        top_k=10,
    )


def test_the_withdrawn_passage_would_win_without_the_filter(
    session: Session, superseded: dict[str, Any]
) -> None:
    """The adversarial premise, established rather than assumed.

    If this fails, every other test in the file is proving something about a passage that would
    not have been retrieved anyway, and the guarantee would be untested while appearing green.
    """
    chunk = superseded["chunk"]
    query = _query(chunk, superseded["after_withdrawal"])

    unfiltered = Retriever().retrieve(session, query, apply_effectivity=False)
    ranked = [item.chunk.chunk_id for item in unfiltered.chunks]

    assert ranked, "the unfiltered retriever returned nothing at all"
    assert ranked[0] == str(chunk["chunk_id"]), (
        "the premise does not hold: querying a passage with its own text did not rank that passage "
        f"first without the filter. Got {ranked[:3]}"
    )


def test_the_withdrawn_passage_is_never_scored_when_the_filter_is_on(
    session: Session, superseded: dict[str, Any]
) -> None:
    """Not demoted, not dropped after ranking — absent from the candidate set.

    Checked against the rows the predicate admits, because a passage that was ranked and then
    discarded is indistinguishable from one that was never ranked if you only look at the output.
    """
    chunk = superseded["chunk"]
    query = _query(chunk, superseded["after_withdrawal"])

    admitted = {c.chunk_id for c in fetch_candidates(session, candidate_filter(query))}
    assert str(chunk["chunk_id"]) not in admitted, (
        "a superseded chunk was admitted to the candidate set, so the filter is not constraining "
        "the rows the ranker sees"
    )

    result = Retriever().retrieve(session, query)
    returned = [item.chunk.chunk_id for item in result.chunks]
    assert str(chunk["chunk_id"]) not in returned


def test_the_filter_shrinks_the_candidate_set_rather_than_the_result(
    session: Session, superseded: dict[str, Any]
) -> None:
    """The structural difference between filtering before and after ranking, stated as a number.

    Filtering before ranking makes the *candidate count* smaller. Filtering after ranking leaves
    the candidate count alone and makes the *result* shorter. This asserts the first and, by
    asserting the result is still full, rules out the second.
    """
    chunk = superseded["chunk"]
    query = _query(chunk, superseded["after_withdrawal"])

    with_filter = candidate_count(session, candidate_filter(query))
    without_filter = candidate_count(session, candidate_filter(query, apply_effectivity=False))

    assert with_filter < without_filter, (
        f"the effectivity predicate admitted as many rows as no predicate at all "
        f"({with_filter} against {without_filter}); it is not constraining anything"
    )

    result = Retriever().retrieve(session, query)
    assert len(result.chunks) == min(query.top_k, with_filter), (
        f"asked for {query.top_k} and got {len(result.chunks)} from {with_filter} eligible "
        "candidates — a short result is the signature of post-filtering a ranked list"
    )


def test_the_predicate_is_in_the_sql_that_ranks(
    session: Session, superseded: dict[str, Any]
) -> None:
    """The claim `effectivity.json` publishes, checked against the statement actually executed."""
    chunk = superseded["chunk"]
    query = _query(chunk, superseded["after_withdrawal"])
    filters = candidate_filter(query)

    statement = dense_sql(filters)
    assert "valid_from" in statement and "valid_to" in statement, (
        "the vector ranking statement does not carry the validity predicate, so ranking happens "
        f"over rows the filter was supposed to remove:\n{statement}"
    )
    assert "known_from" in statement or "known_to" in statement, (
        "the vector ranking statement does not carry the knowledge predicate"
    )
    assert "variant_id" in statement
    assert FILTER_STAGE == "before_ranking"


def test_a_correction_is_not_reachable_by_a_historical_knowledge_query(
    session: Session, superseded: dict[str, Any]
) -> None:
    """The same adversarial shape on the knowledge axis.

    A correction's text is the best possible match for a query made of that text. Asked with a
    knowledge date before the correction existed, it must not be a candidate — even though it is
    the single most similar row in the table.
    """
    path = CORPUS / "documents.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    documents = raw["documents"] if isinstance(raw, dict) else raw
    corrected = sorted(
        (d for d in documents if d.get("corrected_by") and d["language"] == Language.EN.value),
        key=lambda d: str(d["document_id"]),
    )
    if not corrected:
        pytest.skip("this corpus contains no corrections")
    original = corrected[0]
    correction_id = str(original["corrected_by"])

    chunk_raw = json.loads((CORPUS / "chunks.json").read_text(encoding="utf-8"))
    chunks = chunk_raw["chunks"] if isinstance(chunk_raw, dict) else chunk_raw
    correction_chunk = sorted(
        (c for c in chunks if str(c["document_id"]) == correction_id),
        key=lambda c: str(c["chunk_id"]),
    )[0]

    valid_from = date.fromisoformat(str(original["valid_from"]))
    valid_to = date.fromisoformat(str(original["valid_to"]))
    known_to = date.fromisoformat(str(original["known_to"]))
    as_of = valid_from + (valid_to - valid_from) // 2

    query = Query(
        text=str(correction_chunk["text"])[:2000],
        language=Language.EN,
        as_of=as_of,
        known_as_of=known_to - timedelta(days=1),
        variant_id=str(correction_chunk["effectivity"]["variant_id"]),
        top_k=10,
    )

    admitted = {c.chunk_id for c in fetch_candidates(session, candidate_filter(query))}
    assert str(correction_chunk["chunk_id"]) not in admitted, (
        "a correction that did not exist at the asked knowledge date was admitted as a candidate"
    )

    # And the premise: without the filter it is the top hit, so the exclusion above is real.
    unfiltered = Retriever().retrieve(session, query, apply_effectivity=False)
    top = next(i.chunk.chunk_id for i in unfiltered.chunks)
    assert top == str(correction_chunk["chunk_id"])
