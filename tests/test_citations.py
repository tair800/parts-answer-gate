"""Citations, checked the way kill condition D checks them: by offset, not by membership.

The central test in this file is `test_a_quote_that_occurs_elsewhere_in_the_document_still_fails`.
A verifier that asked "does this text appear in that document" would pass a citation that had
drifted to another section of the same manual — the failure a technician cannot detect by reading
the answer, because the words are right and the place is wrong.
"""

from __future__ import annotations

from datetime import date

import pytest

from parts_answer_gate import citations as cite
from parts_answer_gate.domain import (
    Chunk,
    Citation,
    Effectivity,
    Language,
    Query,
    RetrievedChunk,
    SerialRange,
)

AS_OF = date(2026, 3, 1)

TORQUE_PARAGRAPH = (
    "Tighten the drive coupling bolts on the XP-400 to 48 Nm in a cross pattern. "
    "Use only the AB-1234-C fastener kit; a lower grade bolt will not hold this value. "
    "Recheck the torque after the first 50 operating hours."
)

# A realistic document: the cited paragraph sits well inside it, so a citation that forgot to
# translate its offsets would point at the front matter and be caught.
DOCUMENT = (
    "XP-400 SERVICE MANUAL\n"
    "Revision C, in force from 1 January 2024.\n"
    "\n"
    "3.1 Scope. This section applies to XP-400 machines in the serial range 1000 to 4999.\n"
    "\n"
    "3.2 Drive coupling.\n" + TORQUE_PARAGRAPH + "\n"
    "\n"
    "3.3 Hydraulics. The reservoir capacity is 120 L.\n"
)
PARAGRAPH_AT = DOCUMENT.index(TORQUE_PARAGRAPH)

REVISIONS = {"DOC-1": "C", "DOC-2": "D"}


def make_chunk(
    chunk_id: str = "c1",
    text: str = TORQUE_PARAGRAPH,
    *,
    document_id: str = "DOC-1",
    start_offset: int = PARAGRAPH_AT,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        family_id="FAM-1",
        language=Language.EN,
        text=text,
        section="3.2",
        page=7,
        start_offset=start_offset,
        end_offset=start_offset + len(text),
        effectivity=Effectivity(variant_id="XP-400", serials=SerialRange(first=1000, last=4999)),
        valid_from=date(2024, 1, 1),
    )


def found(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, fused_score=0.9, rank=1)


def ask(text: str) -> Query:
    return Query(text=text, language=Language.EN, as_of=AS_OF, variant_id="XP-400")


# ------------------------------------------------------------------------ the graded function


def test_a_citation_built_from_a_chunk_verifies_against_its_document() -> None:
    citation = cite.build_citations([found(make_chunk())], REVISIONS, ask("XP-400 bolt torque?"))[0]

    assert cite.verify_citation(citation, DOCUMENT) is True
    assert DOCUMENT[citation.start_offset : citation.end_offset] == citation.quote


def test_a_quote_that_occurs_elsewhere_in_the_document_still_fails() -> None:
    """Kill condition D is about coordinates, not about words appearing somewhere.

    The quote below really is in the document. The offsets point at the front matter. A membership
    test passes this and a reader following the citation lands on the wrong section.
    """
    citation = cite.build_citations([found(make_chunk())], REVISIONS, ask("XP-400 bolt torque?"))[0]
    drifted = citation.model_copy(
        update={"start_offset": 0, "end_offset": len(citation.quote)},
    )

    assert citation.quote in DOCUMENT, "the fixture no longer tests what it claims to"
    assert cite.verify_citation(drifted, DOCUMENT) is False


def test_an_offset_off_by_one_fails() -> None:
    citation = cite.build_citations([found(make_chunk())], REVISIONS, ask("XP-400 bolt torque?"))[0]
    slipped = citation.model_copy(
        update={
            "start_offset": citation.start_offset + 1,
            "end_offset": citation.end_offset + 1,
        },
    )

    assert cite.verify_citation(slipped, DOCUMENT) is False


def test_a_span_running_past_the_end_of_the_document_fails() -> None:
    citation = cite.build_citations([found(make_chunk())], REVISIONS, ask("XP-400 bolt torque?"))[0]
    overrun = citation.model_copy(
        update={
            "start_offset": len(DOCUMENT) - 5,
            "end_offset": len(DOCUMENT) + len(citation.quote),
        },
    )

    assert cite.verify_citation(overrun, DOCUMENT) is False


def test_a_span_whose_length_disagrees_with_its_quote_fails() -> None:
    citation = cite.build_citations([found(make_chunk())], REVISIONS, ask("XP-400 bolt torque?"))[0]
    stretched = citation.model_copy(update={"end_offset": citation.end_offset + 3})

    assert cite.verify_citation(stretched, DOCUMENT) is False


def test_an_unsupplied_document_counts_as_unverified_rather_than_as_a_pass() -> None:
    citation = cite.build_citations([found(make_chunk())], REVISIONS, ask("XP-400 bolt torque?"))[0]

    assert cite.unfaithful_citations([citation], {"DOC-1": DOCUMENT}) == ()
    assert cite.unfaithful_citations([citation], {}) == (citation,)


def test_unfaithful_citations_reports_the_drifted_one_and_only_it() -> None:
    good = cite.build_citations([found(make_chunk())], REVISIONS, ask("XP-400 bolt torque?"))[0]
    bad = good.model_copy(update={"chunk_id": "c2", "start_offset": 0, "end_offset": 10})

    assert cite.unfaithful_citations([good, bad], {"DOC-1": DOCUMENT}) == (bad,)


# ----------------------------------------------------------------------------- span selection


def test_the_quoted_span_answers_the_question_that_was_asked() -> None:
    fastener = cite.build_citations(
        [found(make_chunk())], REVISIONS, ask("Which fastener kit for the XP-400?")
    )[0]
    torque = cite.build_citations(
        [found(make_chunk())], REVISIONS, ask("What torque for the drive coupling bolts?")
    )[0]

    assert "AB-1234-C fastener kit" in fastener.quote
    assert "48 Nm" in torque.quote
    assert fastener.quote != torque.quote


def test_a_quote_is_a_span_rather_than_the_whole_of_a_long_chunk() -> None:
    """A citation that pastes the chunk makes the reader do the locating work it existed to do."""
    long_text = (
        TORQUE_PARAGRAPH + " Dispose of the used gasket. Record the work order number. "
        "Return the machine to service."
    )

    start, end = cite.select_quote_span(long_text, frozenset({"torque"}))

    assert end - start < len(long_text)
    assert "Recheck the torque" in long_text[start:end]


def test_sentence_spans_are_offsets_into_the_original_text() -> None:
    spans = cite.sentence_spans(TORQUE_PARAGRAPH)

    assert len(spans) == 3
    for start, end in spans:
        assert TORQUE_PARAGRAPH[start:end].strip() == TORQUE_PARAGRAPH[start:end]
    assert TORQUE_PARAGRAPH[spans[0][0] : spans[0][1]].endswith("cross pattern.")


def test_a_chunk_with_no_sentence_boundary_is_still_quotable() -> None:
    table_row = "XP-400 | 48 Nm | AB-1234-C"
    chunk = make_chunk("c9", table_row, start_offset=0)

    citation = cite.build_citation(chunk, "C")

    assert citation.quote == table_row
    assert cite.verify_citation(citation, table_row) is True


def test_an_empty_chunk_produces_an_empty_quote_rather_than_an_exception() -> None:
    citation = cite.build_citation(make_chunk("c9", "   ", start_offset=0), "C")

    assert citation.quote == ""


def test_the_same_inputs_produce_the_same_span_every_time() -> None:
    """Kill condition J reaches into citation selection too: a tie broken at random is a diff."""
    question = ask("What torque for the XP-400 drive coupling bolts?")
    first = cite.build_citations([found(make_chunk())], REVISIONS, question)
    second = cite.build_citations([found(make_chunk())], REVISIONS, question)

    assert first == second


# ------------------------------------------------------------------- refusing to make it up


def test_a_document_with_no_known_revision_cannot_be_cited() -> None:
    """There is no placeholder revision. A project about revision correctness may not invent one."""
    with pytest.raises(cite.MissingRevisionError, match="DOC-7"):
        cite.build_citations([found(make_chunk("c1", document_id="DOC-7"))], REVISIONS)


def test_a_chunk_whose_offsets_disagree_with_its_text_is_refused() -> None:
    """Anything derived from coordinates that are already wrong is fiction."""
    broken = make_chunk().model_copy(update={"end_offset": PARAGRAPH_AT + 5})

    with pytest.raises(ValueError, match="characters of text"):
        cite.build_citation(broken, "C")


def test_a_span_outside_the_chunk_is_refused() -> None:
    with pytest.raises(ValueError, match="outside chunk"):
        cite.build_citation(make_chunk(), "C", (0, len(TORQUE_PARAGRAPH) + 10))


def test_the_citation_carries_the_coordinates_a_reader_needs() -> None:
    citation = cite.build_citations([found(make_chunk())], REVISIONS, ask("XP-400 bolt torque?"))[0]

    assert citation.chunk_id == "c1"
    assert citation.document_id == "DOC-1"
    assert citation.revision == "C"
    assert citation.section == "3.2"
    assert citation.page == 7
    assert isinstance(citation, Citation)
