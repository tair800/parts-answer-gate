"""Citations that can be checked by arithmetic instead of by judgement.

A citation in this system is a **verbatim span plus its coordinates in the source document**. That
is a stronger object than the document-level reference most retrieval systems emit, and the
difference is exactly kill condition D: "the right document was nearby" is what makes bad RAG look
good, and it is caught here by locating the quoted characters at the offset the citation claims.

`verify_citation` is the graded function. It slices the document at `start_offset` and compares —
it does **not** ask whether the quote occurs *somewhere* in the document. A membership test would
pass for a citation that had drifted to a different section of the same manual, which is precisely
the failure a technician cannot detect by reading the answer.

Offsets are absolute into the document's own text. A chunk carries the offset of its own first
character, so a span chosen inside a chunk is translated once, here, and nowhere else.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final

from parts_answer_gate.domain import Chunk, Citation, Query, RetrievedChunk
from parts_answer_gate.gate import content_terms, term_is_covered, tokens_in

__all__ = [
    "MAX_QUOTE_CHARS",
    "MAX_QUOTE_SENTENCES",
    "MissingRevisionError",
    "build_citation",
    "build_citations",
    "select_quote_span",
    "sentence_spans",
    "unfaithful_citations",
    "verify_citation",
]

#: A quote is at most this many sentences. A citation that quotes a whole chunk is not a citation,
#: it is a paste, and it makes the reader do the locating work the citation existed to do.
MAX_QUOTE_SENTENCES: Final[int] = 3

#: And at most this many characters, for the degenerate chunk that contains no sentence boundary at
#: all — a table, or a single unpunctuated line.
MAX_QUOTE_CHARS: Final[int] = 480

#: A sentence: a run starting at a non-space character and ending at terminal punctuation followed
#: by whitespace, or at the end of its line. `.` does not match a newline, so a match can never span
#: two lines and a table row is its own span. Deliberately not a sentence *splitter* — the spans are
#: returned as offsets, because a split that returns strings has already thrown away the coordinates
#: this module exists to preserve.
_SENTENCE = re.compile(r"[^\s].*?(?:[.!?…](?=\s|$)|$)", re.MULTILINE)


class MissingRevisionError(LookupError):
    """Raised when no revision label is known for a document being cited.

    There is no placeholder for this and there must not be one. The single thing this project
    claims is that an answer is revision-correct; a citation carrying `unknown` as its revision
    would be that claim quietly withdrawn while still being displayed.
    """


def sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Sentence boundaries as `(start, end)` offsets into `text`, whitespace already excluded."""
    spans = [(match.start(), match.end()) for match in _SENTENCE.finditer(text) if match.group()]
    return tuple(spans)


def select_quote_span(text: str, terms: frozenset[str] = frozenset()) -> tuple[int, int]:
    """The span of `text` worth quoting: the contiguous sentence window that covers most of `terms`.

    Contiguous on purpose. A citation assembled from two distant sentences would need two offset
    pairs, and a reader checking one of them against the source would find text that reads as
    one passage but never was. Ties go to the shorter window and then to the earlier one, so
    the choice is deterministic — kill condition J requires two runs to produce identical
    output, and "whichever window happened to be yielded first" does not survive that.
    """
    spans = sentence_spans(text)
    if not spans:
        return _fallback_span(text)

    best: tuple[int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    for start_index, (start, _) in enumerate(spans):
        for end_index in range(start_index, min(start_index + MAX_QUOTE_SENTENCES, len(spans))):
            end = spans[end_index][1]
            if end - start > MAX_QUOTE_CHARS and end_index > start_index:
                break
            covered = _covered_terms(text[start:end], terms)
            key = (-covered, end - start, start)
            if best_key is None or key < best_key:
                best_key, best = key, (start, end)
    return best if best is not None else _fallback_span(text)


def _covered_terms(window: str, terms: frozenset[str]) -> int:
    """How many distinct question terms this window contains.

    Uses the gate's own tokeniser and coverage rule rather than a second implementation of the same
    idea: the quote a reader is shown should be chosen by the measure the gate scored, or the
    displayed evidence and the reason for approving it slowly diverge.
    """
    if not terms:
        return 0
    tokens = tokens_in(window)
    return sum(1 for term in terms if term_is_covered(term, tokens))


def _fallback_span(text: str) -> tuple[int, int]:
    """A chunk with no sentence boundary still has to be quotable.

    Truncated at the last whitespace inside the cap rather than at the cap itself, so a quote never
    ends in the middle of a part number — a half-identifier displayed to a technician is worse than
    a shorter quote.
    """
    stripped = text.strip()
    if not stripped:
        return (0, 0)
    start = text.index(stripped[0])
    end = start + len(stripped)
    if end - start <= MAX_QUOTE_CHARS:
        return (start, end)
    cut = text.rfind(" ", start, start + MAX_QUOTE_CHARS)
    return (start, cut if cut > start else start + MAX_QUOTE_CHARS)


def build_citation(chunk: Chunk, revision: str, span: tuple[int, int] | None = None) -> Citation:
    """One citation for one chunk, with the quoted span translated into document coordinates.

    `span` is local to the chunk's own text. The translation happens here and only here: two modules
    that each add `chunk.start_offset` is how an off-by-one becomes an unverifiable citation.
    """
    if chunk.end_offset - chunk.start_offset != len(chunk.text):
        # The chunk's own coordinates disagree with its text, so anything derived from them is
        # fiction. Loud, because the alternative is emitting a citation that cannot be verified and
        # discovering it in the evaluation as a number rather than as a defect.
        raise ValueError(
            f"chunk {chunk.chunk_id} spans {chunk.start_offset}..{chunk.end_offset} "
            f"({chunk.end_offset - chunk.start_offset} characters) but carries "
            f"{len(chunk.text)} characters of text"
        )
    start, end = span if span is not None else select_quote_span(chunk.text)
    if not (0 <= start <= end <= len(chunk.text)):
        raise ValueError(f"span {start}..{end} is outside chunk {chunk.chunk_id}")
    return Citation(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        revision=revision,
        section=chunk.section,
        page=chunk.page,
        quote=chunk.text[start:end],
        start_offset=chunk.start_offset + start,
        end_offset=chunk.start_offset + end,
    )


def build_citations(
    chunks: Sequence[RetrievedChunk],
    revisions: Mapping[str, str],
    query: Query | None = None,
) -> tuple[Citation, ...]:
    """A citation per approved chunk, each quoting the span that best answers `query`.

    `revisions` maps a document id to its revision label. It is required rather than defaulted for
    the reason `MissingRevisionError` documents. The span is chosen with the gate's own notion of a
    content term, so the passage displayed is chosen by the same measure the gate scored coverage
    with rather than by a second, quietly different one.
    """
    terms = content_terms(query.text, query.language) if query is not None else frozenset()
    citations: list[Citation] = []
    for retrieved in chunks:
        chunk = retrieved.chunk
        revision = revisions.get(chunk.document_id)
        if revision is None:
            raise MissingRevisionError(
                f"no revision label for document {chunk.document_id}, cited by chunk "
                f"{chunk.chunk_id}; a citation may not name a revision it was not given"
            )
        citations.append(build_citation(chunk, revision, select_quote_span(chunk.text, terms)))
    return tuple(citations)


def verify_citation(citation: Citation, document_text: str) -> bool:
    """Whether the quote really occupies `start_offset..end_offset` of the document it names.

    This is the function kill condition D is graded by. It locates the span by offset and compares
    the characters; it does not test membership. A citation whose quote appears elsewhere in the
    same document fails here, and should: an offset that points at different text is a coordinate
    that has drifted, and a reader who follows it lands on the wrong paragraph.
    """
    start, end = citation.start_offset, citation.end_offset
    if start < 0 or end < start or end > len(document_text):
        return False
    if end - start != len(citation.quote):
        return False
    return document_text[start:end] == citation.quote


def unfaithful_citations(
    citations: Sequence[Citation],
    documents: Mapping[str, str],
) -> tuple[Citation, ...]:
    """Every citation that does not verify, including any whose document was not supplied.

    An unsupplied document counts as a failure rather than being skipped. Treating "the verifier was
    not given that text" as a pass is how a zero on kill condition D would be reached by omission.
    """
    return tuple(
        citation
        for citation in citations
        if citation.document_id not in documents
        or not verify_citation(citation, documents[citation.document_id])
    )
