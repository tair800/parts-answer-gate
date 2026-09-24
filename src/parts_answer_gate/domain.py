"""The types every other module agrees on, and the two intervals that make this project different.

Most retrieval systems model a chunk as text plus an embedding plus a document id. That is enough
to answer "what does the manual say" and not enough to answer the question a technician actually
asks, which is **"what was the approved procedure on the date of the incident, for this machine"**.

Two things carry that, and they are not the same thing:

- **Effectivity** — which machines a passage applies to: a variant, and a serial range within it. A
  procedure correct for an XP-400 is wrong for an XP-400S, and a reader cannot tell from the prose.
- **Validity** — when a revision was in force. Revisions are *withdrawn, not deleted*, because a
  warranty dispute about an incident in March needs the procedure that was approved in March, not
  the one approved last week.

`Chunk` therefore carries a validity interval as well as an effectivity, and a `Query` carries an
`as_of` date as well as a variant. The filter built from those runs in SQL, before ranking — not as
a post-filter on a ranked list, which silently shrinks the result set and loses recall nobody
measures.

Nothing here imports a retriever, a gate or a model. This module is the vocabulary; the decisions
live where they can be tested separately.
"""

from __future__ import annotations

import enum
import re
from datetime import date
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "PART_NUMBER_PATTERN",
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "Effectivity",
    "GateDecision",
    "GateOutcome",
    "GateSignals",
    "Language",
    "Query",
    "RetrievedChunk",
    "SerialRange",
    "part_numbers_in",
]


class _Frozen(BaseModel):
    """Immutable, and refusing fields nobody declared.

    `extra="forbid"` is doing real work in a retrieval project: a mistyped `varient` silently
    becoming an ignored attribute is how an effectivity filter comes to filter nothing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Language(enum.StrEnum):
    """The three the blueprint scopes. Closed, so an unknown code is an error not a default."""

    EN = "en"
    TR = "tr"
    RU = "ru"


class GateOutcome(enum.StrEnum):
    """What the gate decided, before any text was generated.

    `REVIEW` is not a hedge and not a softer abstention. It is the system saying the evidence
    *conflicts* — two variants, or a supersession the corpus records ambiguously — and that a person
    must decide. A technician told "I am not sure" walks away; a technician told "these two
    bulletins disagree, here they are" has something to act on.
    """

    ANSWER = "answer"
    ABSTAIN = "abstain"
    REVIEW = "review"


#: What a part number looks like in this corpus, and the only thing the groundedness check treats as
#: one. Deliberately strict: a loose pattern that also matched `48 Nm` would make kill condition C
#: pass by never finding anything to check.
PART_NUMBER_PATTERN = re.compile(r"\b[A-Z]{2}-\d{4}-[A-Z0-9]{1,3}\b")


def part_numbers_in(text: str) -> frozenset[str]:
    """Every part number in a piece of text.

    Used on both sides of kill condition C — the answer and the union of its cited chunks — so the
    comparison is between two applications of one function rather than between a parser and a
    different parser.
    """
    return frozenset(PART_NUMBER_PATTERN.findall(text))


class SerialRange(_Frozen):
    """An inclusive serial interval, or an open one.

    Open at either end on purpose: "from serial 4200 onwards" is how a bulletin is actually written,
    and forcing a synthetic upper bound would make a chunk stop applying to machines it applies to.
    """

    first: int | None = None
    last: int | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.first is not None and self.last is not None and self.first > self.last:
            raise ValueError(f"serial range {self.first}..{self.last} is inverted")
        return self

    def covers(self, serial: int | None) -> bool:
        """Whether a machine is in range. An unknown serial is covered only by an unbounded range.

        That asymmetry is deliberate and is the safe direction: a technician who did not supply a
        serial should not be handed a procedure that applies to half the fleet.
        """
        if serial is None:
            return self.first is None and self.last is None
        if self.first is not None and serial < self.first:
            return False
        return not (self.last is not None and serial > self.last)

    def __str__(self) -> str:
        if self.first is None and self.last is None:
            return "all serials"
        return f"{self.first or 'start'}..{self.last or 'end'}"


class Effectivity(_Frozen):
    """Which machines a passage applies to. The `variant` half of a wrong answer."""

    variant_id: str
    serials: SerialRange = SerialRange()

    def applies_to(self, variant_id: str, serial: int | None) -> bool:
        return self.variant_id == variant_id and self.serials.covers(serial)


class Document(_Frozen):
    """One revision of one document. A later revision is a **new** document with an edge to it.

    Modelled as separate documents rather than as versions of one, because that is what makes an
    as-of query expressible at all: the March revision still exists, still has its own chunks, and
    is still retrievable when somebody asks about March.
    """

    document_id: str
    family_id: str
    title: str
    language: Language
    revision: str
    #: **Valid time.** When this revision came into force *for the machine*, and when it stopped.
    #: `None` means still in force.
    valid_from: date
    valid_to: date | None = None
    #: **Knowledge time.** When the organisation learned this, and when it stopped believing it.
    #: `None` means still believed.
    #:
    #: The second axis is what makes this bitemporal rather than merely versioned. Valid time
    #: answers *what was true for the machine in March*; knowledge time answers *what we knew in
    #: March*. They are independent: a correction issued in 2025 about a 2021 procedure is valid in
    #: 2021 and known from 2025, and the two questions have different right answers. Without this
    #: field a correction silently rewrites history, and an audit asking "what did the technician
    #: have in front of them?" cannot be answered at all.
    #:
    #: Defaults to `valid_from`, which is exact for any document that has never been corrected: it
    #: was believed from the day it came into force, and still is. That is the same rule migration
    #: 0002 backfills with, so a row written before the second axis existed and a model constructed
    #: without it agree. A correction must state its own `known_from`, and the validator below
    #: refuses the half-corrected shapes.
    known_from: date = None  # type: ignore[assignment]
    known_to: date | None = None
    #: The document that corrected this one's *knowledge* — same validity period, better
    #: information. Distinct from `superseded_by`, which replaces a revision going forward.
    corrected_by: str | None = None
    #: The revision that replaced this one. The single most important field in the project.
    superseded_by: str | None = None
    source_uri: str
    checksum: str

    @model_validator(mode="after")
    def _supersession_is_consistent(self) -> Self:
        """A withdrawn revision has both an end date and a successor, or neither.

        One without the other is the bug this project exists to prevent, in the data rather than in
        the query: a document with `superseded_by` and no `valid_to` stays in force for ever and an
        as-of query keeps returning it.
        """
        if (self.superseded_by is None) != (self.valid_to is None):
            raise ValueError(
                f"{self.document_id} has superseded_by={self.superseded_by!r} and "
                f"valid_to={self.valid_to!r}; a withdrawn revision needs both, a current one needs "
                "neither"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _default_known_from(cls, data: Any) -> Any:
        """Fill `known_from` from `valid_from` when the caller did not state one."""
        if isinstance(data, dict) and data.get("known_from") is None:
            data = {**data, "known_from": data.get("valid_from")}
        return data

    @model_validator(mode="after")
    def _correction_is_consistent(self) -> Self:
        """A corrected document has both an end to its knowledge and a corrector, or neither.

        The same shape as `_supersession_is_consistent`, and for the same reason: a document with
        `corrected_by` and no `known_to` is still believed for ever, so a knowledge-time query at
        any date keeps returning superseded information alongside the correction.
        """
        if (self.corrected_by is None) != (self.known_to is None):
            raise ValueError(
                f"{self.document_id} has corrected_by={self.corrected_by!r} and "
                f"known_to={self.known_to!r}; a corrected document needs both, a current one "
                "needs neither"
            )
        return self

    def in_force_on(self, as_of: date) -> bool:
        return self.valid_from <= as_of and (self.valid_to is None or as_of < self.valid_to)

    def known_on(self, known_as_of: date | None) -> bool:
        """Whether this was believed at the given *knowledge* date. Half-open, like validity.

        `None` means current knowledge, which is `known_to is None` — still believed, never
        corrected. It is not the same as passing today's date, and the difference matters for a
        document corrected with a future-dated correction.
        """
        if known_as_of is None:
            return self.known_to is None
        return self.known_from <= known_as_of and (
            self.known_to is None or known_as_of < self.known_to
        )

    def visible_at(self, as_of: date, known_as_of: date | None = None) -> bool:
        """Both axes at once: in force for the machine, and known to us."""
        return self.in_force_on(as_of) and self.known_on(known_as_of)


class Chunk(_Frozen):
    """A retrievable passage, with the coordinates a citation needs.

    `start_offset` and `end_offset` are character offsets **into the document's own text**, not into
    some normalised copy. Kill condition D locates a cited span in the source by those offsets, and
    an offset into a string nobody kept is not a coordinate.
    """

    chunk_id: str
    document_id: str
    family_id: str
    language: Language
    text: str
    #: 1-based, as a reader would count sections.
    section: str
    page: int
    start_offset: int
    end_offset: int
    effectivity: Effectivity
    #: Denormalised from the document so the SQL filter needs no join. The generator and the loader
    #: are the only writers, and `test_store.py` asserts they agree with the document they came
    #: from.
    valid_from: date
    valid_to: date | None = None
    #: Denormalised from the document, same as validity. Both axes live on the chunk so the
    #: candidate-set predicate stays a single-table `WHERE` and keeps running before ranking — a
    #: join here would be the thing that tempts somebody to filter afterwards instead.
    #:
    #: Defaults to `valid_from`; see `Document.known_from`.
    known_from: date = None  # type: ignore[assignment]
    known_to: date | None = None
    superseded_by: str | None = None
    corrected_by: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_known_from(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("known_from") is None:
            data = {**data, "known_from": data.get("valid_from")}
        return data

    @model_validator(mode="after")
    def _intervals_are_consistent(self) -> Self:
        """The same two invariants `Document` enforces, on the denormalised copy.

        The chunk carries both intervals so the candidate predicate stays a single-table `WHERE`,
        and a denormalised copy that can hold a state its source refuses is a copy that will
        eventually hold one. A chunk with `superseded_by` and no `valid_to` is in force for ever
        and an as-of query keeps returning it; a chunk with `corrected_by` and no `known_to` is
        believed for ever and a knowledge query returns it beside the correction.
        """
        if (self.superseded_by is None) != (self.valid_to is None):
            raise ValueError(
                f"{self.chunk_id} has superseded_by={self.superseded_by!r} and "
                f"valid_to={self.valid_to!r}; a withdrawn passage needs both, a current one needs "
                "neither"
            )
        if (self.corrected_by is None) != (self.known_to is None):
            raise ValueError(
                f"{self.chunk_id} has corrected_by={self.corrected_by!r} and "
                f"known_to={self.known_to!r}; a corrected passage needs both, a current one needs "
                "neither"
            )
        return self

    @property
    def part_numbers(self) -> frozenset[str]:
        return part_numbers_in(self.text)

    def in_force_on(self, as_of: date) -> bool:
        return self.valid_from <= as_of and (self.valid_to is None or as_of < self.valid_to)

    def known_on(self, known_as_of: date | None) -> bool:
        """`None` means current knowledge. See `Document.known_on`."""
        if known_as_of is None:
            return self.known_to is None
        return self.known_from <= known_as_of and (
            self.known_to is None or known_as_of < self.known_to
        )

    def visible_at(self, as_of: date, known_as_of: date | None = None) -> bool:
        return self.in_force_on(as_of) and self.known_on(known_as_of)


class Query(_Frozen):
    """What a technician asked, and the two things that make the answer theirs rather than general.

    `as_of` defaults to nothing on purpose — it is supplied by the caller, and the service fills it
    with today only at the edge. A default of `date.today()` buried in a model is a hidden clock,
    and a hidden clock makes a replayed evaluation irreproducible.
    """

    text: str = Field(min_length=1, max_length=2000)
    language: Language = Language.EN
    as_of: date
    #: The **knowledge** date: what the organisation knew, as opposed to what was true of the
    #: machine. `None` means *current knowledge* — everything still believed, corrections included.
    #:
    #: Separate from `as_of` because the two questions are genuinely different. A technician asking
    #: *what was the correct torque in March 2021* wants today's best understanding of March 2021,
    #: which is `as_of=2021-03` and this left at `None`. An auditor asking *what did the technician
    #: have in front of them in March 2021* wants the belief of the time, which is `as_of=2021-03`
    #: and `known_as_of=2021-03`. A system with one date can express one of those and silently
    #: answers the other.
    #:
    #: `None` deliberately means current rather than "the same as `as_of`". Defaulting knowledge to
    #: the validity date would hide every correction from every historical question — a technician
    #: asking about a 2021 procedure would be handed the 2021 mistake rather than the fix, which is
    #: the failure mode this project exists to prevent, arrived at from the other direction.
    known_as_of: date | None = None
    variant_id: str | None = None
    serial: int | None = None
    top_k: Annotated[int, Field(ge=1, le=100)] = 10

    @property
    def asks_historical_knowledge(self) -> bool:
        """Whether this pins knowledge time — an audit question rather than a lookup."""
        return self.known_as_of is not None


class RetrievedChunk(_Frozen):
    """A chunk with the scores that produced it, kept separate rather than fused into one number.

    All three are carried because the evaluation attributes error to a stage: a question that failed
    with a high lexical score and a low dense score failed differently from one where both were low,
    and a single fused score cannot tell the two apart.
    """

    chunk: Chunk
    lexical_score: float = 0.0
    dense_score: float = 0.0
    fused_score: float = 0.0
    #: Where in the fused ranking this landed, 1-based.
    rank: int = 0
    #: Set when the deterministic identifier lookup found this, before any embedding was consulted.
    exact_identifier_hit: bool = False


class GateSignals(_Frozen):
    """Everything the gate looked at. Recorded so a decision can be argued with.

    Not one of these is a model output. That is the whole point: the decision to answer is
    deterministic and reproducible, and the model is only ever asked to phrase something the gate
    already permitted.
    """

    top_fused_score: float
    supporting_chunks: int
    #: Share of the question's content terms that appear in the supporting chunks.
    term_coverage: float
    #: True when every supporting chunk agrees about which variant it applies to.
    variant_agreement: bool
    #: True when any supporting chunk was superseded at the query's as-of date. Should be
    #: impossible after filtering, and is measured anyway — a guarantee nobody checks is a belief.
    superseded_present: bool
    #: Two supporting chunks that state different values for the same attribute.
    conflicting_evidence: bool
    exact_identifier_hit: bool


class GateDecision(_Frozen):
    """The outcome, the reason a person can read, and the signals behind it."""

    outcome: GateOutcome
    reason: str
    signals: GateSignals
    #: The chunks the answerer is allowed to see. Empty on ABSTAIN, deliberately: an answerer that
    #: received evidence on an abstention could be asked to use it.
    approved_chunks: tuple[RetrievedChunk, ...] = ()

    @model_validator(mode="after")
    def _abstention_approves_nothing(self) -> Self:
        if self.outcome is GateOutcome.ABSTAIN and self.approved_chunks:
            raise ValueError(
                "an ABSTAIN decision carries approved chunks; the answerer must not be able to see "
                "evidence for a question the gate refused"
            )
        return self


class Citation(_Frozen):
    """A claim traced to a span of a document, by coordinates a reader can check.

    `quote` is the verbatim span. Kill condition D finds it in the source document at
    `start_offset`, so a citation that drifted is caught by arithmetic rather than by judgement.
    """

    chunk_id: str
    document_id: str
    revision: str
    section: str
    page: int
    quote: str
    start_offset: int
    end_offset: int


class Answer(_Frozen):
    """What the service returns, whatever the gate decided.

    An abstention is an `Answer` with `GateOutcome.ABSTAIN` and no text, not a different type and
    not an error. A caller that has to handle two shapes will handle one of them badly, and the one
    they handle badly is always the refusal.
    """

    query: Query
    decision: GateDecision
    text: str | None = None
    citations: tuple[Citation, ...] = ()
    #: EU AI Act Article 50, in force since 2 August 2026. A property of the response rather than of
    #: the user interface, because the obligation follows the content and not the page it is drawn
    #: on. `test_article_50.py` asserts it cannot be absent.
    #:
    #: `min_length` rather than a bare default, because a default is only a suggestion: a caller
    #: could pass the empty string and the obligation would vanish with nothing objecting. A test
    #: caught exactly that, which is the argument for writing the test.
    ai_disclosure: str = Field(
        default=(
            "This answer was generated by an AI system from the cited source passages. "
            "Verify against the referenced document revision before acting on it."
        ),
        min_length=20,
    )
    #: Which answerer produced the text. `extractive` returns spans verbatim and therefore cannot
    #: emit a part number the evidence does not contain; an abstractive arm is post-validated
    #: against the same rule.
    answerer: str = "extractive"

    @model_validator(mode="after")
    def _text_and_outcome_agree(self) -> Self:
        if self.decision.outcome is GateOutcome.ABSTAIN and self.text:
            raise ValueError("an abstention carries answer text")
        if self.decision.outcome is GateOutcome.ANSWER and not self.citations:
            raise ValueError("an answer carries no citation; every factual answer must cite")
        return self

    @property
    def grounded(self) -> bool:
        """Whether every part number in the text occurs in a cited quote.

        Kill condition C in one property, so the service, the evaluation and the tests all ask the
        same question of the same object rather than three near-identical questions.
        """
        if not self.text:
            return True
        cited = (
            frozenset().union(*(part_numbers_in(c.quote) for c in self.citations))
            if (self.citations)
            else frozenset()
        )
        return part_numbers_in(self.text) <= cited
