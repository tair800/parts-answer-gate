"""Turning approved evidence into an answer, without acquiring the ability to invent one.

Claim 3 of ADR-001 is that a part number the retrieved evidence does not contain cannot appear in an
answer. There are two ways to hold that claim and they are not equally strong:

* **Measure it.** Generate freely, then check the output and count the failures. This is what a
  groundedness metric does, and the number it produces is a description of how often the system was
  lucky.
* **Make it impossible.** Assemble the answer out of characters copied from the cited spans. Then
  the set of part numbers in the answer is a subset of the set in the quotes as a matter of string
  arithmetic, and the metric measures a property that could not have been otherwise.

`ExtractiveAnswerer`, the default and the only arm wired in this build, is the second. The
abstractive arm exists as a port so the seam is real rather than described, and it **raises** — no
key exists, nothing was called, and a stub returning plausible text would put a number into the
evaluation that no model produced.

The post-validator runs over *any* answerer's output. The extractive arm cannot trip it; that is not
a reason to stop applying it, because the arm that can trip it is the one it was written for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict

from parts_answer_gate.citations import build_citations
from parts_answer_gate.domain import (
    Answer,
    Citation,
    GateDecision,
    GateOutcome,
    Query,
    part_numbers_in,
)

__all__ = [
    "ANSWER_SEPARATOR",
    "AbstractiveAnswerer",
    "AbstractiveArmUnavailableError",
    "Answerer",
    "EmptyEvidenceError",
    "EvidenceItem",
    "ExtractiveAnswerer",
    "PostValidatedAnswerer",
    "PromptPayload",
    "UngroundedAnswerError",
    "post_validate",
    "ungrounded_part_numbers",
]

#: Quoted spans are joined with a blank line. The separator has to be characters a part number
#: cannot contain, or two adjacent quotes could form an identifier that appears in neither — the one
#: way an extractive answerer could still emit something its evidence does not hold.
ANSWER_SEPARATOR: Final[str] = "\n\n"


class UngroundedAnswerError(ValueError):
    """An answer contains a part number that no cited quote contains. Kill condition C."""


class EmptyEvidenceError(ValueError):
    """The gate approved evidence with nothing quotable in it.

    Raised rather than silently degraded to an abstention: the answerer does not get to change the
    gate's decision, and a corpus that produced an approvable chunk with no text is a defect that
    should surface as one.
    """


class AbstractiveArmUnavailableError(RuntimeError):
    """The abstractive arm was called in a build that has no model behind it."""


class Answerer(Protocol):
    """The port. One method, and it can only ever see what the gate approved.

    The decision is an input, never an output: nothing an implementation returns can change the
    outcome, because the outcome is already on the `GateDecision` it was handed.
    """

    @property
    def name(self) -> str:
        """Recorded on every `Answer`, so a stored answer says which arm produced it."""

    def answer(
        self,
        query: Query,
        decision: GateDecision,
        revisions: Mapping[str, str],
    ) -> Answer: ...


def ungrounded_part_numbers(answer: Answer) -> frozenset[str]:
    """Part numbers in the answer text that no cited quote contains.

    Compared against the **quotes**, not against the approved chunks. A chunk is what the system
    retrieved; a quote is what it showed the technician and what a reader can check. An answer
    grounded in a passage nobody was shown is not a checkable answer.
    """
    if not answer.text:
        return frozenset()
    cited: frozenset[str] = frozenset()
    for citation in answer.citations:
        cited |= part_numbers_in(citation.quote)
    return part_numbers_in(answer.text) - cited


def post_validate(answer: Answer) -> Answer:
    """The rule every arm's output passes through, whatever produced it.

    Two checks, both of which a live model could fail and the extractive arm structurally cannot:
    a part number with no support in the quotes, and a missing Article 50 disclosure. The disclosure
    is checked here rather than only at the API edge because the obligation follows the content, and
    an answer handed to a caller in-process has already left the edge behind.
    """
    if not answer.ai_disclosure.strip():
        raise ValueError(
            "the Article 50 disclosure is empty; every answer this system returns must be labelled "
            "as machine-generated"
        )
    missing = ungrounded_part_numbers(answer)
    if missing:
        raise UngroundedAnswerError(
            f"the answer names {sorted(missing)}, which no cited quote contains; "
            "ADR-001 kill condition C fixes this at zero"
        )
    return answer


class ExtractiveAnswerer:
    """The default arm: the answer **is** its citations, character for character.

    It selects spans from the chunks the gate approved and joins those same spans into the answer
    text. The text is therefore a concatenation of substrings of the quotes it cites, and
    `part_numbers_in(text)` is a subset of the union of `part_numbers_in(quote)` by construction —
    there is no generation step in which an identifier could be altered, completed or invented.

    That is kill condition C made **impossible** rather than measured. The evaluation still counts
    it, because a guarantee that is never checked is a belief about a guarantee, but the count is a
    check on this reasoning rather than on a model's behaviour.

    The cost is fluency: the answer reads like the manual, because it is the manual. For a
    technician looking for a torque figure that is a feature, and it is the trade this project
    chose deliberately over a phrasing model it could not verify.
    """

    @property
    def name(self) -> str:
        return "extractive"

    def answer(
        self,
        query: Query,
        decision: GateDecision,
        revisions: Mapping[str, str],
    ) -> Answer:
        citations = _quotable(build_citations(decision.approved_chunks, revisions, query))
        text: str | None = None
        if decision.outcome is GateOutcome.ANSWER:
            if not citations:
                raise EmptyEvidenceError(
                    f"the gate approved {len(decision.approved_chunks)} chunk(s) for "
                    f"{query.text!r} but none contained quotable text"
                )
            text = ANSWER_SEPARATOR.join(citation.quote for citation in citations)
        # A review keeps its citations and gains no text: the passages that disagree are exactly
        # what the person deciding needs to see, and a summary of them would be the system taking a
        # position it just declined to take.
        return post_validate(
            Answer(
                query=query,
                decision=decision,
                text=text,
                citations=citations,
                answerer=self.name,
            )
        )


class EvidenceItem(BaseModel):
    """One approved chunk, as it would appear in a prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    document_id: str
    revision: str
    text: str


class PromptPayload(BaseModel):
    """Exactly what would be sent to a model, as an object a test can inspect.

    It exists so the claim "the model only ever sees what the gate approved" is checkable without a
    network call or a captured request. `test_answerer.py` asserts the evidence in here is the
    approved set and nothing else.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instruction: str
    question: str
    as_of: str
    evidence: tuple[EvidenceItem, ...]


#: The whole of the instruction that would go to a live model. Written out here rather than composed
#: at call time so that what a reader sees in the repository is what would be sent.
ABSTRACTIVE_INSTRUCTION: Final[str] = (
    "Answer the technician's question using only the passages supplied below. Quote part numbers "
    "and measurements exactly as they appear. If the passages do not contain the answer, say so. "
    "Do not use any knowledge that is not in the passages."
)


class AbstractiveAnswerer:
    """The port for a live model. **Nothing is behind it in this build.**

    No API key exists here, no request has ever been sent, and therefore **no cost, latency or
    answer-quality figure is published for this arm anywhere in this repository**. Calling `answer`
    raises. It does not return a stub, a canned string or an echo of the evidence, because any of
    those would flow into the evaluation and become a number attributed to a model that was never
    called — which ADR-001 forbids in the same breath as it forbids inventing a metric.

    What it does provide is `prompt_payload`: the exact request that would be made. That keeps the
    seam honest — the port is a real shape with a real payload rather than a promise — and it lets a
    test assert the containment property that matters, which is that a live arm would receive the
    gate's approved chunks and nothing else.
    """

    @property
    def name(self) -> str:
        return "abstractive"

    def prompt_payload(
        self,
        query: Query,
        decision: GateDecision,
        revisions: Mapping[str, str],
    ) -> PromptPayload:
        """Everything that would be sent, built only from `decision.approved_chunks`."""
        if not decision.approved_chunks:
            raise EmptyEvidenceError(
                "the gate approved no evidence, so there is no prompt to build; a payload with no "
                "passages is a request to answer from the model's own memory"
            )
        return PromptPayload(
            instruction=ABSTRACTIVE_INSTRUCTION,
            question=query.text,
            as_of=query.as_of.isoformat(),
            evidence=tuple(
                EvidenceItem(
                    chunk_id=retrieved.chunk.chunk_id,
                    document_id=retrieved.chunk.document_id,
                    revision=_revision_for(retrieved.chunk.document_id, revisions),
                    text=retrieved.chunk.text,
                )
                for retrieved in decision.approved_chunks
            ),
        )

    def answer(
        self,
        query: Query,
        decision: GateDecision,
        revisions: Mapping[str, str],
    ) -> Answer:
        raise AbstractiveArmUnavailableError(
            "no model is configured in this build, so the abstractive arm cannot answer "
            f"{query.text!r} from the {len(decision.approved_chunks)} passage(s) the gate "
            f"approved ({len(revisions)} revision labels supplied). Call prompt_payload() to "
            "inspect the exact request it would send; no cost, latency or quality figure is "
            "published for this arm."
        )


class PostValidatedAnswerer:
    """Any answerer, with the grounding rule enforced on the way out.

    Written as a wrapper rather than as a base class so that it applies to an implementation this
    repository does not own — the point of a port is that somebody else's arm can sit behind it, and
    a rule inherited from a base class is a rule that arm can decline to inherit.
    """

    def __init__(self, inner: Answerer) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.name

    def answer(
        self,
        query: Query,
        decision: GateDecision,
        revisions: Mapping[str, str],
    ) -> Answer:
        return post_validate(self._inner.answer(query, decision, revisions))


def _quotable(citations: Sequence[Citation]) -> tuple[Citation, ...]:
    """Citations with something in them. A blank quote cites nothing and displays as nothing."""
    return tuple(citation for citation in citations if citation.quote.strip())


def _revision_for(document_id: str, revisions: Mapping[str, str]) -> str:
    revision = revisions.get(document_id)
    if revision is None:
        raise KeyError(
            f"no revision label for document {document_id}; a prompt may not describe a passage "
            "by a revision it was not given"
        )
    return revision
