"""The deterministic gate: what the system is permitted to say, decided before anything generates.

ADR-001 puts the whole weight of claim 2 here. The gate reads seven signals, **none of which is a
model output**, and returns one of three outcomes. It runs *before* the answerer is called and the
answerer receives only the chunks this module approved, so no model can move a question from an
abstention to an answer — not because it is instructed not to, but because it is never shown the
question in a form that would let it.

Three properties are load-bearing and each is testable on its own:

1. **Exactly one construction site for the answering outcome.** Every other path out of `decide`
   withholds. A second site is a second place for a guard to be forgotten, so `test_gate.py`
   asserts the count over this module's AST rather than trusting review.
2. **Conflict is not weakness.** Two bulletins that disagree, evidence drawn from two variants, or a
   passage that was not in force on the as-of date produce `REVIEW` — a real outcome that puts the
   disagreement in front of a person — rather than a silent shrug.
3. **The thresholds are configuration.** They are named module constants with the reason for each
   value beside it, and `GateThresholds` carries them into a call so the evaluation can sweep the
   primary one and draw the coverage-versus-wrong-answer curve ADR-001 requires.

No database, no retriever, no embedding model is imported here. The gate takes a sequence of
`RetrievedChunk` and is therefore testable, exhaustively, with hand-built fixtures.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Final, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from parts_answer_gate.domain import (
    GateDecision,
    GateOutcome,
    GateSignals,
    Language,
    Query,
    RetrievedChunk,
)

__all__ = [
    "DEFAULT_MAX_APPROVED_CHUNKS",
    "DEFAULT_MIN_SUPPORTING_CHUNKS",
    "DEFAULT_MIN_SUPPORTING_CHUNKS_WITH_EXACT_HIT",
    "DEFAULT_MIN_TERM_COVERAGE",
    "DEFAULT_MIN_TOP_FUSED_SCORE",
    "DEFAULT_SUPPORT_SCORE_FRACTION_OF_TOP",
    "DEFAULT_THRESHOLDS",
    "MEASUREMENT_UNITS",
    "PRIMARY_THRESHOLD_NAME",
    "PRIMARY_THRESHOLD_SWEEP",
    "GateThresholds",
    "compute_signals",
    "content_terms",
    "decide",
    "supporting_chunks",
    "term_is_covered",
    "thresholds_at",
    "tokens_in",
    "withholds_answer",
]

# --------------------------------------------------------------------------------- thresholds

#: **The primary threshold.** Term coverage is the share of the question's content terms that occur
#: in the evidence the gate is about to approve, and it is the knob the evaluation sweeps.
#:
#: It is primary rather than the fused score because it is the only signal on a scale this module
#: owns: coverage is a share, always in [0, 1], whatever fusion produced the ranking. A sweep over
#: the fused score would be a sweep over *another module's units* — RRF puts every top-10 hit inside
#: a band of roughly 0.014 to 0.033, so a curve drawn over it would be flat for most of its length
#: and would have to be redrawn the day the fusion changes.
#:
#: 0.60 as the shipped default: two of every three content terms in a technician's question must be
#: present in the passages being cited. Below that the usual failure is a chunk about the right
#: product and the wrong attribute, which scores well and supports nothing.
DEFAULT_MIN_TERM_COVERAGE: Final[float] = 0.60

#: A top fused score at or below this means nothing was retrieved that scored at all.
#:
#: Deliberately 0.0 and deliberately not tuned. The absolute magnitude of a fused score belongs to
#: the fusion, and a tuned constant here would be a number in units this module does not define; it
#: would silently start abstaining on everything the day the fusion is rescaled. The discrimination
#: is done by term coverage and by corroboration, which are scale-free. Raise this only alongside a
#: fusion whose scale is pinned.
DEFAULT_MIN_TOP_FUSED_SCORE: Final[float] = 0.0

#: ADR-001: "one lucky chunk is not corroboration". Two independent passages must carry the answer.
DEFAULT_MIN_SUPPORTING_CHUNKS: Final[int] = 2

#: Unless the deterministic identifier lookup hit. An exact part-number match is a different kind of
#: evidence from a semantic one — it is a catalogue lookup that either matched or did not — and
#: demanding a second semantic passage to corroborate an exact match would abstain on the questions
#: the system is most certain about.
DEFAULT_MIN_SUPPORTING_CHUNKS_WITH_EXACT_HIT: Final[int] = 1

#: A chunk counts as supporting when its fused score is at least this fraction of the top one.
#:
#: Relative, not absolute, for the same reason the score floor is untuned: a fraction survives a
#: rescaling of the fusion, an absolute cut-off does not. 0.50 keeps the corroborating passages that
#: are genuinely close to the best one and drops the long tail that a top-k always carries.
DEFAULT_SUPPORT_SCORE_FRACTION_OF_TOP: Final[float] = 0.50

#: The most chunks the gate will ever hand to an answerer.
#:
#: This is a containment bound, not a performance one. Kill condition C compares the part numbers in
#: an answer against the part numbers in its *cited quotes*, so every additional approved chunk
#: widens the set of identifiers an answer is permitted to contain. Five passages are more than a
#: technician reads and already more than a wrong answer needs.
DEFAULT_MAX_APPROVED_CHUNKS: Final[int] = 5

#: The name of the swept threshold, so the evaluation artifact can say what its x-axis is without
#: the two files agreeing by coincidence.
PRIMARY_THRESHOLD_NAME: Final[str] = "min_term_coverage"

#: The sweep the coverage-versus-wrong-answer curve is drawn over. Ten points including both ends:
#: at 0.0 the coverage signal is disabled entirely and at 0.9 almost nothing clears it, so the curve
#: spans the whole trade-off rather than a flattering neighbourhood of the shipped value.
PRIMARY_THRESHOLD_SWEEP: Final[tuple[float, ...]] = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)


class GateThresholds(BaseModel):
    """The gate's configuration, carried into a call rather than read from module state.

    Frozen and explicit so that a swept evaluation and the shipped service cannot drift apart: the
    curve is drawn by calling the same `decide` with a different instance of this, not by a
    different code path that happens to resemble it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_term_coverage: float = Field(default=DEFAULT_MIN_TERM_COVERAGE, ge=0.0, le=1.0)
    min_top_fused_score: float = DEFAULT_MIN_TOP_FUSED_SCORE
    min_supporting_chunks: int = Field(default=DEFAULT_MIN_SUPPORTING_CHUNKS, ge=1)
    min_supporting_chunks_with_exact_hit: int = Field(
        default=DEFAULT_MIN_SUPPORTING_CHUNKS_WITH_EXACT_HIT, ge=1
    )
    support_score_fraction_of_top: float = Field(
        default=DEFAULT_SUPPORT_SCORE_FRACTION_OF_TOP, ge=0.0, le=1.0
    )
    max_approved_chunks: int = Field(default=DEFAULT_MAX_APPROVED_CHUNKS, ge=1)

    def with_primary(self, value: float) -> GateThresholds:
        """The same configuration with the swept threshold moved. Used to draw the curve."""
        return self.model_copy(update={PRIMARY_THRESHOLD_NAME: value})


DEFAULT_THRESHOLDS: Final[GateThresholds] = GateThresholds()


def thresholds_at(primary: float) -> GateThresholds:
    """The shipped configuration with only the primary threshold moved to `primary`."""
    return DEFAULT_THRESHOLDS.with_primary(primary)


# ------------------------------------------------------------------------------- term coverage

#: A token is a run of word characters, keeping internal hyphens, so `XP-400` and `AB-1234-C`
#: survive tokenisation as single terms. Underscores are excluded: nothing in this corpus's prose
#: uses them and they otherwise glue identifiers to their neighbours.
_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*")

#: Below this length a term must match exactly. Short tokens are where a prefix rule does damage:
#: `bar` would cover `barrier`.
_MIN_PREFIX_MATCH: Final[int] = 4


def _words(listing: str) -> frozenset[str]:
    """A whitespace-separated word list. Written as prose rather than as a list of quoted strings
    because a stopword list is read and edited as a sentence of words, not as fifty literals."""
    return frozenset(listing.split())


# The `noqa: RUF001` markers below are the price of a multilingual project: ruff reads several
# Cyrillic and Turkish letters as ASCII look-alikes smuggled into source. That is the right
# default and the wrong call here, because these are two of the three languages the blueprint
# requires the system to answer in.
#
# Function words, per language. Small and hand-written rather than pulled from an NLP package: the
# gate needs to know which tokens carry no evidence, not to do linguistics, and a stopword list is
# not worth a dependency that has to be vendored into a container and patched for the next decade.
_STOPWORDS: Final[dict[Language, frozenset[str]]] = {
    Language.EN: _words(
        "a an and any are as at be been by can do does each for from had has have how i if in "
        "into is it its me must my no not of on or per should that the their then there these "
        "this to use used was were what when which while with you your"
    ),
    Language.TR: _words(
        "ama ancak bir bu çok da daha de değil en gibi hangi için ile ise "
        "kadar kaç ki mi mı mu mü nasıl ne nedir o olan olarak önce sonra "  # noqa: RUF001
        "şu üzerinde üzerine var ve veya ya yok"
    ),
    Language.RU: _words(
        "а бы быть в для до его если есть же за и из или к как какой ли "  # noqa: RUF001
        "может на не нет ну о об от по при с сколько то тот у что чтобы это"  # noqa: RUF001
    ),
}

#: Turkish `İ` casefolds to `i` plus U+0307 COMBINING DOT ABOVE, so `İLE` and `ile` compare unequal
#: unless the mark is dropped. Applied to both sides of every comparison, so the question and the
#: corpus are always folded the same way.
_COMBINING_DOT_ABOVE: Final[str] = "̇"


def _fold(text: str) -> str:
    return text.casefold().replace(_COMBINING_DOT_ABOVE, "")


def tokens_in(text: str) -> frozenset[str]:
    """Every distinct token in a piece of text, folded for comparison.

    Exported so `citations.py` chooses a quote with the same tokeniser the gate scored coverage
    with. Two tokenisers would eventually disagree about what a part number is.
    """
    return frozenset(_fold(match.group()) for match in _TOKEN.finditer(text))


def content_terms(text: str, language: Language = Language.EN) -> frozenset[str]:
    """The terms in a question that carry evidence, folded for comparison.

    Exported because `citations.py` chooses which span of a chunk to quote by the *same* notion of a
    content term the gate scored coverage with. Two different notions would let the gate approve on
    one set of words and the citation display another.
    """
    stopwords = _STOPWORDS.get(language, frozenset())
    return frozenset(
        token for token in tokens_in(text) if token not in stopwords and len(token) > 1
    )


def term_is_covered(term: str, tokens: frozenset[str]) -> bool:
    """Whether one question term appears in the evidence.

    Exact match, or a prefix match in either direction once both sides are at least four characters
    long. The prefix rule is a deliberate approximation of stemming for Turkish and Russian, where
    a Turkish or Russian noun and its inflected form are the same word, and an exact-match rule
    would score coverage near zero on two of the three languages the blueprint requires --
    abstaining on everything non-English and calling it caution. A real stemmer per language was
    rejected: three more dependencies, three more models in the image, and a morphology this
    synthetic corpus does not actually exercise. It errs towards covering, which is why it is
    only one of five signals and never sufficient on its own.
    """
    if term in tokens:
        return True
    if len(term) < _MIN_PREFIX_MATCH:
        return False
    return any(
        token.startswith(term) or (len(token) >= _MIN_PREFIX_MATCH and term.startswith(token))
        for token in tokens
    )


def _term_coverage(query: Query, chunks: Sequence[RetrievedChunk]) -> float:
    terms = content_terms(query.text, query.language)
    if not terms or not chunks:
        # A question made entirely of function words cannot be shown to be covered by anything, and
        # reporting 1.0 for an empty denominator would answer it on no evidence at all.
        return 0.0
    tokens = frozenset[str]().union(*(tokens_in(chunk.chunk.text) for chunk in chunks))
    return sum(1 for term in terms if term_is_covered(term, tokens)) / len(terms)


# --------------------------------------------------------------------------- conflicting values

#: Every unit spelling the corpus writes, mapped to the quantity it names.
#:
#: Localised spellings are listed because a disagreement between two Russian passages is the same
#: disagreement as between two English ones, and a units table that only knew `Nm` would find none
#: of them — which would leave the conflict signal permanently false off English and make the
#: multilingual evaluation look better than the system is. Mapping spellings onto a shared quantity
#: also means an inflected Russian month form is not read as a different unit from another.
#:
#: These are the spellings `corpus/catalogue.py` renders. The gate deliberately does **not** import
#: that module: a gate that depends on the corpus generator cannot be tested without it, and the
#: whole argument for this file is that it is testable from hand-built fixtures alone. Extending
#: this table is a configuration change, and a considered one — a unit the gate cannot parse is a
#: conflict the gate cannot see.
MEASUREMENT_UNITS: Final[Mapping[str, str]] = {
    "Nm": "torque",
    "N·m": "torque",
    "Н·м": "torque",  # noqa: RUF001
    "bar": "bar",
    "бар": "bar",  # noqa: RUF001
    "L": "litres",
    "л": "litres",
    "months": "months",
    "ay": "months",
    "месяц": "months",
    "месяца": "months",
    "месяцев": "months",
    "A": "amps",
    "°C": "celsius",
    "mm": "millimetres",
}

#: Longest spelling first, so `месяцев` is not read as `месяц`, and a lookahead that refuses any
#: following word character, so a month count matches and a longer Turkish word beginning with
#: the same two letters does not.
_MEASUREMENT = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*("
    + "|".join(re.escape(unit) for unit in sorted(MEASUREMENT_UNITS, key=len, reverse=True))
    + r")(?!\w)"
)

#: Quantities written with the unit *before* the number. `ISO VG 46` is a viscosity grade in all
#: three languages and the suffix pattern above cannot see it at all.
_PREFIXED_QUANTITIES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"ISO\s+VG\s*(\d+)"), "iso_vg"),
)


def _normalised(value: str) -> str:
    """`48,0`, `48.0` and `48` are one value. Compared as text, because two decimal strings that
    should be equal are a bad reason to introduce float tolerance into a boolean signal."""
    text = value.replace(",", ".")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _measurements(text: str) -> dict[str, frozenset[str]]:
    found: dict[str, set[str]] = {}
    for match in _MEASUREMENT.finditer(text):
        found.setdefault(MEASUREMENT_UNITS[match.group(2)], set()).add(_normalised(match.group(1)))
    for pattern, quantity in _PREFIXED_QUANTITIES:
        for match in pattern.finditer(text):
            found.setdefault(quantity, set()).add(_normalised(match.group(1)))
    return {quantity: frozenset(values) for quantity, values in found.items()}


def _conflicting_evidence(chunks: Sequence[RetrievedChunk]) -> bool:
    """Two supporting passages that state different values for the same quantity.

    Detected on units rather than on prose: if one bulletin says 48 Nm and another says 52 Nm, the
    technician has a decision to make and the system has no business picking one. Chunks whose
    value sets overlap are not in conflict — a passage listing several torques and a passage
    listing one of them agree about that one.

    An LLM-judged contradiction check was rejected outright. ADR-001 requires every gate signal
    to be computable without a model, or the gate becomes a thing a model can talk past.
    """
    per_chunk = [_measurements(chunk.chunk.text) for chunk in chunks]
    for index, first in enumerate(per_chunk):
        for second in per_chunk[index + 1 :]:
            for unit in first.keys() & second.keys():
                if first[unit].isdisjoint(second[unit]):
                    return True
    return False


# ---------------------------------------------------------------------------------- the signals


class _Evidence(NamedTuple):
    signals: GateSignals
    supporting: tuple[RetrievedChunk, ...]


def supporting_chunks(
    retrieved: Sequence[RetrievedChunk],
    thresholds: GateThresholds = DEFAULT_THRESHOLDS,
) -> tuple[RetrievedChunk, ...]:
    """The evidence the gate would approve, ordered and capped.

    Ordered by fused score, then rank, then chunk id. The final tie-break is not cosmetic: kill
    condition J requires two runs to produce byte-identical gate decisions, and a set iterated in
    whatever order the retriever happened to emit is exactly how that guarantee is lost.
    """
    if not retrieved:
        return ()
    top = max(chunk.fused_score for chunk in retrieved)
    floor = top * thresholds.support_score_fraction_of_top
    supporting = [
        chunk
        for chunk in retrieved
        # An exact identifier hit is kept whatever it scored. The deterministic lookup runs before
        # any embedding is consulted and a catalogue match that the fusion ranked low is still a
        # catalogue match.
        if chunk.exact_identifier_hit or chunk.fused_score >= floor
    ]
    supporting.sort(key=lambda chunk: (-chunk.fused_score, chunk.rank, chunk.chunk.chunk_id))
    return tuple(supporting[: thresholds.max_approved_chunks])


def _weigh(
    query: Query,
    retrieved: Sequence[RetrievedChunk],
    thresholds: GateThresholds,
) -> _Evidence:
    approved = supporting_chunks(retrieved, thresholds)
    variants = {chunk.chunk.effectivity.variant_id for chunk in approved}
    signals = GateSignals(
        top_fused_score=max((chunk.fused_score for chunk in retrieved), default=0.0),
        # Counted over the evidence the gate will actually hand over, which is capped. A count that
        # included passages the answerer never sees would describe a different decision from the one
        # being taken.
        supporting_chunks=len(approved),
        term_coverage=_term_coverage(query, approved),
        variant_agreement=len(variants) <= 1,
        # A chunk that was not in force on the as-of date should be impossible here: the
        # effectivity filter runs in SQL before ranking. It is measured anyway, because a
        # guarantee nobody checks downstream is a belief. This catches the withdrawn revision
        # and the not-yet-effective one alike — neither is the text approved on the date asked
        # about.
        superseded_present=any(not chunk.chunk.in_force_on(query.as_of) for chunk in approved),
        conflicting_evidence=_conflicting_evidence(approved),
        exact_identifier_hit=any(chunk.exact_identifier_hit for chunk in approved),
    )
    return _Evidence(signals, approved)


def compute_signals(
    query: Query,
    retrieved: Sequence[RetrievedChunk],
    thresholds: GateThresholds = DEFAULT_THRESHOLDS,
) -> GateSignals:
    """The seven signals, without the decision. Exposed so the evaluation can record them."""
    return _weigh(query, retrieved, thresholds).signals


# ------------------------------------------------------------------------------------ the rules


class _GateContext(NamedTuple):
    """Everything a rule may look at. No field of it is a model output, and none is mutable."""

    s: GateSignals
    t: GateThresholds


class _Rule(NamedTuple):
    fires: Callable[[_GateContext], bool]
    outcome: GateOutcome
    reason: str


def _below_support_floor(context: _GateContext) -> bool:
    required = (
        context.t.min_supporting_chunks_with_exact_hit
        if context.s.exact_identifier_hit
        else context.t.min_supporting_chunks
    )
    return context.s.supporting_chunks < required


#: Evaluated in order, first match wins, and the order is the argument.
#:
#: `superseded_present` is checked ahead of the strength rules on purpose. If a passage that was not
#: in force on the as-of date reached the gate at all, the SQL effectivity filter failed, and
#: abstaining would bury the one failure this project exists to prevent under a message about weak
#: evidence. Every other conflict is checked *after* the strength rules, the other way round: a
#: contradiction between two passages too thin to support any answer is not worth a person's
#: attention, and a review queue that fills with those is a review queue nobody reads.
_RULES: Final[tuple[_Rule, ...]] = (
    _Rule(
        lambda context: context.s.supporting_chunks == 0,
        GateOutcome.ABSTAIN,
        "no passage was retrieved for this question at the as-of date asked about",
    ),
    _Rule(
        lambda context: context.s.superseded_present,
        GateOutcome.REVIEW,
        "a supporting passage was not in force on the as-of date, so the effectivity filter did "
        "not constrain this result; a person must confirm which revision applies",
    ),
    _Rule(
        lambda context: (
            context.s.top_fused_score <= context.t.min_top_fused_score
            and not context.s.exact_identifier_hit
        ),
        GateOutcome.ABSTAIN,
        "nothing retrieved scored above {t.min_top_fused_score} and no exact identifier matched",
    ),
    _Rule(
        _below_support_floor,
        GateOutcome.ABSTAIN,
        "{s.supporting_chunks} passage(s) above the support floor; one passage is not "
        "corroboration",
    ),
    _Rule(
        lambda context: context.s.term_coverage < context.t.min_term_coverage,
        GateOutcome.ABSTAIN,
        "the retrieved passages cover {s.term_coverage:.2f} of the question's terms, below the "
        "{t.min_term_coverage:.2f} floor; the evidence is about something adjacent",
    ),
    _Rule(
        lambda context: not context.s.variant_agreement,
        GateOutcome.REVIEW,
        "the supporting passages apply to different variants; one of them is the wrong machine and "
        "the text does not say which",
    ),
    _Rule(
        lambda context: context.s.conflicting_evidence,
        GateOutcome.REVIEW,
        "two supporting passages state different values for the same quantity; both are shown "
        "rather than one being chosen",
    ),
)


def decide(
    query: Query,
    retrieved: Sequence[RetrievedChunk],
    thresholds: GateThresholds = DEFAULT_THRESHOLDS,
) -> GateDecision:
    """Whether this question may be answered, decided before a word is generated.

    The loop withholds; only the fall-through answers. That shape is the point: to add a new reason
    to answer you would have to delete a rule, which is visible in a diff, rather than add a branch,
    which is not.
    """
    evidence = _weigh(query, retrieved, thresholds)
    context = _GateContext(evidence.signals, thresholds)
    for rule in _RULES:
        if rule.fires(context):
            return GateDecision(
                outcome=rule.outcome,
                reason=rule.reason.format(s=evidence.signals, t=thresholds),
                signals=evidence.signals,
                # An abstention approves nothing, so an answerer cannot be handed evidence for a
                # question the gate refused. A review carries the evidence, because the whole value
                # of a review is that the person sees the passages that disagree.
                approved_chunks=() if rule.outcome is GateOutcome.ABSTAIN else evidence.supporting,
            )
    return GateDecision(
        outcome=GateOutcome.ANSWER,
        reason=(
            f"{evidence.signals.supporting_chunks} passages in force on {query.as_of.isoformat()} "
            f"agree on one variant and cover "
            f"{evidence.signals.term_coverage:.2f} of the question's terms"
        ),
        signals=evidence.signals,
        approved_chunks=evidence.supporting,
    )


def withholds_answer(outcome: GateOutcome) -> bool:
    """Whether this outcome means nothing is presented to the technician as an answer.

    Both an abstention and a review withhold, and the evaluation must count them together when it
    measures coverage. Written once, here, so that the service, the curve and the abstention rate
    are not three slightly different opinions about what `REVIEW` means.
    """
    return outcome in {GateOutcome.ABSTAIN, GateOutcome.REVIEW}
