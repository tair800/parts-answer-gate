"""The benchmark question set, including the ninety-plus questions that have no answer.

Ground truth here is **construction metadata**, not a judgement. The generator knows which chunk
supports a question because it wrote the chunk and then wrote the question from it, and it knows a
question is unanswerable because it arranged for the answer to be absent. Nothing is labelled by a
model, and nothing is labelled by reading the corpus back.

The unanswerable set is the part that decides whether the abstention claim means anything, so it is
built as seven distinct failures rather than as one:

| kind | what the retriever sees |
|---|---|
| `absent_specification` | a real product, an ordinary question, a figure nobody published |
| `attribute_absent_for_existing_product` | the sibling variants document it; this one does not |
| `near_match_different_identifier` | a part number one digit from a real one |
| `superseded_without_replacement_asked` | a part that was in force and is not any more |
| `contradictory_sources` | two documents, both in force, disagreeing |
| `different_product_family` | a product this corpus has never heard of |
| `malformed_part_number` | an identifier the catalogue's own pattern rejects |

Only the first is the easy case — nothing scores well. The other six all retrieve something with a
high score, which is the point: a system that abstains only when retrieval returns nothing has not
implemented abstention, it has implemented an empty-result check.

Every question exists in all three languages with the **same** ground truth, each pointing at the
supporting chunk in its own language's document. A Turkish question whose gold chunk is English
would measure translation, not retrieval.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

from parts_answer_gate.corpus import text as phrases
from parts_answer_gate.corpus.builder import MANUAL, BuiltCorpus, Revision
from parts_answer_gate.corpus.catalogue import FAMILIES, TOPICS, Family, Topic, Unit
from parts_answer_gate.corpus.identifiers import (
    is_well_formed,
    malformed_variants,
    near_miss_part_number,
)
from parts_answer_gate.corpus.rng import stream
from parts_answer_gate.corpus.splits import split_of
from parts_answer_gate.domain import Language

__all__ = [
    "UNANSWERABLE_KINDS",
    "QuestionSeed",
    "build_questions",
    "question_records",
]

#: The seven ADR-001 names, in the order the artifact reports them. A kind with no questions is a
#: contract failure, not an empty bucket — `generate.py` asserts all seven are populated.
UNANSWERABLE_KINDS: Final = (
    "absent_specification",
    "attribute_absent_for_existing_product",
    "near_match_different_identifier",
    "superseded_without_replacement_asked",
    "contradictory_sources",
    "different_product_family",
    "malformed_part_number",
)

_TOPICS_PER_VARIANT = 8

#: Per-kind counts, sized against ADR-001's floor of 90 unanswerable questions counted as
#: **distinct** questions.
#:
#: The first corpus sized these so that the *trilingual* set cleared the floor — 6 per kind gives
#: 44 distinct negatives rendered as 132 rows, and 132 was the number published. A Turkish
#: translation of a question is not a second question, and ADR-002 records the correction. Fifteen
#: per kind clears the floor on distinct content with the margin the floor was meant to have.
_NEGATIVES_PER_KIND: Final = {
    "absent_specification": 15,
    "attribute_absent_for_existing_product": 15,
    "near_match_different_identifier": 15,
    "superseded_without_replacement_asked": 15,
    "different_product_family": 15,
    "malformed_part_number": 15,
}


@dataclass(frozen=True)
class QuestionSeed:
    """One question, already resolved in all three languages."""

    seed_id: str
    family_id: str
    #: False for the `different_product_family` negatives, whose family genuinely does not exist.
    #: The split is still computed from the identifier, because the rule is a pure function of a
    #: string and a question has to land on one side of the split or it cannot be scored.
    family_in_corpus: bool
    variant_id: str | None
    topic_key: str | None
    as_of: date
    serial: int | None
    answerable: bool
    unanswerable_kind: str | None
    text: Mapping[Language, str]
    supporting: Mapping[Language, tuple[str, ...]]
    span: Mapping[Language, str | None]
    #: Why this question is answerable or not, in one line, carried into the artifact. A label
    #: nobody can trace back to a reason is a label nobody can dispute.
    rationale: str


# ------------------------------------------------------------------------------------- helpers


def _variant_pairs() -> tuple[tuple[Family, int, str], ...]:
    return tuple(
        (family, index, variant.variant_id)
        for family in FAMILIES
        for index, variant in enumerate(family.variants)
    )


def _date_in(revision: Revision, chooser: random.Random) -> date:
    """A query date inside a revision's in-force window.

    The open-ended current revision is sampled over a bounded span rather than an unbounded one,
    so a question's as-of date never drifts past every other document in the corpus.
    """
    if revision.valid_to is None:
        return revision.valid_from + timedelta(days=chooser.randrange(0, 400))
    span = (revision.valid_to - revision.valid_from).days
    return revision.valid_from + timedelta(days=chooser.randrange(0, span))


def _serial_for(first: int | None, last: int | None) -> int | None:
    """A machine inside a bounded effectivity window, or nothing when the passage is fleet-wide.

    `SerialRange.covers(None)` is deliberately false for a bounded range, so a question about a
    serial-limited passage that supplied no serial would be unanswerable by accident rather than by
    design — and would be scored as a retrieval failure it is not.
    """
    if first is None or last is None:
        return None
    return first + (last - first) // 2


def _pick(pool_size: int, used: set[int], *purpose: str) -> int:
    """An unused index into a pool, chosen by hash and advanced on collision.

    Advancing rather than re-rolling keeps the choice a function of the purpose string: two
    negatives of the same kind never land on the same target, and adding a kind does not move the
    targets of the kinds beside it.
    """
    index = stream(*purpose).randrange(pool_size)
    while index in used:
        index = (index + 1) % pool_size
    used.add(index)
    return index


def _empty_support() -> Mapping[Language, tuple[str, ...]]:
    empty: tuple[str, ...] = ()
    return dict.fromkeys(Language, empty)


def _no_span() -> Mapping[Language, str | None]:
    return dict.fromkeys(Language, None)


# -------------------------------------------------------------------------- answerable questions


def _answerable_revisions(
    corpus: BuiltCorpus, family: Family, variant_id: str, topic: Topic
) -> tuple[Revision, ...]:
    """Revisions whose window a question may be asked in.

    A revision is excluded when the field bulletin contradicts this exact variant and topic and is
    in force during it. Asking an answerable question there would be asking a question the corpus
    answers two ways — which is a `contradictory_sources` negative, not a positive whose gold chunk
    happens to have a rival.
    """
    revisions = corpus.revisions[family.family_id]
    bulletin = corpus.bulletins.get(family.family_id)
    if bulletin is None:
        return revisions
    contradicted = any(
        entry.variant_id == variant_id and entry.topic_key == topic.key
        for entry in bulletin.entries
    )
    if not contradicted:
        return revisions
    return tuple(
        revision
        for revision in revisions
        if revision.valid_to is not None and revision.valid_to <= bulletin.valid_from
    )


def _answerable_seeds(corpus: BuiltCorpus) -> list[QuestionSeed]:
    seeds: list[QuestionSeed] = []
    for family, _, variant_id in _variant_pairs():
        omitted = corpus.omitted_topics[(family.family_id, variant_id)]
        available = [topic for topic in TOPICS if topic.key != omitted]
        picker = stream("question-topics", family.family_id, variant_id)
        for topic in picker.sample(available, _TOPICS_PER_VARIANT):
            candidates = _answerable_revisions(corpus, family, variant_id, topic)
            if not candidates:
                continue
            chooser = stream("question-revision", family.family_id, variant_id, topic.key)
            revision = candidates[chooser.randrange(len(candidates))]
            seeds.append(_answerable_seed(corpus, family, variant_id, topic, revision))
    return seeds


def _answerable_seed(
    corpus: BuiltCorpus,
    family: Family,
    variant_id: str,
    topic: Topic,
    revision: Revision,
) -> QuestionSeed:
    spec = corpus.specs[(family.family_id, variant_id, topic.key, revision.index)]
    dates = stream("question-as-of", family.family_id, variant_id, topic.key)
    support: dict[Language, tuple[str, ...]] = {}
    span: dict[Language, str | None] = {}
    render: dict[Language, str] = {}
    for language in Language:
        key = (family.family_id, MANUAL, revision.index, language.value, variant_id, topic.key)
        support[language] = (corpus.chunk_by_key[key].chunk_id,)
        span[language] = spec.value(topic.unit, language)
        render[language] = phrases.TOPIC_QUESTION[topic.key].of(language).format(variant=variant_id)
    return QuestionSeed(
        seed_id=f"q-pos-{variant_id}-{topic.key}".lower(),
        family_id=family.family_id,
        family_in_corpus=True,
        variant_id=variant_id,
        topic_key=topic.key,
        as_of=_date_in(revision, dates),
        serial=_serial_for(spec.serials.first, spec.serials.last),
        answerable=True,
        unanswerable_kind=None,
        text=render,
        supporting=support,
        span=span,
        rationale=(
            f"revision {revision.code} of the {family.family_id} manual states this for "
            f"{variant_id}, and it was in force at the query date"
        ),
    )


# ------------------------------------------------------------------------ unanswerable questions


def _absent_specification_seeds(corpus: BuiltCorpus) -> list[QuestionSeed]:
    pairs = _variant_pairs()
    used: set[int] = set()
    seeds: list[QuestionSeed] = []
    kind = "absent_specification"
    for index in range(_NEGATIVES_PER_KIND[kind]):
        family, _, variant_id = pairs[_pick(len(pairs), used, "negative", kind, str(index))]
        spec = phrases.ABSENT_SPECIFICATIONS[index % len(phrases.ABSENT_SPECIFICATIONS)]
        current = corpus.revisions[family.family_id][-1]
        render = {
            language: phrases.UNANSWERABLE_TEMPLATES[kind]
            .of(language)
            .format(spec=spec.of(language), variant=variant_id)
            for language in Language
        }
        seeds.append(
            QuestionSeed(
                seed_id=f"q-neg-absent-{index}",
                family_id=family.family_id,
                family_in_corpus=True,
                variant_id=variant_id,
                topic_key=None,
                as_of=current.valid_from + timedelta(days=30),
                serial=None,
                answerable=False,
                unanswerable_kind=kind,
                text=render,
                supporting=_empty_support(),
                span=_no_span(),
                rationale=f"no document in this corpus states {spec.en!r} for any variant",
            )
        )
    return seeds


def _attribute_absent_seeds(corpus: BuiltCorpus) -> list[QuestionSeed]:
    pairs = _variant_pairs()
    used: set[int] = set()
    seeds: list[QuestionSeed] = []
    kind = "attribute_absent_for_existing_product"
    for index in range(_NEGATIVES_PER_KIND[kind]):
        family, _, variant_id = pairs[_pick(len(pairs), used, "negative", kind, str(index))]
        missing = corpus.omitted_topics[(family.family_id, variant_id)]
        current = corpus.revisions[family.family_id][-1]
        # The ordinary phrasing, deliberately: the question is indistinguishable from an answerable
        # one and only the corpus decides. A negative recognisable from its wording tests wording.
        render = {
            language: phrases.TOPIC_QUESTION[missing].of(language).format(variant=variant_id)
            for language in Language
        }
        seeds.append(
            QuestionSeed(
                seed_id=f"q-neg-attribute-{index}",
                family_id=family.family_id,
                family_in_corpus=True,
                variant_id=variant_id,
                topic_key=missing,
                as_of=current.valid_from + timedelta(days=45),
                serial=None,
                answerable=False,
                unanswerable_kind=kind,
                text=render,
                supporting=_empty_support(),
                span=_no_span(),
                rationale=(
                    f"{variant_id} exists and its siblings document {missing}, but no revision of "
                    f"the {family.family_id} manual states it for {variant_id}"
                ),
            )
        )
    return seeds


def _near_match_seeds(corpus: BuiltCorpus) -> list[QuestionSeed]:
    pairs = _variant_pairs()
    used: set[int] = set()
    seeds: list[QuestionSeed] = []
    kind = "near_match_different_identifier"
    for index in range(_NEGATIVES_PER_KIND[kind]):
        family, _, variant_id = pairs[_pick(len(pairs), used, "negative", kind, str(index))]
        current = corpus.revisions[family.family_id][-1]
        real = _a_real_part(corpus, family.family_id, variant_id, current.index)
        near = near_miss_part_number(real, corpus.issued_parts) if real else None
        if near is None:
            continue
        render = {
            language: phrases.UNANSWERABLE_TEMPLATES[kind]
            .of(language)
            .format(pn=near, variant=variant_id)
            for language in Language
        }
        seeds.append(
            QuestionSeed(
                seed_id=f"q-neg-nearmatch-{index}",
                family_id=family.family_id,
                family_in_corpus=True,
                variant_id=variant_id,
                topic_key=None,
                as_of=current.valid_from + timedelta(days=60),
                serial=None,
                answerable=False,
                unanswerable_kind=kind,
                text=render,
                supporting=_empty_support(),
                span=_no_span(),
                rationale=(
                    f"{near} is one digit from {real}, is well-formed, and is issued to nothing in "
                    "this corpus"
                ),
            )
        )
    return seeds


def _a_real_part(corpus: BuiltCorpus, family_id: str, variant_id: str, revision: int) -> str | None:
    for topic in TOPICS:
        if topic.unit is not Unit.PART:
            continue
        spec = corpus.specs.get((family_id, variant_id, topic.key, revision))
        if spec is not None and spec.part is not None:
            return spec.part
    return None


@dataclass(frozen=True)
class _Withdrawal:
    family_id: str
    variant_id: str
    topic_key: str
    part: str
    withdrawn_on: date


def withdrawn_parts(corpus: BuiltCorpus) -> tuple[_Withdrawal, ...]:
    """Part numbers that were in force and are not any more, with the date they stopped.

    Read out of the specifications rather than out of the text, so a withdrawal is a fact about the
    construction of the corpus and not a string search that could miss a hyphenation.
    """
    found: list[_Withdrawal] = []
    for family, _, variant_id in _variant_pairs():
        revisions = corpus.revisions[family.family_id]
        for topic in TOPICS:
            if topic.unit is not Unit.PART:
                continue
            for step in range(1, len(revisions)):
                before = corpus.specs[(family.family_id, variant_id, topic.key, step - 1)]
                after = corpus.specs[(family.family_id, variant_id, topic.key, step)]
                if before.part != after.part and before.part is not None:
                    found.append(
                        _Withdrawal(
                            family.family_id,
                            variant_id,
                            topic.key,
                            before.part,
                            revisions[step].valid_from,
                        )
                    )
    return tuple(found)


def _superseded_seeds(corpus: BuiltCorpus) -> list[QuestionSeed]:
    pool = withdrawn_parts(corpus)
    used: set[int] = set()
    seeds: list[QuestionSeed] = []
    kind = "superseded_without_replacement_asked"
    for index in range(min(_NEGATIVES_PER_KIND[kind], len(pool))):
        item = pool[_pick(len(pool), used, "negative", kind, str(index))]
        heading = phrases.TOPIC_HEADING[item.topic_key]
        render = {
            language: phrases.UNANSWERABLE_TEMPLATES[kind]
            .of(language)
            .format(pn=item.part, variant=item.variant_id, heading=heading.of(language))
            for language in Language
        }
        seeds.append(
            QuestionSeed(
                seed_id=f"q-neg-superseded-{index}",
                family_id=item.family_id,
                family_in_corpus=True,
                variant_id=item.variant_id,
                topic_key=item.topic_key,
                # After the withdrawal, so every document naming this part is out of force at the
                # query date. The replacement is deliberately not asked about: the technician is
                # holding the old part and the honest answer is that nothing in force describes it.
                as_of=item.withdrawn_on + timedelta(days=30),
                serial=None,
                answerable=False,
                unanswerable_kind=kind,
                text=render,
                supporting=_empty_support(),
                span=_no_span(),
                rationale=(
                    f"{item.part} was withdrawn on {item.withdrawn_on.isoformat()} and appears "
                    "only in revisions that were out of force at the query date"
                ),
            )
        )
    return seeds


def _contradictory_seeds(corpus: BuiltCorpus) -> list[QuestionSeed]:
    seeds: list[QuestionSeed] = []
    kind = "contradictory_sources"
    for family in FAMILIES:
        bulletin = corpus.bulletins.get(family.family_id)
        if bulletin is None:
            continue
        current = corpus.revisions[family.family_id][-1]
        for entry in bulletin.entries:
            spec = corpus.specs[
                (family.family_id, entry.variant_id, entry.topic_key, current.index)
            ]
            render = {
                language: phrases.TOPIC_QUESTION[entry.topic_key]
                .of(language)
                .format(variant=entry.variant_id)
                for language in Language
            }
            seeds.append(
                QuestionSeed(
                    seed_id=f"q-neg-conflict-{entry.variant_id}-{entry.topic_key}".lower(),
                    family_id=family.family_id,
                    family_in_corpus=True,
                    variant_id=entry.variant_id,
                    topic_key=entry.topic_key,
                    as_of=bulletin.valid_from + timedelta(days=30),
                    # Supplied when the manual passage is serial-limited, or the contradiction
                    # would be invisible: the manual chunk would be filtered out and the bulletin
                    # would stand unopposed.
                    serial=_serial_for(spec.serials.first, spec.serials.last),
                    answerable=False,
                    unanswerable_kind=kind,
                    text=render,
                    supporting=_empty_support(),
                    span=_no_span(),
                    rationale=(
                        f"the manual and the field bulletin are both in force and state different "
                        f"values for {entry.topic_key} on {entry.variant_id}"
                    ),
                )
            )
    return seeds


def _decoy_family_id(decoy_en: str) -> str:
    slug = "".join(character if character.isalnum() else "-" for character in decoy_en.lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"fam-{slug.strip('-')}"


def _different_family_seeds(corpus: BuiltCorpus) -> list[QuestionSeed]:
    seeds: list[QuestionSeed] = []
    kind = "different_product_family"
    for index in range(_NEGATIVES_PER_KIND[kind]):
        decoy = phrases.DECOY_FAMILIES[index % len(phrases.DECOY_FAMILIES)]
        topic = TOPICS[stream("negative", kind, "topic", str(index)).randrange(len(TOPICS))]
        anchor = FAMILIES[index % len(FAMILIES)]
        render = {
            language: phrases.UNANSWERABLE_TEMPLATES[kind]
            .of(language)
            .format(decoy=decoy.of(language), heading=phrases.TOPIC_HEADING[topic.key].of(language))
            for language in Language
        }
        seeds.append(
            QuestionSeed(
                seed_id=f"q-neg-otherfamily-{index}",
                family_id=_decoy_family_id(decoy.en),
                family_in_corpus=False,
                variant_id=None,
                topic_key=topic.key,
                as_of=corpus.revisions[anchor.family_id][-1].valid_from + timedelta(days=30),
                serial=None,
                answerable=False,
                unanswerable_kind=kind,
                text=render,
                supporting=_empty_support(),
                span=_no_span(),
                rationale=f"{decoy.en} is not a product family this corpus contains",
            )
        )
    return seeds


def _malformed_seeds(corpus: BuiltCorpus) -> list[QuestionSeed]:
    pairs = _variant_pairs()
    used: set[int] = set()
    seeds: list[QuestionSeed] = []
    kind = "malformed_part_number"
    for index in range(_NEGATIVES_PER_KIND[kind]):
        family, _, variant_id = pairs[_pick(len(pairs), used, "negative", kind, str(index))]
        current = corpus.revisions[family.family_id][-1]
        real = _a_real_part(corpus, family.family_id, variant_id, current.index)
        if real is None:
            continue
        broken = malformed_variants(real)[index % 4]
        if is_well_formed(broken):
            raise ValueError(f"{broken} was meant to be malformed and the pattern accepts it")
        render = {
            language: phrases.UNANSWERABLE_TEMPLATES[kind]
            .of(language)
            .format(pn=broken, variant=variant_id)
            for language in Language
        }
        seeds.append(
            QuestionSeed(
                seed_id=f"q-neg-malformed-{index}",
                family_id=family.family_id,
                family_in_corpus=True,
                variant_id=variant_id,
                topic_key=None,
                as_of=current.valid_from + timedelta(days=15),
                serial=None,
                answerable=False,
                unanswerable_kind=kind,
                text=render,
                supporting=_empty_support(),
                span=_no_span(),
                rationale=f"{broken} is rejected by the corpus' own part-number pattern",
            )
        )
    return seeds


# ----------------------------------------------------------------------------------- assembly


def build_questions(corpus: BuiltCorpus) -> tuple[QuestionSeed, ...]:
    """Every seed, positives first, in a fixed order that does not depend on a dict iteration."""
    seeds: list[QuestionSeed] = list(_answerable_seeds(corpus))
    seeds.extend(_absent_specification_seeds(corpus))
    seeds.extend(_attribute_absent_seeds(corpus))
    seeds.extend(_near_match_seeds(corpus))
    seeds.extend(_superseded_seeds(corpus))
    seeds.extend(_contradictory_seeds(corpus))
    seeds.extend(_different_family_seeds(corpus))
    seeds.extend(_malformed_seeds(corpus))
    return tuple(seeds)


def question_records(seeds: Sequence[QuestionSeed]) -> list[dict[str, object]]:
    """One JSON record per seed per language, in the order the generator produced them."""
    records: list[dict[str, object]] = []
    for seed in seeds:
        for language in Language:
            records.append(
                {
                    "question_id": f"{seed.seed_id}-{language.value}",
                    "text": seed.text[language],
                    "language": language.value,
                    "family_id": seed.family_id,
                    "family_in_corpus": seed.family_in_corpus,
                    "variant_id": seed.variant_id,
                    "serial": seed.serial,
                    "as_of": seed.as_of.isoformat(),
                    "answerable": seed.answerable,
                    "supporting_chunk_ids": list(seed.supporting[language]),
                    "expected_answer_span": seed.span[language],
                    "unanswerable_kind": seed.unanswerable_kind,
                    "topic": seed.topic_key,
                    "split": split_of(seed.family_id),
                    # The parallel-set key: an English question and its Turkish twin share it, so
                    # the multilingual comparison is a paired one rather than three populations.
                    "parallel_group": seed.seed_id,
                    "rationale": seed.rationale,
                }
            )
    return records
