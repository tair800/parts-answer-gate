"""Documents, revisions and chunks — and the character offsets that make a citation checkable.

Three things in here are load-bearing.

**Revisions are separate documents.** Revision B of a manual is a new `Document` with its own id,
its own chunks and its own validity interval; revision A keeps its text, gains a `valid_to` and a
`superseded_by`, and is never deleted. That is the only shape in which *"what was the approved
procedure in March"* is a question the data can answer at all.

**Offsets are true offsets.** The document's full text is assembled by a cursor that also records
where each chunk started and stopped, so `document_text[start:end] == chunk.text` holds by
construction rather than by a later re-scan. Kill condition D locates a cited span in the source by
exactly those numbers; an offset computed against a normalised or re-joined copy of the text is a
coordinate into a string nobody kept. `generate.py` re-checks every one of them anyway, because a
guarantee nobody tests is a belief.

**A value changes between revisions only sometimes.** A corpus where every revision rewrites every
figure makes supersession trivially detectable — any older chunk is stale — and makes incremental
re-embedding meaningless, since everything changed. Here a specification changes at a revision with
a fixed probability, drawn from a stream keyed by that specification and that revision, so the set
of changed chunks is both realistic and reproducible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta

from parts_answer_gate.corpus import text as phrases
from parts_answer_gate.corpus.catalogue import (
    FAMILIES,
    MEASURED_TOPIC_KEYS,
    TOPICS,
    Family,
    Topic,
    Unit,
    Variant,
    render_value,
    spec_options,
    topic_by_key,
)
from parts_answer_gate.corpus.identifiers import part_number
from parts_answer_gate.corpus.rng import stream
from parts_answer_gate.domain import Chunk, Document, Effectivity, Language, SerialRange

__all__ = [
    "BULLETIN",
    "MANUAL",
    "BuiltCorpus",
    "BuiltDocument",
    "Bulletin",
    "BulletinEntry",
    "ChunkKey",
    "Revision",
    "Spec",
    "build_corpus",
    "omitted_topic",
    "plan_bulletin",
    "plan_revisions",
    "spec_for",
]

MANUAL = "manual"
BULLETIN = "bulletin"

#: How often a specification is revised at a revision boundary. Low enough that most chunks survive
#: a revision unchanged — which is what gives the index-lifecycle work something real to measure —
#: and high enough that every family accumulates withdrawn part numbers.
_CHANGE_RATE = 0.35

_MIN_REVISIONS = 3
_MAX_REVISIONS = 4

#: Chunks per printed page. Arbitrary, but a citation without a page number is harder to check
#: against a paper manual than one with it, and the whole project is about checkable citations.
_CHUNKS_PER_PAGE = 4

#: (family_id, series, revision_index, language, variant_id, topic_key)
ChunkKey = tuple[str, str, int, str, str, str]


@dataclass(frozen=True)
class Revision:
    """One revision of one family's manual, before it is turned into three documents."""

    index: int
    code: str
    valid_from: date
    valid_to: date | None


@dataclass(frozen=True)
class Spec:
    """What one manual revision states for one variant and one topic."""

    generation: int
    number: int | None
    part: str | None
    aux_part: str
    aux_number: int
    serials: SerialRange

    def value(self, unit: Unit, language: Language) -> str:
        """The value as written in the prose, which is also the expected answer span."""
        if unit is Unit.PART:
            assert self.part is not None
            return self.part
        assert self.number is not None
        return render_value(unit, self.number, language)


@dataclass(frozen=True)
class BulletinEntry:
    """One planted contradiction: a value in force at the same time as a different one."""

    variant_id: str
    variant_index: int
    topic_key: str
    number: int


@dataclass(frozen=True)
class Bulletin:
    family_id: str
    valid_from: date
    entries: tuple[BulletinEntry, ...]


@dataclass(frozen=True)
class BuiltDocument:
    document: Document
    series: str
    revision_index: int
    text: str
    chunks: tuple[Chunk, ...]
    #: The topic each chunk states, positionally parallel to `chunks`. Kept beside the chunks
    #: rather than on them because `Chunk` is the shared vocabulary and a generator-only field has
    #: no business in it.
    topic_keys: tuple[str, ...]


#: (family_id, variant_id, topic_key, revision_index) -> Spec
_SpecMap = dict[tuple[str, str, str, int], Spec]


@dataclass(frozen=True)
class BuiltCorpus:
    documents: tuple[BuiltDocument, ...]
    revisions: Mapping[str, tuple[Revision, ...]]
    bulletins: Mapping[str, Bulletin]
    #: (family_id, variant_id, topic_key, revision_index) -> Spec
    specs: Mapping[tuple[str, str, str, int], Spec]
    #: (family_id, variant_id) -> the one topic this variant's manual does not cover
    omitted_topics: Mapping[tuple[str, str], str]
    chunk_by_key: Mapping[ChunkKey, Chunk]
    issued_parts: frozenset[str]


# ------------------------------------------------------------------------------------- planning


def plan_revisions(family: Family) -> tuple[Revision, ...]:
    """When each revision came into force. Every gap is drawn from its own stream."""
    count = stream("revision-count", family.family_id).randint(_MIN_REVISIONS, _MAX_REVISIONS)
    launch = stream("launch", family.family_id)
    issued = [date(launch.randint(2019, 2021), launch.randint(1, 12), launch.randint(1, 28))]
    for index in range(1, count):
        gap = stream("revision-gap", family.family_id, str(index)).randint(240, 480)
        issued.append(issued[-1] + timedelta(days=gap))

    return tuple(
        Revision(
            index=index,
            code=chr(ord("A") + index),
            valid_from=issued[index],
            # The last revision is still in force, so it has neither an end date nor a successor.
            # `Document` rejects one without the other, which is the consistency this relies on.
            valid_to=issued[index + 1] if index + 1 < count else None,
        )
        for index in range(count)
    )


def omitted_topic(family: Family, variant: Variant) -> str:
    """The one topic this variant's manual never covers.

    Without it, `attribute_absent_for_existing_product` could not be a real negative: the product
    has to exist, and the attribute has to be genuinely missing for it while present for its
    siblings. Omitting a whole topic from the corpus would be a different (and easier) test.
    """
    picker = stream("omitted-topic", family.family_id, variant.variant_id)
    return TOPICS[picker.randrange(len(TOPICS))].key


def _generation(family_id: str, variant_id: str, topic_key: str, revision_index: int) -> int:
    """How many times this specification has been revised by the given revision."""
    generation = 0
    for boundary in range(1, revision_index + 1):
        roll = stream("spec-change", family_id, variant_id, topic_key, str(boundary))
        if roll.random() < _CHANGE_RATE:
            generation += 1
    return generation


def _number_at(
    unit: Unit, variant_index: int, family_id: str, variant_id: str, topic_key: str, generation: int
) -> int:
    """The measurement at a given generation, guaranteed different from the one before it.

    Walking the generations rather than jumping to the last one costs nothing at these sizes and
    buys the property that matters: a revision that *changed* a value really changed it. A value
    that re-rolled to itself would leave a supersession edge with no observable difference, and the
    evaluation would score a correct answer as revision-incorrect for no visible reason.
    """
    options = spec_options(unit, variant_index)
    previous: int | None = None
    chosen = options[0]
    for step in range(generation + 1):
        roller = stream("spec-number", family_id, variant_id, topic_key, f"g{step}")
        index = roller.randrange(len(options))
        if previous is not None and options[index] == previous:
            index = (index + 1) % len(options)
        chosen = options[index]
        previous = chosen
    return chosen


def _serials(family_id: str, variant: Variant, topic: Topic) -> SerialRange:
    """A bounded serial window for the serial-limited topics, unbounded for the rest."""
    if not topic.serial_limited:
        return SerialRange()
    window = stream("serial-window", family_id, variant.variant_id, topic.key)
    first = variant.serial_first + window.randrange(0, 4_000, 100)
    return SerialRange(first=first, last=first + window.randrange(2_000, 5_000, 100))


def spec_for(
    family: Family, variant: Variant, variant_index: int, topic: Topic, revision_index: int
) -> Spec:
    family_id, variant_id = family.family_id, variant.variant_id
    generation = _generation(family_id, variant_id, topic.key, revision_index)
    aux = stream("aux-number", family_id, variant_id, topic.key)
    number: int | None = None
    part: str | None = None
    if topic.unit is Unit.PART:
        part = part_number("spec-part", family_id, variant_id, topic.key, f"g{generation}")
    else:
        number = _number_at(topic.unit, variant_index, family_id, variant_id, topic.key, generation)
    return Spec(
        generation=generation,
        number=number,
        part=part,
        # The auxiliary consumable is keyed without a generation: a sealing ring does not change
        # every time the element it seals does, and a corpus where every identifier moves together
        # would let a retriever key on "the numbers that changed" instead of on the question.
        aux_part=part_number("aux-part", family_id, variant_id, topic.key),
        aux_number=topic.aux_choices[aux.randrange(len(topic.aux_choices))],
        serials=_serials(family_id, variant, topic),
    )


def plan_bulletin(family: Family, revisions: tuple[Revision, ...], specs: _SpecMap) -> Bulletin:
    """A field bulletin in force alongside the current manual, disagreeing with it.

    The disagreement is drawn from the *same variant's* band, so the conflict is a genuine
    contradiction about one machine rather than another variant's figure that wandered in. Those
    are different failures and the gate treats them differently — one is `REVIEW`, the other is the
    wrong-variant answer the project exists to prevent.
    """
    current = revisions[-1]
    offset = stream("bulletin-date", family.family_id).randint(45, 150)
    entries: list[BulletinEntry] = []
    for variant_index, variant in enumerate(family.variants[:2]):
        omitted = omitted_topic(family, variant)
        available = tuple(key for key in MEASURED_TOPIC_KEYS if key != omitted)
        picker = stream("bulletin-topic", family.family_id, variant.variant_id)
        topic_key = available[picker.randrange(len(available))]
        topic = topic_by_key(topic_key)
        in_manual = specs[(family.family_id, variant.variant_id, topic_key, current.index)].number
        options = spec_options(topic.unit, variant_index)
        chooser = stream("bulletin-value", family.family_id, variant.variant_id, topic_key)
        index = chooser.randrange(len(options))
        if options[index] == in_manual:
            index = (index + 1) % len(options)
        entries.append(
            BulletinEntry(variant.variant_id, variant_index, topic_key, options[index])
        )
    return Bulletin(family.family_id, current.valid_from + timedelta(days=offset), tuple(entries))


# ------------------------------------------------------------------------------------ rendering


class _Cursor:
    """Assembles a document's text while recording where every piece landed.

    The alternative — render the document, then search it for each chunk — finds the *first*
    occurrence of a repeated passage and silently attributes a citation to the wrong section.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._at = 0

    def add(self, piece: str) -> tuple[int, int]:
        start = self._at
        self._parts.append(piece)
        self._at += len(piece)
        return start, self._at

    @property
    def text(self) -> str:
        return "".join(self._parts)


def _serial_phrase(serials: SerialRange, language: Language) -> str:
    if serials.first is None and serials.last is None:
        return phrases.SERIAL_ALL.of(language)
    return phrases.SERIAL_RANGE.of(language).format(first=serials.first, last=serials.last)


def _section_block(
    *,
    language: Language,
    section: str,
    topic_key: str,
    variant_id: str,
    body: str,
    serials: SerialRange,
) -> str:
    heading = phrases.TOPIC_HEADING[topic_key].of(language)
    applies = phrases.APPLICABILITY.of(language).format(
        variant=variant_id, serials=_serial_phrase(serials, language)
    )
    return f"{section}. {heading} — {variant_id}\n{body}\n{applies}"


def _header(
    *, language: Language, title: str, revision: str, document_id: str, valid_from: date
) -> str:
    return phrases.HEADER_TEMPLATE.of(language).format(
        title=title,
        revision=revision,
        document_id=document_id,
        valid_from=valid_from.isoformat(),
        notice=phrases.SYNTHETIC_NOTICE.of(language),
    )


def _document_id(family: Family, series: str, code: str, language: Language) -> str:
    short = family.family_id.removeprefix("fam-")
    return f"doc-{short}-{series}-{code.lower()}-{language.value}"


def _checksum(body: str) -> str:
    return hashlib.blake2b(body.encode("utf-8"), digest_size=16).hexdigest()


def _source_uri(family: Family, series: str, code: str, language: Language) -> str:
    """A URI scheme that cannot be mistaken for a real publication, because it resolves nowhere."""
    return f"synthetic://parts-answer-gate/{family.family_id}/{series}/{code}/{language.value}"


def _render_manual(
    family: Family,
    revision: Revision,
    revisions: tuple[Revision, ...],
    specs: _SpecMap,
    language: Language,
) -> BuiltDocument:
    """One revision of one family's manual, in one language."""
    document_id = _document_id(family, "man", revision.code, language)
    successor = (
        _document_id(family, "man", revisions[revision.index + 1].code, language)
        if revision.valid_to is not None
        else None
    )

    cursor = _Cursor()
    cursor.add(
        _header(
            language=language,
            title=phrases.MANUAL_TITLE.of(language).format(product=family.product.of(language)),
            revision=revision.code,
            document_id=document_id,
            valid_from=revision.valid_from,
        )
    )

    chunks: list[Chunk] = []
    topic_keys: list[str] = []
    for variant_index, variant in enumerate(family.variants):
        omitted = omitted_topic(family, variant)
        for topic_index, topic in enumerate(TOPICS):
            if topic.key == omitted:
                continue
            spec = specs[(family.family_id, variant.variant_id, topic.key, revision.index)]
            block = _section_block(
                language=language,
                section=f"{variant_index + 1}.{topic_index + 1}",
                topic_key=topic.key,
                variant_id=variant.variant_id,
                body=phrases.TOPIC_BODY[topic.key]
                .of(language)
                .format(
                    variant=variant.variant_id,
                    value=spec.value(topic.unit, language),
                    part=spec.aux_part,
                    n=spec.aux_number,
                ),
                serials=spec.serials,
            )
            cursor.add("\n\n")
            start, end = cursor.add(block)
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-c{ordinal:03d}",
                    document_id=document_id,
                    family_id=family.family_id,
                    language=language,
                    text=block,
                    section=f"{variant_index + 1}.{topic_index + 1}",
                    page=1 + ordinal // _CHUNKS_PER_PAGE,
                    start_offset=start,
                    end_offset=end,
                    effectivity=Effectivity(variant_id=variant.variant_id, serials=spec.serials),
                    # Denormalised from the document on purpose: the effectivity filter is a SQL
                    # predicate that must not need a join to decide whether a chunk was in force.
                    valid_from=revision.valid_from,
                    valid_to=revision.valid_to,
                    superseded_by=successor,
                )
            )
            topic_keys.append(topic.key)

    body = cursor.text
    document = Document(
        document_id=document_id,
        family_id=family.family_id,
        title=phrases.MANUAL_TITLE.of(language).format(product=family.product.of(language)),
        language=language,
        revision=revision.code,
        valid_from=revision.valid_from,
        valid_to=revision.valid_to,
        superseded_by=successor,
        source_uri=_source_uri(family, MANUAL, revision.code, language),
        checksum=_checksum(body),
    )
    return BuiltDocument(document, MANUAL, revision.index, body, tuple(chunks), tuple(topic_keys))


def _render_bulletin(family: Family, bulletin: Bulletin, language: Language) -> BuiltDocument:
    """The field bulletin that disagrees with the manual currently in force.

    It is a normal in-force document with no supersession edge: the conflict is between two
    documents that are both valid, which is precisely the case a system that only checks dates
    cannot see.
    """
    document_id = _document_id(family, "bul", "fb1", language)
    title = phrases.BULLETIN_TITLE.of(language).format(product=family.product.of(language))

    cursor = _Cursor()
    cursor.add(
        _header(
            language=language,
            title=title,
            revision="FB1",
            document_id=document_id,
            valid_from=bulletin.valid_from,
        )
    )

    chunks: list[Chunk] = []
    topic_keys: list[str] = []
    for index, entry in enumerate(bulletin.entries):
        topic = topic_by_key(entry.topic_key)
        fleet = stream("bulletin-units", family.family_id, entry.variant_id)
        body = phrases.BULLETIN_BODY.of(language).format(
            variant=entry.variant_id,
            heading=phrases.TOPIC_HEADING[entry.topic_key].of(language),
            value=render_value(topic.unit, entry.number, language),
            # Fleet sizes chosen to agree with the Russian genitive plural in the template; the
            # count is cosmetic, a grammatically wrong corpus is not.
            n=(8, 12, 14, 17)[fleet.randrange(4)],
        )
        block = _section_block(
            language=language,
            section=f"B.{index + 1}",
            topic_key=entry.topic_key,
            variant_id=entry.variant_id,
            body=body,
            serials=SerialRange(),
        )
        cursor.add("\n\n")
        start, end = cursor.add(block)
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}-c{index:03d}",
                document_id=document_id,
                family_id=family.family_id,
                language=language,
                text=block,
                section=f"B.{index + 1}",
                page=1,
                start_offset=start,
                end_offset=end,
                effectivity=Effectivity(variant_id=entry.variant_id, serials=SerialRange()),
                valid_from=bulletin.valid_from,
                valid_to=None,
                superseded_by=None,
            )
        )
        topic_keys.append(entry.topic_key)

    body_text = cursor.text
    document = Document(
        document_id=document_id,
        family_id=family.family_id,
        title=title,
        language=language,
        revision="FB1",
        valid_from=bulletin.valid_from,
        valid_to=None,
        superseded_by=None,
        source_uri=_source_uri(family, BULLETIN, "fb1", language),
        checksum=_checksum(body_text),
    )
    return BuiltDocument(document, BULLETIN, 0, body_text, tuple(chunks), tuple(topic_keys))


# --------------------------------------------------------------------------------------- build


def _plan_specs() -> tuple[dict[str, tuple[Revision, ...]], _SpecMap, dict[tuple[str, str], str]]:
    revisions: dict[str, tuple[Revision, ...]] = {}
    specs: _SpecMap = {}
    omitted: dict[tuple[str, str], str] = {}
    for family in FAMILIES:
        revisions[family.family_id] = plan_revisions(family)
        for variant_index, variant in enumerate(family.variants):
            omitted[(family.family_id, variant.variant_id)] = omitted_topic(family, variant)
            for topic in TOPICS:
                for revision in revisions[family.family_id]:
                    key = (family.family_id, variant.variant_id, topic.key, revision.index)
                    specs[key] = spec_for(family, variant, variant_index, topic, revision.index)
    return revisions, specs, omitted


def _collect_parts(specs: _SpecMap) -> frozenset[str]:
    """Every part number the corpus issues, with a uniqueness check rather than a hope.

    `identifiers.part_number` is collision-free by size, not by construction. Two different keys
    mapping to one number would silently make a part belong to two variants, so the assumption is
    tested here and the build stops if it ever fails.
    """
    owners: dict[str, tuple[str, ...]] = {}
    for (family_id, variant_id, topic_key, _), spec in specs.items():
        issued = [("aux-part", family_id, variant_id, topic_key, spec.aux_part)]
        if spec.part is not None:
            generation = f"spec-part-g{spec.generation}"
            issued.append((generation, family_id, variant_id, topic_key, spec.part))
        for *key, number in issued:
            existing = owners.setdefault(number, tuple(key))
            if existing != tuple(key):
                raise ValueError(
                    f"part number {number} was issued to both {existing} and {tuple(key)}; the "
                    "identifier space is too small or a key is not unique"
                )
    return frozenset(owners)


def build_corpus() -> BuiltCorpus:
    """The whole corpus, in memory, before anything is written or validated."""
    revisions, specs, omitted = _plan_specs()
    bulletins = {
        family.family_id: plan_bulletin(family, revisions[family.family_id], specs)
        for family in FAMILIES
        if family.has_bulletin
    }

    documents: list[BuiltDocument] = []
    chunk_by_key: dict[ChunkKey, Chunk] = {}
    for family in FAMILIES:
        for language in Language:
            for revision in revisions[family.family_id]:
                documents.append(
                    _render_manual(family, revision, revisions[family.family_id], specs, language)
                )
            if family.family_id in bulletins:
                documents.append(_render_bulletin(family, bulletins[family.family_id], language))

    for built in documents:
        for chunk, topic_key in zip(built.chunks, built.topic_keys, strict=True):
            key: ChunkKey = (
                chunk.family_id,
                built.series,
                built.revision_index,
                chunk.language.value,
                chunk.effectivity.variant_id,
                topic_key,
            )
            if key in chunk_by_key:
                raise ValueError(f"two chunks claim the same coordinates: {key}")
            chunk_by_key[key] = chunk

    return BuiltCorpus(
        documents=tuple(documents),
        revisions=revisions,
        bulletins=bulletins,
        specs=specs,
        omitted_topics=omitted,
        chunk_by_key=chunk_by_key,
        issued_parts=_collect_parts(specs),
    )
