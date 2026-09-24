"""Assemble, check, and write the corpus — in that order, and refusing to write a corpus that fails.

The checks in here are not tests. Tests run when somebody runs them; these run every time the
corpus is built, and a failure stops the build rather than producing files that a later stage has
to discover are wrong. Three of them earn their place:

- **Offsets.** `document_text[start:end] == chunk.text`, for every chunk. Kill condition D is graded
  on this arithmetic and the generator is the only thing that can guarantee it.
- **The split.** A hold-out question whose gold chunk lives in a development document is the leak
  ADR-001's condition L exists to catch, and it is checked against the *references* rather than
  against the family assignment — the family assignment cannot disagree with itself.
- **The contract minimums.** Six counts and seven negative kinds, from ADR-001's table. The
  generator fails loudly rather than emitting a corpus that is quietly one document short.

Nothing written here contains a timestamp, a hostname or a path. Two runs must be byte-identical
and a `generated_at` field would make that impossible for the most boring reason imaginable.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from parts_answer_gate.corpus.builder import BuiltCorpus, BuiltDocument, build_corpus
from parts_answer_gate.corpus.catalogue import FAMILIES, TOPICS, Unit, spec_options
from parts_answer_gate.corpus.questions import (
    UNANSWERABLE_KINDS,
    QuestionSeed,
    build_questions,
    question_records,
)
from parts_answer_gate.corpus.rng import CORPUS_SEED, GENERATOR_VERSION
from parts_answer_gate.corpus.splits import DEVELOPMENT, HOLDOUT, SPLIT_BY, SPLIT_RULE, split_of
from parts_answer_gate.domain import Chunk, Language, part_numbers_in

__all__ = [
    "CorpusContractError",
    "GeneratedCorpus",
    "corpus_artifact",
    "generate_corpus",
    "write_artifact",
]

JsonDict = dict[str, Any]

# ADR-001's corpus table. Restated here so the generator fails before the kill test does, with a
# message that names what is short rather than an assertion on a number in a file.
MIN_VARIANTS: Final = 6
MIN_DOCUMENTS: Final = 40
MIN_SUPERSESSION_EDGES: Final = 25
MIN_CHUNKS: Final = 1_500
MIN_QUESTIONS: Final = 300
MIN_UNANSWERABLE: Final = 90

_NOTICE: Final = (
    "Synthetic corpus generated from a committed seed for evaluation. Not real-world data, not a "
    "manufacturer publication, and not reviewed by a native speaker of Turkish or Russian."
)


class CorpusContractError(RuntimeError):
    """The generated corpus does not meet ADR-001. Raised instead of writing the files."""


@dataclass(frozen=True)
class GeneratedCorpus:
    documents: JsonDict
    chunks: JsonDict
    questions: JsonDict
    manifest: JsonDict
    artifact: JsonDict

    @property
    def counts(self) -> JsonDict:
        return {key: self.artifact[key] for key in _COUNT_KEYS}


_COUNT_KEYS: Final = (
    "variants",
    "documents",
    "supersession_edges",
    "chunks",
    "questions",
    "unanswerable_questions",
)


# ------------------------------------------------------------------------------------- checks


def _check_offsets(documents: Sequence[BuiltDocument]) -> int:
    """Every chunk's offsets must index its own document's text exactly."""
    checked = 0
    for built in documents:
        for chunk in built.chunks:
            located = built.text[chunk.start_offset : chunk.end_offset]
            if located != chunk.text:
                raise CorpusContractError(
                    f"{chunk.chunk_id} claims offsets "
                    f"[{chunk.start_offset}:{chunk.end_offset}] of {chunk.document_id}, which "
                    f"hold {located[:60]!r}, not {chunk.text[:60]!r}"
                )
            checked += 1
    return checked


def _check_denormalised_validity(documents: Sequence[BuiltDocument]) -> None:
    """A chunk's copy of its document's validity must agree with the document.

    The copy exists so the effectivity filter is a SQL predicate with no join. A copy that drifts
    is worse than no copy: the filter would run, return quickly, and be wrong.
    """
    for built in documents:
        document = built.document
        for chunk in built.chunks:
            same = (
                chunk.valid_from == document.valid_from
                and chunk.valid_to == document.valid_to
                and chunk.superseded_by == document.superseded_by
                and chunk.family_id == document.family_id
                and chunk.language == document.language
            )
            if not same:
                raise CorpusContractError(
                    f"{chunk.chunk_id} disagrees with {document.document_id} about validity, "
                    "family or language"
                )


def _check_part_numbers(corpus: BuiltCorpus) -> int:
    """Every part-number-shaped token in the corpus must be one the generator issued.

    A token that looks like a part number but was not issued would be evidence the templates are
    producing identifiers by accident, and kill condition C would then be checking a mixture of
    real identifiers and coincidences.

    This function previously stated that invariant in its docstring and never evaluated it: it
    accumulated the tokens found in the text, returned how many there were, and never consulted
    `BuiltCorpus.issued_parts` at all. The data happened to satisfy the invariant, so nothing was
    wrong except that nothing was checked — which is the arrangement ADR-001's falsifiability
    clause exists to rule out. ADR-002.
    """
    seen: set[str] = set()
    for built in corpus.documents:
        seen |= part_numbers_in(built.text)

    invented = seen - corpus.issued_parts
    if invented:
        raise ValueError(
            f"{len(invented)} part-number-shaped tokens appear in the corpus text that the "
            f"generator never issued, so kill condition C would be scoring coincidences as "
            f"identifiers: {sorted(invented)[:10]}"
        )
    return len(seen)


_MAX_VARIANTS_PER_FAMILY = max(len(family.variants) for family in FAMILIES)


def _check_aux_collisions() -> int:
    """No passage may state its auxiliary figure and its answer as the same number.

    A body reads "tighten to {value} ... re-check after {n} operating hours". If `{value}` renders
    as 40 Nm and `{n}` is also 40, the passage carries one number where the reader needs two, the
    expected answer span stops being unambiguous, and a retriever that found the wrong figure
    scores as correct. The content review that produced ADR-002's corpus found five topics whose
    auxiliary choices overlapped their own value band exactly this way.

    Checked here rather than left to review, because the next topic somebody adds will not get a
    review.
    """
    collisions: list[str] = []
    for topic in TOPICS:
        if topic.unit is Unit.PART:
            # The answer is a part number, so no integer auxiliary can collide with it.
            continue
        for variant_index in range(_MAX_VARIANTS_PER_FAMILY):
            band = set(spec_options(topic.unit, variant_index))
            shared = band & set(topic.aux_choices)
            if shared:
                collisions.append(
                    f"{topic.key} at variant position {variant_index}: {sorted(shared)} is both a "
                    f"value and an auxiliary figure"
                )
    if collisions:
        raise ValueError(
            "a passage would state its answer and its auxiliary figure as the same number:\n  "
            + "\n  ".join(collisions)
        )
    return len(TOPICS)


def _check_answerable(seeds: Sequence[QuestionSeed], chunks: dict[str, Chunk]) -> None:
    for seed in seeds:
        if not seed.answerable:
            if any(seed.supporting[language] for language in Language):
                raise CorpusContractError(f"{seed.seed_id} is unanswerable and cites a chunk")
            continue
        for language in Language:
            _check_one_support(seed, language, chunks)


def _check_one_support(seed: QuestionSeed, language: Language, chunks: dict[str, Chunk]) -> None:
    supporting = seed.supporting[language]
    if len(supporting) != 1:
        raise CorpusContractError(f"{seed.seed_id} ({language}) has {len(supporting)} gold chunks")
    chunk = chunks[supporting[0]]
    if chunk.language is not language:
        raise CorpusContractError(
            f"{seed.seed_id} asks in {language} and is supported by a {chunk.language} chunk; the "
            "parallel set is not parallel"
        )
    if not chunk.in_force_on(seed.as_of):
        raise CorpusContractError(
            f"{seed.seed_id} is dated {seed.as_of} and its gold chunk {chunk.chunk_id} was not in "
            f"force then ({chunk.valid_from}..{chunk.valid_to})"
        )
    if chunk.effectivity.variant_id != seed.variant_id:
        raise CorpusContractError(
            f"{seed.seed_id} asks about {seed.variant_id} and its gold chunk applies to "
            f"{chunk.effectivity.variant_id}"
        )
    if not chunk.effectivity.serials.covers(seed.serial):
        raise CorpusContractError(
            f"{seed.seed_id} supplies serial {seed.serial}, outside "
            f"{chunk.effectivity.serials} on its gold chunk"
        )
    span = seed.span[language]
    if span is None or span not in chunk.text:
        raise CorpusContractError(
            f"{seed.seed_id} ({language}) expects the span {span!r}, which does not occur in "
            f"{chunk.chunk_id}"
        )


def _check_contract(artifact: JsonDict, by_kind: dict[str, int]) -> None:
    floors = (
        ("variants", MIN_VARIANTS),
        ("documents", MIN_DOCUMENTS),
        ("supersession_edges", MIN_SUPERSESSION_EDGES),
        ("chunks", MIN_CHUNKS),
        ("questions", MIN_QUESTIONS),
        ("unanswerable_questions", MIN_UNANSWERABLE),
    )
    for key, floor in floors:
        if artifact[key] < floor:
            raise CorpusContractError(
                f"ADR-001 requires at least {floor} {key}; this corpus has {artifact[key]}"
            )
    missing = [kind for kind in UNANSWERABLE_KINDS if by_kind.get(kind, 0) == 0]
    if missing:
        raise CorpusContractError(
            f"ADR-001 names seven kinds of unanswerable question and these have none: {missing}"
        )
    if artifact["documents_in_both_splits"] != 0:
        raise CorpusContractError(
            f"kill condition L: {artifact['leaked_documents']} appear in both splits"
        )


# --------------------------------------------------------------------------------- assembly


def _document_records(documents: Sequence[BuiltDocument]) -> list[JsonDict]:
    records: list[JsonDict] = []
    for built in documents:
        record: JsonDict = built.document.model_dump(mode="json")
        record["series"] = built.series
        record["revision_index"] = built.revision_index
        record["split"] = split_of(built.document.family_id)
        # The full text ships with the documents so a citation's offsets can be checked against the
        # source without rebuilding the corpus. Kill condition D is graded on exactly this string.
        record["text"] = built.text
        records.append(record)
    return records


def _chunk_records(documents: Sequence[BuiltDocument]) -> list[JsonDict]:
    records: list[JsonDict] = []
    for built in documents:
        for chunk, topic_key in zip(built.chunks, built.topic_keys, strict=True):
            record: JsonDict = chunk.model_dump(mode="json")
            record["topic"] = topic_key
            record["series"] = built.series
            record["revision"] = built.document.revision
            record["revision_index"] = built.revision_index
            record["split"] = split_of(chunk.family_id)
            records.append(record)
    return records


def _leaked_documents(
    questions: Sequence[JsonDict],
    chunks: dict[str, Chunk],
    documents: Sequence[JsonDict],
) -> list[str]:
    """Documents reachable from both splits, resolved through the references rather than the rule.

    The previous version of this function could not report a leak under any corpus this generator
    could produce, and its docstring said so while the body did the opposite. It read
    `split_of(chunk.family_id) != record["split"]`, and `record["split"]` was itself assigned as
    `split_of(seed.family_id)`; because every supporting chunk is looked up by the question's own
    family id, the comparison reduced to `split_of(F) != split_of(F)`. It returned zero for
    structural reasons, kill condition L graded that zero, and ADR-001's falsifiability clause —
    a breach must change behaviour and a test must observe it — had nothing to observe. ADR-002
    records it.

    What follows is a real check, and it is deliberately built the way a sceptic would build it:
    **the split label is never recomputed from the rule.** Each document collects the set of splits
    that actually reach it, taken from the questions that cite it, and a document reached from both
    is a leak whatever any rule says. A document whose own family says one thing while a question
    of the other split depends on it is exactly the failure the hold-out exists to prevent, and it
    is caught here because the two facts are compared rather than derived from one another.

    Three separate leaks are reported, all as document ids:

    - a document cited by questions of both splits;
    - a document whose family's split disagrees with the split of a question citing it;
    - a document whose family owns documents on both sides, which would mean the protected unit is
      not the family at all.
    """
    reached_by: dict[str, set[str]] = {}
    for record in questions:
        asking = str(record["split"])
        for chunk_id in record["supporting_chunk_ids"]:
            chunk = chunks[str(chunk_id)]
            reached_by.setdefault(chunk.document_id, set()).add(asking)

    leaked = {document_id for document_id, splits in reached_by.items() if len(splits) > 1}

    # The reference and the rule, compared rather than one derived from the other.
    for document_id, splits in reached_by.items():
        owning = split_of(chunks_family_of(document_id, chunks))
        if splits != {owning}:
            leaked.add(document_id)

    # The protected unit really is the family: no family may own documents on both sides.
    splits_per_family: dict[str, set[str]] = {}
    for record in documents:
        splits_per_family.setdefault(str(record["family_id"]), set()).add(str(record["split"]))
    for family_id, splits in splits_per_family.items():
        if len(splits) > 1:
            leaked.update(
                str(record["document_id"])
                for record in documents
                if str(record["family_id"]) == family_id
            )

    return sorted(leaked)


def chunks_family_of(document_id: str, chunks: dict[str, Chunk]) -> str:
    """The family a document belongs to, read off its chunks rather than off the document table.

    Deliberately the long way round. Reading `document["family_id"]` would take the split check
    back to trusting the same record it is checking; resolving through the chunks means a document
    whose chunks disagree with it about its family shows up as a leak rather than as agreement.
    """
    for chunk in chunks.values():
        if chunk.document_id == document_id:
            return chunk.family_id
    raise KeyError(f"no chunk belongs to {document_id}; the corpus is not self-consistent")


def _distinct(records: Sequence[JsonDict], *key: str) -> int:
    """How many distinct pieces of *content* these records hold, ignoring the language rendering.

    The reason this function exists is ADR-002. The first corpus reported 108 documents, 1,941
    chunks, 432 questions, 132 unanswerable questions and 69 supersession edges, and cleared every
    ADR-001 floor. Counted as distinct content it held 36, 647, 144, 44 and 23, and missed five of
    the six floors — the supersession floor on any reading. The corpus was trilingual, and each of
    those numbers was one fact rendered three times.

    A translation of a passage is not a new passage. The artifact now reports these as the graded
    numbers, with the per-language totals kept beside them under `*_rows` so nothing is hidden.
    """
    return len({tuple(str(record[field]) for field in key) for record in records})


def _artifact(
    *,
    documents: Sequence[JsonDict],
    chunks: Sequence[JsonDict],
    questions: Sequence[JsonDict],
    leaked: Sequence[str],
    by_kind: dict[str, int],
) -> JsonDict:
    holdout = sorted(f.family_id for f in FAMILIES if split_of(f.family_id) == HOLDOUT)
    development = sorted(f.family_id for f in FAMILIES if split_of(f.family_id) == DEVELOPMENT)
    unanswerable = [record for record in questions if not record["answerable"]]
    superseded = [record for record in documents if record["superseded_by"]]
    return {
        "is_synthetic": True,
        "notice": _NOTICE,
        "generator_version": GENERATOR_VERSION,
        "seed": CORPUS_SEED,
        "split_rule": SPLIT_RULE,
        "split_by": SPLIT_BY,
        "variants": sum(len(family.variants) for family in FAMILIES),
        # --- the graded counts: distinct content, not rows ---------------------------------------
        "documents": _distinct(documents, "family_id", "series", "revision"),
        "supersession_edges": _distinct(superseded, "family_id", "revision"),
        "chunks": _distinct(chunks, "family_id", "series", "revision", "section"),
        "questions": _distinct(questions, "parallel_group"),
        "unanswerable_questions": _distinct(unanswerable, "parallel_group"),
        # --- the same corpus counted as rows, one per language -----------------------------------
        "counted_by": (
            "distinct content; a translation of a passage is not a second passage. The *_rows "
            "figures below count one row per language rendering."
        ),
        "document_rows": len(documents),
        "supersession_edge_rows": len(superseded),
        "chunk_rows": len(chunks),
        "question_rows": len(questions),
        "unanswerable_question_rows": len(unanswerable),
        "languages": [language.value for language in Language],
        "documents_in_both_splits": len(leaked),
        "leaked_documents": list(leaked),
        "families": len(FAMILIES),
        "holdout_families": holdout,
        "development_families": development,
        "documents_by_split": _tally(documents, "split"),
        "chunks_by_split": _tally(chunks, "split"),
        "questions_by_split": _tally(questions, "split"),
        "questions_by_language": _tally(questions, "language"),
        "unanswerable_by_kind": {kind: by_kind.get(kind, 0) for kind in UNANSWERABLE_KINDS},
    }


def _tally(records: Sequence[JsonDict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def _manifest(artifact: JsonDict, offsets_checked: int, distinct_parts: int) -> JsonDict:
    return {
        "is_synthetic": True,
        "notice": _NOTICE,
        "generator_version": GENERATOR_VERSION,
        "seed": CORPUS_SEED,
        "split_rule": SPLIT_RULE,
        "split_by": SPLIT_BY,
        "languages": [language.value for language in Language],
        # The task allowed ASCII transliteration as a fallback. It was not needed: the documents
        # and questions are written in Turkish and Russian orthography and the files are UTF-8.
        "script_fidelity": "native (Turkish diacritics and Cyrillic), files encoded UTF-8",
        "judge_calibration_performed": False,
        "ground_truth": "construction metadata — the generator knows, nothing was labelled",
        "counts": {key: artifact[key] for key in _COUNT_KEYS},
        "chunk_offsets_verified": offsets_checked,
        "distinct_part_numbers_in_text": distinct_parts,
        "unanswerable_by_kind": artifact["unanswerable_by_kind"],
        "holdout_families": artifact["holdout_families"],
        "development_families": artifact["development_families"],
        "files": ["documents.json", "chunks.json", "questions.json", "manifest.json"],
        # Deliberately absent: a generation timestamp. Two runs must be byte-identical, and a clock
        # in the output is the cheapest possible way to fail kill condition J.
        "contains_timestamp": False,
    }


def _payload(kind: str, records: Sequence[JsonDict]) -> JsonDict:
    """Every file states, in its own body, that it is synthetic. ADR-001 requires it by name."""
    return {
        "is_synthetic": True,
        "notice": _NOTICE,
        "generator_version": GENERATOR_VERSION,
        "seed": CORPUS_SEED,
        "count": len(records),
        kind: list(records),
    }


def build() -> GeneratedCorpus:
    """The corpus in memory, fully checked. Raises `CorpusContractError` rather than returning."""
    corpus: BuiltCorpus = build_corpus()
    offsets_checked = _check_offsets(corpus.documents)
    _check_denormalised_validity(corpus.documents)
    distinct_parts = _check_part_numbers(corpus)
    _check_aux_collisions()

    by_id = {chunk.chunk_id: chunk for built in corpus.documents for chunk in built.chunks}
    seeds = build_questions(corpus)
    _check_answerable(seeds, by_id)

    documents = _document_records(corpus.documents)
    chunks = _chunk_records(corpus.documents)
    questions = question_records(seeds)
    by_kind = Counter(
        str(record["unanswerable_kind"]) for record in questions if not record["answerable"]
    )
    leaked = _leaked_documents(questions, by_id, documents)
    artifact = _artifact(
        documents=documents,
        chunks=chunks,
        questions=questions,
        leaked=leaked,
        by_kind=dict(by_kind),
    )
    _check_contract(artifact, dict(by_kind))

    return GeneratedCorpus(
        documents=_payload("documents", documents),
        chunks=_payload("chunks", chunks),
        questions=_payload("questions", questions),
        manifest=_manifest(artifact, offsets_checked, distinct_parts),
        artifact=artifact,
    )


def corpus_artifact() -> JsonDict:
    """The `artifacts/corpus.json` body, for a caller that wants it without writing files."""
    return build().artifact


# ---------------------------------------------------------------------------------- writing


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
    # `newline="\n"` because the default on Windows rewrites every newline to CRLF, and a corpus
    # whose bytes depend on the operating system cannot be compared across two runs on two machines.
    path.write_text(body + "\n", encoding="utf-8", newline="\n")


def write_artifact(artifact: JsonDict, path: Path) -> None:
    _write_json(path, artifact)


def generate_corpus(out_dir: Path, *, artifact_path: Path | None = None) -> GeneratedCorpus:
    """Build, check and write the corpus. Returns what was written."""
    generated = build()
    _write_json(out_dir / "documents.json", generated.documents)
    _write_json(out_dir / "chunks.json", generated.chunks)
    _write_json(out_dir / "questions.json", generated.questions)
    _write_json(out_dir / "manifest.json", generated.manifest)
    if artifact_path is not None:
        write_artifact(generated.artifact, artifact_path)
    return generated
