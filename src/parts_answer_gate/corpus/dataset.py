"""Reading the generated files back as domain objects.

`Document` and `Chunk` forbid unknown fields, and the written records deliberately carry a few the
domain does not model — the document's full text, the split, the topic, the series. That is the
right trade: the shared vocabulary stays about retrieval, and the generator still ships everything a
consumer needs to check a citation or partition a score.

The cost is one place that has to know which keys are generator-only. This is that place, so the
store loader, the evaluation and any future consumer all strip the same set rather than each
maintaining a list that drifts. Everything here is a read; nothing writes.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from parts_answer_gate.domain import Chunk, Document

__all__ = [
    "CHUNK_EXTRA_KEYS",
    "DOCUMENT_EXTRA_KEYS",
    "LoadedDocument",
    "read_chunks",
    "read_documents",
    "read_questions",
]

#: Fields the generator adds beside the domain model. `text` is the one that matters: kill
#: condition D locates a cited span in the document by offset, and that needs the document's own
#: text rather than a reconstruction from its chunks.
DOCUMENT_EXTRA_KEYS: Final = frozenset({"series", "revision_index", "split", "text"})
CHUNK_EXTRA_KEYS: Final = frozenset({"topic", "series", "revision", "revision_index", "split"})


@dataclass(frozen=True)
class LoadedDocument:
    document: Document
    text: str
    series: str
    split: str


def _records(path: Path, key: str) -> Iterator[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("is_synthetic"):
        raise ValueError(
            f"{path} does not declare is_synthetic; ADR-001 forbids treating a corpus this "
            "project generated as anything else"
        )
    records: list[dict[str, Any]] = payload[key]
    yield from records


def read_documents(corpus_dir: Path) -> list[LoadedDocument]:
    loaded: list[LoadedDocument] = []
    for record in _records(corpus_dir / "documents.json", "documents"):
        fields = {k: v for k, v in record.items() if k not in DOCUMENT_EXTRA_KEYS}
        loaded.append(
            LoadedDocument(
                document=Document.model_validate(fields),
                text=record["text"],
                series=record["series"],
                split=record["split"],
            )
        )
    return loaded


def read_chunks(corpus_dir: Path) -> list[Chunk]:
    return [
        Chunk.model_validate({k: v for k, v in record.items() if k not in CHUNK_EXTRA_KEYS})
        for record in _records(corpus_dir / "chunks.json", "chunks")
    ]


def read_questions(corpus_dir: Path) -> list[dict[str, Any]]:
    """Questions stay dictionaries: they are benchmark rows, not something the service returns."""
    return list(_records(corpus_dir / "questions.json", "questions"))
