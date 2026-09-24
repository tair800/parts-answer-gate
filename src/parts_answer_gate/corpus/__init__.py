"""The synthetic multi-revision service-manual corpus: catalogue, documents, chunks, questions.

ADR-001 says the corpus is synthetic and openly so, because no redistributable corpus of
multi-revision service manuals with variant effectivity, serial ranges, supersession edges and
parallel EN/TR/RU text exists — real ones are a manufacturer's commercial property. Generating it
buys something a real corpus could not give either: the ground truth is **construction metadata**.
The generator knows which revision superseded which and which questions have no answer, because it
built them that way, so no label in the benchmark is anybody's opinion.

Entry points:

- `generate_corpus(out_dir, artifact_path=...)` writes `documents.json`, `chunks.json`,
  `questions.json` and `manifest.json`, and optionally `artifacts/corpus.json`.
- `corpus_artifact()` returns the `artifacts/corpus.json` body without writing anything, for a
  caller that assembles the artifacts itself.
- `build()` returns everything in memory, already checked.
- `dataset.read_documents` / `read_chunks` / `read_questions` read the written files back as domain
  objects, stripping the generator-only fields in one place instead of in every consumer.
"""

from __future__ import annotations

from parts_answer_gate.corpus.dataset import (
    LoadedDocument,
    read_chunks,
    read_documents,
    read_questions,
)
from parts_answer_gate.corpus.generate import (
    CorpusContractError,
    GeneratedCorpus,
    build,
    corpus_artifact,
    generate_corpus,
    write_artifact,
)
from parts_answer_gate.corpus.rng import CORPUS_SEED, GENERATOR_VERSION
from parts_answer_gate.corpus.splits import SPLIT_BY, SPLIT_RULE, is_holdout, split_of

__all__ = [
    "CORPUS_SEED",
    "GENERATOR_VERSION",
    "SPLIT_BY",
    "SPLIT_RULE",
    "CorpusContractError",
    "GeneratedCorpus",
    "LoadedDocument",
    "build",
    "corpus_artifact",
    "generate_corpus",
    "is_holdout",
    "read_chunks",
    "read_documents",
    "read_questions",
    "split_of",
    "write_artifact",
]
