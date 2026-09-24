"""Measuring incremental re-embedding, by doing it rather than by describing it.

The skill matrix assigns "index lifecycle and incremental re-embedding" to this project alone, and
the honest form of that claim is a number: how many chunks the store holds, how many changed, how
many were re-encoded, and how long a changed chunk spent carrying a vector that no longer described
it. The last of those is `staleness_seconds`, and it is computed from two timestamps the loader
wrote — `content_changed_at` when the new text arrived and `embedded_at` when its vector existed —
not from a wall clock read around the outside of the call.

**Why this rolls back.** The measurement has to change real chunk text to be a measurement at all.
But the same rows are the evidence kill conditions C and D are graded against: a cited span must
occur in the document it is attributed to, and a lifecycle probe that left `[lifecycle-probe]`
appended to forty passages would break that for no benefit. So the whole probe runs inside a
savepoint that is rolled back, and the corpus is exactly as it was afterwards. The embeddings were
genuinely computed, the SQL genuinely executed, and the numbers are genuinely measured; only the
residue is discarded.

The production re-embedding path is `store.loader.load_chunks`, which is what an operator runs and
what this probe calls. There is no second implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from parts_answer_gate.domain import Chunk
from parts_answer_gate.retrieval.embeddings import EMBEDDING_MODEL_NAME, Embedder
from parts_answer_gate.store.loader import chunk_count, load_chunks
from parts_answer_gate.store.schema import CHUNK_READ_COLUMNS, chunk_from_mapping

__all__ = ["LIFECYCLE_PROBE_MARKER", "measure_index_lifecycle"]

#: Appended to the sampled chunks so their content hash genuinely differs. Visible and obviously
#: synthetic: if it ever survived a rollback, a reader would know immediately what put it there.
LIFECYCLE_PROBE_MARKER: Final = " [lifecycle-probe]"

_SAMPLE_SQL: Final = (
    # Column list from a module constant, LIMIT bound: nothing a caller supplies reaches the text.
    f"SELECT {', '.join(f'chunk.{name}' for name in CHUNK_READ_COLUMNS)} "  # noqa: S608
    "FROM chunk ORDER BY chunk.chunk_id LIMIT :sample"
)

_STALENESS_SQL: Final = """
SELECT max(EXTRACT(EPOCH FROM (embedded_at - content_changed_at))) AS worst,
       avg(EXTRACT(EPOCH FROM (embedded_at - content_changed_at))) AS mean
FROM chunk
WHERE chunk_id = ANY(CAST(:chunk_ids AS text[]))
"""

#: The incremental claim, asked of the database rather than of the loader's own counters: no chunk
#: outside the changed set may have acquired a new vector while the probe ran.
_COLLATERAL_SQL: Final = """
SELECT count(*)
FROM chunk
WHERE NOT (chunk_id = ANY(CAST(:chunk_ids AS text[])))
  AND embedded_at >= CAST(:since AS timestamptz)
"""


def measure_index_lifecycle(
    session: Session,
    embedder: Embedder,
    *,
    sample_size: int = 40,
) -> dict[str, Any]:
    """Change `sample_size` chunks, re-embed only those, and report what actually happened.

    The sample is the lowest `sample_size` chunk ids. Deterministic rather than random because kill
    condition J compares two runs of the whole pipeline byte for byte, and a randomly sampled
    lifecycle report would differ between them for a reason that has nothing to do with retrieval.
    """
    total = chunk_count(session)
    if total == 0:
        raise RuntimeError(
            "the store is empty; a lifecycle measurement over no chunks is a fiction"
        )
    sample_size = min(sample_size, max(total - 1, 0))
    if sample_size == 0:
        raise RuntimeError(
            f"the store holds {total} chunk(s); the lifecycle report must leave at least one chunk "
            "un-re-embedded or it is measuring a full rebuild"
        )

    rows = session.execute(sql_text(_SAMPLE_SQL), {"sample": sample_size}).mappings().all()
    sampled = [chunk_from_mapping(row) for row in rows]
    chunk_ids = [chunk.chunk_id for chunk in sampled]

    probe_started = datetime.now(UTC)
    savepoint = session.begin_nested()
    try:
        changed = _with_changed_text(sampled)
        report = load_chunks(session, changed, embedder)
        # Flush rather than commit: the timestamps must be readable by the SQL below, and the
        # savepoint still has to be discardable.
        session.flush()

        staleness = session.execute(
            sql_text(_STALENESS_SQL), {"chunk_ids": chunk_ids}
        ).mappings().one()
        collateral = int(
            session.execute(
                sql_text(_COLLATERAL_SQL), {"chunk_ids": chunk_ids, "since": probe_started}
            ).scalar_one()
        )

        result: dict[str, Any] = {
            "chunks_total": total,
            "chunks_changed": report.chunks_changed + report.chunks_new,
            "chunks_re_embedded": report.chunks_re_embedded,
            "chunks_unchanged_left_alone": total - report.chunks_re_embedded,
            "staleness_seconds": _as_float(staleness["worst"]),
            "staleness_seconds_mean": _as_float(staleness["mean"]),
            "staleness_definition": (
                "seconds between a chunk's new text being handed to the loader "
                "(content_changed_at) and its replacement vector existing (embedded_at), worst "
                "case over the changed chunks"
            ),
            "embedding_seconds": round(report.embed_seconds, 4),
            "write_seconds": round(report.write_seconds, 4),
            "seconds_per_re_embedded_chunk": (
                round(report.embed_seconds / report.chunks_re_embedded, 5)
                if report.chunks_re_embedded
                else None
            ),
            "model": EMBEDDING_MODEL_NAME,
            "cost_basis": (
                "local CPU ONNX inference; no metered API is called, so the cost of a re-embed is "
                "wall-clock time and nothing is billed. No monetary figure is published because "
                "none was measured."
            ),
            "chunks_re_embedded_outside_the_changed_set": collateral,
            "probe": (
                "the sampled chunks' text was modified, re-embedded through the production loader, "
                "measured, and rolled back; the corpus is unchanged"
            ),
            "rolled_back": True,
        }
    finally:
        savepoint.rollback()

    return result


def _with_changed_text(chunks: Sequence[Chunk]) -> list[Chunk]:
    """Chunks whose text — and therefore whose content hash — genuinely differs."""
    return [
        chunk.model_copy(update={"text": chunk.text + LIFECYCLE_PROBE_MARKER}) for chunk in chunks
    ]


def _as_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]
