"""The second storage backend, and the several ways this comparison could have been a lie.

The blueprint asks for a managed vector store comparison. ADR-001 records what was actually built
instead: **a local Qdrant container, not Qdrant Cloud.** A cloud account, its credentials and its
network variance are exactly the days the blueprint says to spend elsewhere, so they were not spent.
Everything this module produces is therefore a measurement of one process on this host against
another process on this host, and `storage_comparison.json` says so in its own body — `deployment`,
`qdrant_cloud_tested` and a prose note — rather than in a footnote a reader may not reach. No
figure here may be read as a managed-cloud production result, because none was produced.

What a second backend behind one port *does* buy is the part of the comparison that has evidence in
it: a real recall@10 and a real latency distribution over **the same corpus and the same queries**.
Four things had to be true for that to be worth publishing, and each one is a decision below.

**The two backends must hold the same vectors.** Qdrant is loaded from the vectors already stored in
`chunk.embedding`, read back out of PostgreSQL — never re-encoded. Re-embedding for the second
backend would have made the two differ by the encoder's own nondeterminism and by nothing that has
to do with storage, and a recall gap caused by a second ONNX run is not a fact about Qdrant.

**The two backends must see the same candidate set.** The effectivity predicate is this project's
sole-home skill and a comparison that applied it to one side only would be comparing a filtered
retriever against an unfiltered one. So the same rule is expressed twice — as the SQL `WHERE`
fragment `store.effectivity` builds, and as a Qdrant payload filter — and because two spellings of
one rule is the thing `store.effectivity` warns about, the artifact does not assert they agree: for
every query it pulls the admitted chunk ids out of *both* engines and compares the sets, and
publishes how many queries disagreed. A stated equivalence nobody measured would be the weakest
sentence in the file.

**The two filters are not the same mechanism, and that is a finding rather than a caveat.** In
PostgreSQL the predicate is a SQL clause in the same statement as the distance ordering: the planner
is free to satisfy it with a btree index scan and never consult HNSW at all, which at this corpus
size is frequently what it does — `pgvector.json` publishes the plan, and this artifact records
whether the plan it measured mentioned the vector index. In Qdrant the predicate is a *payload
filter* evaluated against payload indexes while the vector index is searched; Qdrant calls this
filtered search rather than post-filtering, and it is genuinely not the naive rank-then-discard this
project exists to avoid — but the vector index is still built over every point, and when the filter
is selective Qdrant abandons the graph and rescores the admitted set exactly. Both engines, at this
corpus size and left alone, decline to walk an approximate graph. That is the honest headline of the
comparison and it is in `findings`.

**The two indexes must be configured alike, and neither engine may be pushed off the plan it would
choose.** The Qdrant collection is built from the same constants the pgvector index is built from —
`HNSW_BUILD_PARAMETERS` for `m` and `ef_construction`, `SESSION_SETTINGS["hnsw.ef_search"]` for the
search beam — rather than from a second set of numbers typed in here. Exactly one Qdrant default is
overridden: `indexing_threshold`, without which a corpus this size never builds a vector index at
all and the collection would have no HNSW graph for the comparison to be about. Every default is
**read off the server** rather than recalled, because a finding that argues from a default is only
as good as the number it names.

`full_scan_threshold` was overridden too, for one iteration, and the result is why it no longer is.
Forcing the filtered search onto the graph made Qdrant's recall@10 swing between 0.93 and 0.73
across two runs of the identical workload — because the effectivity predicate admits around twenty
of twelve thousand points, a graph traversal at 0.2% selectivity misses most of them, and Qdrant
rebuilds its graph nondeterministically. Meanwhile PostgreSQL's planner was quietly doing the
sensible thing: index-scanning the twenty admitted rows and sorting them exactly. So the override
was not making the engines comparable, it was handicapping one of them into an approximate answer
while the other answered exactly, and it was publishing a recall figure that changed between runs.
Both engines now choose their own plan for each query, which is the comparison worth having — and at
this corpus size both choose an exact pass over the admitted set. `findings` says so.

**Cost is arithmetic, and the arithmetic is not filled in.** ADR-001 commits to published list price
arithmetic clearly labelled, and that is what `cost` is: the quantities are measured on this host,
the formula is written out, and the price per unit is `null` because no vendor price list was read
during this build. Inventing a plausible dollar figure would be the same defect as publishing an
unmeasured latency, and it is the one this artifact would be least likely to be caught at.

**If Qdrant is not reachable this raises.** It does not fall back to a single-backend artifact and
it does not fabricate a second column. `tests/test_kill_criteria.py` requires two backends; the
correct outcome of an unreachable container is that the evidence is absent and the test fails, not
that it passes on a file describing a comparison nobody ran.

`qdrant-client` is a **development** dependency, not a runtime one: the serving API has no reason to
carry a gRPC stack for an artifact it only ever reads off disk. Every import of it below is
therefore inside the function that needs it, so that `config.py` can import this module's URL
constants and a production install with no `qdrant-client` still starts.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Final

from sqlalchemy import select
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from parts_answer_gate.domain import Language, Query
from parts_answer_gate.evaluation.metrics import RETRIEVAL_K, recall_at_k
from parts_answer_gate.retrieval.dense import DISTANCE_OPERATOR, dense_search, explain_dense
from parts_answer_gate.retrieval.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    Embedder,
)
from parts_answer_gate.store.effectivity import CandidateFilter, candidate_filter
from parts_answer_gate.store.engine import SESSION_SETTINGS
from parts_answer_gate.store.schema import HNSW_BUILD_PARAMETERS, ChunkRow

if TYPE_CHECKING:  # pragma: no cover - import-time typing only, see the module docstring
    from qdrant_client import QdrantClient
    from qdrant_client import models as qmodels

__all__ = [
    "COLLECTION_NAME",
    "DEFAULT_QDRANT_URL",
    "QDRANT_URL_ENV",
    "ComparisonQuestion",
    "QdrantUnreachableError",
    "build_storage_comparison_artifact",
    "comparison_questions",
    "qdrant_url",
]

QDRANT_URL_ENV: Final = "PAG_QDRANT_URL"

#: Matches the `qdrant` service in `docker-compose.yml`. 16333 rather than Qdrant's own 6333 for the
#: same reason PostgreSQL sits on 15440: the other repositories in this workspace already hold
#: 15432/15433/15434/15437/15440, and a developer running two of them must not have to stop one.
DEFAULT_QDRANT_URL: Final = "http://127.0.0.1:16333"

#: Recreated from scratch on every run. A collection that survived a corpus change would be a recall
#: figure measured against vectors the database no longer holds, which is the exact failure the
#: `docker-compose.yml` service avoids by mounting no volume.
COLLECTION_NAME: Final = "pag_storage_comparison"

#: NULL bounds, carried into the payload as sentinels rather than as JSON nulls.
#:
#: The SQL predicate spells an open interval as `valid_to IS NULL OR :as_of < valid_to`. Qdrant can
#: express that — `IsNull` nested under a `should` — but the result is a differently-shaped boolean
#: expression from the one PostgreSQL evaluates, and the whole point of this module is that the two
#: engines admit the same rows. A sentinel makes the Qdrant side a plain `Range`, identical in shape
#: to the closed case, so the two filters differ in syntax and not in structure.
#:
#: Both sentinels are deliberately **outside the representable range of the column they stand in
#: for**: `date.max.toordinal() + 1` is the ordinal of no date at all, and `serial_first` and
#: `serial_last` are PostgreSQL `integer` columns so no real serial reaches 2**62. A sentinel a
#: real value could
#: collide with would silently admit the wrong chunks, and the candidate-set comparison this module
#: publishes is what would catch it.
_OPEN_DATE: Final = date.max.toordinal() + 1
_OPEN_SERIAL_FIRST: Final = -(2**62)
_OPEN_SERIAL_LAST: Final = 2**62

#: Qdrant builds no vector index until a segment's vectors exceed this **size in kilobytes** — not a
#: count of vectors, which is the easy misreading. Its own default is far above what a corpus this
#: size reaches, so the collection would serve every query by exact search and the p95 published
#: beside pgvector's would be the p95 of a different algorithm. Lowered so the index the comparison
#: claims to be measuring actually exists. The default is quoted from memory nowhere in this module:
#: `_qdrant_defaults` reads it off the server and the artifact publishes what it read.
_INDEXING_THRESHOLD_KB: Final = 100

#: How long to wait for Qdrant to finish optimising after the upload. Indexing is asynchronous, and
#: querying a collection whose status is still `yellow` would measure a half-built index.
_INDEX_TIMEOUT_SECONDS: Final = 120.0

_ADMITTED_SQL: Final = (
    # The `where` fragment is assembled from constants in `store.effectivity` and every
    # caller-supplied value arrives as a bound parameter, which is why this is not an injection
    # site.
    # `embedding IS NOT NULL` mirrors `retrieval.dense`: Qdrant holds only chunks that have a
    # vector, so a comparison against every admitted row would charge PostgreSQL with rows the other
    # backend was never given.
    "SELECT chunk.chunk_id FROM chunk WHERE ({where}) AND chunk.embedding IS NOT NULL"
)


class QdrantUnreachableError(RuntimeError):
    """The second backend did not answer.

    Raised rather than degraded. A one-backend `storage_comparison.json` would fail the kill test on
    a missing key, which is recoverable; a two-backend one with invented numbers would pass it, and
    nothing downstream could tell.
    """


def qdrant_url() -> str:
    """The comparison's Qdrant endpoint. Same shape as `store.engine.database_url`."""
    return os.environ.get(QDRANT_URL_ENV, DEFAULT_QDRANT_URL)


# ------------------------------------------------------------------------------ the query workload


@dataclass(frozen=True)
class ComparisonQuestion:
    """One question of the shared workload, with the passages that genuinely support it.

    `relevant_ids` is the corpus's own construction metadata — the generator knows which chunk
    answers which question because it built them together — so both backends are scored against one
    ground truth that neither of them produced.
    """

    question_id: str
    query: Query
    relevant_ids: frozenset[str]


def comparison_questions(
    questions: Iterable[Mapping[str, Any]], *, max_questions: int
) -> list[ComparisonQuestion]:
    """The workload, taken in corpus order and truncated.

    Questions with no supporting passage are dropped. The hold-out deliberately contains questions
    the corpus cannot answer and `metrics.recall_at_k` refuses to score them, for the reason argued
    there: a vacuous 1.0 apiece would lift both backends' recall without either of them retrieving
    anything, and the comparison would be between two identical inflations.

    Corpus order rather than a sample, and no seed: the selection has to be identical across runs
    for kill condition J, and it has to be identical *between backends* or the comparison is of two
    different workloads. Both backends receive this exact list, in this order, so a selection that
    happens to be easy or hard is a property of the comparison and not of either engine.
    """
    selected: list[ComparisonQuestion] = []
    for question in questions:
        supporting = frozenset(str(item) for item in (question.get("supporting_chunk_ids") or []))
        if not supporting:
            continue
        selected.append(
            ComparisonQuestion(
                question_id=str(question["question_id"]),
                query=Query(
                    text=str(question["text"]),
                    language=Language(question["language"]),
                    as_of=date.fromisoformat(str(question["as_of"])),
                    variant_id=question.get("variant_id") or None,
                    serial=question.get("serial"),
                ),
                relevant_ids=supporting,
            )
        )
        if len(selected) >= max_questions:
            break
    return selected


# --------------------------------------------------------------- reading the vectors back out of PG


@dataclass(frozen=True)
class _StoredChunk:
    """A chunk as both backends will see it: one vector and the effectivity coordinates."""

    chunk_id: str
    vector: list[float]
    payload: dict[str, Any]


def _ordinal(value: date | None) -> int:
    return _OPEN_DATE if value is None else value.toordinal()


def _read_stored_chunks(session: Session) -> list[_StoredChunk]:
    """Every chunk that already carries a vector, with the vector.

    **The vectors are read, never recomputed.** That is the single most important line in this
    module: the comparison is of two storage engines over one set of embeddings, and an engine fed
    its own fresh encoding of the same text would differ from the other by the encoder as well as by
    the index.

    Read through the declared `Vector` column rather than through raw SQL, which is why this needs
    none of the pgvector-wire-format parsing `store.diagnostics` carries. That module shares one
    textual `WHERE` fragment with the retrieval path and therefore has to issue raw SQL; this one is
    a bulk export of a whole column with no fragment to share, so the ORM's own type decodes it.

    Ordered by `chunk_id` so the point-id assignment below is reproducible.
    """
    statement = (
        select(
            ChunkRow.chunk_id,
            ChunkRow.embedding,
            ChunkRow.variant_id,
            ChunkRow.language,
            ChunkRow.serial_first,
            ChunkRow.serial_last,
            ChunkRow.valid_from,
            ChunkRow.valid_to,
            ChunkRow.known_from,
            ChunkRow.known_to,
        )
        .where(ChunkRow.embedding.is_not(None))
        .order_by(ChunkRow.chunk_id)
    )
    stored: list[_StoredChunk] = []
    for row in session.execute(statement).mappings():
        vector = [float(value) for value in row["embedding"]]
        if len(vector) != EMBEDDING_DIM:
            raise RuntimeError(
                f"chunk {row['chunk_id']} carries a {len(vector)}-dimension vector where the "
                f"schema declares {EMBEDDING_DIM}; the two backends would index different spaces"
            )
        stored.append(
            _StoredChunk(
                chunk_id=str(row["chunk_id"]),
                vector=vector,
                payload={
                    # Carried so the collection is legible to somebody who opens it in Qdrant's own
                    # dashboard. The measured search does not request it back — see `_search`.
                    "chunk_id": str(row["chunk_id"]),
                    "variant_id": str(row["variant_id"]),
                    "language": str(row["language"]),
                    "valid_from": _ordinal(row["valid_from"]),
                    "valid_to": _ordinal(row["valid_to"]),
                    "known_from": _ordinal(row["known_from"]),
                    "known_to": _ordinal(row["known_to"]),
                    "serial_first": (
                        _OPEN_SERIAL_FIRST
                        if row["serial_first"] is None
                        else int(row["serial_first"])
                    ),
                    "serial_last": (
                        _OPEN_SERIAL_LAST if row["serial_last"] is None else int(row["serial_last"])
                    ),
                },
            )
        )
    return stored


def _admitted_in_postgres(session: Session, filters: CandidateFilter) -> set[str]:
    """The chunk ids the SQL predicate admits: what the Qdrant payload filter is checked against."""
    rows = session.execute(sql_text(_ADMITTED_SQL.format(where=filters.sql)), filters.params).all()
    return {str(row[0]) for row in rows}


# --------------------------------------------------------------------------------- the Qdrant side


def _versions(client: QdrantClient) -> dict[str, str]:
    """Which Qdrant answered, and which client asked.

    Both are recorded rather than one. `docker-compose.yml` pins the server at v1.12.4 and
    `qdrant-client` resolves to whatever the lock file holds, so the two can drift apart — the
    client emits a compatibility warning when they do, and a reader of the artifact deserves to see
    the same pair of numbers the warning is about rather than have to reproduce it.
    """
    from importlib.metadata import version  # noqa: PLC0415 - one call, at artifact-build time only

    return {
        "server_version": str(client.info().version),
        "client_version": version("qdrant-client"),
    }


def _qdrant_defaults(client: QdrantClient) -> dict[str, Any]:
    """Qdrant's own defaults, read off the server rather than quoted from memory.

    This module overrides two of them and `findings` argues from what they would otherwise have
    been, so without this the artifact would be asserting numbers nobody checked. A throwaway
    collection created with no overrides and immediately deleted is cheap, and it turns "Qdrant's
    default is X" from a recollection into a value this run read back from the server that produced
    the measurements.
    """
    from qdrant_client import models  # noqa: PLC0415 - dev-only dependency, see module docs

    probe = f"{COLLECTION_NAME}_defaults_probe"
    client.delete_collection(probe)
    client.create_collection(
        collection_name=probe,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
    )
    config = client.get_collection(probe).config
    client.delete_collection(probe)
    return {
        "indexing_threshold_kb": config.optimizer_config.indexing_threshold,
        "full_scan_threshold_kb": config.hnsw_config.full_scan_threshold,
        "m": config.hnsw_config.m,
        "ef_construct": config.hnsw_config.ef_construct,
        "read_from": "a throwaway collection created with no overrides and deleted again",
    }


def _connect(url: str) -> QdrantClient:
    """Open the client and prove the server answers, or raise with the URL in the message."""
    from qdrant_client import QdrantClient  # noqa: PLC0415 - dev-only dependency, see module docs

    client = QdrantClient(url=url, timeout=30)
    try:
        client.get_collections()
    except Exception as error:
        raise QdrantUnreachableError(
            f"Qdrant did not answer at {url}: {error}. Start it with "
            "`docker compose up -d qdrant`. "
            "This raises rather than emitting a one-backend artifact: a storage comparison with an "
            "invented second column would pass the kill test, and an absent one only fails it."
        ) from error
    return client


def _effectivity_payload_filter(
    query: Query, *, languages: Iterable[Language] | None = None
) -> qmodels.Filter:
    """The same admission rule as `store.effectivity.candidate_filter`, in Qdrant's filter language.

    Clause for clause, in the same order, so the two can be read side by side. Every clause is a
    `must`, because the SQL fragment is a conjunction and an `OR` introduced here to work around a
    NULL would be a structural difference between the two engines rather than a syntactic one — see
    the sentinel constants for why the open intervals need no `should` at all.

    This is the second implementation of a rule `store.effectivity` says should have exactly one,
    and the module does not ask to be trusted about it: `build_storage_comparison_artifact`
    compares the admitted id sets from both engines for every query in the workload, and publishes
    the result.
    """
    from qdrant_client import models  # noqa: PLC0415 - dev-only dependency, see module docs

    as_of = query.as_of.toordinal()
    must: list[qmodels.Condition] = [
        # valid_from <= as_of AND (valid_to IS NULL OR as_of < valid_to)
        models.FieldCondition(key="valid_from", range=models.Range(lte=as_of)),
        models.FieldCondition(key="valid_to", range=models.Range(gt=as_of)),
    ]

    # Knowledge time. Current knowledge is `known_to IS NULL`, which is the sentinel exactly — not
    # "today", for the reason `domain.Query.known_as_of` argues at length.
    if query.known_as_of is None:
        must.append(
            models.FieldCondition(key="known_to", match=models.MatchValue(value=_OPEN_DATE))
        )
    else:
        known = query.known_as_of.toordinal()
        must.append(models.FieldCondition(key="known_from", range=models.Range(lte=known)))
        must.append(models.FieldCondition(key="known_to", range=models.Range(gt=known)))

    selected = list(languages) if languages is not None else [query.language]
    if selected:
        must.append(
            models.FieldCondition(
                key="language", match=models.MatchAny(any=[item.value for item in selected])
            )
        )

    if query.variant_id is not None:
        must.append(
            models.FieldCondition(key="variant_id", match=models.MatchValue(value=query.variant_id))
        )
        # Mirrors the SQL asymmetry deliberately: a technician who named the machine but not its
        # serial gets only passages that apply to every serial. Losing that here would make Qdrant
        # admit chunks PostgreSQL refuses, and the candidate-set comparison would report it.
        if query.serial is None:
            must.append(
                models.FieldCondition(
                    key="serial_first", match=models.MatchValue(value=_OPEN_SERIAL_FIRST)
                )
            )
            must.append(
                models.FieldCondition(
                    key="serial_last", match=models.MatchValue(value=_OPEN_SERIAL_LAST)
                )
            )
        else:
            must.append(
                models.FieldCondition(key="serial_first", range=models.Range(lte=query.serial))
            )
            must.append(
                models.FieldCondition(key="serial_last", range=models.Range(gte=query.serial))
            )

    return models.Filter(must=must)


def _search_params() -> qmodels.SearchParams:
    """The search beam, taken from the constant the pgvector session is configured with.

    `hnsw.ef_search = 100` is what every PostgreSQL connection in this project sets, so Qdrant is
    asked for the same beam rather than for its own default. Two engines searched at different
    widths would differ in recall for a reason that is a configuration choice, not a property of
    either store.
    """
    from qdrant_client import models  # noqa: PLC0415 - dev-only dependency, see module docs

    return models.SearchParams(hnsw_ef=int(SESSION_SETTINGS["hnsw.ef_search"]), exact=False)


def _load_collection(client: QdrantClient, stored: Sequence[_StoredChunk]) -> dict[int, str]:
    """Recreate the collection, fill it, and return the point-id to chunk-id map.

    Recreated rather than upserted into. An index left over from a previous corpus is the storage
    equivalent of a stale artifact: it answers, it looks measured, and it describes data that is no
    longer there.

    The HNSW parameters come from `store.schema.HNSW_BUILD_PARAMETERS`, the same dictionary the
    PostgreSQL index is declared from. Cosine, because the encoder's output is not unit-norm — see
    `EMBEDDING_POLICY["normalization"]`, which is a measurement — and the pgvector index is built
    with `vector_cosine_ops` for that reason.

    Payload indexes are created for every field the filter touches. They are the counterpart of the
    btree indexes `store.schema` declares for the same columns: without them Qdrant would still
    filter correctly and would do it by scanning payloads, and the latency published beside
    PostgreSQL's indexed predicate would be a comparison of an index against no index.
    """
    from qdrant_client import models  # noqa: PLC0415 - dev-only dependency, see module docs

    client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM, distance=models.Distance.COSINE, on_disk=False
        ),
        hnsw_config=models.HnswConfigDiff(
            m=int(HNSW_BUILD_PARAMETERS["m"]),
            ef_construct=int(HNSW_BUILD_PARAMETERS["ef_construction"]),
        ),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=_INDEXING_THRESHOLD_KB),
    )
    for field, schema in (
        ("variant_id", models.PayloadSchemaType.KEYWORD),
        ("language", models.PayloadSchemaType.KEYWORD),
        ("valid_from", models.PayloadSchemaType.INTEGER),
        ("valid_to", models.PayloadSchemaType.INTEGER),
        ("known_from", models.PayloadSchemaType.INTEGER),
        ("known_to", models.PayloadSchemaType.INTEGER),
        ("serial_first", models.PayloadSchemaType.INTEGER),
        ("serial_last", models.PayloadSchemaType.INTEGER),
    ):
        client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)

    batch = 512
    for start in range(0, len(stored), batch):
        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=[
                models.PointStruct(id=start + offset, vector=item.vector, payload=item.payload)
                for offset, item in enumerate(stored[start : start + batch])
            ],
        )
    # Qdrant point ids must be unsigned integers or UUIDs and this corpus keys chunks by strings
    # like `doc-xp400-hydraulic-pump-man-f-en-c010`, so the mapping is positional over `stored` —
    # which `_read_stored_chunks` ordered by chunk id. Reproducible across runs on purpose: kill
    # condition J compares two runs byte for byte, and an id assignment that depended on dictionary
    # iteration order would break that on its own.
    return {index: item.chunk_id for index, item in enumerate(stored)}


def _wait_for_index(client: QdrantClient) -> float:
    """Block until optimisation finishes, returning how long it took.

    Qdrant indexes asynchronously and answers queries the whole time. A workload started against a
    `yellow` collection would measure a half-built graph and would measure a different one on the
    next run, which is a latency figure with no reproducible meaning.
    """
    deadline = time.monotonic() + _INDEX_TIMEOUT_SECONDS
    started = time.monotonic()
    while time.monotonic() < deadline:
        info = client.get_collection(COLLECTION_NAME)
        if str(info.status) in {"green", "CollectionStatus.GREEN"}:
            return time.monotonic() - started
        time.sleep(0.5)
    raise QdrantUnreachableError(
        f"the Qdrant collection was still optimising after {_INDEX_TIMEOUT_SECONDS}s; a latency "
        "measured against a half-built index is not a measurement"
    )


def _search(
    client: QdrantClient,
    vector: Sequence[float],
    query_filter: qmodels.Filter,
    params: qmodels.SearchParams,
    *,
    limit: int,
    names: Mapping[int, str],
) -> tuple[str, ...]:
    """Top `limit` chunk ids from Qdrant, within the filter.

    `with_payload=False`: the point id is mapped back to a chunk id in Python from the assignment
    this module made. The alternative — asking Qdrant to return the payload — would put a JSON
    object per hit on the wire where the pgvector query returns an id and a float, and the latency
    difference would be a serialisation choice reported as a storage property.
    """
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=list(vector),
        query_filter=query_filter,
        search_params=params,
        limit=limit,
        with_payload=False,
        with_vectors=False,
    )
    return tuple(names[int(point.id)] for point in response.points)


def _admitted_in_qdrant(
    client: QdrantClient, query_filter: qmodels.Filter, names: Mapping[int, str]
) -> set[str]:
    """Every chunk id the payload filter admits, paginated.

    Counted by enumeration rather than by `count`, because the question the artifact answers is
    whether the two engines admit *the same chunks*, and two equal counts over different sets would
    read as agreement.
    """
    admitted: set[str] = set()
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query_filter,
            limit=1024,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        admitted.update(names[int(point.id)] for point in points)
        if offset is None:
            return admitted


# ------------------------------------------------------------------------------------- the measure


@dataclass(frozen=True)
class _QuestionResult:
    """What both backends returned for one question, and what each of them cost."""

    question_id: str
    relevant_ids: frozenset[str]
    pgvector_ids: tuple[str, ...]
    qdrant_ids: tuple[str, ...]
    pgvector_ms: float
    qdrant_ms: float
    admitted_postgres: int
    admitted_qdrant: int
    admitted_identical: bool


def _percentile(samples: Sequence[float], quantile: float) -> float:
    """Nearest-rank percentile: the ⌈q·n⌉-th smallest sample, no interpolation.

    No interpolation on purpose. An interpolated p95 invents a latency between two observations,
    and every number in this artifact is supposed to be one the host actually produced. The cost is
    that with a workload of n queries the p95 is a single order statistic — at n=30 it is the 29th
    of 30 — so the artifact publishes `queries_measured` next to it and a reader can see how much
    tail that figure is entitled to describe.
    """
    if not samples:
        raise ValueError("a percentile over no samples is not a measurement")
    ordered = sorted(samples)
    rank = max(1, min(len(ordered), math.ceil(quantile * len(ordered))))
    return ordered[rank - 1]


def _latency_block(samples: Sequence[float]) -> dict[str, Any]:
    return {
        "p50_ms": round(_percentile(samples, 0.50), 3),
        "p95_ms": round(_percentile(samples, 0.95), 3),
        "mean_ms": round(sum(samples) / len(samples), 3),
        "max_ms": round(max(samples), 3),
        "queries_measured": len(samples),
    }


def _round_trip_floor_ms(work: Callable[[], object], *, runs: int = 21) -> float:
    """Cheapest observed cost of the cheapest call this backend accepts — the transport floor.

    Published beside each backend's p50 because it changes how the whole table reads. The
    effectivity predicate admits a couple of dozen chunks for a question that names its variant, and
    ranking twenty-one vectors is microseconds of arithmetic. Whatever sits above this floor is
    small, so the number a reader sees is mostly the cost of crossing a container boundary on this
    host, and a latency table that did not say so would invite them to read it as the cost of an
    index. Neither figure is a statement about either engine's search speed at production scale.

    The **minimum** rather than the median, which this took first and which was wrong. A floor is a
    lower bound; a median over a sample taken while the host is busy is not one, and on a loaded
    laptop it came out *above* the p50 it was supposed to sit under — a figure that reads as a
    contradiction rather than as context. The minimum over twenty-one calls finds a quiet moment
    and states a bound that holds.
    """
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        work()
        samples.append((time.perf_counter() - started) * 1000.0)
    return min(samples)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _run_workload(
    session: Session,
    client: QdrantClient,
    workload: Sequence[ComparisonQuestion],
    names: Mapping[int, str],
    *,
    embedder: Embedder,
    limit: int,
) -> list[_QuestionResult]:
    """Put one query vector through both backends, question by question.

    **One vector per question, used twice.** Encoding separately for each backend would put the
    encoder's cost and its nondeterminism into the difference between them.

    The two calls are interleaved rather than run as two passes. Anything that drifts on this host
    while the workload runs — another container, a background index build, the laptop deciding to
    throttle — then lands on both backends at roughly the same rate instead of on whichever one
    happened to be measured during it.

    A warm-up query is issued against each backend before the loop and is not recorded. The first
    query pays for a cold buffer cache in PostgreSQL and a cold mmap in Qdrant, and publishing that
    as the p95 of either would be a measurement of a container start.
    """
    params = _search_params()
    first = workload[0]
    warm_vector = embedder.embed_query(first.query.text)
    warm_filter = candidate_filter(first.query)
    dense_search(session, warm_filter, warm_vector, limit=limit)
    _search(
        client,
        warm_vector,
        _effectivity_payload_filter(first.query),
        params,
        limit=limit,
        names=names,
    )

    results: list[_QuestionResult] = []
    for question in workload:
        vector = embedder.embed_query(question.query.text)
        filters = candidate_filter(question.query)
        payload_filter = _effectivity_payload_filter(question.query)

        started = time.perf_counter()
        pg_hits = dense_search(session, filters, vector, limit=limit)
        pgvector_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        qdrant_ids = _search(client, vector, payload_filter, params, limit=limit, names=names)
        qdrant_ms = (time.perf_counter() - started) * 1000.0

        # Outside both timings on purpose: this is the like-for-like audit, not the workload.
        admitted_pg = _admitted_in_postgres(session, filters)
        admitted_qd = _admitted_in_qdrant(client, payload_filter, names)

        results.append(
            _QuestionResult(
                question_id=question.question_id,
                relevant_ids=question.relevant_ids,
                pgvector_ids=tuple(hit.chunk_id for hit in pg_hits),
                qdrant_ids=qdrant_ids,
                pgvector_ms=pgvector_ms,
                qdrant_ms=qdrant_ms,
                admitted_postgres=len(admitted_pg),
                admitted_qdrant=len(admitted_qd),
                admitted_identical=admitted_pg == admitted_qd,
            )
        )
    return results


# ------------------------------------------------------------------------------------ the artifact


def _backend_block(
    name: str,
    retrieved: Sequence[tuple[str, ...]],
    latencies: Sequence[float],
    results: Sequence[_QuestionResult],
    *,
    note: str,
    filtering: str,
    index: Mapping[str, Any],
    round_trip_floor_ms: float,
) -> dict[str, Any]:
    """One column of the comparison. `recall_at_10` and `p95_ms` are what the kill test reads."""
    recalls = [
        recall_at_k(ids, result.relevant_ids, RETRIEVAL_K)
        for ids, result in zip(retrieved, results, strict=True)
    ]
    return {
        "backend": name,
        "recall_at_10": round(_mean(recalls), 4),
        **_latency_block(latencies),
        "round_trip_floor_ms": round(round_trip_floor_ms, 3),
        "round_trip_floor_note": (
            "cheapest observed cost of the cheapest call this backend accepts, on this host, in "
            "this run. Subtract it from the percentiles above to see what the search itself cost. "
            "A lower bound on transport, not a typical one: it is sampled after the workload, so a "
            "busy host inflates the percentiles more than it inflates this."
        ),
        "note": note,
        "filtering": filtering,
        "index": dict(index),
    }


def build_storage_comparison_artifact(
    session: Session,
    questions: Sequence[Mapping[str, Any]],
    *,
    embedder: Embedder,
    qdrant_endpoint: str | None = None,
    max_questions: int = 120,
) -> dict[str, Any]:
    """Produce `storage_comparison.json`: pgvector against a local Qdrant, one corpus, one workload.

    `k` is not a parameter. The artifact's key is spelled `recall_at_10` and a function that took
    `k=5` would write that number under a name claiming ten — the same argument
    `evaluation.metrics.by_language` makes, and the same reason it takes no `k` either.

    Raises `QdrantUnreachableError` when the container is not up, and `RuntimeError` when the store
    holds no vectors or the workload is empty. All three are the honest outcome: the kill test is
    supposed to fail on absent evidence, not pass on a fabricated column.
    """
    stored = _read_stored_chunks(session)
    if not stored:
        raise RuntimeError(
            "no chunk in the store carries an embedding, so there is nothing to load into a second "
            "backend; run `make index` before building this artifact"
        )
    workload = comparison_questions(questions, max_questions=max_questions)
    if not workload:
        raise RuntimeError(
            "no question in the input has a supporting passage, so neither backend can be scored; "
            "a comparison over unanswerable questions alone is not a comparison"
        )

    endpoint = qdrant_endpoint or qdrant_url()
    client = _connect(endpoint)
    versions = _versions(client)
    defaults = _qdrant_defaults(client)
    names = _load_collection(client, stored)
    index_seconds = _wait_for_index(client)
    collection = client.get_collection(COLLECTION_NAME)
    if collection.points_count != len(stored):
        raise RuntimeError(
            f"Qdrant holds {collection.points_count} points against {len(stored)} vectors read "
            "from PostgreSQL; the two backends would not be indexing the same corpus"
        )

    results = _run_workload(session, client, workload, names, embedder=embedder, limit=RETRIEVAL_K)
    postgres_floor = _round_trip_floor_ms(
        lambda: session.execute(sql_text("SELECT 1")).scalar_one()
    )
    # `info` is the root GET: a version string and nothing else. `get_collections` was the
    # obvious choice and is not a floor — the server enumerates a twelve-thousand-point
    # collection to answer it, and the number moved by a factor of five between runs.
    qdrant_floor = _round_trip_floor_ms(client.info)

    # The plan PostgreSQL actually chose for one of these queries. Reported because the comparison's
    # most interesting finding is about *which* index each engine consulted, and a p95 with no plan
    # beside it invites the reader to assume HNSW was walked.
    first = workload[0]
    plan = explain_dense(
        session,
        candidate_filter(first.query),
        embedder.embed_query(first.query.text),
        limit=RETRIEVAL_K,
    )

    disagreeing = [item.question_id for item in results if not item.admitted_identical]
    overlaps = [
        len(set(item.pgvector_ids) & set(item.qdrant_ids)) / RETRIEVAL_K
        for item in results
        if item.pgvector_ids
    ]
    vector_bytes = len(stored) * EMBEDDING_DIM * 4
    table_bytes = int(
        session.execute(sql_text("SELECT pg_total_relation_size('chunk')")).scalar_one()
    )
    hnsw_bytes = int(
        session.execute(
            sql_text("SELECT pg_relation_size(:index)"),
            {"index": str(HNSW_BUILD_PARAMETERS["index_name"])},
        ).scalar_one()
    )

    return {
        "topology": (
            "One ranking contract, two storage engines, both on this host: the project's pgvector "
            "column in the PostgreSQL container, and a Qdrant container loaded with the very same "
            "vectors read back out of that column. Same corpus, same query vectors, same "
            "effectivity predicate, same top-k."
        ),
        "deployment": "local container",
        "qdrant_cloud_tested": False,
        "managed_cloud_note": (
            "QDRANT CLOUD WAS NOT TESTED. No managed vector service was provisioned, no account "
            "was created, no credential was used and no request left this machine. Both backends "
            "are containers on one developer laptop, so the latency figures below contain no "
            "network and no multi-tenant contention, and neither of them is evidence about how "
            "either product behaves as a managed service. ADR-001 records this substitution "
            "deliberately: the blueprint says to delete the comparison rather than let it consume "
            "days, and a cloud account with its credentials and its network variance is exactly "
            "that risk."
        ),
        "corpus": {
            "chunks_indexed": len(stored),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "dimensions": EMBEDDING_DIM,
            "vectors_source": (
                "read back out of the PostgreSQL `chunk.embedding` column and uploaded to Qdrant "
                "unchanged; nothing was re-encoded for the second backend, so the two engines "
                "differ by storage and by nothing else"
            ),
        },
        "workload": {
            "questions": len(workload),
            "selection": (
                "the first answerable questions in corpus order, identical for both backends and "
                "reproducible across runs; questions with no supporting passage are excluded "
                "because recall is undefined for them"
            ),
            "variant_constrained": sum(1 for item in workload if item.query.variant_id is not None),
            "languages": sorted({item.query.language.value for item in workload}),
            "k": RETRIEVAL_K,
        },
        "backends": {
            "pgvector": _backend_block(
                "pgvector",
                [item.pgvector_ids for item in results],
                [item.pgvector_ms for item in results],
                results,
                note=(
                    "PostgreSQL 16 with pgvector, the shipped retrieval path: "
                    f"`store.dense_search`, cosine distance `{DISTANCE_OPERATOR}`, effectivity as "
                    "a SQL predicate in the same statement as the ordering."
                ),
                filtering=(
                    "A SQL `WHERE` clause in the ranking statement. The planner may satisfy it "
                    "with a btree index on the effectivity columns and never consult the vector "
                    "index — the plan measured here "
                    + (
                        "did name the HNSW index."
                        if plan.mentions_index
                        else "did NOT name the HNSW index, so this column's latency is a filtered "
                        "scan with a top-N sort rather than a graph traversal."
                    )
                ),
                index={
                    **HNSW_BUILD_PARAMETERS,
                    "ef_search": SESSION_SETTINGS["hnsw.ef_search"],
                    "iterative_scan": SESSION_SETTINGS["hnsw.iterative_scan"],
                    "plan_mentions_index": plan.mentions_index,
                    "plan_indexes_named": list(plan.index_names),
                },
                round_trip_floor_ms=postgres_floor,
            ),
            "qdrant": _backend_block(
                "qdrant",
                [item.qdrant_ids for item in results],
                [item.qdrant_ms for item in results],
                results,
                note=(
                    f"Qdrant {versions['server_version']} in a LOCAL container at {endpoint}, "
                    "loaded from the vectors PostgreSQL already held. Not Qdrant Cloud, not a "
                    "managed service, not over a network."
                ),
                filtering=(
                    "A payload filter evaluated against payload indexes while the vector index "
                    "is searched. Qdrant calls this filtered search and it is not the naive "
                    "rank-then-discard this project exists to avoid — but the graph is built "
                    "over every point regardless of the filter, and below "
                    "`full_scan_threshold` Qdrant abandons the graph and rescores the admitted "
                    "set exactly, which at this filter selectivity is what it did. That is the "
                    "substantive difference from PostgreSQL, where the predicate can change "
                    "which index the planner uses at all."
                ),
                index={
                    "engine": "qdrant",
                    **versions,
                    "distance": "cosine",
                    "m": HNSW_BUILD_PARAMETERS["m"],
                    "ef_construct": HNSW_BUILD_PARAMETERS["ef_construction"],
                    "hnsw_ef": int(SESSION_SETTINGS["hnsw.ef_search"]),
                    "indexing_threshold_kb_override": _INDEXING_THRESHOLD_KB,
                    "indexed_vectors": collection.indexed_vectors_count,
                    "points": collection.points_count,
                    "index_build_wait_seconds": round(index_seconds, 3),
                    "defaults_read_from_server": defaults,
                },
                round_trip_floor_ms=qdrant_floor,
            ),
        },
        "like_for_like": {
            "same_vectors": True,
            "same_query_vectors": True,
            "same_effectivity_filter": not disagreeing,
            "queries_with_identical_candidate_sets": len(results) - len(disagreeing),
            "queries_compared": len(results),
            "queries_disagreeing": disagreeing[:10],
            # Published because a candidate set of six is a different measurement from one of six
            # hundred, and a recall figure with no admitted-set size beside it hides which it was.
            "mean_candidates_admitted_postgres": round(
                _mean([float(item.admitted_postgres) for item in results]), 2
            ),
            "mean_candidates_admitted_qdrant": round(
                _mean([float(item.admitted_qdrant) for item in results]), 2
            ),
            "how_checked": (
                "for every query in the workload the admitted chunk ids were enumerated from both "
                "engines and the sets compared — not the counts, the ids. The effectivity rule is "
                "written twice, once as SQL and once as a Qdrant payload filter, and "
                "`store.effectivity` is explicit that two implementations of one rule need "
                "evidence they are one rule."
            ),
            "mean_top_10_overlap": round(_mean(overlaps), 4),
            "overlap_note": (
                "share of each backend's top-10 that the other also returned, averaged over the "
                "workload. 1.0 is what both engines taking an exact pass over an identical "
                "candidate set should produce, and is therefore a check on this file rather than a "
                "result about either product. Below 1.0 is not a defect either: it is what an "
                "approximate index on one side or a different tie-break on the other looks like, "
                "and each backend's `filtering` field says which path it took."
            ),
        },
        "cost_basis": "published list price arithmetic, not a measured bill",
        "cost": {
            "what_was_measured": {
                "vectors": len(stored),
                "raw_vector_bytes": vector_bytes,
                "raw_vector_bytes_formula": (
                    f"{len(stored)} vectors x {EMBEDDING_DIM} dims x 4 bytes"
                ),
                "postgres_chunk_table_total_bytes": table_bytes,
                "postgres_hnsw_index_bytes": hnsw_bytes,
                "postgres_note": (
                    "`pg_total_relation_size('chunk')` covers the passage text, the identifier "
                    "array and every index as well as the vectors, so it is not a vector-storage "
                    "figure and must not be quoted as one."
                ),
            },
            "formula": (
                "monthly_cost = stored_gib * price_per_gib_month + queries_per_month * "
                "price_per_query; both prices are inputs a reader supplies from the vendor's "
                "current published list price."
            ),
            "price_per_gib_month": None,
            "price_per_query": None,
            "prices_read_on": None,
            "monetary_estimate": None,
            "why_no_number": (
                "NO PRICE WAS READ AND NO BILL WAS MEASURED. No vendor price list was consulted "
                "during this build, so publishing a monetary figure would mean inventing the one "
                "input the arithmetic needs — the same defect as publishing an unmeasured latency, "
                "and a much harder one to catch. The quantities above are real and measured on "
                "this host; the formula is stated so a reader can finish the arithmetic against "
                "whatever the published list price is on the day they read it. Nothing here is an "
                "invoice, an estimate of one, or evidence about what either product costs."
            ),
        },
        "findings": [
            "Neither engine walked an approximate graph for these queries, and both decided "
            "that for themselves. PostgreSQL's planner prices a 384-dimension cosine distance "
            "at procost=1 and index-scanned the rows the effectivity filter admits; Qdrant "
            "kept its own `full_scan_threshold` default, which puts a filtered set of this "
            "size below the line at which it traverses the graph, so it rescored the admitted "
            "points exactly. A question naming its variant leaves around twenty of twelve "
            "thousand chunks, and an exact pass over twenty vectors is the right plan. The one "
            "Qdrant default this run overrides is `indexing_threshold`, lowered to "
            f"{_INDEXING_THRESHOLD_KB} KB so that an HNSW graph is built at all; every default "
            "is published under `defaults_read_from_server`, read off this server rather than "
            "recalled. At a production corpus two orders of magnitude larger, neither engine "
            "would be taking the exact path and this table would be measuring something else.",
            "The two filters are different mechanisms and the difference is the substantive "
            "result. In PostgreSQL the effectivity predicate and the distance ordering are one "
            "statement, so the predicate can change which index the planner uses at all. In Qdrant "
            "the predicate is a payload filter over a vector index built across every point; it is "
            "applied during the search rather than after it, but it cannot change what the vector "
            "index contains.",
            "A like-for-like comparison required writing the effectivity rule a second time, in a "
            "second filter language. That is the duplication `store.effectivity` exists to "
            "prevent, and it is the real cost of a second backend in a system whose correctness "
            "lives in a predicate. `like_for_like` reports the measured agreement rather than "
            "asserting it.",
            "Both backends are containers on one laptop. Nothing here measures a network, a "
            "managed service, a noisy neighbour, replication, or a failure mode of either product "
            "in production. Qdrant Cloud was not tested.",
            "Neither p50 in this table is mostly search. The effectivity predicate admits a few "
            "dozen chunks for a question that names its variant, so both engines rank a set small "
            "enough to be microseconds of arithmetic, and what the percentiles measure is "
            "overwhelmingly the cost of crossing a container boundary on a Docker Desktop host. "
            "`round_trip_floor_ms` is published beside each column so the reader can subtract it "
            "rather than take a transport number for an index number.",
            "The recall figures compare the two stores' DENSE retrieval only. They are not the "
            "project's headline recall: the shipped pipeline fuses a deterministic identifier "
            "lookup and a lexical stage with this one, and `evaluation.json` is where the number "
            "kill condition F is graded on lives.",
        ],
    }
