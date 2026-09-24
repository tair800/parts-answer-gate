"""Ask the deployed database what it is, and write the answer down.

    PAG_DATABASE_URL=... python scripts/live_pgvector.py

Writes `artifacts/live_pgvector.json`.

`artifacts/pgvector.json` is measured against the local `pgvector/pgvector:pg16` container, which is
the right place to grade kill condition K because it is the configuration CI can reproduce. It says
nothing about the database the public deployment actually queries. A deployment can look identical
from the outside while serving from a float array in a text column, a different extension version,
or no vector index at all, and the visitor would never know.

So this asks the server. Every fact below is read out of `pg_catalog` or out of the planner's own
output for the statement the retrieval path issues — not out of a migration file that was supposed
to have been applied.

**The credential is never printed.** The DSN is read from the environment, the host is replaced by
its provider and region, and the artifact this writes is committed. Check it with `git diff` before
pushing, as with everything else here.

**This does not change kill condition K.** K requires the query plan to use a *vector index scan*.
The effectivity predicate cuts the candidate set to a couple of dozen rows and PostgreSQL correctly
prefers a sequential scan over so few, so K fails whatever the infrastructure is. That the extension
is real, the column is a `vector`, the HNSW index exists and the distance operator is `<=>` is worth
recording and is a different claim from the one K makes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sqlalchemy import create_engine, text  # noqa: E402

from parts_answer_gate.store.engine import normalise_database_url  # noqa: E402
from parts_answer_gate.store.schema import HNSW_BUILD_PARAMETERS  # noqa: E402

ARTIFACT = REPO_ROOT / "artifacts" / "live_pgvector.json"

#: Enough to identify the provider and the region for a reader, and nothing that could be used to
#: connect. A host like `ep-something-123456.eu-central-1.aws.neon.tech` becomes
#: `<neon>.eu-central-1.aws`.
HOST_PATTERN = re.compile(r"@([^/:@]+)")
REGION_PATTERN = re.compile(r"\.([a-z]{2}-[a-z]+-\d)\.([a-z]+)\.")


def describe_host(dsn: str) -> str:
    """Name the provider and region, never the endpoint."""
    host_match = HOST_PATTERN.search(dsn)
    if not host_match:
        return "<not disclosed>"
    host = host_match.group(1)
    provider = "neon" if "neon.tech" in host else "<managed postgresql>"
    region = REGION_PATTERN.search(host)
    if region:
        return f"<{provider}>.{region.group(1)}.{region.group(2)}"
    return f"<{provider}>"


#: The statement the dense stage issues, reduced to the shape the planner sees. It is written here
#: rather than imported because `retrieval.dense` builds it against a `CandidateFilter` assembled
#: from a `Query`, and constructing one would drag the whole retrieval path into a script whose job
#: is to look at the server.
PROBE_SQL = """
EXPLAIN (ANALYZE, VERBOSE, FORMAT TEXT)
SELECT chunk_id, embedding <=> (SELECT embedding FROM chunk WHERE embedding IS NOT NULL LIMIT 1)
FROM chunk
WHERE embedding IS NOT NULL
ORDER BY 2
LIMIT 10
"""

FACTS: dict[str, str] = {
    "server_version": "SHOW server_version",
    "vector_extension_version": ("SELECT extversion FROM pg_extension WHERE extname = 'vector'"),
    "embedding_column_type": (
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'chunk'::regclass AND attname = 'embedding'"
    ),
    "hnsw_index_definition": (
        "SELECT indexdef FROM pg_indexes WHERE tablename = 'chunk' AND indexdef LIKE '%hnsw%'"
    ),
    "documents": "SELECT count(*) FROM document",
    "chunks": "SELECT count(*) FROM chunk",
    "chunks_with_an_embedding": "SELECT count(*) FROM chunk WHERE embedding IS NOT NULL",
    "superseded_documents": "SELECT count(*) FROM document WHERE superseded_by IS NOT NULL",
    "corrected_documents": "SELECT count(*) FROM document WHERE corrected_by IS NOT NULL",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("PAG_DATABASE_URL", ""))
    args = parser.parse_args(argv)

    if not args.database_url:
        print("set PAG_DATABASE_URL (it is never printed or written)", file=sys.stderr)
        return 1

    engine = create_engine(normalise_database_url(args.database_url), pool_pre_ping=True)
    observed: dict[str, Any] = {}
    with engine.connect() as connection:
        for name, statement in FACTS.items():
            observed[name] = connection.execute(text(statement)).scalar()
        plan = "\n".join(str(row[0]) for row in connection.execute(text(PROBE_SQL)).fetchall())

    index_definition = str(observed["hnsw_index_definition"] or "")
    checks = {
        "extension_is_installed": bool(observed["vector_extension_version"]),
        "column_is_a_vector_type": str(observed["embedding_column_type"]).startswith("vector"),
        "hnsw_index_exists": "USING hnsw" in index_definition,
        "index_uses_cosine_ops": "vector_cosine_ops" in index_definition,
        # Only the two build parameters, because the rest of `HNSW_BUILD_PARAMETERS` names the
        # index, the method and the opclass, which the three checks above already read directly out
        # of the definition. PostgreSQL stores reloptions as quoted strings, hence `m='16'`.
        "index_carries_the_declared_build_parameters": all(
            f"{key}='{HNSW_BUILD_PARAMETERS[key]}'" in index_definition
            for key in ("m", "ef_construction")
        ),
        "every_chunk_is_embedded": observed["chunks"] == observed["chunks_with_an_embedding"],
        "distance_operator_is_used": "<=>" in plan,
        "both_time_axes_have_rows": bool(observed["superseded_documents"])
        and bool(observed["corrected_documents"]),
    }

    payload = {
        "is_synthetic_corpus": True,
        "host": describe_host(args.database_url),
        "what_this_is": (
            "the deployed database describing itself, read from pg_catalog and from the planner's "
            "own output, so that the public deployment's vector retrieval is evidenced rather than "
            "asserted"
        ),
        "does_not_affect_kill_condition_K": (
            "K requires a vector INDEX SCAN in the plan. The effectivity predicate leaves a couple "
            "of dozen candidate rows and PostgreSQL prefers a sequential scan over so few, so K "
            "FAILS here exactly as it does locally. A real extension, a real vector column and a "
            "real HNSW index are a different claim from the one K makes, and neither rescues it"
        ),
        "declared_build_parameters": dict(HNSW_BUILD_PARAMETERS),
        "observed": observed,
        "checks": checks,
        "all_checks_hold": all(checks.values()),
        "probe_plan": plan,
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for name, value in observed.items():
        print(f"  {name}: {value}")
    print()
    for name, holds in checks.items():
        print(f"  {'ok  ' if holds else 'FAIL'} {name}")
    print(f"\nwrote {ARTIFACT.relative_to(REPO_ROOT)} (host recorded as {payload['host']})")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
