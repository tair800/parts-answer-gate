"""Both bitemporal axes, measured against the live database, into `artifacts/bitemporal.json`.

    python scripts/bitemporal_evidence.py

`tests/test_bitemporal.py` proves the behaviour; this publishes it. The difference matters: a claim
whose only witness is a test a reader has to clone, install and run is weaker than a number sitting
in an artifact next to the others, and this project's second axis is the thing about it that is
least common and easiest to assert without evidence.

Three claims, measured separately:

- **valid time** — at a date inside a revision's window the system returns that revision, and at a
  date past its withdrawal it does not;
- **knowledge time** — pinning `known_as_of` returns what was believed then, which is a different
  document from what is believed now wherever a correction has landed;
- **a later correction does not rewrite history** — the correction is already in the table, and the
  historical query still returns the belief it replaced.

Everything is read through `candidate_filter`, the same predicate the ranking statements
interpolate, so what is measured is the thing the service uses rather than a re-implementation.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sqlalchemy import text as sql_text  # noqa: E402

from parts_answer_gate.domain import Language, Query  # noqa: E402
from parts_answer_gate.store.effectivity import candidate_filter  # noqa: E402
from parts_answer_gate.store.engine import (  # noqa: E402
    build_engine,
    database_url,
    session_scope,
)
from parts_answer_gate.store.queries import fetch_candidates  # noqa: E402

CORPUS = REPO_ROOT / "data" / "generated"
ARTIFACTS = REPO_ROOT / "artifacts"


def _records(name: str, key: str) -> list[dict[str, Any]]:
    raw = json.loads((CORPUS / name).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = raw[key] if isinstance(raw, dict) else raw
    return records


def main() -> int:
    if not (CORPUS / "documents.json").is_file():
        print("no corpus; run scripts/generate_corpus.py", file=sys.stderr)
        return 1

    documents = _records("documents.json", "documents")
    chunks = _records("chunks.json", "chunks")

    corrected = sorted(
        (
            d
            for d in documents
            if d.get("corrected_by") and d.get("known_to") and d["language"] == Language.EN.value
        ),
        key=lambda d: str(d["document_id"]),
    )
    superseded = sorted(
        (
            d
            for d in documents
            if d.get("superseded_by") and d.get("valid_to") and d["language"] == Language.EN.value
        ),
        key=lambda d: str(d["document_id"]),
    )
    if not corrected or not superseded:
        print("the corpus contains no corrections or no supersessions", file=sys.stderr)
        return 1

    engine = build_engine(database_url())
    with session_scope(engine) as session:
        if not session.execute(sql_text("SELECT count(*) FROM chunk")).scalar_one():
            print("the chunk table is empty; run scripts/seed_index.py", file=sys.stderr)
            return 1

        def documents_for(variant: str, as_of: date, known_as_of: date | None) -> set[str]:
            query = Query(
                text="specification",
                language=Language.EN,
                as_of=as_of,
                known_as_of=known_as_of,
                variant_id=variant,
                top_k=50,
            )
            return {c.document_id for c in fetch_candidates(session, candidate_filter(query))}

        # ---- valid time ---------------------------------------------------------------------
        target = superseded[0]
        variant = next(
            str(c["effectivity"]["variant_id"])
            for c in chunks
            if str(c["document_id"]) == str(target["document_id"])
        )
        valid_from = date.fromisoformat(str(target["valid_from"]))
        valid_to = date.fromisoformat(str(target["valid_to"]))
        inside = valid_from + (valid_to - valid_from) // 2
        after = valid_to + timedelta(days=1)

        in_window = documents_for(variant, inside, None)
        past_window = documents_for(variant, after, None)

        valid_time = {
            "claim": "an as-of query returns the revision in force for the machine at that date",
            "family_id": str(target["family_id"]),
            "variant_id": variant,
            "revision": str(target["revision"]),
            "valid_from": valid_from.isoformat(),
            "valid_to": valid_to.isoformat(),
            "asked_inside_the_window": inside.isoformat(),
            "asked_after_the_withdrawal": after.isoformat(),
            "returned_inside_the_window": str(target["document_id"]) in in_window,
            "returned_after_the_withdrawal": str(target["document_id"]) in past_window,
            "holds": str(target["document_id"]) in in_window
            and str(target["document_id"]) not in past_window,
        }

        # ---- knowledge time -----------------------------------------------------------------
        original = corrected[0]
        correction_id = str(original["corrected_by"])
        c_variant = next(
            str(c["effectivity"]["variant_id"])
            for c in chunks
            if str(c["document_id"]) == str(original["document_id"])
        )
        c_from = date.fromisoformat(str(original["valid_from"]))
        c_to = date.fromisoformat(str(original["valid_to"]))
        known_to = date.fromisoformat(str(original["known_to"]))
        as_of = c_from + (c_to - c_from) // 2
        family = str(original["family_id"])

        believed_then = documents_for(c_variant, as_of, known_to - timedelta(days=1))
        believed_now = documents_for(c_variant, as_of, None)
        on_the_day = documents_for(c_variant, as_of, known_to)

        knowledge_time = {
            "claim": (
                "the same machine at the same as-of date returns what was believed then when a "
                "knowledge date is pinned, and what is believed now when it is not"
            ),
            "family_id": family,
            "variant_id": c_variant,
            "valid_from": c_from.isoformat(),
            "valid_to": c_to.isoformat(),
            "correction_known_from": known_to.isoformat(),
            "as_of": as_of.isoformat(),
            "original_document": str(original["document_id"]),
            "correction_document": correction_id,
            "at_historical_knowledge": {
                "known_as_of": (known_to - timedelta(days=1)).isoformat(),
                "original_returned": str(original["document_id"]) in believed_then,
                "correction_returned": correction_id in believed_then,
            },
            "at_current_knowledge": {
                "known_as_of": None,
                "original_returned": str(original["document_id"]) in believed_now,
                "correction_returned": correction_id in believed_now,
            },
            "on_the_day_the_correction_landed": {
                "known_as_of": known_to.isoformat(),
                "exactly_one_of_the_pair": len(
                    on_the_day & {str(original["document_id"]), correction_id}
                )
                == 1,
                "correction_returned": correction_id in on_the_day,
                "note": "the knowledge interval is half-open, so the correction applies from its "
                "own known_from",
            },
            "holds": (
                str(original["document_id"]) in believed_then
                and correction_id not in believed_then
                and correction_id in believed_now
                and str(original["document_id"]) not in believed_now
            ),
        }

        # ---- a later correction does not rewrite history --------------------------------------
        original_text, correction_text = session.execute(
            sql_text(
                "SELECT"
                " (SELECT string_agg(text, '' ORDER BY chunk_id) FROM chunk WHERE document_id=:a),"
                " (SELECT string_agg(text, '' ORDER BY chunk_id) FROM chunk WHERE document_id=:b)"
            ),
            {"a": str(original["document_id"]), "b": correction_id},
        ).one()

        no_rewrite = {
            "claim": (
                "the correction is already stored, and the historical query still returns the "
                "belief it replaced -- which an UPDATE would have destroyed"
            ),
            "correction_is_present_in_the_database": correction_text is not None,
            "historical_query_returns_the_original": str(original["document_id"]) in believed_then,
            "the_two_documents_differ": original_text != correction_text,
            "holds": (
                correction_text is not None
                and str(original["document_id"]) in believed_then
                and original_text != correction_text
            ),
        }

    payload = {
        "measured_against": "the live database, through store.effectivity.candidate_filter -- the "
        "same predicate the lexical and pgvector ranking statements interpolate",
        "corrections_in_corpus": len(corrected),
        "supersessions_in_corpus": len(superseded),
        "valid_time": valid_time,
        "knowledge_time": knowledge_time,
        "a_later_correction_does_not_rewrite_history": no_rewrite,
        "all_three_hold": valid_time["holds"] and knowledge_time["holds"] and no_rewrite["holds"],
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "bitemporal.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )

    for name in ("valid_time", "knowledge_time", "a_later_correction_does_not_rewrite_history"):
        block = payload[name]
        assert isinstance(block, dict)
        print(f"  {'HOLDS  ' if block['holds'] else 'FAILS  '} {name}")
    return 0 if payload["all_three_hold"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
