"""How many candidates the eligibility filter leaves, over the whole question set.

    python scripts/candidate_profile.py

This is the check that the benchmark is not structurally trivial, and it is the measurement the
first iteration never took. ADR-002: the first corpus left a **median of 9** eligible chunks after
effectivity filtering, against `top_k` = 10, with 84.3% of hold-out questions leaving fewer
candidates than k. When the candidate set is smaller than k, every arm that shares the filter
returns the same set and ranking cannot change recall@10 at all. Four arms scored exactly 1.0000
and the number meant nothing.

So this runs before any score is read, and what it measures is a property of the **corpus and the
filter** — not of the system's performance. Nothing here depends on how well retrieval works, which
is why taking it after the hold-out was frozen cannot have influenced the split. It writes
`artifacts/candidate_profile.json`, and `tests/test_kill_criteria.py` does not grade it; ADR-002
does, by being readable next to it.

Needs a seeded index.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sqlalchemy import text as sql_text  # noqa: E402

from parts_answer_gate.domain import Language, Query  # noqa: E402
from parts_answer_gate.holdout import load_frozen  # noqa: E402
from parts_answer_gate.store.effectivity import candidate_filter  # noqa: E402
from parts_answer_gate.store.engine import (  # noqa: E402
    build_engine,
    database_url,
    session_scope,
)
from parts_answer_gate.store.queries import candidate_count  # noqa: E402

CORPUS = REPO_ROOT / "data" / "generated"
ARTIFACTS = REPO_ROOT / "artifacts"
TOP_K = 10


def _percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile. Exact on a sorted list of integers, with no interpolation to argue
    about when the sample is small."""
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def _profile(sizes: list[int]) -> dict[str, Any]:
    below = sum(1 for n in sizes if n <= TOP_K)
    return {
        "questions": len(sizes),
        "min": min(sizes) if sizes else 0,
        "p50": int(statistics.median(sizes)) if sizes else 0,
        "p90": _percentile(sizes, 0.90),
        "p95": _percentile(sizes, 0.95),
        "max": max(sizes) if sizes else 0,
        "mean": round(statistics.fmean(sizes), 2) if sizes else 0.0,
        "at_or_below_top_k": below,
        "fraction_at_or_below_top_k": round(below / len(sizes), 4) if sizes else 0.0,
    }


def main() -> int:
    raw = json.loads((CORPUS / "questions.json").read_text(encoding="utf-8"))
    questions = raw["questions"] if isinstance(raw, dict) else raw
    frozen = load_frozen(ARTIFACTS)
    if frozen is None:
        print("artifacts/holdout.json is missing; freeze the hold-out first", file=sys.stderr)
        return 1
    holdout_ids = set(frozen.questions)

    engine = build_engine(database_url())
    sizes: dict[str, list[int]] = {"holdout": [], "development": []}
    by_language: dict[str, list[int]] = {}

    with session_scope(engine) as session:
        total = session.execute(sql_text("SELECT count(*) FROM chunk")).scalar_one()
        if not total:
            print("the chunk table is empty; run scripts/seed_index.py", file=sys.stderr)
            return 1

        for record in questions:
            query = Query(
                text=str(record["text"]),
                language=Language(record["language"]),
                as_of=date.fromisoformat(str(record["as_of"])),
                variant_id=record.get("variant_id") or None,
                serial=record.get("serial"),
                top_k=TOP_K,
            )
            count = candidate_count(session, candidate_filter(query))
            split = "holdout" if str(record["question_id"]) in holdout_ids else "development"
            sizes[split].append(count)
            by_language.setdefault(str(record["language"]), []).append(count)

    everything = sizes["holdout"] + sizes["development"]
    payload = {
        "measured_over": "every question, against the live index, before any score was read",
        "top_k": TOP_K,
        "why": (
            "when the eligible candidate set is no larger than top_k, ranking cannot change "
            "recall@k and every arm sharing the filter scores identically. The first benchmark had "
            "a median of 9 against top_k 10 and published four identical 1.0000s. ADR-002."
        ),
        "chunks_indexed": total,
        "all": _profile(everything),
        "holdout": _profile(sizes["holdout"]),
        "development": _profile(sizes["development"]),
        "by_language": {lang: _profile(values) for lang, values in sorted(by_language.items())},
        "distribution": dict(sorted(Counter(everything).items())),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "candidate_profile.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    for name in ("all", "holdout", "development"):
        p = payload[name]
        print(
            f"  {name:<12} n={p['questions']:<5} min {p['min']:<4} p50 {p['p50']:<4} "
            f"p90 {p['p90']:<4} p95 {p['p95']:<4} max {p['max']:<4} "
            f"<=top_k {p['fraction_at_or_below_top_k']:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
