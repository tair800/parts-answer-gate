"""The candidate-set predicate. This module is the sole-home skill.

Every other retrieval system in this portfolio ranks first and filters afterwards, because that is
what a vector database makes easy. Post-filtering is wrong here in a way that does not show up in
the output: ask an index for ten, discard six that belong to another variant or to a withdrawn
revision, and you answer from four while the evaluation still records that ten were retrieved. The
recall you lost is invisible, and the four survivors look like a confident result.

So the predicate below goes into the `WHERE` clause of *both* ranking queries — the lexical
candidate fetch and the pgvector search — and a chunk outside the asked-for variant, serial range or
in-force window is never scored at all. `effectivity.json` can then say `before_ranking` truthfully,
because there is no code path where a score is computed for a row this predicate rejects.

The Python mirror `applies_in_python` is here so a test can assert the SQL and `domain.Effectivity`
agree on the same chunk. Two implementations of one rule need a test that they are one rule.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Final

from parts_answer_gate.domain import Chunk, Language, Query

__all__ = [
    "AS_OF_PREDICATE_SQL",
    "FILTER_STAGE",
    "CandidateFilter",
    "applies_in_python",
    "candidate_filter",
]

#: The value `effectivity.json` publishes. A constant rather than a literal typed into the artifact
#: builder, so the claim and the code that implements it are the same string.
FILTER_STAGE: Final = "before_ranking"

#: The bitemporal half, written once. Half-open on purpose — `as_of < valid_to` rather than `<=` —
#: so the day a revision is withdrawn belongs to its successor and not to both. An inclusive upper
#: bound makes exactly one day of the year return two revisions of the same procedure, which is the
#: kind of defect that is found in production by a warranty dispute.
AS_OF_PREDICATE_SQL: Final = (
    "{a}.valid_from <= :as_of AND ({a}.valid_to IS NULL OR :as_of < {a}.valid_to)"
)


@dataclass(frozen=True)
class CandidateFilter:
    """A `WHERE` fragment and its bound values, plus what it actually constrained.

    The booleans are not decoration: `effectivity.json` has to report whether a replay genuinely
    exercised the variant predicate, and a filter that constrained nothing because the question
    named no machine is a different result from one that constrained everything.
    """

    sql: str
    params: dict[str, Any] = field(default_factory=dict)
    as_of_constrained: bool = True
    variant_constrained: bool = False
    serial_constrained: bool = False
    language_constrained: bool = False

    def where(self) -> str:
        return self.sql


def candidate_filter(
    query: Query,
    *,
    alias: str = "c",
    languages: Iterable[Language] | None = None,
) -> CandidateFilter:
    """Build the predicate that constrains the candidate set before any ranking.

    `languages` defaults to the query's own language. It is a parameter rather than a fixed rule
    because the multilingual evaluation needs to run one arm with the language constraint widened —
    that is the ablated cross-lingual fix — and an ablation that cannot be expressed through the
    production API is an ablation of something else.
    """
    clauses = [AS_OF_PREDICATE_SQL.format(a=alias)]
    params: dict[str, Any] = {"as_of": query.as_of}

    selected = list(languages) if languages is not None else [query.language]
    language_constrained = bool(selected)
    if language_constrained:
        clauses.append(f"{alias}.language = ANY(:languages)")
        params["languages"] = [language.value for language in selected]

    variant_constrained = query.variant_id is not None
    serial_constrained = False
    if variant_constrained:
        clauses.append(f"{alias}.variant_id = :variant_id")
        params["variant_id"] = query.variant_id

        # The serial predicate mirrors `domain.SerialRange.covers` exactly, *including* its
        # asymmetry: a technician who named the machine but not its serial gets only passages that
        # apply to every serial. That is deliberate in the domain model — handing somebody a
        # procedure that applies to half the fleet is the wrong-answer mode this project exists to
        # prevent — and re-deciding it here would mean the SQL and the model disagree about which
        # chunks apply to a machine.
        serial_constrained = True
        if query.serial is None:
            clauses.append(f"({alias}.serial_first IS NULL AND {alias}.serial_last IS NULL)")
        else:
            clauses.append(
                f"({alias}.serial_first IS NULL OR :serial >= {alias}.serial_first) "
                f"AND ({alias}.serial_last IS NULL OR :serial <= {alias}.serial_last)"
            )
            params["serial"] = query.serial

    return CandidateFilter(
        sql=" AND ".join(f"({clause})" for clause in clauses),
        params=params,
        as_of_constrained=True,
        variant_constrained=variant_constrained,
        serial_constrained=serial_constrained,
        language_constrained=language_constrained,
    )


def applies_in_python(
    chunk: Chunk,
    query: Query,
    *,
    languages: Iterable[Language] | None = None,
) -> bool:
    """The same rule, evaluated against a domain object.

    Never used by the retrieval path — the point of the sole-home skill is that the database does
    this. It exists so a test can walk the corpus and assert that the rows SQL returned are exactly
    the chunks the domain model says apply. A guarantee with one implementation is a guarantee
    nobody has checked.
    """
    if not chunk.in_force_on(query.as_of):
        return False
    selected = list(languages) if languages is not None else [query.language]
    if selected and chunk.language not in selected:
        return False
    if query.variant_id is None:
        return True
    return chunk.effectivity.applies_to(query.variant_id, query.serial)
