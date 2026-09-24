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
    "CURRENT_KNOWLEDGE_PREDICATE_SQL",
    "FILTER_STAGE",
    "KNOWN_AS_OF_PREDICATE_SQL",
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

#: The **knowledge-time** half, and the second axis that makes this bitemporal rather than
#: versioned. Same half-open shape as validity, for the same reason.
#:
#: Two forms, because "what do we currently believe" and "what did we believe on 4 March" are
#: different questions and only one of them is expressible as a date. Current knowledge is
#: `known_to IS NULL`; it is *not* the same as passing today's date, which would also admit a
#: correction that has already been superseded by a later one if the dates happened to line up.
KNOWN_AS_OF_PREDICATE_SQL: Final = (
    "{a}.known_from <= :known_as_of AND ({a}.known_to IS NULL OR :known_as_of < {a}.known_to)"
)
CURRENT_KNOWLEDGE_PREDICATE_SQL: Final = "{a}.known_to IS NULL"


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
    knowledge_constrained: bool = True
    #: True only when the query pinned a historical knowledge date, as opposed to taking current
    #: knowledge. `effectivity.json` reports it so a replay that never varied the second axis
    #: cannot be described as a bitemporal test.
    historical_knowledge: bool = False
    variant_constrained: bool = False
    serial_constrained: bool = False
    language_constrained: bool = False
    #: False only for the `hybrid_without_effectivity` baseline. It is recorded rather than
    #: implied so the artifact can state which arm produced a candidate set.
    effectivity_applied: bool = True

    def where(self) -> str:
        return self.sql


def candidate_filter(
    query: Query,
    *,
    alias: str = "chunk",
    languages: Iterable[Language] | None = None,
    apply_effectivity: bool = True,
) -> CandidateFilter:
    """Build the predicate that constrains the candidate set before any ranking.

    The default qualifier is the table's own name rather than a short alias, because every query
    that uses this fragment writes `FROM chunk` without aliasing. One fragment used by three
    statements has to agree with all three, and an alias only one of them declares is a runtime
    error waiting for the first query that takes a different branch.

    `languages` defaults to the query's own language. It is a parameter rather than a fixed rule
    because the multilingual evaluation needs to run one arm with the language constraint widened —
    that is the ablated cross-lingual fix — and an ablation that cannot be expressed through the
    production API is an ablation of something else.

    `apply_effectivity=False` builds the `hybrid_without_effectivity` baseline by **removing the
    predicate**, which is the only honest way to measure what it buys. An earlier version of that
    baseline simulated removal by moving `as_of` to 2099-12-31 and leaving the predicate in place.
    That is strictly *more* filtered, not less: at a date past every withdrawal only `valid_to IS
    NULL` rows survive, so the arm silently deleted the gold passage for every question whose
    answer had since been superseded, and the recall it reported measured a query issued at the
    wrong date. Language stays constrained — ADR-001 scopes this baseline to the temporal and
    variant constraints, and widening language too would make it a different ablation.
    """
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if apply_effectivity:
        clauses.append(AS_OF_PREDICATE_SQL.format(a=alias))
        params["as_of"] = query.as_of

        # The knowledge axis. Current knowledge is a predicate with no bound value, which is why
        # this is a branch rather than a parameter substitution.
        if query.known_as_of is None:
            clauses.append(CURRENT_KNOWLEDGE_PREDICATE_SQL.format(a=alias))
        else:
            clauses.append(KNOWN_AS_OF_PREDICATE_SQL.format(a=alias))
            params["known_as_of"] = query.known_as_of

    selected = list(languages) if languages is not None else [query.language]
    language_constrained = bool(selected)
    if language_constrained:
        clauses.append(f"{alias}.language = ANY(:languages)")
        params["languages"] = [language.value for language in selected]

    variant_constrained = apply_effectivity and query.variant_id is not None
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

    # A filter with no clauses at all would produce `WHERE ()`. `TRUE` keeps the callers' string
    # interpolation uniform rather than making three query builders each handle an empty fragment.
    return CandidateFilter(
        sql=" AND ".join(f"({clause})" for clause in clauses) if clauses else "TRUE",
        params=params,
        as_of_constrained=apply_effectivity,
        knowledge_constrained=apply_effectivity,
        historical_knowledge=apply_effectivity and query.known_as_of is not None,
        variant_constrained=variant_constrained,
        serial_constrained=serial_constrained,
        language_constrained=language_constrained,
        effectivity_applied=apply_effectivity,
    )


def applies_in_python(
    chunk: Chunk,
    query: Query,
    *,
    languages: Iterable[Language] | None = None,
    apply_effectivity: bool = True,
) -> bool:
    """The same rule, evaluated against a domain object.

    Never used by the retrieval path — the point of the sole-home skill is that the database does
    this. It exists so a test can walk the corpus and assert that the rows SQL returned are exactly
    the chunks the domain model says apply. A guarantee with one implementation is a guarantee
    nobody has checked.
    """
    selected = list(languages) if languages is not None else [query.language]
    if selected and chunk.language not in selected:
        return False
    if not apply_effectivity:
        return True
    if not chunk.visible_at(query.as_of, query.known_as_of):
        return False
    if query.variant_id is None:
        return True
    return chunk.effectivity.applies_to(query.variant_id, query.serial)
