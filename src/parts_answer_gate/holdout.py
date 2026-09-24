"""The hold-out rule, in one place, and the freeze that makes it a hold-out rather than a split.

ADR-001 fixes the rule before anything was built:

    a product family is held out iff blake2b(family_id, digest_size=8) % 100 < 34

Three properties matter, and each is here for a reason somebody has been caught by:

**It partitions by product family, not by question.** Two questions about the same pump share the
same passages, so a random question split tests whether the system memorised a document it was
already tuned on. Splitting by family moves the documents too.

**It consults no seed and no score.** There is nothing to re-draw. A split that took a seed could be
re-rolled until it flattered a result, and nobody reading the number afterwards could tell.

**It lives in exactly one module.** The corpus generator, the freezer and the evaluation all import
it. Projects 5 and 6 both shipped a rule implemented twice that came to disagree; the second copy is
always the one nobody edits.

The freeze is separate from the rule and is the part that has teeth. :func:`freeze` materialises the
membership into `artifacts/holdout.json` — every family, document and question id, with a digest —
and refuses to overwrite an existing freeze whose content would change. Once that file is committed,
the hold-out is a fact in git history with a timestamp, and any later score can be checked against
it by a reader who does not trust the author.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "HOLDOUT_RULE",
    "HOLDOUT_SHARE",
    "FrozenHoldout",
    "HoldoutDriftError",
    "freeze",
    "is_held_out",
    "load_frozen",
    "split_of",
]

#: Out of a hundred. Chosen before any corpus existed, from what a hold-out needs to be: large
#: enough that a recall figure over it is not decided by three questions, small enough to leave the
#: development set able to show a fault. Never tuned — moving it after a score would be re-drawing
#: the hold-out with extra steps.
HOLDOUT_SHARE: Final = 34
_BUCKETS: Final = 100

HOLDOUT_RULE: Final = (
    "a product family is held out iff blake2b(family_id, digest_size=8) % 100 < 34; "
    "the assignment uses no seed and no score, so it cannot be re-drawn"
)

#: The filename is part of the contract: `tests/test_kill_criteria.py` and the evaluation both look
#: for it here, and a freeze written somewhere else is a freeze nobody checks.
FREEZE_FILENAME: Final = "holdout.json"


class HoldoutDriftError(RuntimeError):
    """A freeze already exists and the corpus would now produce a different one.

    This is the error that protects the whole evaluation. It fires when the corpus changed in a way
    that moves a family, a document or a question across the line — which may be entirely innocent,
    and is never something to resolve by deleting the file.
    """


def is_held_out(family_id: str) -> bool:
    """Whether a product family is in the hold-out. The rule, and nothing else."""
    digest = hashlib.blake2b(family_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _BUCKETS < HOLDOUT_SHARE


def split_of(family_id: str) -> str:
    return "holdout" if is_held_out(family_id) else "development"


@dataclass(frozen=True)
class FrozenHoldout:
    """What was held out, enumerated rather than described.

    A rule plus a corpus implies a membership, but only if both are unchanged. Enumerating the ids
    means a reader can check a later score against the actual set without regenerating anything.
    """

    families: tuple[str, ...]
    documents: tuple[str, ...]
    questions: tuple[str, ...]
    digest: str
    rule: str = HOLDOUT_RULE

    def as_json(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "share_percent": HOLDOUT_SHARE,
            "split_by": "product_family",
            "families": list(self.families),
            "documents": list(self.documents),
            "questions": list(self.questions),
            "counts": {
                "families": len(self.families),
                "documents": len(self.documents),
                "questions": len(self.questions),
            },
            "digest": self.digest,
            "what_this_file_is": (
                "the materialised membership of the hold-out, written and committed before any "
                "score over it existed. Its position in git history is the evidence that the "
                "split was not chosen after seeing a result."
            ),
        }


def _digest_of(families: Iterable[str], documents: Iterable[str], questions: Iterable[str]) -> str:
    payload = json.dumps(
        {
            "families": sorted(families),
            "documents": sorted(documents),
            "questions": sorted(questions),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute(
    documents: Iterable[Mapping[str, Any]], questions: Iterable[Mapping[str, Any]]
) -> FrozenHoldout:
    """Apply the rule to a corpus and enumerate what falls on the hold-out side.

    Families are collected from **questions as well as documents**, and the difference is not
    cosmetic. ADR-001's unanswerable set includes questions about a product family that does not
    exist in the corpus at all — that is the whole point of the `different_product_family` kind —
    and such a family owns no documents. Enumerating from documents alone therefore dropped every
    one of those questions out of the hold-out silently, whichever side of the line the rule put
    them on, removing one of the seven unanswerable kinds from the graded set entirely.

    The rule itself is unchanged and is still a pure function of the identifier. This is the
    enumeration being corrected, not the draw being re-taken: the same `is_held_out` decides the
    same families, and the questions it was already deciding for are now actually collected.
    """
    documents = list(documents)
    questions = list(questions)

    named = {str(d["family_id"]) for d in documents} | {str(q["family_id"]) for q in questions}
    families = sorted(family for family in named if is_held_out(family))
    held_families = set(families)
    held_documents = sorted(
        str(d["document_id"]) for d in documents if str(d["family_id"]) in held_families
    )
    held_questions = sorted(
        str(q["question_id"]) for q in questions if str(q["family_id"]) in held_families
    )

    return FrozenHoldout(
        families=tuple(families),
        documents=tuple(held_documents),
        questions=tuple(held_questions),
        digest=_digest_of(families, held_documents, held_questions),
    )


def verify_partition(
    documents: Iterable[Mapping[str, Any]],
    questions: Iterable[Mapping[str, Any]],
    chunk_to_document: Mapping[str, str] | None = None,
) -> list[str]:
    """Every way the split could leak, checked rather than assumed. Returns the problems found.

    Kill condition L is a zero, and the only way to earn it is to look: a document whose family is
    held out but which also appears under a development family is invisible in every aggregate
    number the evaluation produces.
    """
    documents = list(documents)
    questions = list(questions)
    chunk_to_document = chunk_to_document or {}
    problems: list[str] = []

    splits_per_document: dict[str, set[str]] = {}
    for document in documents:
        splits_per_document.setdefault(str(document["document_id"]), set()).add(
            split_of(str(document["family_id"]))
        )
    for document_id, splits in sorted(splits_per_document.items()):
        if len(splits) > 1:
            problems.append(f"document {document_id} appears in both splits: {sorted(splits)}")

    # Supporting evidence is named per **chunk** by the corpus, and a caller may hand either form.
    # Both are read.
    #
    # This read only `supporting_document_ids` — a key the corpus does not use — so the loop
    # iterated nothing and the leak check passed without inspecting a single question. A guard that
    # examines an empty sequence is indistinguishable from a guard that found nothing wrong, which
    # is the worst property a guard can have.
    document_by_id = {str(d["document_id"]): d for d in documents}
    for question in questions:
        question_split = split_of(str(question["family_id"]))
        named = [
            *(question.get("supporting_document_ids") or []),
            *(
                chunk_to_document.get(str(chunk_id), "")
                for chunk_id in (question.get("supporting_chunk_ids") or [])
            ),
        ]
        for document_id in (str(item) for item in named if item):
            supporting = document_by_id.get(document_id)
            if supporting is None:
                problems.append(
                    f"question {question['question_id']} cites unknown document {document_id}"
                )
                continue
            if split_of(str(supporting["family_id"])) != question_split:
                problems.append(
                    f"question {question['question_id']} is {question_split} but its supporting "
                    f"document {document_id} is {split_of(str(supporting['family_id']))}"
                )

    return problems


def load_frozen(artifacts_dir: Path) -> FrozenHoldout | None:
    path = artifacts_dir / FREEZE_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FrozenHoldout(
        families=tuple(payload["families"]),
        documents=tuple(payload["documents"]),
        questions=tuple(payload["questions"]),
        digest=payload["digest"],
        rule=payload["rule"],
    )


def freeze(
    artifacts_dir: Path,
    documents: Iterable[Mapping[str, Any]],
    questions: Iterable[Mapping[str, Any]],
    *,
    chunk_to_document: Mapping[str, str] | None = None,
    allow_refreeze: bool = False,
) -> FrozenHoldout:
    """Materialise the membership, or confirm it has not moved.

    On a first run this writes the file. On every later run it recomputes and compares, and raises
    :class:`HoldoutDriftError` if the membership changed — because a hold-out that silently follows
    the corpus is not frozen, and the number measured over it means whatever the last edit decided.

    `allow_refreeze` exists for the case where the corpus legitimately grew and the hold-out is
    being re-drawn deliberately. It is not a way to make an inconvenient drift go away: re-freezing
    invalidates every score taken against the old membership, and ADR-001 requires those to be
    re-run rather than carried over.
    """
    documents = list(documents)
    questions = list(questions)

    # The chunk map matters. `verify_partition`'s reference check resolves a question's supporting
    # chunks to the documents that own them, and without the map it has nothing to resolve — so
    # freezing ran a guard that inspected none of the references it exists to inspect, and reported
    # clean because it had looked at nothing. The caller has the map; it is now passed.
    leaks = verify_partition(documents, questions, chunk_to_document)
    if leaks:
        raise HoldoutDriftError(
            "the split does not partition cleanly, so freezing it would freeze a leak:\n  "
            + "\n  ".join(leaks[:10])
        )

    computed = compute(documents, questions)
    existing = load_frozen(artifacts_dir)

    if existing is not None and existing.digest != computed.digest and not allow_refreeze:
        raise HoldoutDriftError(
            f"artifacts/{FREEZE_FILENAME} already holds a different membership.\n"
            f"  frozen:   {len(existing.families)} families, {len(existing.documents)} documents, "
            f"{len(existing.questions)} questions, digest {existing.digest[:16]}\n"
            f"  computed: {len(computed.families)} families, {len(computed.documents)} documents, "
            f"{len(computed.questions)} questions, digest {computed.digest[:16]}\n"
            "The corpus moved a family, a document or a question across the line. That may be "
            "entirely innocent, and it still invalidates every score taken against the old "
            "membership. Re-run them, or pass allow_refreeze and say so in DECISIONS.md."
        )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / FREEZE_FILENAME).write_text(
        json.dumps(computed.as_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return computed
