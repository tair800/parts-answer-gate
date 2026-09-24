"""Part numbers, derived from what they identify rather than drawn from a counter.

Three properties are needed and they pull in different directions.

1. **Format.** Every issued number must match `domain.PART_NUMBER_PATTERN`, because the groundedness
   check (kill condition C) only treats a matching token as a part number. A corpus whose part
   numbers do not match the pattern would make C pass by giving it nothing to check.
2. **Order independence.** A number is a pure function of its key, so inserting a variant does not
   renumber anything else. A registry with re-rolling on collision would have been simpler to write
   and would have made the corpus order-dependent, which is exactly what ADR-001's determinism
   clause is about.
3. **Uniqueness.** Two different keys must not produce the same number, or "this part belongs to
   the XP-400" stops being true and the wrong-variant measurement silently softens.

(2) and (3) are reconciled by making the space large enough that a collision is not expected —
26^2 * 10^4 * 36^3, about 3.1e11, against roughly 400 issued numbers — and then *checking*
rather than assuming: `generate.py` asserts every issued number is unique and fails the build
loudly if not. An unproven assumption about a hash is still an assumption.
"""

from __future__ import annotations

import hashlib
import string

from parts_answer_gate.domain import PART_NUMBER_PATTERN

__all__ = ["is_well_formed", "malformed_variants", "near_miss_part_number", "part_number"]

_LETTERS = string.ascii_uppercase
_ALNUM = string.ascii_uppercase + string.digits
_SEPARATOR = "\x1f"


def part_number(*purpose: str) -> str:
    """A part number for one identified thing, e.g. ("spec-part", family, variant, topic, "g2")."""
    key = _SEPARATOR.join(purpose).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")

    prefix = _LETTERS[value % 26] + _LETTERS[(value // 26) % 26]
    value //= 676
    digits = value % 10_000
    value //= 10_000
    length = 1 + value % 3
    value //= 3
    tail = ""
    for _ in range(length):
        tail += _ALNUM[value % 36]
        value //= 36
    return f"{prefix}-{digits:04d}-{tail}"


def is_well_formed(candidate: str) -> bool:
    """Whether the corpus' own pattern accepts this token, start to end."""
    match = PART_NUMBER_PATTERN.fullmatch(candidate)
    return match is not None


def near_miss_part_number(real: str, issued: frozenset[str]) -> str | None:
    """A well-formed number one digit away from a real one and absent from the corpus.

    This is the `near_match_different_identifier` negative: it looks exactly like something the
    catalogue contains, which is what makes a retriever confident about it. Returns `None` when
    every single-digit neighbour happens to be a real part, so the caller can move on instead of
    fabricating one that is not actually a near miss.
    """
    prefix, digits, tail = real.split("-")
    for position in range(4):
        for shift in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            replacement = str((int(digits[position]) + shift) % 10)
            candidate = f"{prefix}-{digits[:position]}{replacement}{digits[position + 1 :]}-{tail}"
            if candidate not in issued and is_well_formed(candidate):
                return candidate
    return None


def malformed_variants(real: str) -> tuple[str, ...]:
    """Tokens the pattern rejects, built by breaking one rule of a real number each.

    Each is a mistake a technician actually makes — a digit dropped off a stamped plate, a number
    read off a lower-case label, a suffix that ran on. The check that they really are malformed is
    `is_well_formed`, the same function the rest of the corpus uses, so a loosened pattern would
    make these fail loudly rather than silently reclassify the questions as answerable.
    """
    prefix, digits, tail = real.split("-")
    return (
        f"{prefix}-{digits[:3]}-{tail}",  # one digit short
        f"{prefix}-{digits}-{tail}X4Z",  # suffix ran over three characters
        f"{prefix.lower()}-{digits}-{tail.lower()}",  # read off a lower-case label
        f"{prefix}{digits}{tail}",  # separators lost
    )
