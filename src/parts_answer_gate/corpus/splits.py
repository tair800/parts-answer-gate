"""The hold-out rule, exactly as ADR-001 fixes it, and nothing that could be tuned.

ADR-001:

    A product family is held out if ``blake2b(family_id, digest_size=8) % 100 < 34``.

The rule consults no score, no count and no balance target. It is a pure function of the family
identifier, which is why the split can be recomputed by anyone reading the corpus and compared with
the one the generator claims.

Splitting by **family** rather than by question row is the whole point. Two questions about the
same pump are supported by the same passages; a random row split puts one of them in training and
the other in the hold-out, and then measures how well the system remembers a document it was tuned
on. That number is always good and always meaningless.

The digest is read big-endian. The ADR does not name a byte order, so one is chosen here, stated in
the rule string that ships in the artifact, and never changed — a split that silently reverses is a
new experiment wearing the old one's numbers.
"""

from __future__ import annotations

import hashlib
from typing import Final

__all__ = [
    "DEVELOPMENT",
    "HOLDOUT",
    "HOLDOUT_THRESHOLD",
    "SPLIT_BY",
    "SPLIT_RULE",
    "family_split_score",
    "is_holdout",
    "split_of",
]

#: Must start with "blake2b(family_id" — `tests/test_kill_criteria.py::test_L` reads this string out
#: of the artifact and checks it against the ADR.
SPLIT_RULE: Final = "blake2b(family_id, digest_size=8) % 100 < 34  (digest read big-endian)"
SPLIT_BY: Final = "product_family"
HOLDOUT_THRESHOLD: Final = 34

HOLDOUT: Final = "holdout"
DEVELOPMENT: Final = "development"


def family_split_score(family_id: str) -> int:
    """The 0..99 score the rule thresholds. Exposed so a reviewer can recompute it by hand."""
    digest = hashlib.blake2b(family_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 100


def is_holdout(family_id: str) -> bool:
    return family_split_score(family_id) < HOLDOUT_THRESHOLD


def split_of(family_id: str) -> str:
    """The split name carried on every document, chunk and question record.

    Every artifact stores the split it computed rather than expecting a consumer to recompute it,
    so a disagreement between two modules shows up as a mismatch in the data instead of as two
    modules quietly disagreeing.
    """
    return HOLDOUT if is_holdout(family_id) else DEVELOPMENT
