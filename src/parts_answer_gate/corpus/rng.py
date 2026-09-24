"""Seeded randomness, keyed by what the stream is *for*.

Every random choice in this corpus comes from a generator seeded by a hash of the thing being
decided — the family, the variant, the topic, the revision — and never from a counter and never
from the module-level `random` functions.

The difference matters the first time the catalogue changes. With a counter, inserting one variant
shifts every subsequent draw, so a corpus regenerated after a one-line edit differs everywhere, the
diff is unreadable and the previous evaluation numbers are no longer comparable to the new ones.
Keying by identity means an inserted variant draws its own values and nothing else moves.

`random.Random` is the Mersenne Twister, which is fixed by the language rather than by the
platform, so the same seed gives the same stream on any machine running the same CPython. That is
what kill condition J needs; cryptographic quality is irrelevant here and would be slower.
"""

from __future__ import annotations

import hashlib
import random

__all__ = ["CORPUS_SEED", "GENERATOR_VERSION", "derive_seed", "stream"]

#: The committed constant every stream descends from. Changing it regenerates the whole corpus and
#: invalidates every measurement taken against the old one, so it is written here and nowhere else.
CORPUS_SEED = "parts-answer-gate/ADR-001/corpus/v1"

#: Bumped whenever the *shape* of the output changes — a new field, a new topic, a different chunk
#: boundary. Recorded in every artifact so a stored number can be traced to the corpus that
#: produced it rather than to whatever the generator emits today.
GENERATOR_VERSION = "corpus-1.0.0"

#: Separator that cannot occur in any identifier used here, so ("ab", "c") and ("a", "bc") cannot
#: hash to the same key. A plain join on "-" would collide, and a collision between two streams is
#: a silent duplicate value in the corpus.
_SEPARATOR = "\x1f"


def derive_seed(*purpose: str) -> int:
    """A 64-bit seed from the purpose of the stream."""
    key = _SEPARATOR.join((CORPUS_SEED, *purpose)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def stream(*purpose: str) -> random.Random:
    """An owned generator for one decision.

    Callers pass what the decision is about, most specific last, e.g.
    ``stream("spec-number", family_id, variant_id, topic_key, "g2")``.
    """
    return random.Random(derive_seed(*purpose))
