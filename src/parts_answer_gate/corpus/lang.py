"""The three-language carrier, and the one grammatical rule that cannot be faked by templating.

Every human-readable string in this corpus exists three times and the three are the *same fact*.
`Trilingual` holds them together so a template cannot be added in English and forgotten in Turkish
— a missing translation is a missing attribute, not a silently shorter corpus.

`russian_count` exists because Russian agrees the noun with the number, and "24 месяцев" is the
tell that a multilingual benchmark was machine-assembled by string substitution. The multilingual
result this project publishes is a measurement over synthetic text, and it is worth nothing if the
non-English text is visibly not the language it claims to be.

The text is written in the native scripts, not transliterated, and the generated files are UTF-8.
"""

# The Turkish and Russian text below trips ruff's ambiguous-character rules on almost every
# line: `ı`, `İ`, `Н`, `б`, `а`, `р` and their neighbours are exactly the characters those
# rules warn about, and here they are the point rather than a typo. Suppressed for the file,
# because a per-string noqa on a parallel corpus is noise that hides a real one.
# ruff: noqa: RUF003

from __future__ import annotations

from dataclasses import dataclass

from parts_answer_gate.domain import Language

__all__ = ["Trilingual", "russian_count"]


@dataclass(frozen=True)
class Trilingual:
    """One fact in three languages."""

    en: str
    tr: str
    ru: str

    def of(self, language: Language) -> str:
        # An if-chain rather than a dict literal: this is called once per chunk per language, and
        # rebuilding a mapping for every call is the kind of waste that only shows up at 2,000
        # chunks times three languages times two determinism runs.
        if language is Language.EN:
            return self.en
        if language is Language.TR:
            return self.tr
        return self.ru


def russian_count(number: int, one: str, few: str, many: str) -> str:
    """`number` with the correct Russian case of the noun.

    The teens are irregular in a way the last-digit rule gets wrong — 11 through 14 take the
    genitive plural despite ending in 1..4 — so they are tested first.
    """
    if 11 <= number % 100 <= 14:
        return f"{number} {many}"
    last = number % 10
    if last == 1:
        return f"{number} {one}"
    if 2 <= last <= 4:
        return f"{number} {few}"
    return f"{number} {many}"
