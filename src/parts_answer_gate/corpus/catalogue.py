"""The product catalogue and the specification topics, written out rather than rolled.

Two things are deliberately **constants rather than random draws**: which families exist, and which
variants each family has. They decide the hold-out split, and a split that moves when a seed moves
is not a committed split. Everything downstream of them — dates, values, part numbers, which
revision changed what — is derived from `rng.stream`.

The part of this file that carries the project's whole premise is `spec_options`. Variants of one
family get **disjoint bands** of values, not two draws from one range. If an XP-400 took 48 Nm and
an XP-400S could also take 48 Nm, then answering an XP-400S question from XP-400 evidence would
sometimes be right by accident, kill condition B would be measuring a coin flip, and the honest
headline — *a wrong-variant answer is genuinely wrong* — would not be true of the corpus. Disjoint
bands make every cross-variant answer wrong by construction.
"""

# The Turkish and Russian text below trips ruff's ambiguous-character rules on almost every
# line: `ı`, `İ`, `Н`, `б`, `а`, `р` and their neighbours are exactly the characters those
# rules warn about, and here they are the point rather than a typo. Suppressed for the file,
# because a per-string noqa on a parallel corpus is noise that hides a real one.
# ruff: noqa: RUF001, RUF003

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final

from parts_answer_gate.corpus.lang import Trilingual, russian_count
from parts_answer_gate.domain import Language

__all__ = [
    "FAMILIES",
    "TOPICS",
    "Family",
    "Topic",
    "Unit",
    "Variant",
    "render_value",
    "spec_options",
    "topic_by_key",
]

#: Serial numbers per variant block. Wide enough that a bulletin can carve a sub-range out of it
#: without touching the neighbouring variant's machines.
SERIAL_BLOCK: Final = 10_000


class Unit(enum.StrEnum):
    """What kind of value a topic states, which decides both rendering and the answer span."""

    TORQUE = "torque"
    LITRES = "litres"
    MONTHS = "months"
    BAR = "bar"
    AMPS = "amps"
    GRADE = "grade"
    PART = "part"


@dataclass(frozen=True)
class Topic:
    """One specification a service manual states, once per variant."""

    key: str
    unit: Unit
    #: Whether this topic's chunks carry a bounded serial range rather than applying to the fleet.
    #: Two of ten do, so a query without a serial still retrieves most of the corpus while the
    #: serial filter is genuinely exercised.
    serial_limited: bool
    #: The secondary number the prose mentions — running hours, a temperature, a flow. It gives the
    #: passage a second plausible figure, so a retriever cannot reach the right answer by finding
    #: the only number on the page.
    aux_choices: tuple[int, ...]


TOPICS: Final = (
    Topic("drive_coupling_torque", Unit.TORQUE, False, (50, 100, 150, 250)),
    Topic("filter_element", Unit.PART, False, (250, 500, 1000)),
    Topic("reservoir_capacity", Unit.LITRES, False, (55, 60, 65, 70)),
    Topic("seal_kit", Unit.PART, False, (16, 20, 25)),
    Topic("calibration_interval", Unit.MONTHS, False, (2, 3, 4)),
    Topic("control_fuse", Unit.AMPS, False, (12, 24, 48)),
    Topic("pump_shaft_bearing", Unit.PART, False, (90, 100, 110, 120)),
    Topic("relief_valve_setting", Unit.BAR, True, (40, 60, 80, 120)),
    Topic("gearbox_oil_grade", Unit.GRADE, False, (10, 15, 20, 25)),
    Topic("inspection_serial_note", Unit.PART, True, (2, 3, 4)),
)

#: Topics whose value is a measurement rather than a part number. Only these are used for the
#: planted contradictions, which keeps the `superseded_without_replacement_asked` negatives clean:
#: a part number that disappeared from the in-force revisions has then disappeared from the corpus
#: entirely, rather than surviving in a bulletin nobody thought about.
MEASURED_TOPIC_KEYS: Final = tuple(t.key for t in TOPICS if t.unit is not Unit.PART)


@dataclass(frozen=True)
class Variant:
    """A machine variant. The `serial_first` block is what a serial-limited bulletin carves up."""

    variant_id: str
    serial_first: int

    @property
    def serial_last(self) -> int:
        return self.serial_first + SERIAL_BLOCK - 1


@dataclass(frozen=True)
class Family:
    """A product family: the unit the hold-out split is taken on."""

    family_id: str
    product: Trilingual
    variants: tuple[Variant, ...]
    #: Whether a field bulletin, in force at the same time as the current manual revision,
    #: contradicts it. Planted on purpose: `REVIEW` is a declared gate outcome and a corpus with no
    #: conflicting evidence would never reach it.
    has_bulletin: bool


FAMILIES: Final = (
    Family(
        family_id="fam-xp400-hydraulic-pump",
        product=Trilingual(
            en="XP-400 Hydraulic Pump Unit",
            tr="XP-400 Hidrolik Pompa Ünitesi",
            ru="Гидравлический насосный агрегат XP-400",
        ),
        variants=(
            Variant("XP-400", 100_000),
            Variant("XP-400S", 110_000),
            Variant("XP-400H", 120_000),
        ),
        has_bulletin=True,
    ),
    Family(
        family_id="fam-mk7-gearbox",
        product=Trilingual(
            en="MK7-120 Gearbox Assembly",
            tr="MK7-120 Şanzıman Grubu",
            ru="Редукторный узел MK7-120",
        ),
        variants=(Variant("MK7-120", 200_000), Variant("MK7-120L", 210_000)),
        has_bulletin=False,
    ),
    Family(
        family_id="fam-tr9-compressor",
        product=Trilingual(
            en="TR9-55 Screw Compressor",
            tr="TR9-55 Vidalı Kompresör",
            ru="Винтовой компрессор TR9-55",
        ),
        variants=(Variant("TR9-55", 300_000), Variant("TR9-55X", 310_000)),
        has_bulletin=True,
    ),
    Family(
        family_id="fam-dl3-valve-block",
        product=Trilingual(
            en="DL3-210 Control Valve Block",
            tr="DL3-210 Kumanda Valf Bloğu",
            ru="Блок управляющих клапанов DL3-210",
        ),
        variants=(Variant("DL3-210", 400_000), Variant("DL3-215", 410_000)),
        has_bulletin=False,
    ),
    Family(
        family_id="fam-qs6-actuator",
        product=Trilingual(
            en="QS6-90 Linear Actuator",
            tr="QS6-90 Doğrusal Aktüatör",
            ru="Линейный привод QS6-90",
        ),
        variants=(
            Variant("QS6-90", 500_000),
            Variant("QS6-90R", 510_000),
            Variant("QS6-92", 520_000),
        ),
        has_bulletin=True,
    ),
    Family(
        family_id="fam-vb2-cooling-unit",
        product=Trilingual(
            en="VB2-300 Oil Cooling Unit",
            tr="VB2-300 Yağ Soğutma Ünitesi",
            ru="Блок масляного охлаждения VB2-300",
        ),
        variants=(Variant("VB2-300", 600_000), Variant("VB2-305", 610_000)),
        has_bulletin=False,
    ),
    Family(
        family_id="fam-nk5-winch-drive",
        product=Trilingual(
            en="NK5-140 Winch Drive",
            tr="NK5-140 Vinç Tahrik Ünitesi",
            ru="Приводной механизм лебёдки NK5-140",
        ),
        variants=(Variant("NK5-140", 700_000), Variant("NK5-140S", 710_000)),
        has_bulletin=True,
    ),
    Family(
        family_id="fam-hz8-filtration-skid",
        product=Trilingual(
            en="HZ8-70 Filtration Skid",
            tr="HZ8-70 Filtrasyon Ünitesi",
            ru="Фильтрационная установка HZ8-70",
        ),
        variants=(Variant("HZ8-70", 800_000), Variant("HZ8-72", 810_000)),
        has_bulletin=False,
    ),
    Family(
        family_id="fam-ct4-drive-controller",
        product=Trilingual(
            en="CT4-25 Drive Controller",
            tr="CT4-25 Sürücü Kontrol Ünitesi",
            ru="Контроллер привода CT4-25",
        ),
        variants=(Variant("CT4-25", 900_000), Variant("CT4-25E", 910_000)),
        has_bulletin=False,
    ),
)

_TOPICS_BY_KEY: Final = {topic.key: topic for topic in TOPICS}


def topic_by_key(key: str) -> Topic:
    return _TOPICS_BY_KEY[key]


def spec_options(unit: Unit, variant_index: int) -> tuple[int, ...]:
    """The values a variant may take for a unit — disjoint from its siblings' by construction.

    See the module docstring: overlapping bands would let a wrong-variant answer be accidentally
    correct, and kill condition B is worth nothing against a corpus where that can happen.
    """
    if unit is Unit.TORQUE:
        base = 40 + variant_index * 24
        return tuple(range(base, base + 21, 4))
    if unit is Unit.LITRES:
        base = 40 + variant_index * 30
        return tuple(range(base, base + 26, 5))
    if unit is Unit.MONTHS:
        return ((6, 9), (12, 18), (24, 36))[variant_index]
    if unit is Unit.BAR:
        base = 150 + variant_index * 60
        return tuple(range(base, base + 41, 10))
    if unit is Unit.AMPS:
        return ((4, 6), (10, 16), (20, 25))[variant_index]
    if unit is Unit.GRADE:
        return ((32, 46), (68, 100), (150, 220))[variant_index]
    raise ValueError(f"{unit} states a part number, which is not drawn from a numeric band")


#: How a measurement is written in each language. Units are localised where a technician would
#: localise them and left alone where they would not: an ISO viscosity grade and an ampere rating
#: are written the same way in all three, and translating them would make the corpus less
#: plausible rather than more multilingual.
_UNIT_TEXT: Final[dict[Unit, Trilingual]] = {
    Unit.TORQUE: Trilingual(en="{n} Nm", tr="{n} Nm", ru="{n} Н·м"),
    Unit.LITRES: Trilingual(en="{n} L", tr="{n} L", ru="{n} л"),
    Unit.BAR: Trilingual(en="{n} bar", tr="{n} bar", ru="{n} бар"),
    Unit.AMPS: Trilingual(en="{n} A", tr="{n} A", ru="{n} A"),
    Unit.GRADE: Trilingual(en="ISO VG {n}", tr="ISO VG {n}", ru="ISO VG {n}"),
}


def _months(number: int, language: Language) -> str:
    """Out of the table because Russian inflects the noun and a format string cannot."""
    if language is Language.EN:
        return f"{number} months"
    if language is Language.TR:
        return f"{number} ay"
    return russian_count(number, "месяц", "месяца", "месяцев")


def render_value(unit: Unit, number: int, language: Language) -> str:
    """The value as it appears in the prose, which is also the expected answer span."""
    if unit is Unit.PART:
        raise ValueError(f"{unit} has no numeric rendering; its value is a part number")
    if unit is Unit.MONTHS:
        return _months(number, language)
    return _UNIT_TEXT[unit].of(language).format(n=number)
