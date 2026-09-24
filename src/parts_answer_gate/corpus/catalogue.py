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
    #: Pressures that are genuinely low — case drain, crankcase — which share `bar` as a written
    #: unit with `BAR` but must not share its 150-310 band.
    BAR_LOW = "bar_low"
    AMPS = "amps"
    VOLTS = "volts"
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
    Topic("drive_coupling_torque", Unit.TORQUE, False, (150, 250, 400, 600)),
    Topic("filter_element", Unit.PART, False, (250, 500, 1000)),
    Topic("reservoir_capacity", Unit.LITRES, False, (82, 88, 94)),
    Topic("seal_kit", Unit.PART, False, (16, 20, 25)),
    Topic("calibration_interval", Unit.MONTHS, False, (2, 3, 4)),
    Topic("control_fuse", Unit.AMPS, False, (230, 400, 690)),
    Topic("pump_shaft_bearing", Unit.PART, False, (90, 100, 110, 120)),
    Topic("relief_valve_setting", Unit.BAR, True, (40, 60, 80, 120)),
    Topic("gearbox_oil_grade", Unit.GRADE, False, (10, 15, 20, 25)),
    Topic("inspection_serial_note", Unit.PART, True, (2, 3, 4)),
    # --- the confusable clusters ------------------------------------------------
    # Each of the following shares vocabulary with one of the topics above. A corpus
    # whose topics use disjoint vocabulary lets a lexical retriever separate them
    # without understanding anything, which is what made the first benchmark trivial:
    # BM25 alone reached recall@10 of 1.0000 and there was nothing left for ranking
    # to do. Three torque topics, three filter topics and three sealing topics mean a
    # question naming one of them has near neighbours it must be ranked above.
    Topic("flange_bolt_torque", Unit.TORQUE, False, (45, 55, 65)),
    Topic("mounting_foot_torque", Unit.TORQUE, False, (150, 250, 500)),
    Topic("suction_strainer", Unit.PART, False, (250, 500, 1000)),
    Topic("breather_filter", Unit.PART, False, (500, 750, 1500)),
    Topic("shaft_seal_kit", Unit.PART, False, (85, 95, 105)),
    Topic("valve_seal_kit", Unit.PART, False, (12, 18, 25)),
    Topic("lubrication_interval", Unit.MONTHS, False, (35, 40, 45)),
    Topic("inspection_interval", Unit.MONTHS, False, (5, 7, 10)),
    Topic("heater_fuse", Unit.AMPS, False, (230, 400, 690)),
    Topic("sensor_supply_voltage", Unit.VOLTS, False, (100, 150, 250)),
    Topic("idler_bearing", Unit.PART, False, (4000, 6000, 8000)),
    Topic("pilot_pressure_setting", Unit.BAR, False, (12, 18, 25)),
    Topic("case_drain_limit", Unit.BAR_LOW, False, (80, 90, 100)),
    Topic("coolant_capacity", Unit.LITRES, False, (82, 88, 94)),
    Topic("retrofit_bracket_note", Unit.PART, True, (2, 3, 4)),
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
    Family(
        family_id="fam-ax7-accumulator-station",
        product=Trilingual(
            en="AX7-160 Hydraulic Accumulator Station",
            tr="AX7-160 Hidrolik Akümülatör İstasyonu",
            ru="Гидроаккумуляторная станция AX7-160",
        ),
        variants=(
            Variant("AX7-160", 1000000),
            Variant("AX7-160P", 1010000),
            Variant("AX7-165", 1020000),
        ),
        has_bulletin=True,
    ),
    Family(
        family_id="fam-rm2-piston-motor",
        product=Trilingual(
            en="RM2-45 Radial Piston Motor",
            tr="RM2-45 Radyal Pistonlu Motor",
            ru="Радиально-поршневой гидромотор RM2-45",
        ),
        variants=(
            Variant("RM2-45", 1100000),
            Variant("RM2-45D", 1110000),
        ),
        has_bulletin=False,
    ),
    Family(
        family_id="fam-bd6-disc-brake",
        product=Trilingual(
            en="BD6-320 Wet Multi-Disc Brake",
            tr="BD6-320 Yağ Banyolu Çok Diskli Fren",
            ru="Многодисковый тормоз в масляной ванне BD6-320",
        ),
        variants=(
            Variant("BD6-320", 1200000),
            Variant("BD6-325", 1210000),
        ),
        has_bulletin=True,
    ),
    Family(
        family_id="fam-sw3-rotary-union",
        product=Trilingual(
            en="SW3-80 Hydraulic Rotary Union",
            tr="SW3-80 Hidrolik Döner Rakor",
            ru="Гидравлическое вращающееся соединение SW3-80",
        ),
        variants=(
            Variant("SW3-80", 1300000),
            Variant("SW3-80T", 1310000),
        ),
        has_bulletin=False,
    ),
    Family(
        family_id="fam-lg1-lubrication-unit",
        product=Trilingual(
            en="LG1-18 Automatic Lubrication Unit",
            tr="LG1-18 Otomatik Yağlama Ünitesi",
            ru="Автоматическая станция смазки LG1-18",
        ),
        variants=(
            Variant("LG1-18", 1400000),
            Variant("LG1-24", 1410000),
        ),
        has_bulletin=False,
    ),
)

_TOPICS_BY_KEY: Final = {topic.key: topic for topic in TOPICS}


def topic_by_key(key: str) -> Topic:
    return _TOPICS_BY_KEY[key]


#: `unit -> (first value for variant 0, gap between variants, step within a band, values per band)`.
#:
#: One table rather than seven branches, so a new unit is a row and not a code path. Every band is
#: `base = start + variant_index * gap`, and `gap` is always wider than the band the step and count
#: produce — which is what makes the bands **disjoint**. See the module docstring: overlapping
#: bands would let a wrong-variant answer be accidentally correct, and kill condition B is worth
#: nothing against a corpus where that can happen.
_BANDS: Final[dict[Unit, tuple[int, int, int, int]]] = {
    Unit.TORQUE: (40, 24, 4, 6),
    Unit.LITRES: (40, 30, 5, 6),
    Unit.MONTHS: (6, 9, 3, 2),
    Unit.BAR: (150, 60, 10, 5),
    Unit.BAR_LOW: (2, 3, 1, 2),
    Unit.AMPS: (4, 8, 2, 2),
    Unit.VOLTS: (24, 24, 6, 3),
    Unit.GRADE: (32, 46, 14, 2),
}


def spec_options(unit: Unit, variant_index: int) -> tuple[int, ...]:
    """The values a variant may take for a unit — disjoint from its siblings' by construction."""
    if unit not in _BANDS:
        raise ValueError(f"{unit} states a part number, which is not drawn from a numeric band")
    start, gap, step, count = _BANDS[unit]
    base = start + variant_index * gap
    return tuple(base + step * offset for offset in range(count))


#: How a measurement is written in each language. Units are localised where a technician would
#: localise them and left alone where they would not: an ISO viscosity grade and an ampere rating
#: are written the same way in all three, and translating them would make the corpus less
#: plausible rather than more multilingual.
_UNIT_TEXT: Final[dict[Unit, Trilingual]] = {
    Unit.TORQUE: Trilingual(en="{n} Nm", tr="{n} Nm", ru="{n} Н·м"),
    Unit.LITRES: Trilingual(en="{n} L", tr="{n} L", ru="{n} л"),
    Unit.BAR: Trilingual(en="{n} bar", tr="{n} bar", ru="{n} бар"),
    Unit.BAR_LOW: Trilingual(en="{n} bar", tr="{n} bar", ru="{n} бар"),
    Unit.AMPS: Trilingual(en="{n} A", tr="{n} A", ru="{n} A"),
    Unit.VOLTS: Trilingual(en="{n} V", tr="{n} V", ru="{n} В"),
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
