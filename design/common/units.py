"""Presentation unit system, modeled after the ETABS unit codes (1..16).

Internal numerics across the design package are MPa, mm, kN, kN·m.
This module never converts those internals — it only provides multipliers
for *displaying* numbers in user-friendly units. Plots, reports and
summaries multiply the internal values by the factors below.

ETABS codes:
    1  lb_in_F      2  lb_ft_F       3  kip_in_F     4  kip_ft_F
    5  kN_mm_C      6  kN_m_C        7  kgf_mm_C     8  kgf_m_C
    9  N_mm_C      10  N_m_C        11  Tonf_mm_C   12  Tonf_m_C
   13  kN_cm_C     14  kgf_cm_C     15  N_cm_C      16  Tonf_cm_C
"""
from __future__ import annotations
from dataclasses import dataclass


# kN -> target force
_FORCE_FROM_kN: dict[str, float] = {
    "N":    1_000.0,
    "kN":   1.0,
    "kgf":  101.9716,
    "Tonf": 0.1019716,
    "lb":   224.8089,
    "kip":  0.2248089,
}

# mm -> target length
_LENGTH_FROM_mm: dict[str, float] = {
    "mm": 1.0,
    "cm": 0.1,
    "m":  0.001,
    "in": 0.0393701,
    "ft": 0.00328084,
}

# m -> target length (used to convert kN·m to target moment unit)
_LENGTH_FROM_m: dict[str, float] = {
    "mm": 1_000.0,
    "cm": 100.0,
    "m":  1.0,
    "in": 39.3701,
    "ft": 3.28084,
}

# ETABS code -> (force, length, temperature)
_ETABS_CODES: dict[int, tuple[str, str, str]] = {
    1:  ("lb",   "in", "F"),
    2:  ("lb",   "ft", "F"),
    3:  ("kip",  "in", "F"),
    4:  ("kip",  "ft", "F"),
    5:  ("kN",   "mm", "C"),
    6:  ("kN",   "m",  "C"),
    7:  ("kgf",  "mm", "C"),
    8:  ("kgf",  "m",  "C"),
    9:  ("N",    "mm", "C"),
    10: ("N",    "m",  "C"),
    11: ("Tonf", "mm", "C"),
    12: ("Tonf", "m",  "C"),
    13: ("kN",   "cm", "C"),
    14: ("kgf",  "cm", "C"),
    15: ("N",    "cm", "C"),
    16: ("Tonf", "cm", "C"),
}

_NAME_TO_CODE = {f"{f}_{L}_{T}": c for c, (f, L, T) in _ETABS_CODES.items()}


@dataclass(frozen=True, slots=True)
class UnitSystem:
    code: int
    force: str
    length: str
    temperature: str

    @property
    def name(self) -> str:
        return f"{self.force}_{self.length}_{self.temperature}"

    @property
    def force_factor(self) -> float:
        return _FORCE_FROM_kN[self.force]

    @property
    def length_factor(self) -> float:
        return _LENGTH_FROM_mm[self.length]

    @property
    def moment_factor(self) -> float:
        # internal moments are stored in kN·m
        return _FORCE_FROM_kN[self.force] * _LENGTH_FROM_m[self.length]

    @property
    def area_factor(self) -> float:
        return self.length_factor ** 2

    @property
    def stress_factor(self) -> float:
        # internal stresses (fc, fy) are stored in MPa = N/mm²
        # convert to target_force / target_length²
        return _FORCE_FROM_kN[self.force] / 1_000.0 / (self.length_factor ** 2)

    # ---- conversion FROM this system TO the internal one ----
    # internal = MPa, mm, kN, kN·m
    def to_internal_force(self, value: float) -> float:
        """`value` in self.force → kN."""
        return value / _FORCE_FROM_kN[self.force]

    def to_internal_length(self, value: float) -> float:
        """`value` in self.length → mm."""
        return value / _LENGTH_FROM_mm[self.length]

    def to_internal_moment(self, value: float) -> float:
        """`value` in self.force · self.length → kN·m."""
        return value / (_FORCE_FROM_kN[self.force] * _LENGTH_FROM_m[self.length])

    def to_internal_stress(self, value: float) -> float:
        """`value` in self.force / self.length² → MPa."""
        return value * 1_000.0 * (self.length_factor ** 2) / _FORCE_FROM_kN[self.force]


def to_internal(kind: str, value: float, *, units: int | str | UnitSystem) -> float:
    """Convert a value from a given unit system to the internal system.

    `kind` is one of 'force', 'length', 'moment', 'stress'.
    `units` is an ETABS code (1..16), a name like 'Tonf_m_C', or a UnitSystem.
    """
    u = units if isinstance(units, UnitSystem) else units_from(units)
    method = {
        "force":  u.to_internal_force,
        "length": u.to_internal_length,
        "moment": u.to_internal_moment,
        "stress": u.to_internal_stress,
    }.get(kind)
    if method is None:
        raise ValueError(f"Unknown kind {kind!r}; expected force/length/moment/stress.")
    return method(value)


def units_from(code_or_name: int | str) -> UnitSystem:
    """Build a UnitSystem from an ETABS code (1..16) or a name like 'Tonf_m_C'."""
    if isinstance(code_or_name, (int,)):
        if code_or_name not in _ETABS_CODES:
            raise ValueError(f"Unknown ETABS unit code {code_or_name!r}; expected 1..16.")
        f, L, T = _ETABS_CODES[code_or_name]
        return UnitSystem(code=code_or_name, force=f, length=L, temperature=T)
    name = str(code_or_name).strip()
    if name in _NAME_TO_CODE:
        code = _NAME_TO_CODE[name]
        f, L, T = _ETABS_CODES[code]
        return UnitSystem(code=code, force=f, length=L, temperature=T)
    raise ValueError(
        f"Unknown unit system {code_or_name!r}. Use a code 1..16 or a name like 'Tonf_m_C'."
    )


# Convenient default for the design package
DEFAULT_UNITS = units_from(5)   # kN_mm_C
