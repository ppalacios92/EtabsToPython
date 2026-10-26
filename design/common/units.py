"""Presentation unit system, modeled after the ETABS unit codes (1..16).

Internal numerics across the design package are MPa, mm, kN, kN·m.
This module never converts those internals — it only provides multipliers
(``*_factor``) and the SINGLE shared value formatters (``fmt_length``,
``fmt_area``, ``fmt_force``, ``fmt_moment``, ``fmt_stress``) used by every
print/plot in the design package. All formatter output is ASCII-only so it
never crashes a cp1252 (Windows) console. ``set_units(...)`` changes ONLY
presentation; the internal numerics stay mm / MPa / kN / kN·m.

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

# Decimals per length unit for presentation (single homogeneous policy).
_LEN_DECIMALS: dict[str, int] = {"m": 3, "cm": 1, "mm": 0, "in": 2, "ft": 3}


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

    # ------------------------------------------------------------------ #
    # Presentation: ASCII labels + value formatters.
    # The SINGLE source of truth for how the whole design package renders
    # units in prints, tables and plot labels. Internal numerics NEVER
    # change (mm / MPa / kN / kN·m) — these only format for display. Every
    # token is ASCII so output never crashes a cp1252 (Windows) console.
    # Decimals (homogeneous): length m=3/cm=1/mm=0/in=2/ft=3, force=1,
    # moment=1, stress=2, area=0. Areas ALWAYS print in mm^2.
    # ------------------------------------------------------------------ #
    @property
    def length_label(self) -> str:
        return self.length

    @property
    def area_label(self) -> str:
        # Areas are ALWAYS shown in mm^2 (m^2/cm^2 collapse to a useless 0.0).
        return "mm^2"

    @property
    def force_label(self) -> str:
        return self.force

    @property
    def moment_label(self) -> str:
        # Dot separator (ASCII), e.g. "kN.m", "Tonf.m".
        return f"{self.force}.{self.length}"

    @property
    def stress_label(self) -> str:
        # force / length^2 (ASCII), e.g. "N/mm^2", "Tonf/cm^2".
        return f"{self.force}/{self.length}^2"

    def fmt_length(self, value_mm: float, *, unit: bool = True) -> str:
        dp = _LEN_DECIMALS.get(self.length, 0)
        s = f"{value_mm * self.length_factor:.{dp}f}"
        return f"{s} {self.length_label}" if unit else s

    def fmt_area(self, value_mm2: float, *, unit: bool = True) -> str:
        # Always mm^2, independent of self.units (locked policy).
        s = f"{value_mm2:.0f}"
        return f"{s} {self.area_label}" if unit else s

    def fmt_force(self, value_kN: float, *, unit: bool = True) -> str:
        s = f"{value_kN * self.force_factor:.1f}"
        return f"{s} {self.force_label}" if unit else s

    def fmt_moment(self, value_kNm: float, *, unit: bool = True) -> str:
        s = f"{value_kNm * self.moment_factor:.1f}"
        return f"{s} {self.moment_label}" if unit else s

    def fmt_stress(self, value_MPa: float, *, unit: bool = True) -> str:
        s = f"{value_MPa * self.stress_factor:.2f}"
        return f"{s} {self.stress_label}" if unit else s

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


def to_internal(
    kind: str, value: float, *, units: int | str | UnitSystem
) -> float:
    """Convert a value from a given unit system to the internal system.

    `kind` is one of 'force', 'length', 'moment', 'stress'.
    `units` is REQUIRED — an ETABS code (1..16), a name like 'Tonf_m_C', or a
    UnitSystem. There is no global fallback: every conversion is explicit and
    driven by the unit system of the object that owns the data.
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


# --------------------------------------------------------------------------- #
# Default unit system
# --------------------------------------------------------------------------- #
# Immutable fallback used by every element when no ``units=`` is supplied.
# There is NO mutable global active system: units live on each object, and
# govern both how its input is interpreted and how its output is presented.
# ``N_mm_C`` (code 9) is fully self-consistent (N/mm² = MPa, length mm), so
# omitting ``units=`` reads/reports forces in N, lengths in mm, stress in MPa.
DEFAULT_UNITS: UnitSystem = units_from(9)   # N_mm_C
