"""Phase 0 — the single shared presentation formatter on UnitSystem.

These tests pin the homogeneous display policy:
  - length decimals: m=3, cm=1, mm=0, in=2, ft=3
  - area ALWAYS in mm^2 (ASCII), 0 decimals
  - force=1 dp, moment=1 dp, stress=2 dp
  - moment separator is a dot ("kN.m"), area is "mm^2" (never the ² glyph)
  - EVERY output is ASCII (never crashes a cp1252 Windows console)
  - default unit system is N_mm_C
"""
import pytest

from design.common.units import units_from, DEFAULT_UNITS, _FORCE_FROM_kN


def _assert_ascii(s: str) -> None:
    s.encode("ascii")  # raises UnicodeEncodeError if any non-ASCII char


def test_default_is_N_mm_C():
    assert DEFAULT_UNITS.name == "N_mm_C"


def test_length_decimals_and_no_collapse():
    u_m = units_from(6)   # kN_m_C  -> length in m, 3 decimals
    assert u_m.fmt_length(400) == "0.400 m"
    assert u_m.fmt_length(300) == "0.300 m"   # 300 mm must NOT collapse to "0 m"
    assert u_m.fmt_length(800) == "0.800 m"   # the old "0x1 m" bug is gone
    assert u_m.fmt_length(400, unit=False) == "0.400"

    u_mm = units_from(5)  # kN_mm_C
    assert u_mm.fmt_length(400) == "400 mm"

    u_cm = units_from(13)  # kN_cm_C
    assert u_cm.fmt_length(400) == "40.0 cm"


def test_area_always_mm2():
    for code in (5, 6, 9, 12, 16):
        u = units_from(code)
        assert u.fmt_area(1963) == "1963 mm^2"     # never m^2, never 0.0
        assert u.area_label == "mm^2"
        assert u.fmt_area(1963, unit=False) == "1963"


def test_force():
    assert units_from(6).fmt_force(488.6) == "488.6 kN"
    tonf = units_from(12)
    assert tonf.fmt_force(488.6) == f"{488.6 * _FORCE_FROM_kN['Tonf']:.1f} Tonf"


def test_moment_dot_separator():
    s = units_from(12).fmt_moment(332.2)        # Tonf_m_C
    assert s.endswith(" Tonf.m")
    assert "·" not in s                      # no middle dot
    assert units_from(6).moment_label == "kN.m"


def test_stress_uses_factor():
    # N_mm_C: stress_factor == 1 -> identity, label is force/length^2 = N/mm^2
    assert units_from(9).fmt_stress(28) == "28.00 N/mm^2"
    u = units_from(16)  # Tonf_cm_C
    assert u.fmt_stress(28, unit=False) == f"{28 * u.stress_factor:.2f}"
    assert u.stress_label == "Tonf/cm^2"


def test_every_output_is_ascii_for_all_codes():
    for code in range(1, 17):
        u = units_from(code)
        outputs = [
            u.fmt_length(400), u.fmt_area(1963), u.fmt_force(488.6),
            u.fmt_moment(332.2), u.fmt_stress(28),
            u.length_label, u.area_label, u.force_label,
            u.moment_label, u.stress_label,
        ]
        for s in outputs:
            _assert_ascii(s)
