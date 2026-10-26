"""Tests for design.columns.slenderness — ACI 318-25 sign convention.

ACI 318-25 §6.6.4.5.3a (Cm) and §6.2.5 (slenderness threshold) use the
right-hand-rule convention where M1/M2 is **negative for single curvature**
and **positive for double (reverse) curvature**. These tests pin that
convention down:

    single curvature (M1/M2 -> -1)  => Cm -> 1.0  (worst case)
    double  curvature (M1/M2 -> +1)  => Cm reduced (relieved)
    non-sway threshold = 34 + 12*(M1/M2), capped at 40.
"""
from __future__ import annotations
import pytest

from design.columns.slenderness import is_slender, cm_factor, moment_magnifier, Pc


# --- Cm, §6.6.4.5.3a --------------------------------------------------- #
def test_cm_single_curvature_tends_to_one():
    # M1/M2 = -1 (equal end moments, single curvature) -> Cm = 0.6 + 0.4 = 1.0
    cm = cm_factor(M1=-100.0, M2=100.0)
    assert cm == pytest.approx(1.0)


def test_cm_double_curvature_is_lower():
    # M1/M2 = +1 (reverse curvature) -> Cm = 0.6 - 0.4 = 0.2, floored to 0.4
    cm_double = cm_factor(M1=100.0, M2=100.0)
    cm_single = cm_factor(M1=-100.0, M2=100.0)
    assert cm_double < cm_single
    assert cm_double == pytest.approx(0.4)   # 0.2 raw, clamped to the 0.4 floor


def test_cm_pure_single_vs_pure_double_ordering():
    # Sweep the ratio from single (-1) to double (+1): Cm must decrease.
    ratios = [-1.0, -0.5, 0.0, 0.5, 1.0]
    cms = [cm_factor(M1=r * 200.0, M2=200.0) for r in ratios]
    # Non-increasing
    assert all(a >= b - 1e-9 for a, b in zip(cms, cms[1:]))
    assert cms[0] == pytest.approx(1.0)      # single curvature
    assert cms[2] == pytest.approx(0.6)      # M1 = 0 -> Cm = 0.6


def test_cm_transverse_loaded_is_unity():
    assert cm_factor(M1=50.0, M2=100.0, transverse_loaded=True) == 1.0


# --- slenderness threshold, §6.2.5 ------------------------------------- #
def test_is_slender_single_curvature_threshold_22():
    # M1/M2 = -1 -> limit = 34 - 12 = 22.
    # klu/r = 21.9 (< 22) -> NOT slender; klu/r = 22.1 (> 22) -> slender.
    common = dict(M1=-100.0, M2=100.0)
    assert not is_slender(k=1.0, lu=2190.0, r=100.0, **common)
    assert is_slender(k=1.0, lu=2210.0, r=100.0, **common)


def test_is_slender_double_curvature_threshold_capped_40():
    # M1/M2 = +1 -> 34 + 12 = 46, capped at 40.
    common = dict(M1=100.0, M2=100.0)
    assert not is_slender(k=1.0, lu=3990.0, r=100.0, **common)   # 39.9 < 40
    assert is_slender(k=1.0, lu=4010.0, r=100.0, **common)       # 40.1 > 40


def test_is_slender_double_curvature_more_lenient_than_single():
    # Same geometry, klu/r = 30: slender in single curvature (limit 22),
    # NOT slender in double curvature (limit 46->40).
    geom = dict(k=1.0, lu=3000.0, r=100.0)   # klu/r = 30
    assert is_slender(**geom, M1=-100.0, M2=100.0)       # single
    assert not is_slender(**geom, M1=100.0, M2=100.0)    # double


def test_is_slender_sway_uses_22():
    assert not is_slender(k=1.0, lu=2190.0, r=100.0, M1=0.0, M2=100.0, sway=True)
    assert is_slender(k=1.0, lu=2210.0, r=100.0, M1=0.0, M2=100.0, sway=True)


# --- magnifier sanity -------------------------------------------------- #
def test_moment_magnifier_at_least_one():
    delta = moment_magnifier(Cm=0.4, Pu=100.0, Pc=10000.0)
    assert delta >= 1.0


def test_moment_magnifier_grows_with_axial():
    d_low = moment_magnifier(Cm=1.0, Pu=1000.0, Pc=10000.0)
    d_high = moment_magnifier(Cm=1.0, Pu=5000.0, Pc=10000.0)
    assert d_high > d_low


def test_pc_positive():
    assert Pc(EI=5.0e4, k=1.0, lu=3.0) > 0
