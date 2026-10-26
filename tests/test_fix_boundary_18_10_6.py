"""Regression tests for special boundary-element triggers, ACI 318-25 §18.10.6.2.

Fixes covered:
  - Eq. 18.10.6.2a: the 1.5 factor on the design displacement was missing.
  - The 0.005 floor on delta_u/hwcs is a FLOOR, not a skip.
  - delta_u <= 0 (no displacement demand) -> displacement method N/A.
  - Eq. 18.10.6.2b drift-capacity check (was absent).
  - §18.10.6.2(b)(ii) width check b >= sqrt(c*lw/40) (was absent).
"""
from __future__ import annotations
from math import sqrt
import pytest

from design.walls.boundary import (
    boundary_element_required_displacement,
    wall_drift_capacity,
    boundary_drift_capacity_ok,
    boundary_width_ok,
)


# --- §18.10.6.2(a) trigger with the 1.5 factor -------------------------- #
def test_trigger_includes_1p5_factor():
    lw, hw = 4000.0, 20000.0
    drift = 0.01
    du = drift * hw
    # threshold c = lw / (600 * 1.5 * drift)
    c_thr = lw / (600.0 * 1.5 * drift)
    assert c_thr == pytest.approx(444.44, abs=0.1)
    assert boundary_element_required_displacement(c=c_thr + 1, lw=lw, delta_u=du, hw=hw)
    assert not boundary_element_required_displacement(c=c_thr - 1, lw=lw, delta_u=du, hw=hw)


def test_trigger_no_displacement_demand_is_false():
    # Without delta_u the displacement method is not applicable.
    assert not boundary_element_required_displacement(
        c=5000.0, lw=4000.0, delta_u=0.0, hw=20000.0)


def test_trigger_drift_floor_0p005():
    # A very small drift is floored at 0.005, so a large c still trips it.
    lw, hw = 4000.0, 20000.0
    du = 0.001 * hw                      # below the floor
    c_thr_floor = lw / (600.0 * 1.5 * 0.005)   # = 888.9 mm
    assert boundary_element_required_displacement(
        c=c_thr_floor + 1, lw=lw, delta_u=du, hw=hw)


# --- Eq. 18.10.6.2b drift capacity ------------------------------------- #
def test_drift_capacity_formula():
    lw, b, c = 4000.0, 300.0, 800.0
    Ve, Acv, fc = 2000.0, 4000.0 * 300.0, 28.0
    v_norm = (Ve * 1000.0) / (0.66 * sqrt(fc) * Acv)
    expected = (4.0 - (1.0 / 50.0) * (lw / b) * (c / b) - v_norm) / 100.0
    assert wall_drift_capacity(c=c, lw=lw, b=b, Ve=Ve, Acv=Acv, fc=fc) == pytest.approx(expected)


def test_drift_capacity_floored_at_0p015():
    # Degenerate inputs floor at 0.015.
    assert wall_drift_capacity(c=0.0, lw=0.0, b=0.0, Ve=0.0, Acv=0.0, fc=0.0) == 0.015


def test_drift_capacity_check_requires_1p5x_demand():
    lw, hw, b = 4000.0, 20000.0, 300.0
    Ve, Acv, fc = 2000.0, 4000.0 * 300.0, 28.0
    # demand 0.01 -> required 0.015; capacity ~0.0281 -> OK
    ok, cap, req = boundary_drift_capacity_ok(
        c=800.0, lw=lw, b=b, Ve=Ve, Acv=Acv, fc=fc, delta_u=0.01 * hw, hw=hw)
    assert req == pytest.approx(1.5 * 0.01)
    assert ok and cap >= req
    # demand 0.02 -> required 0.03 > capacity -> FAIL
    ok2, cap2, req2 = boundary_drift_capacity_ok(
        c=800.0, lw=lw, b=b, Ve=Ve, Acv=Acv, fc=fc, delta_u=0.02 * hw, hw=hw)
    assert not ok2


# --- §18.10.6.2(b)(ii) width check ------------------------------------- #
def test_width_check():
    ok, b_req = boundary_width_ok(b=300.0, c=800.0, lw=4000.0)
    assert b_req == pytest.approx(sqrt(800.0 * 4000.0 / 40.0))
    assert ok                    # 300 >= 282.8
    assert not boundary_width_ok(b=250.0, c=800.0, lw=4000.0)[0]
