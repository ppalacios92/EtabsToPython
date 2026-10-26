"""Regression tests for the integral wall design (ACI 318-25 §18.10.6).

Before the fix the wall was not "integral":
  - capacity() / check() carried no boundary-element information;
  - the BE longitudinal was a fixed 1% target, not grown from a baseline;
  - the full wall Pu was passed to the BE column (no proration);
  - the drift-capacity / width checks (§18.10.6.2b) were absent.

These tests pin the integral behaviour.
"""
from __future__ import annotations
import pytest

from design.walls.wall import Wall, WallDemands
from design.common.materials import Concrete, Steel


def _wall_with_be(units=None):
    return Wall.rectangular(
        lw=4000, tw=300, hw=24000, lu=3000,
        concrete=Concrete(fc=28), steel=Steel(fy=420, grade=60),
        n_y_web=12, db_web=12,
        be_top_length=600, be_bot_length=600, n_y_be=5, db_be=25,
        cover=40, seismic=True, units=units,
    )


def test_capacity_evaluates_boundary_elements():
    w = _wall_with_be()
    cap = w.capacity()
    assert cap.has_boundary_elements
    # Each BE is a real Column -> its capacity is reported
    assert cap.be_top_capacity is not None
    assert cap.be_bot_capacity is not None


def test_capacity_reports_mpr_in_out_and_at_zero():
    w = _wall_with_be()
    cap = w.capacity()
    assert cap.Mpr_in_plane is not None and cap.Mpr_in_plane > 0
    assert cap.Mpr_in_plane_at_zero is not None and cap.Mpr_in_plane_at_zero > 0
    # out-of-plane Mpr is much smaller than in-plane (weak axis) and finite
    assert cap.Mpr_out_of_plane is not None
    assert 0 < cap.Mpr_out_of_plane < cap.Mpr_in_plane_at_zero
    # at zero axial the moment is below the moment at the (compressive) gravity Pu
    assert cap.Mpr_in_plane_at_zero < cap.Mpr_in_plane


def test_check_verifies_boundary_element_and_drift():
    w = _wall_with_be()
    dem = WallDemands(Pu=3000, Mu=8000, Vu=1500, delta_u=0.008 * 24000)
    chk = w.check(dem)
    assert chk.be_required_disp                 # displacement governs here
    assert chk.be_present
    # drift-capacity check ran and is a number compared against 1.5x demand
    assert chk.drift_capacity is not None
    assert chk.drift_capacity_required == pytest.approx(1.5 * 0.008)
    # width check ran
    assert chk.be_width_required > 0
    # the longitudinal BE rho floor is the column floor 0.01 (not 6 sqrt/fy)
    assert chk.be_long_rho_min == pytest.approx(0.01)


def test_be_axial_is_prorated_not_full_wall_pu():
    w = _wall_with_be()
    dem = WallDemands(Pu=6000, Mu=8000, Vu=1500)
    be_dem = w._be_column_demands(dem, "top")
    # the prorated BE axial must be a fraction of the wall axial plus the
    # moment couple — never the full 6000 kN
    assert be_dem is not None
    assert be_dem.Pu < dem.Pu
    # gravity share alone: Pu * A_be/Ag
    s = w.section
    share = (s.be_thickness_top * s.be_length_top) / s.gross_area()
    assert be_dem.Pu >= dem.Pu * share          # share + moment couple


def test_design_grows_be_and_exposes_ash():
    w = _wall_with_be()
    dem = WallDemands(Pu=3000, Mu=8000, Vu=1500, delta_u=0.008 * 24000)
    res = w.design(dem)
    assert res.converged
    # a BE is proposed with finite geometry
    assert res.envelope.be_length_proposed > 0
    # BE confinement Ash is surfaced (delegated to the BE Column)
    bep = res.envelope.be_top_proposal
    assert bep is not None
    assert bep.ash_required_x > 0 and bep.ash_required_y > 0
