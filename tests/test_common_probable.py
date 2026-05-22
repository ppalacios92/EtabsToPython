"""Tests for design.common.probable — Mpr (probable strength).

We build a small rectangular column with a known reinforcement and
verify that ``mpr_envelope`` returns a Mpr larger than the regular
phi*Mn (since fy_pr = 1.25 fy and phi = 1.0).
"""
from __future__ import annotations
import numpy as np
import pytest

from design.common.materials import Concrete, Steel, Bar
from design.common.probable import (
    probable_interaction_diagram,
    mpr_envelope,
)
from design.columns.interaction import interaction_diagram
from design.sections.rectangular import RectangularSection
from design.sections.reinforcement import RebarLayout, perimeter_bars


def _make_section() -> RectangularSection:
    c = Concrete(fc=28.0)
    s = Steel(fy=420.0, grade=60)
    group = perimeter_bars(
        b=500.0, h=500.0, cover=50.0,
        n_x=4, n_y=4,
        bar=Bar(diameter=25.0), steel=s,
    )
    return RectangularSection(
        b=500.0, h=500.0, concrete=c,
        rebar=RebarLayout(groups=(group,)),
    )


def test_probable_diagram_has_higher_capacity_than_nominal():
    """At fy_pr = 1.25 fy, the diagram envelope should exceed the
    regular phi*Mn at all neutral-axis positions."""
    sec = _make_section()
    nom = interaction_diagram(sec, n_points=40, angle_deg=0.0)
    pr = probable_interaction_diagram(sec, n_points=40, angle_deg=0.0)

    # Same number of points (anchors are added the same way)
    assert len(pr.phi_Mn) == len(nom.phi_Mn)

    # Probable max Mn should exceed nominal phi_Mn max by a meaningful margin
    assert max(pr.Mn) > max(nom.phi_Mn) * 1.05


def test_probable_diagram_phi_is_unity():
    sec = _make_section()
    pr = probable_interaction_diagram(sec, n_points=40, angle_deg=0.0)
    # All points report phi = 1.0
    assert all(p.phi == 1.0 for p in pr.points)
    # phi_Mn == Mn
    assert np.allclose(pr.phi_Mn, pr.Mn)


def test_probable_fy_factor_unity_matches_nominal_phi_one():
    """With fy_factor = 1.0, probable should match interaction_diagram
    in shape, only differing by phi = 1.0 (nominal Mn agreement)."""
    sec = _make_section()
    nom = interaction_diagram(sec, n_points=40, angle_deg=0.0)
    pr = probable_interaction_diagram(sec, n_points=40, angle_deg=0.0,
                                      fy_factor=1.0)
    # Nominal Mn should match (phi has been stripped on the probable side).
    # Same indices, same anchors -> compare arrays.
    assert np.allclose(pr.Mn, nom.Mn, rtol=1e-6, atol=1.0)


def test_mpr_envelope_returns_finite_positive():
    sec = _make_section()
    Mpr_max, Pu_at_max = mpr_envelope(sec, angle_deg=0.0, n_points=40)
    assert Mpr_max > 0
    assert np.isfinite(Mpr_max)
    assert np.isfinite(Pu_at_max)


def test_mpr_envelope_filters_by_pu_range():
    sec = _make_section()
    Mpr_full, _ = mpr_envelope(sec, angle_deg=0.0, n_points=40)
    # Narrow Pu range — Mpr should be <= full sweep
    Mpr_narrow, _ = mpr_envelope(sec, angle_deg=0.0, n_points=40,
                                 Pu_range=(0.0, 1000.0))
    assert Mpr_narrow <= Mpr_full + 1e-6
