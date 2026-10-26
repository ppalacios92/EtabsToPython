"""Regression tests for the rotated-section PM sweep fix.

Root cause (fixed): ``pm_curve`` / ``probable_interaction_diagram`` bounded
the neutral-axis sweep ``c`` by ``section.height()`` (the un-rotated
bounding-box height). For a wall bent out-of-plane (angle_deg = 90) the
real depth is the thickness ``tw``, not the length ``lw``, so ``c_max``
was over-scaled by ``lw/tw`` and the curved part of the diagram was
starved — the wall out-of-plane interaction diagram (and Mpr) came out
wrong. The fix bounds ``c`` by ``rotated_depth(section, angle_deg)``.

ACI 318-25 §22.2 (strain compatibility), §18.10 (walls).
"""
from __future__ import annotations
import numpy as np
import pytest

from design.common.materials import Concrete, Steel, Bar
from design.sections.rectangular import RectangularSection
from design.sections.wall import WallSection
from design.sections.reinforcement import RebarLayout, perimeter_bars
from design.analysis.section_analysis import pm_curve, rotated_depth


def _wall(lw=3000.0, tw=300.0, fc=28.0, fy=420.0):
    conc = Concrete(fc=fc)
    steel = Steel(fy=fy, grade=60)
    bars = perimeter_bars(b=tw, h=lw, cover=40.0, n_x=2, n_y=12,
                          bar=Bar(16.0), steel=steel)
    return WallSection(lw=lw, tw=tw, concrete=conc, rebar=RebarLayout(groups=(bars,)))


def test_rotated_depth_in_plane_is_lw():
    w = _wall()
    assert rotated_depth(w, 0.0) == pytest.approx(3000.0)


def test_rotated_depth_out_of_plane_is_tw():
    w = _wall()
    # out-of-plane (angle 90) depth is the wall thickness, not the length
    assert rotated_depth(w, 90.0) == pytest.approx(300.0)


def test_wall_out_of_plane_matches_equivalent_rectangular():
    """A wall bent out-of-plane (90°) must match the rectangular section
    with the same geometry/bars at 90° — the rotation must not change the
    physics, only the sampling. Before the fix the ratio dropped to ~0.94
    at low n_points."""
    conc = Concrete(fc=28.0)
    steel = Steel(fy=420.0, grade=60)
    lw, tw = 3000.0, 300.0
    bars = perimeter_bars(b=tw, h=lw, cover=40.0, n_x=2, n_y=12,
                          bar=Bar(16.0), steel=steel)
    wall = WallSection(lw=lw, tw=tw, concrete=conc, rebar=RebarLayout(groups=(bars,)))
    rect_bars = perimeter_bars(b=tw, h=lw, cover=40.0, n_x=2, n_y=12,
                               bar=Bar(16.0), steel=steel)
    rect = RectangularSection(b=tw, h=lw, concrete=conc,
                              rebar=RebarLayout(groups=(rect_bars,)))
    for npts in (19, 30, 60):
        mw = max(p.Mn for p in pm_curve(wall, n_points=npts, angle_deg=90.0).points)
        mr = max(p.Mn for p in pm_curve(rect, n_points=npts, angle_deg=90.0).points)
        assert mw == pytest.approx(mr, rel=1e-6), f"n_points={npts}"


def test_out_of_plane_samples_curved_region():
    """The out-of-plane sweep must reach the balanced/tension region, i.e.
    the diagram must contain points with both tension (Pn<0) and
    compression (Pn>0). Before the fix everything collapsed to the
    compression tail."""
    w = _wall()
    res = pm_curve(w, n_points=40, angle_deg=90.0)
    Pn = [p.Pn for p in res.points]
    assert min(Pn) < 0.0 < max(Pn)
