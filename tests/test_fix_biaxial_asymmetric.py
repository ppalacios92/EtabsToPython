"""Regression tests for biaxial moment in asymmetric sections.

Before the fix the kernel reported only the moment perpendicular to the
neutral axis (Mn) and the PMM surface reconstructed (Mx, My) = Mn·(cosθ,
sinθ) — valid only for doubly-symmetric sections. For asymmetric layouts
(C/L/T, offset BE, unbalanced bars) the resultant moment is not aligned
with the NA angle, so the surface was biased. The kernel now reports the
true global components (Mx, My) and the surface sweeps 0..360° for
asymmetric sections and checks demands by point-in-polygon.

ACI 318-25 §22.4 (combined axial + flexure), R22.4.3.1 (biaxial).
"""
from __future__ import annotations
import numpy as np
import pytest

from design import Concrete, Steel, Bar, RectangularSection
from design.sections.reinforcement import (
    RebarLayout, RebarGroup, Rebar, perimeter_bars,
)
from design.analysis.section_analysis import pm_at_neutral_axis
from design.columns.interaction_surface import (
    interaction_surface, _is_doubly_symmetric,
)


def _sym():
    c = Concrete(fc=28); s = Steel(fy=420, grade=60)
    g = perimeter_bars(b=500, h=500, cover=50, n_x=4, n_y=4, bar=Bar(25), steel=s)
    return RectangularSection(b=500, h=500, concrete=c,
                              rebar=RebarLayout(groups=(g,)))


def _asym():
    """Extra #32 bars on the top edge only -> asymmetric about X."""
    c = Concrete(fc=28); s = Steel(fy=420, grade=60)
    base = perimeter_bars(b=500, h=500, cover=50, n_x=4, n_y=4, bar=Bar(20), steel=s)
    extra = RebarGroup(
        bars=tuple(Rebar(x=x, y=200.0, area=Bar(32).area) for x in (-150, -50, 50, 150)),
        steel=s)
    return RectangularSection(b=500, h=500, concrete=c,
                              rebar=RebarLayout(groups=(base, extra)))


# --- kernel: global moment components -------------------------------- #
def test_symmetric_has_no_transverse_moment_at_0_and_90():
    sec = _sym()
    p0 = pm_at_neutral_axis(sec, c=200, angle_deg=0.0)
    p90 = pm_at_neutral_axis(sec, c=200, angle_deg=90.0)
    assert abs(p0.My) < 1e-6 * max(abs(p0.Mx), 1.0)
    assert abs(p90.Mx) < 1e-6 * max(abs(p90.My), 1.0)
    # Mn (legacy, perpendicular to NA) equals the in-axis component at 0°
    assert p0.Mn == pytest.approx(p0.Mx)


def test_symmetric_square_at_45_has_equal_components():
    sec = _sym()
    p = pm_at_neutral_axis(sec, c=200, angle_deg=45.0)
    # square + symmetric layout: (Mx, My) both positive and equal
    assert p.Mx == pytest.approx(p.My, rel=1e-6)
    assert p.Mx > 0 and p.My > 0
    # magnitude equals the perpendicular moment (no transverse component)
    assert np.hypot(p.Mx, p.My) == pytest.approx(abs(p.Mn), rel=1e-6)


def test_asymmetric_has_transverse_moment():
    sec = _asym()
    # low axial (small c) -> the off-axis component is pronounced
    p = pm_at_neutral_axis(sec, c=150, angle_deg=45.0)
    # the resultant exceeds the perpendicular-only Mn (discarded component)
    assert np.hypot(p.Mx, p.My) > abs(p.Mn) * 1.05


# --- surface: symmetry detection and sweep range -------------------- #
def test_symmetry_detection():
    assert _is_doubly_symmetric(_sym())
    assert not _is_doubly_symmetric(_asym())


def test_surface_sweep_range_depends_on_symmetry():
    s_sym = interaction_surface(_sym(), n_angles=13)
    assert s_sym.symmetric
    assert s_sym.angles_deg[-1] == pytest.approx(180.0)
    s_asym = interaction_surface(_asym(), n_angles=25)
    assert not s_asym.symmetric
    assert s_asym.angles_deg[-1] == pytest.approx(360.0)


def test_asymmetric_envelope_is_not_centrally_symmetric():
    surf = interaction_surface(_asym(), n_angles=37)
    mx, my = surf.envelope_at_Pu(1500.0)
    # extra steel on top -> capacity about X differs for +Mx vs -Mx
    assert abs(mx.max()) != pytest.approx(abs(mx.min()), rel=0.02)


def test_asymmetric_check_is_directional():
    surf = interaction_surface(_asym(), n_angles=37)
    _, r_pos = surf.check_demand(Pu=1500.0, Mux=350.0, Muy=0.0)
    _, r_neg = surf.check_demand(Pu=1500.0, Mux=-350.0, Muy=0.0)
    # the two opposite demands must give different ratios (asymmetry)
    assert abs(r_pos - r_neg) > 0.03


def test_symmetric_check_still_symmetric():
    surf = interaction_surface(_sym(), n_angles=19)
    _, r_pos = surf.check_demand(Pu=1500.0, Mux=300.0, Muy=0.0)
    _, r_neg = surf.check_demand(Pu=1500.0, Mux=-300.0, Muy=0.0)
    # symmetric section: opposite demands give the same ratio
    assert r_pos == pytest.approx(r_neg, rel=1e-6)
