"""Regression tests for the corrected common-layer formulas.

  - vc_detailed: axial term is ADDITIVE Nu/(6 Ag), capped 0.05 fc'
    (ACI 318-25 §22.5.5.1) — was the superseded (1+Nu/14Ag) multiplier.
  - Concrete.sqrt_fc: capped at 8.3 MPa (= 100 psi) per §22.5.3.1.
  - Concrete.fr: modulus of rupture 0.62 λ sqrt(fc') per §19.2.3.1.
  - ash_required: expression (c) 0.2 kf kn Pu/(fyt Ach), Table 18.7.5.4.
  - rho_s_spiral_required: §25.7.3.3.
  - mpr_at_axial(Nu=0): probable moment at zero axial (beam Mpr).
"""
from __future__ import annotations
from math import sqrt
import pytest

from design.common.materials import Concrete, Steel, Bar
from design.common.shear import vc_detailed, vc_simplified
from design.common.spacing import ash_required, rho_s_spiral_required


# --- vc_detailed additive axial term ----------------------------------- #
def test_vc_detailed_axial_is_additive_with_cap():
    c = Concrete(fc=28.0)
    b, d, Ag, rho_w = 300.0, 540.0, 300.0 * 600.0, 0.01
    base = vc_detailed(b=b, d=d, Nu=0.0, Ag=Ag, rho_w=rho_w, concrete=c)
    with_axial = vc_detailed(b=b, d=d, Nu=500.0, Ag=Ag, rho_w=rho_w, concrete=c)
    # additive term Nu/(6 Ag) capped at 0.05 fc'
    nu_term = min(500.0 * 1000.0 / (6.0 * Ag), 0.05 * c.fc)
    expected = with_axial - base
    assert expected == pytest.approx(nu_term * b * d / 1000.0, rel=1e-9)
    # axial increases Vc (compression helps), not multiplies the rho term
    assert with_axial > base


def test_vc_detailed_zero_rho_zero_axial_is_zero():
    c = Concrete(fc=28.0)
    assert vc_detailed(b=300.0, d=540.0, Nu=0.0, Ag=1.8e5, rho_w=0.0, concrete=c) == 0.0


# --- sqrt_fc cap and fr ------------------------------------------------- #
def test_sqrt_fc_capped_8p3():
    assert Concrete(fc=100.0).sqrt_fc == pytest.approx(8.3)
    assert Concrete(fc=28.0).sqrt_fc == pytest.approx(sqrt(28.0))


def test_modulus_of_rupture():
    c = Concrete(fc=28.0)
    assert c.fr == pytest.approx(0.62 * 1.0 * sqrt(28.0))


# --- Ash expression (c), Table 18.7.5.4 -------------------------------- #
def test_ash_expression_c_triggers_on_high_axial():
    kw = dict(s=100.0, bc=400.0, Ag=500.0 * 500.0, Ach=440.0 * 440.0,
              fc=28.0, fyt=420.0)
    low = ash_required(**kw, Pu_over_Ag_fc=0.10, nl=12)   # only (a),(b)
    high = ash_required(**kw, Pu_over_Ag_fc=0.55, nl=12)  # adds (c)
    # closed-form (c): Ash = 0.2 kf kn Pu s bc / (fyt Ach)
    kf = max(28.0 / 172.4 + 0.6, 1.0)
    kn = 12 / (12 - 2)
    Pu_N = 0.55 * (500.0 * 500.0) * 28.0
    ash_c = 0.2 * kf * kn * Pu_N * 100.0 * 400.0 / (420.0 * 440.0 * 440.0)
    ash_ab = max(0.3 * ((500.0 * 500.0) / (440.0 * 440.0) - 1.0) * 100.0 * 400.0 * 28.0 / 420.0,
                 0.09 * 100.0 * 400.0 * 28.0 / 420.0)
    assert high == pytest.approx(max(ash_ab, ash_c))
    assert high > low                # (c) governs at high axial


def test_ash_low_axial_only_a_b():
    kw = dict(s=100.0, bc=400.0, Ag=500.0 * 500.0, Ach=440.0 * 440.0,
              fc=28.0, fyt=420.0)
    ash = ash_required(**kw, Pu_over_Ag_fc=0.10, nl=12)
    ash_ab = max(0.3 * ((500.0 * 500.0) / (440.0 * 440.0) - 1.0) * 100.0 * 400.0 * 28.0 / 420.0,
                 0.09 * 100.0 * 400.0 * 28.0 / 420.0)
    assert ash == pytest.approx(ash_ab)


# --- spiral rho_s, §25.7.3.3 ------------------------------------------- #
def test_rho_s_spiral_required():
    Ag, Ach, fc, fyt = 500.0 * 500.0, 440.0 * 440.0, 28.0, 420.0
    rho = rho_s_spiral_required(Ag=Ag, Ach=Ach, fc=fc, fyt=fyt)
    rho_d = 0.45 * (Ag / Ach - 1.0) * fc / fyt
    rho_e = 0.12 * fc / fyt
    assert rho == pytest.approx(max(rho_d, rho_e))


# --- Mpr at zero axial (beam Mpr), §18.6.5 ----------------------------- #
def test_mpr_at_axial_zero_is_positive_and_probable():
    from design.sections.rectangular import RectangularSection
    from design.sections.reinforcement import RebarLayout, perimeter_bars
    from design.common.probable import mpr_at_axial
    from design.analysis.section_analysis import pm_curve

    conc = Concrete(fc=28.0)
    steel = Steel(fy=420.0, grade=60)
    bars = perimeter_bars(b=400.0, h=600.0, cover=40.0, n_x=3, n_y=3,
                          bar=Bar(20.0), steel=steel)
    sec = RectangularSection(b=400.0, h=600.0, concrete=conc,
                             rebar=RebarLayout(groups=(bars,)))
    mpr0 = mpr_at_axial(sec, Nu=0.0)
    assert mpr0 > 0.0
    # probable (1.25 fy) at P=0 exceeds the nominal moment at P=0
    nom = pm_curve(sec, n_points=80, angle_deg=0.0)
    Pn = [p.Pn for p in nom.points]
    Mn = [p.Mn for p in nom.points]
    import numpy as np
    mn0 = float(np.interp(0.0, np.array(Pn)[np.argsort(Pn)],
                          np.array(Mn)[np.argsort(Pn)]))
    assert mpr0 > mn0
