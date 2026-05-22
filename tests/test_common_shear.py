"""Tests for design.common.shear — ACI 318-25 §22.5 + §18.x."""
from __future__ import annotations
from math import sqrt
import pytest

from design.common.materials import Concrete
from design.common.shear import (
    vc_simplified,
    vc_detailed,
    vs_capacity,
    vs_max,
    av_s_required,
    av_s_minimum,
    vc_zero_seismic,
    alpha_c,
    vn_wall,
    vn_wall_max,
    ve_in_plane_capacity_wall,
)


# ---------------------------------------------------------------------------
# vc_simplified — ACI 318-25 §22.5.5.1
# ---------------------------------------------------------------------------
def test_vc_simplified_normalweight():
    """Vc = 0.17 * lam * sqrt(fc) * b * d / 1000  [kN]."""
    c = Concrete(fc=28.0)
    Vc = vc_simplified(b=300.0, d=540.0, concrete=c)
    expected = 0.17 * 1.0 * sqrt(28.0) * 300.0 * 540.0 / 1000.0
    assert Vc == pytest.approx(expected, rel=1e-9)
    # Sanity: ~146 kN for fc=28 / b=300 / d=540
    assert 140.0 < Vc < 150.0


def test_vc_simplified_lightweight_drops_by_lambda():
    c_n = Concrete(fc=28.0)
    c_l = Concrete(fc=28.0, lightweight=True)
    Vc_n = vc_simplified(b=300.0, d=540.0, concrete=c_n)
    Vc_l = vc_simplified(b=300.0, d=540.0, concrete=c_l)
    # All-lightweight => lam = 0.75
    assert Vc_l == pytest.approx(0.75 * Vc_n, rel=1e-9)


def test_vc_simplified_sqrt_fc_capped_at_70():
    """§22.5.3.1: sqrt(fc) capped at sqrt(70 MPa)."""
    c_high = Concrete(fc=100.0)
    Vc = vc_simplified(b=300.0, d=540.0, concrete=c_high)
    expected = 0.17 * 1.0 * sqrt(70.0) * 300.0 * 540.0 / 1000.0
    assert Vc == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# vc_detailed
# ---------------------------------------------------------------------------
def test_vc_detailed_zero_axial_zero_rho_returns_zero():
    """rho_w = 0 -> rho_term = 0 -> Vc = 0 (vacuous detailed form)."""
    c = Concrete(fc=28.0)
    Vc = vc_detailed(b=300.0, d=540.0, Nu=0.0, Ag=300.0*600.0,
                     rho_w=0.0, concrete=c)
    assert Vc == pytest.approx(0.0)


def test_vc_detailed_positive_with_rho():
    c = Concrete(fc=28.0)
    Vc = vc_detailed(b=300.0, d=540.0, Nu=0.0, Ag=300.0*600.0,
                     rho_w=0.01, concrete=c)
    assert Vc > 0


# ---------------------------------------------------------------------------
# vs_capacity / vs_max
# ---------------------------------------------------------------------------
def test_vs_capacity_zero_spacing_returns_zero():
    assert vs_capacity(Av=142.0, fyt=420.0, d=540.0, s=0.0) == 0.0


def test_vs_capacity_value():
    Vs = vs_capacity(Av=142.0, fyt=420.0, d=540.0, s=150.0)
    expected = 142.0 * 420.0 * 540.0 / 150.0 / 1000.0
    assert Vs == pytest.approx(expected)


def test_vs_max():
    c = Concrete(fc=28.0)
    Vs_max = vs_max(b=300.0, d=540.0, concrete=c)
    expected = 0.66 * sqrt(28.0) * 300.0 * 540.0 / 1000.0
    assert Vs_max == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# av_s_required
# ---------------------------------------------------------------------------
def test_av_s_required_none_when_no_demand():
    assert av_s_required(Vu=0.0, Vc=150.0, fyt=420.0, d=540.0) is None
    assert av_s_required(Vu=-10.0, Vc=150.0, fyt=420.0, d=540.0) is None


def test_av_s_required_zero_when_vc_sufficient():
    # Vu/phi = 100/0.75 = 133.3 < Vc = 200 -> 0
    res = av_s_required(Vu=100.0, Vc=200.0, fyt=420.0, d=540.0)
    assert res == 0.0


def test_av_s_required_positive_when_vc_short():
    # Vu/phi - Vc > 0
    res = av_s_required(Vu=300.0, Vc=100.0, fyt=420.0, d=540.0)
    # phi=0.75 default
    Vs_req_kN = 300.0 / 0.75 - 100.0
    expected = (Vs_req_kN * 1000.0) / (420.0 * 540.0)
    assert res == pytest.approx(expected, rel=1e-9)


def test_av_s_required_custom_phi():
    res = av_s_required(Vu=300.0, Vc=100.0, fyt=420.0, d=540.0, phi=0.9)
    Vs_req_kN = 300.0 / 0.9 - 100.0
    expected = (Vs_req_kN * 1000.0) / (420.0 * 540.0)
    assert res == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# av_s_minimum
# ---------------------------------------------------------------------------
def test_av_s_minimum_returns_larger_of_two():
    c = Concrete(fc=28.0)
    res = av_s_minimum(b=300.0, concrete=c, fyt=420.0)
    t1 = 0.062 * sqrt(28.0) * 300.0 / 420.0
    t2 = 0.35 * 300.0 / 420.0
    assert res == pytest.approx(max(t1, t2), rel=1e-9)


# ---------------------------------------------------------------------------
# vc_zero_seismic — §18.6.5.2 / §18.7.6.2.1
# ---------------------------------------------------------------------------
def test_vc_zero_when_seismic_share_and_light_axial():
    # V_seismic >= 0.5 V_total  AND  Pu < Ag·fc/20
    Ag = 500.0 * 500.0   # mm²
    fc = 28.0            # MPa
    threshold_kN = Ag * fc / 20.0 / 1000.0  # = 350 kN
    assert vc_zero_seismic(
        V_seismic=150.0, V_total=200.0,
        Pu=threshold_kN - 1.0,
        Ag=Ag, fc=fc,
    ) is True


def test_vc_zero_false_when_axial_high():
    Ag = 500.0 * 500.0
    fc = 28.0
    threshold_kN = Ag * fc / 20.0 / 1000.0
    assert vc_zero_seismic(
        V_seismic=150.0, V_total=200.0,
        Pu=threshold_kN + 1.0,
        Ag=Ag, fc=fc,
    ) is False


def test_vc_zero_false_when_seismic_share_low():
    Ag = 500.0 * 500.0
    fc = 28.0
    assert vc_zero_seismic(
        V_seismic=30.0, V_total=200.0,
        Pu=10.0,
        Ag=Ag, fc=fc,
    ) is False


def test_vc_zero_false_when_v_total_zero():
    assert vc_zero_seismic(
        V_seismic=0.0, V_total=0.0,
        Pu=10.0, Ag=1.0, fc=28.0,
    ) is False


# ---------------------------------------------------------------------------
# Walls
# ---------------------------------------------------------------------------
def test_alpha_c_endpoints_and_interpolation():
    assert alpha_c(1.0) == 0.25
    assert alpha_c(1.5) == 0.25
    assert alpha_c(2.0) == 0.17
    assert alpha_c(3.0) == 0.17
    # Midpoint
    mid = alpha_c(1.75)
    assert mid == pytest.approx(0.25 + (0.17 - 0.25) * (0.25 / 0.5), rel=1e-9)


def test_vn_wall_value():
    Acv = 300.0 * 4000.0   # mm² (web of a 4m wall, 300 thick)
    fc = 35.0
    rho_t = 0.003
    fyt = 420.0
    hw_over_lw = 2.5
    Vn = vn_wall(Acv=Acv, fc=fc, rho_t=rho_t, fyt=fyt,
                 hw_over_lw=hw_over_lw, lam=1.0)
    a_c = 0.17  # hw/lw >= 2
    expected = (a_c * 1.0 * sqrt(fc) + rho_t * fyt) * Acv / 1000.0
    assert Vn == pytest.approx(expected, rel=1e-9)


def test_vn_wall_max():
    Acv = 300.0 * 4000.0
    fc = 35.0
    assert vn_wall_max(Acv=Acv, fc=fc) == pytest.approx(
        0.83 * sqrt(fc) * Acv / 1000.0, rel=1e-9
    )


def test_ve_in_plane_capacity_wall_falls_back_to_omega():
    Ve = ve_in_plane_capacity_wall(Mpr=10000.0, Mu=None, Vu=200.0,
                                   omega_v_factor=1.5)
    assert Ve == pytest.approx(1.5 * 200.0)


def test_ve_in_plane_capacity_wall_takes_max():
    # Mpr/Mu = 1.2 -> Mpr-based = 1.2*Vu; omega-based = 1.5*Vu. Max = 1.5*Vu.
    Ve = ve_in_plane_capacity_wall(Mpr=12000.0, Mu=10000.0, Vu=200.0,
                                   omega_v_factor=1.5)
    assert Ve == pytest.approx(max(1.5 * 200.0, (12000.0/10000.0) * 200.0))


def test_ve_in_plane_capacity_wall_mpr_dominates():
    Ve = ve_in_plane_capacity_wall(Mpr=20000.0, Mu=10000.0, Vu=200.0,
                                   omega_v_factor=1.5)
    # Mpr/Mu = 2 -> 2*Vu = 400 > 1.5*Vu = 300
    assert Ve == pytest.approx(400.0)
