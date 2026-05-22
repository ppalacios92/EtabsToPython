"""In-plane shear strength of structural walls. ACI 318-25 §18.10.3 / §18.10.4."""
from __future__ import annotations
from math import sqrt

from design.common.factors import phi_shear


def alpha_c(hw_over_lw: float) -> float:
    """Coefficient alpha_c. ACI 318-25 Table 18.10.4.1.

    SI units (MPa, mm):
        hw/lw <= 1.5 -> 0.25
        hw/lw >= 2.0 -> 0.17
        1.5 < hw/lw < 2.0 -> linear interpolation
    """
    if hw_over_lw <= 1.5:
        return 0.25
    if hw_over_lw >= 2.0:
        return 0.17
    return 0.25 + (0.17 - 0.25) * (hw_over_lw - 1.5) / (2.0 - 1.5)


def vn_wall(*, Acv: float, fc: float, rho_t: float, fyt: float,
            hw_over_lw: float, lam: float = 1.0) -> float:
    """Vn = (alpha_c * lambda * sqrt(fc) + rho_t * fyt) * Acv. §18.10.4.1.

    Inputs MPa / mm^2; returns kN.
    """
    a_c = alpha_c(hw_over_lw)
    Vn = (a_c * lam * sqrt(fc) + rho_t * fyt) * Acv
    return Vn / 1000.0


def vn_wall_max(*, Acv: float, fc: float) -> float:
    """Cap Vn <= 0.83 sqrt(fc) Acv. §18.10.4.4 (SI form ~ 10 sqrt psi)."""
    return 0.83 * sqrt(fc) * Acv / 1000.0


def omega_v(
    *,
    n_stories: int | None = None,
    hw_over_lw: float | None = None,
) -> float:
    """Capacity-design shear amplifier. ACI 318-25 §18.10.3.1.2.

    Simplified form: 1.5 by default. Drop to 1.0 for very short walls
    (n_stories <= 2 and hw_over_lw <= 2). The strict height-dependent
    form of §18.10.3.1.2(b) is left as a manual override by the caller.
    """
    if n_stories is not None and hw_over_lw is not None:
        if n_stories <= 2 and hw_over_lw <= 2.0:
            return 1.0
    return 1.5


def av_s_required_wall(
    *,
    Vu: float,
    Vc: float,
    fyt: float,
    d_eff: float,
) -> float | None:
    """Av/s in mm²/mm required to resist Vu given Vc and effective depth d_eff.

    Wall effective depth (§11.5.4.3): d_eff = 0.8 * lw. Vu and Vc in kN.
    Returns 0.0 if Vc alone suffices.
    """
    if Vu is None or Vu <= 0:
        return None
    phi = phi_shear()
    Vs_req = Vu / phi - Vc
    if Vs_req <= 0:
        return 0.0
    return (Vs_req * 1000.0) / (fyt * d_eff)


def rho_distributed_min() -> tuple[float, float]:
    """Minimum distributed reinforcement ratios for walls.

    ACI 318-25 §11.6 (general) and §18.10.2 (special seismic walls).
    Returns (rho_l_min, rho_t_min) = (0.0025, 0.0025).
    """
    return 0.0025, 0.0025


def web_double_curtain_required(
    *,
    Vu: float,
    Acv: float,
    fc: float,
    lam: float = 1.0,
    hw_over_lw: float = 2.0,
) -> bool:
    """Two curtains of reinforcement required when Vu exceeds the threshold.

    ACI 318-25 §18.10.2.2: two curtains required when
        Vu > 0.17 * lambda * sqrt(fc) * Acv   (SI)
    or when hw/lw >= 2.0 (commonly enforced by §11.7.2 for slenderness).
    """
    Vu_N = Vu * 1000.0
    threshold = 0.17 * lam * sqrt(fc) * Acv
    return Vu_N > threshold or hw_over_lw >= 2.0


def ve_in_plane_capacity(
    *,
    Mpr: float,
    Mu: float | None = None,
    Vu: float | None = None,
    omega_v_factor: float = 1.5,
) -> float:
    """Capacity-design in-plane shear Ve. §18.10.3.1.

    If (Mu, Vu) are provided: Ve = max(omega_v * Vu, (Mpr / Mu) * Vu).
    Otherwise Ve = omega_v * (Vu or 0). Returns kN.
    """
    base = omega_v_factor * (Vu or 0.0)
    if Mu is not None and Mu > 0 and Vu is not None:
        mpr_based = (Mpr / Mu) * Vu
        return max(base, mpr_based)
    return base
