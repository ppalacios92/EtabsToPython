"""Centralized shear formulas for beams, columns and walls.

ACI 318-25 references:
    §22.5         — one-way shear (general)
    §22.5.5.1     — Vc simplified
    §22.5.10      — Vs (shear reinforcement)
    §22.5.1.2     — Vs,max (web crushing)
    §9.6.3.4      — minimum transverse reinforcement
    §18.6.5.2     — Vc=0 in beam plastic hinges
    §18.7.6.2.1   — Vc=0 in column plastic hinges
    §18.10.3.1    — wall capacity-design shear
    §18.10.4.1    — wall Vn (alpha_c · sqrt(fc) + rho_t·fyt)
    §18.10.4.4    — wall Vn cap
"""
from __future__ import annotations
from math import sqrt

from design.common.materials import Concrete
from design.common.factors import phi_shear


# ---------------------------------------------------------------------------
# Concrete contribution Vc
# ---------------------------------------------------------------------------
def vc_simplified(*, b: float, d: float, concrete: Concrete) -> float:
    """Vc simplified form. ACI 318-25 §22.5.5.1.

        Vc = 0.17 * lambda * sqrt(fc) * b * d   [SI, N]

    Returns kN. Inputs: b, d in mm; concrete provides lam and sqrt_fc.
    """
    return 0.17 * concrete.lam * concrete.sqrt_fc * b * d / 1000.0


def vc_detailed(
    *,
    b: float,
    d: float,
    Nu: float,
    Ag: float,
    rho_w: float,
    concrete: Concrete,
) -> float:
    """Detailed Vc per §22.5.5.1 (with axial + rho_w effects).

    Nu in kN (positive compression); Ag in mm². Returns kN.
    """
    rho_w = max(rho_w, 0.0)
    rho_w_term = (rho_w) ** (1.0 / 3.0) if rho_w > 0 else 0.0
    Nu_N = Nu * 1000.0
    Nu_factor = max(1.0 + Nu_N / (14.0 * Ag), 0.0) if Ag > 0 else 1.0
    Vc = 0.66 * concrete.lam * rho_w_term * concrete.sqrt_fc * b * d * Nu_factor
    return Vc / 1000.0


# ---------------------------------------------------------------------------
# Steel contribution Vs and cap
# ---------------------------------------------------------------------------
def vs_capacity(*, Av: float, fyt: float, d: float, s: float) -> float:
    """Stirrup contribution Vs. ACI 318-25 §22.5.10.5.3.

        Vs = Av * fyt * d / s   [N]

    Av in mm², fyt in MPa, d / s in mm. Returns kN.
    """
    if s <= 0:
        return 0.0
    return Av * fyt * d / s / 1000.0


def vs_max(*, b: float, d: float, concrete: Concrete) -> float:
    """Cap on Vs to avoid web crushing. ACI 318-25 §22.5.1.2.

        Vs_max = 0.66 * sqrt(fc) * b * d   [SI]

    Returns kN.
    """
    return 0.66 * concrete.sqrt_fc * b * d / 1000.0


# ---------------------------------------------------------------------------
# Required Av/s and minimum
# ---------------------------------------------------------------------------
def av_s_required(
    *,
    Vu: float,
    Vc: float,
    fyt: float,
    d: float,
    phi: float | None = None,
) -> float | None:
    """Required Av/s [mm²/mm] given a factored shear Vu [kN] and Vc [kN].

    - Returns ``None`` if Vu <= 0 (no demand).
    - Returns ``0.0`` if Vc alone (with phi) covers Vu.
    - Otherwise returns the positive Av/s [mm²/mm].

    `phi` defaults to `phi_shear()` (0.75).
    """
    if Vu is None or Vu <= 0:
        return None
    if phi is None:
        phi = phi_shear()
    Vs_req = Vu / phi - Vc
    if Vs_req <= 0:
        return 0.0
    return (Vs_req * 1000.0) / (fyt * d)


def av_s_minimum(*, b: float, concrete: Concrete, fyt: float) -> float:
    """Minimum transverse reinforcement Av/s. ACI 318-25 §9.6.3.4.

        Av/s >= max(0.062 * sqrt(fc) * b / fyt, 0.35 * b / fyt)  [SI]

    Returns mm²/mm.
    """
    t1 = 0.062 * sqrt(concrete.fc) * b / fyt
    t2 = 0.35 * b / fyt
    return max(t1, t2)


# ---------------------------------------------------------------------------
# Vc=0 sismic trigger (beams §18.6.5.2 / columns §18.7.6.2.1)
# ---------------------------------------------------------------------------
def vc_zero_seismic(
    *,
    V_seismic: float,
    V_total: float,
    Pu: float,
    Ag: float,
    fc: float,
) -> bool:
    """Returns True if Vc must be taken as 0 in the confined zone (`lo`).

    Applies to:
        - §18.6.5.2 (SMF beams)
        - §18.7.6.2.1 (SMF columns)

    Conditions (both must hold):
        (a) Seismic-induced shear is at least 50% of total: V_seismic >= 0.5 V_total
        (b) Axial load is light: Pu < Ag * fc / 20

    Units:
        V_seismic, V_total in kN (any consistent ratio works).
        Pu in kN, Ag in mm², fc in MPa.  Ag * fc / 20 is converted to kN.
    """
    if V_total <= 0:
        return False
    seismic_share = V_seismic / V_total
    cond_a = seismic_share >= 0.5
    # Ag·fc / 20 está en N (mm²·MPa = N). Para comparar con Pu en kN
    # convertimos dividiendo por 1000.
    cond_b = Pu < (Ag * fc / 20.0) / 1000.0
    return cond_a and cond_b


# ---------------------------------------------------------------------------
# Walls (§18.10)
# ---------------------------------------------------------------------------
def alpha_c(hw_over_lw: float) -> float:
    """Coefficient alpha_c. ACI 318-25 Table 18.10.4.1.

        hw/lw <= 1.5  -> 0.25
        hw/lw >= 2.0  -> 0.17
        1.5 < hw/lw < 2.0 -> linear interpolation
    """
    if hw_over_lw <= 1.5:
        return 0.25
    if hw_over_lw >= 2.0:
        return 0.17
    return 0.25 + (0.17 - 0.25) * (hw_over_lw - 1.5) / (2.0 - 1.5)


def vn_wall(
    *,
    Acv: float,
    fc: float,
    rho_t: float,
    fyt: float,
    hw_over_lw: float,
    lam: float = 1.0,
) -> float:
    """Wall nominal shear. ACI 318-25 §18.10.4.1.

        Vn = (alpha_c * lambda * sqrt(fc) + rho_t * fyt) * Acv  [N]

    Returns kN. Acv in mm², fc / fyt in MPa.
    """
    a_c = alpha_c(hw_over_lw)
    Vn = (a_c * lam * sqrt(fc) + rho_t * fyt) * Acv
    return Vn / 1000.0


def vn_wall_max(*, Acv: float, fc: float) -> float:
    """Cap on wall Vn. ACI 318-25 §18.10.4.4.

        Vn <= 0.83 * sqrt(fc) * Acv   [SI form ~ 10 sqrt(fc psi)]

    Returns kN.
    """
    return 0.83 * sqrt(fc) * Acv / 1000.0


def ve_in_plane_capacity_wall(
    *,
    Mpr: float,
    Mu: float | None = None,
    Vu: float | None = None,
    omega_v_factor: float = 1.5,
) -> float:
    """Capacity-design in-plane shear Ve for walls. ACI 318-25 §18.10.3.1.

    If (Mu, Vu) are provided: Ve = max(omega_v * Vu, (Mpr / Mu) * Vu).
    Otherwise Ve = omega_v * (Vu or 0). Returns kN.
    """
    base = omega_v_factor * (Vu or 0.0)
    if Mu is not None and Mu > 0 and Vu is not None:
        mpr_based = (Mpr / Mu) * Vu
        return max(base, mpr_based)
    return base
