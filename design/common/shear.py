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
    §22.6         — two-way (punching) shear
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


def lambda_s(d: float) -> float:
    """Size-effect factor for one-way shear, ACI 318-25 Eq. 22.5.5.1.3.

    ``d`` is the effective depth in mm.  The ACI expression is limited to
    one; non-positive depths retain that upper-bound value so callers with
    incomplete geometry do not receive an artificial size-effect increase.
    """
    if d <= 0.0:
        return 1.0
    return min(sqrt(2.0 / (1.0 + d / 254.0)), 1.0)


def vc_one_way(
    *,
    b: float,
    d: float,
    rho_w: float,
    concrete: Concrete,
    av_ge_av_min: bool,
    Nu: float = 0.0,
    Ag: float = 0.0,
) -> float:
    """Concrete one-way shear strength ``Vc`` per Table 22.5.5.1 [kN].

    ``Nu`` is positive in compression and is capped at ``0.05 fc`` in the
    additive axial-stress term.  With at least minimum transverse
    reinforcement, the lesser of the two tabulated expressions governs;
    otherwise the size-effect expression applies.  The result is bounded by
    zero and by ``0.42 lambda sqrt(fc) b d``.  Units are MPa, mm and kN.
    """
    rho_term = max(rho_w, 0.0) ** (1.0 / 3.0)
    nu_term = 0.0
    if Nu > 0.0 and Ag > 0.0:
        nu_term = min(Nu * 1000.0 / (6.0 * Ag), 0.05 * concrete.fc)

    common = concrete.lam * concrete.sqrt_fc
    if av_ge_av_min:
        vc_stress = min(
            0.17 * common + nu_term,
            0.66 * rho_term * common + nu_term,
        )
    else:
        vc_stress = 0.66 * lambda_s(d) * rho_term * common + nu_term

    vc_n = vc_stress * b * d
    vc_cap_n = 0.42 * common * b * d
    return max(0.0, min(vc_n, vc_cap_n)) / 1000.0


def vc_detailed(
    *,
    b: float,
    d: float,
    Nu: float,
    Ag: float,
    rho_w: float,
    concrete: Concrete,
    lambda_s: float = 1.0,
) -> float:
    """Detailed Vc per ACI 318-25 §22.5.5.1, Table 22.5.5.1.

        Vc = [0.66 * lambda_s * lambda * rho_w^(1/3) * sqrt(fc')
              + Nu/(6 Ag)] * bw * d                       [SI, N]

    The axial term ``Nu/(6 Ag)`` is **additive** (Nu positive in
    compression) and capped at ``0.05 fc'`` per §22.5.5.1.2 — NOT a
    ``(1 + Nu/14Ag)`` multiplier (that was the superseded 318-14 form).
    ``lambda_s`` is the size-effect factor (§22.5.5.1.3); it is 1.0 when
    at least minimum shear reinforcement is provided (Av >= Av,min).

    Nu in kN (positive compression); Ag in mm². Returns kN.
    """
    rho_w = max(rho_w, 0.0)
    rho_w_term = (rho_w) ** (1.0 / 3.0) if rho_w > 0 else 0.0
    Nu_N = Nu * 1000.0  # kN -> N
    nu_term = min(Nu_N / (6.0 * Ag), 0.05 * concrete.fc) if Ag > 0 else 0.0
    Vc = (0.66 * lambda_s * concrete.lam * rho_w_term * concrete.sqrt_fc
          + nu_term) * b * d
    return max(Vc, 0.0) / 1000.0


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
# Two-way (punching) shear (§22.6)
# ---------------------------------------------------------------------------
def bo_rectangular(
    *,
    c1: float,
    c2: float,
    d: float,
    position: str = "interior",
) -> float:
    """Return the rectangular critical perimeter ``bo`` in mm (§22.6.4).

    The critical section is at ``d/2`` from the supported-area faces.  For
    an edge or corner support, ``c1`` is perpendicular to the free edge;
    ``c2`` is parallel to it.  ``position`` is ``"interior"``, ``"edge"``,
    or ``"corner"``.
    """
    if position == "interior":
        return 2.0 * (c1 + d) + 2.0 * (c2 + d)
    if position == "edge":
        return 2.0 * (c1 + d / 2.0) + (c2 + d)
    if position == "corner":
        return (c1 + d / 2.0) + (c2 + d / 2.0)
    raise ValueError("position must be 'interior', 'edge', or 'corner'.")


def _alpha_s_two_way(position: str) -> float:
    """Return the §22.6.5.3 alpha_s coefficient for support position."""
    try:
        return {"interior": 40.0, "edge": 30.0, "corner": 20.0}[position]
    except KeyError as exc:
        raise ValueError("position must be 'interior', 'edge', or 'corner'.") from exc


def vc_two_way(
    *,
    bo: float,
    d: float,
    concrete: Concrete,
    beta: float = 1.0,
    position: str = "interior",
) -> float:
    """Return two-way concrete shear stress ``vc`` [MPa] (§22.6.5.3).

    The least of the three applicable expressions governs.  ``beta`` is the
    ratio of the long to short side of the supported area, and ``bo`` and
    ``d`` are in mm.
    """
    if bo <= 0.0 or d <= 0.0:
        raise ValueError("bo and d must be positive.")
    if beta <= 0.0:
        raise ValueError("beta must be positive.")
    common = lambda_s(d) * concrete.lam * concrete.sqrt_fc
    alpha_s = _alpha_s_two_way(position)
    return min(
        0.33 * common,
        0.17 * (1.0 + 2.0 / beta) * common,
        0.083 * (2.0 + alpha_s * d / bo) * common,
    )


def vc_two_way_force(
    *,
    bo: float,
    d: float,
    concrete: Concrete,
    beta: float = 1.0,
    position: str = "interior",
) -> float:
    """Return the two-way concrete shear strength ``Vc`` in kN (§22.6.5.3)."""
    return vc_two_way(
        bo=bo, d=d, concrete=concrete, beta=beta, position=position,
    ) * bo * d / 1000.0


def vc_two_way_max_with_stirrups(*, d: float, concrete: Concrete) -> float:
    """Return max ``vc`` [MPa] for two-way shear reinforcement (§22.6.6.1).

    The concrete term is ``0.17 lambda_s lambda sqrt(fc)``.  The associated
    steel contribution is evaluated separately as ``vs = Av*fyt/(bo*s)``.
    """
    if d <= 0.0:
        raise ValueError("d must be positive.")
    return 0.17 * lambda_s(d) * concrete.lam * concrete.sqrt_fc


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


def alpha_sh(*, acf: float = 0.0, bcf: float | None = None, bw: float) -> float:
    """Return the wall shear coefficient ``alpha_sh`` of §18.10.4.4.

    ``acf`` is the area of the flange in compression and ``bcf`` its
    effective width.  A wall without a flange has ``acf=0`` and therefore
    returns 1.0.  Supplying a positive ``acf`` requires a positive ``bcf``.
    """
    if acf <= 0.0:
        return 1.0
    if bcf is None or bcf <= 0.0 or bw <= 0.0:
        raise ValueError("bcf and bw must be positive when acf is positive.")
    value = 0.7 * (1.0 + sqrt(acf / (bcf * bw)))
    return min(1.2, max(1.0, value))


def vn_wall_max(*, Acv: float, fc: float, alpha_sh: float = 1.0) -> float:
    """Cap on wall Vn. ACI 318-25 §18.10.4.4.

        Vn <= alpha_sh * 0.83 * sqrt(fc) * Acv
        [SI form ~ 10 sqrt(fc psi)]

    Returns kN.
    """
    return alpha_sh * 0.83 * sqrt(fc) * Acv / 1000.0


def vn_wall_group_max(
    *,
    acv_list: list[float] | tuple[float, ...],
    fc: float,
    alpha_sh_list: list[float] | tuple[float, ...] | None = None,
) -> float:
    """Return the group wall-shear upper bound of §18.10.4.4 in kN.

    ``acv_list`` contains the vertical concrete areas of the wall segments
    that share a common lateral force.  ``alpha_sh_list`` defaults to 1.0
    for every segment.
    """
    if alpha_sh_list is None:
        alpha_sh_list = [1.0] * len(acv_list)
    if len(alpha_sh_list) != len(acv_list):
        raise ValueError("alpha_sh_list must have one entry per Acv value.")
    return 0.66 * sqrt(fc) * sum(
        alpha * acv for alpha, acv in zip(alpha_sh_list, acv_list)
    ) / 1000.0


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
