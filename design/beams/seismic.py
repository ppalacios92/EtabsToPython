"""Capacity-design helpers for SMF beams.

ACI 318-25 §18.6.5.1. Mpr uses fy_pr = 1.25 fy and phi = 1.0. The
induced shear Ve combines Mpr at both ends with the gravity shear.
"""
from __future__ import annotations

from design.common.materials import Concrete, Steel


def v_seismic_from_mpr(
    *,
    Mpr_pos_i: float, Mpr_neg_i: float,
    Mpr_pos_j: float, Mpr_neg_j: float,
    ln: float,
) -> float:
    """Constant seismic-induced shear from end-moment couple. §18.6.5.1.

    Considers both sway directions (Fig. R18.6.5 Note 2):
        sway right: (Mpr_neg_i + Mpr_pos_j) / ln
        sway left:  (Mpr_pos_i + Mpr_neg_j) / ln
    Returns the larger magnitude. This shear is constant along the beam.
    """
    if ln <= 0:
        raise ValueError(f"Clear span ln must be positive, got {ln}")
    Ve_R = (Mpr_neg_i + Mpr_pos_j) / ln
    Ve_L = (Mpr_pos_i + Mpr_neg_j) / ln
    return max(Ve_R, Ve_L)


def mpr(*, b: float, d: float, As: float,
        concrete: Concrete, steel: Steel) -> float:
    """Probable moment for one bending sense. §18.6.5.1.

    Mpr = As * (1.25 fy) * (d - a/2)  with a = As*1.25fy / (0.85 fc b).
    Inputs MPa, mm, mm². Returns kN·m.

    Only the tension-side area is used (the compression area is
    conservatively ignored — standard §18.6.5.1 interpretation).
    """
    fy_pr = 1.25 * steel.fy
    a = (As * fy_pr) / (0.85 * concrete.fc * b)
    Mpr = As * fy_pr * (d - a / 2.0)
    return Mpr / 1e6


def ve_capacity_design(  # noqa: D401
    *,
    Mpr_pos_i: float, Mpr_neg_i: float,
    Mpr_pos_j: float, Mpr_neg_j: float,
    ln: float, Vg: float = 0.0,
) -> float:
    """Design shear Ve per ACI 318-25 §18.6.5.1 + Fig. R18.6.5.

    Two seismic directions must be considered (Note 2 of Fig. R18.6.5).
    In each direction one end plastifies with top steel in tension
    (Mpr_neg) and the other with bottom steel in tension (Mpr_pos):

        Sway right (clockwise):
            Mpr at i = Mpr_neg_i   (top yields)
            Mpr at j = Mpr_pos_j   (bottom yields)
            Ve_R = (Mpr_neg_i + Mpr_pos_j) / ln + Vg

        Sway left (counter-clockwise):
            Mpr at i = Mpr_pos_i   (bottom yields)
            Mpr at j = Mpr_neg_j   (top yields)
            Ve_L = (Mpr_pos_i + Mpr_neg_j) / ln + Vg

    Ve_design = max(Ve_R, Ve_L).

    For continuous reinforcement (Mpr at i == Mpr at j) both directions
    are equal: Ve = (Mpr_pos + Mpr_neg) / ln + Vg.

    ln in m, moments in kN·m, Vg in kN. Returns kN.
    """
    if ln <= 0:
        raise ValueError(f"Clear span ln must be positive, got {ln}")
    Ve_right = (Mpr_neg_i + Mpr_pos_j) / ln
    Ve_left  = (Mpr_pos_i + Mpr_neg_j) / ln
    return max(Ve_right, Ve_left) + Vg
