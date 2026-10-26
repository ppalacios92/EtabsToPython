"""Slenderness and moment magnification for columns.

ACI 318-25 §6.2.5 (slender vs non-slender threshold), §6.6.4
(non-sway moment magnification).
"""
from __future__ import annotations
from math import pi


def is_slender(
    *,
    k: float,
    lu: float,
    r: float,
    M1: float,
    M2: float,
    sway: bool = False,
) -> bool:
    """Return True if slenderness effects must be considered.

    ACI 318-25 §6.2.5 (non-sway): k lu / r > 34 + 12*(M1/M2), capped at 40.
    Sway (§6.2.5b): k lu / r > 22.

    SIGN CONVENTION (ACI 318-25, right-hand rule): M2 is the larger
    end moment (|M2| >= |M1|). The ratio M1/M2 is **NEGATIVE for single
    curvature** and **POSITIVE for double (reverse) curvature**. With
    this convention the limit grows from 22 (single curvature, M1/M2=-1)
    up to the 40 cap (double curvature), matching §6.2.5.
    """
    klu_r = k * lu / r
    if sway:
        return klu_r > 22.0
    limit = 34.0 + 12.0 * (M1 / M2 if abs(M2) > 1e-9 else 0.0)
    return klu_r > min(limit, 40.0)


def cm_factor(*, M1: float, M2: float, transverse_loaded: bool = False) -> float:
    """Equivalent uniform moment factor Cm. ACI 318-25 §6.6.4.5.3a.

    If transverse loads act between supports: Cm = 1.0 (§6.6.4.5.3b).
    Otherwise Cm = 0.6 - 0.4 * (M1/M2), clamped to a 0.4 floor.

    SIGN CONVENTION (ACI 318-25 §6.6.4.5.3, right-hand rule): M2 is the
    larger end moment (|M2| >= |M1|). The ratio M1/M2 is **NEGATIVE for
    single curvature** and **POSITIVE for double (reverse) curvature**.
    Hence single curvature (M1/M2 -> -1) drives Cm -> 1.0 (worst case),
    while double curvature (M1/M2 -> +1) drives Cm -> 0.2 (relieved,
    floored at 0.4). This is the post-2014 convention; pre-2014 editions
    wrote 0.6 + 0.4*(M1/M2) with the opposite sign on M1/M2.
    """
    if transverse_loaded:
        return 1.0
    cm = 0.6 - 0.4 * (M1 / M2 if abs(M2) > 1e-9 else 0.0)
    return max(cm, 0.4)


def moment_magnifier(
    *,
    Cm: float,
    Pu: float,
    Pc: float,
) -> float:
    """delta_ns = Cm / (1 - Pu/(0.75 Pc)). §6.6.4.5.

    Pu, Pc in kN. Returns dimensionless magnifier (>= 1.0).
    """
    denom = 1.0 - Pu / (0.75 * Pc)
    if denom <= 0:
        return float("inf")
    delta = Cm / denom
    return max(delta, 1.0)


def Pc(*, EI: float, k: float, lu: float) -> float:
    """Critical buckling load Pc = pi^2 EI / (k lu)^2.

    EI in kN·m^2, lu in m. Returns kN.
    """
    return pi ** 2 * EI / (k * lu) ** 2
