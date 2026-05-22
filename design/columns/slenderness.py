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

    Non-sway: k lu / r > 34 - 12*(M1/M2). Cap 40.
    Sway:     k lu / r > 22.
    """
    klu_r = k * lu / r
    if sway:
        return klu_r > 22.0
    limit = 34.0 - 12.0 * (M1 / M2 if abs(M2) > 1e-9 else 0.0)
    return klu_r > min(limit, 40.0)


def cm_factor(*, M1: float, M2: float, transverse_loaded: bool = False) -> float:
    """Equivalent uniform moment factor Cm. §6.6.4.5.3.

    If transverse loads act between supports: Cm = 1.0.
    Otherwise Cm = 0.6 + 0.4 * (M1/M2), where M1/M2 is positive for
    single-curvature bending.
    """
    if transverse_loaded:
        return 1.0
    cm = 0.6 + 0.4 * (M1 / M2 if abs(M2) > 1e-9 else 0.0)
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
