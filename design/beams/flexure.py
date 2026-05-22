"""Flexural strength of rectangular reinforced concrete beams.

ACI 318-25 §9, §22.2.
"""
from __future__ import annotations
from math import sqrt

from design.common.materials import Concrete, Steel
from design.common.factors import beta1


def rho_min_beam(fc: float, fy: float) -> float:
    """Minimum flexural tension reinforcement ratio. ACI 318-25 §9.6.1.2."""
    rho_1 = 0.25 * sqrt(fc) / fy
    rho_2 = 1.4 / fy
    return max(rho_1, rho_2)


def rho_balanced(fc: float, fy: float,
                 *, beta1_val: float | None = None,
                 Es: float = 200_000.0, eps_cu: float = 0.003) -> float:
    """Balanced reinforcement ratio.

    rho_b = 0.85 * beta1 * fc / fy * (eps_cu / (eps_cu + eps_y))

    With fy in MPa, eps_cu = 0.003 and Es = 200_000 MPa.
    """
    from design.common.factors import beta1
    b1 = beta1_val if beta1_val is not None else beta1(fc)
    eps_y = fy / Es
    return 0.85 * b1 * (fc / fy) * (eps_cu / (eps_cu + eps_y))


def rho_max_beam(fc: float, fy: float) -> float:
    """Maximum tension reinforcement ratio used by this module.

    rho_max = 0.5 * rho_balanced

    This is more restrictive than ACI §18.6.3.1 (0.025 cap for SMF) for
    typical material strengths and forces an ample ductility margin.
    For Grade 60 / fc=28 MPa it returns ~0.014.
    """
    return 0.5 * rho_balanced(fc, fy)


def rho_max_seismic(fy: float) -> float:
    """Legacy ACI §18.6.3.1 cap (0.025 / 0.020). Kept for reporting only.

    The module-wide rho_max is `rho_max_beam` (0.5 * rho_balanced),
    which is more restrictive.
    """
    return 0.025 if fy <= 420.0 else 0.020


def mn_singly(*, b: float, d: float, As: float,
              concrete: Concrete, steel: Steel) -> float:
    """Nominal flexural strength of a singly reinforced rectangular section.

    Assumes tension steel yields. Returns Mn in kN·m. Inputs in MPa, mm,
    output via T = As fy and lever arm (d - a/2).
    """
    a = (As * steel.fy) / (0.85 * concrete.fc * b)
    Mn = As * steel.fy * (d - a / 2.0)
    return Mn / 1e6


def mn_doubly(*, b: float, d: float, d_prime: float,
              As: float, As_prime: float,
              concrete: Concrete, steel: Steel) -> float:
    """Doubly reinforced rectangular beam. ACI 318-25 §22.2.

    Three branches:
      1. As_prime <= 0  →  defer to mn_singly (only tension steel).
      2. As ≈ As' (symmetric)  →  both steels yield, concrete block
         vanishes: Mn = As · fy · (d − d').
      3. General case: try with As' at fy; if that is consistent
         (eps_s' >= eps_y) use it. Otherwise iterate with elastic
         compression steel.
    """
    if As_prime <= 0.0:
        return mn_singly(b=b, d=d, As=As, concrete=concrete, steel=steel)

    Es = steel.Es
    fy = steel.fy

    # Symmetric or near-symmetric — closed-form limit.
    a_yield = (As - As_prime) * fy / (0.85 * concrete.fc * b)
    if abs(a_yield) < 0.5:
        Mn = min(As, As_prime) * fy * (d - d_prime)
        return Mn / 1e6

    # Standard branch: assume As' yields.
    if a_yield > 0:
        c_yield = a_yield / concrete.beta1
        eps_sp = (concrete.eps_cu * (c_yield - d_prime) / c_yield
                  if c_yield > 0 else 0.0)
        if eps_sp >= fy / Es:
            a = a_yield
            Mn = ((0.85 * concrete.fc * b * a) * (d - a / 2.0)
                  + As_prime * fy * (d - d_prime))
            return Mn / 1e6

    # Iterate with elastic compression steel.
    a = max(a_yield, 5.0)
    for _ in range(30):
        c = a / concrete.beta1
        if c <= d_prime:
            fs_p = 0.0
        else:
            fs_p = min(concrete.eps_cu * (c - d_prime) / c * Es, fy)
        a_new = (As * fy - As_prime * fs_p) / (0.85 * concrete.fc * b)
        if a_new <= 0:
            break
        if abs(a_new - a) < 0.1:
            a = a_new
            break
        a = a_new

    c = a / concrete.beta1
    fs_p = (min(concrete.eps_cu * (c - d_prime) / c * Es, fy)
            if c > d_prime else 0.0)
    Mn = ((0.85 * concrete.fc * b * a) * (d - a / 2.0)
          + As_prime * fs_p * (d - d_prime))
    return Mn / 1e6


def as_required_singly(*, Mu: float, b: float, d: float,
                       concrete: Concrete, steel: Steel,
                       phi: float = 0.9) -> float:
    """Solve As for a target Mu assuming tension-controlled.

    Inputs: Mu in kN·m, b, d in mm. Returns As in mm². The quadratic
    derived from Mu = phi * As fy (d - a/2) with a = As fy / (0.85 fc b).
    """
    Mu_Nmm = Mu * 1e6
    A = 0.5 * steel.fy * steel.fy / (0.85 * concrete.fc * b)
    B = -steel.fy * d
    C = Mu_Nmm / phi

    disc = B * B - 4 * A * C
    if disc < 0:
        # Section needs compression steel or bigger geometry
        return float("inf")
    As = (-B - sqrt(disc)) / (2 * A)
    return As


def is_tension_controlled(*, b: float, d: float, As: float,
                          concrete: Concrete, steel: Steel) -> bool:
    """Return True if a singly reinforced section is tension-controlled."""
    a = (As * steel.fy) / (0.85 * concrete.fc * b)
    c = a / concrete.beta1
    eps_t = concrete.eps_cu * (d - c) / c
    return eps_t >= steel.eps_ty + 0.003
