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

    Strain compatibility with the Whitney block (§22.2.2): the neutral-axis
    depth ``c`` is found by bisection on force equilibrium,

        0.85 fc b a + As' (fs' - 0.85 fc [if d' < a]) = As fs,   a = beta1 c,

    with fs = Es eps_s and fs' = Es eps_s' limited to +/- fy and the strains
    from eps_cu = 0.003 at the compression face. The "compression" layer may
    end up in tension when c < d' (thin slabs with equal top and bottom bars);
    it then adds to the tension force instead of being assumed to yield in
    compression. Moments are taken about the tension steel. Returns kN·m.
    """
    if As_prime <= 0.0:
        return mn_singly(b=b, d=d, As=As, concrete=concrete, steel=steel)

    Es = steel.Es
    fy = steel.fy
    fc = concrete.fc
    b1 = concrete.beta1
    ecu = concrete.eps_cu

    def forces(c: float):
        a = min(b1 * c, d)
        Cc = 0.85 * fc * a * b
        es_p = ecu * (c - d_prime) / c
        fs_p = max(-fy, min(fy, es_p * Es))
        if d_prime < a:
            fs_p -= 0.85 * fc                 # displaced concrete
        es = ecu * (d - c) / c
        fs = max(-fy, min(fy, es * Es))       # + tension
        return a, Cc, fs_p, fs

    def residual(c: float) -> float:
        _, Cc, fs_p, fs = forces(c)
        return Cc + As_prime * fs_p - As * fs

    lo, hi = 1e-6 * d, d
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-9 * d:
            break
    c = 0.5 * (lo + hi)
    a, Cc, fs_p, fs = forces(c)
    # moments about the tension steel
    Mn = Cc * (d - a / 2.0) + As_prime * fs_p * (d - d_prime)
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
