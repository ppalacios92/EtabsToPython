"""Special boundary elements. ACI 318-25 §18.10.6."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from design.common.spacing import ash_required as _ash_required_column
from design.columns.interaction import InteractionDiagram


# ---------------------------------------------------------------------- #
# §18.10.6 BE-required checks
# ---------------------------------------------------------------------- #
def boundary_element_required_displacement(
    *,
    c: float,
    lw: float,
    delta_u: float,
    hw: float,
) -> bool:
    """Displacement-based BE check. ACI 318-25 §18.10.6.2(a), Eq. 18.10.6.2a.

    A special boundary element is required where

        c >= lw / (600 * (1.5 * delta_u/hwcs))            (Eq. 18.10.6.2a)

    Note the **1.5** factor on the design displacement (added in ACI
    318-14): the wall must confine its compression zone for 1.5x the
    design displacement. The ratio ``delta_u/hwcs`` is **floored at
    0.005** (a floor, NOT a skip — stiff walls with low drift but large c
    still need a BE, R18.10.6.2(c)).

    Returns False when no displacement demand is given (``delta_u <= 0``)
    — the displacement method is not applicable without delta_u; use the
    stress method §18.10.6.3 in that case.
    """
    if delta_u <= 0 or hw <= 0:
        return False
    drift_ratio = max(delta_u / hw, 0.005)        # §18.10.6.2 floor
    c_threshold = lw / (600.0 * 1.5 * drift_ratio)
    return c >= c_threshold


def wall_drift_capacity(
    *,
    c: float,
    lw: float,
    b: float,
    Ve: float,
    Acv: float,
    fc: float,
) -> float:
    """Wall drift capacity delta_c/hwcs. ACI 318-25 Eq. 18.10.6.2b.

        delta_c/hwcs = (1/100) * (4 - (1/50)(lw/b)(c/b)
                                  - Ve/(8*sqrt(fc')*Acv))

    The normalized shear term ``Ve/(8 sqrt(fc') Acv)`` uses the US-unit
    coefficient 8*sqrt(fc' psi), which equals ``0.66*sqrt(fc' MPa)`` in SI
    (8 sqrt(psi) = 0.8 * 10 sqrt(psi) = 0.8 * 0.83 = 0.664). The value
    need not be taken less than 0.015 (§18.10.6.2). ``Ve`` in kN, ``Acv``
    in mm^2, ``fc`` in MPa, lengths in mm. Returns the (dimensionless)
    drift-capacity ratio.
    """
    from math import sqrt
    if b <= 0 or Acv <= 0 or fc <= 0:
        return 0.015
    Ve_N = abs(Ve) * 1000.0
    v_norm = Ve_N / (0.66 * sqrt(fc) * Acv)        # = Ve/(8 sqrt(fc' psi) Acv)
    cap = (4.0 - (1.0 / 50.0) * (lw / b) * (c / b) - v_norm) / 100.0
    return max(cap, 0.015)                          # §18.10.6.2 floor


def boundary_drift_capacity_ok(
    *,
    c: float,
    lw: float,
    b: float,
    Ve: float,
    Acv: float,
    fc: float,
    delta_u: float,
    hw: float,
) -> tuple[bool, float, float]:
    """§18.10.6.2(b)(iii): wall drift capacity must exceed 1.5x the drift
    demand, i.e. ``delta_c/hwcs >= 1.5*(delta_u/hwcs)``.

    Returns ``(ok, drift_capacity, required)`` where ``required`` is
    ``1.5 * delta_u/hwcs``. When there is no displacement demand the check
    is vacuously satisfied (returns True with required = 0).
    """
    cap = wall_drift_capacity(c=c, lw=lw, b=b, Ve=Ve, Acv=Acv, fc=fc)
    drift_demand = (delta_u / hw) if hw > 0 else 0.0
    required = 1.5 * drift_demand
    return cap >= required, cap, required


def boundary_width_ok(*, b: float, c: float, lw: float) -> tuple[bool, float]:
    """§18.10.6.2(b)(ii): compression-zone width check ``b >= sqrt(c*lw/40)``.

    Returns ``(ok, b_required)``.
    """
    from math import sqrt
    b_req = sqrt(c * lw / 40.0) if (c > 0 and lw > 0) else 0.0
    return b >= b_req, b_req


def boundary_element_required_stress(
    *,
    sigma_max_compressive: float,
    fc: float,
) -> bool:
    """Stress-based BE check. §18.10.6.3.

    BE required if extreme compressive fiber stress >= 0.20 fc'.
    Discontinued when it drops below 0.15 fc' but that hysteresis is
    not modeled here (single-pass check).
    """
    return sigma_max_compressive >= 0.20 * fc


# ---------------------------------------------------------------------- #
# §18.10.6.4 BE geometric requirements
# ---------------------------------------------------------------------- #
def boundary_extension_length(*, c: float, lw: float) -> float:
    """Required BE length along the wall. §18.10.6.4(a).

    Length >= max(c - 0.1 lw, c/2).
    """
    return max(c - 0.1 * lw, c / 2.0)


def be_thickness_minimum(*, lu: float, hw_over_lw: float, c: float, lw: float) -> float:
    """Minimum BE thickness. §18.10.6.4(c).

    >= 300 mm if hw/lw >= 2.0 and c/lw >= 3/8, else lu/16 (conservative).
    """
    base = lu / 16.0
    if hw_over_lw >= 2.0 and (c / lw) >= 0.375:
        return max(base, 300.0)
    return base


# ---------------------------------------------------------------------- #
# Proposal — what BE geometry to build when one is required
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BEGeometryProposal:
    length: float            # mm, along wall length lw
    thickness: float         # mm, across wall thickness tw
    reason: str              # 'displacement' | 'stress' | 'manual' | 'none'
    c_used: float            # neutral-axis depth that drove the proposal
    extension_above: float   # mm above critical section (§18.10.6.2)
    extension_below: float   # mm below critical section
    required: bool           # True if §18.10.6.2 or §18.10.6.3 dispatched it


def propose_be_geometry(
    *,
    c: float,
    lw: float,
    tw: float,
    hw: float,
    lu: float,
    delta_u: float,
    sigma_max_compressive: float,
    fc: float,
    min_length: float = 300.0,
    min_thickness: float | None = None,
    extension_factor: float = 1.0,
    Mu: float | None = None,
    Vu: float | None = None,
) -> BEGeometryProposal:
    """Minimum-ACI proposal of BE length and thickness given the demand state.

    `c` is the neutral-axis depth at the demand point (computed by the
    caller from the in-plane PM diagram). Returns a proposal even when
    no BE is strictly required — in that case `required=False` and the
    caller can ignore it.
    """
    hw_over_lw = hw / lw if lw > 0 else float("inf")
    be_d = boundary_element_required_displacement(
        c=c, lw=lw, delta_u=delta_u, hw=hw,
    )
    be_s = boundary_element_required_stress(
        sigma_max_compressive=sigma_max_compressive, fc=fc,
    )
    required = be_d or be_s

    length = max(boundary_extension_length(c=c, lw=lw), min_length)
    thickness = be_thickness_minimum(lu=lu, hw_over_lw=hw_over_lw, c=c, lw=lw)
    if min_thickness is not None:
        thickness = max(thickness, min_thickness)
    thickness = max(thickness, tw)

    # Vertical extension above/below the critical section. §18.10.6.2(b)(i):
    # the transverse reinforcement of the BE must extend at least
    # max(lw, Mu/(4 Vu)). When Mu/Vu are given we use that; otherwise we
    # fall back to lw (conservative — the floor of the requirement).
    if Mu is not None and Vu is not None and Vu > 0:
        lp = max(lw, abs(Mu) * 1000.0 / (4.0 * Vu))   # Mu kN·m, Vu kN -> mm
    else:
        lp = lw
    lp *= extension_factor
    extension_above = lp
    extension_below = lp

    reason = "displacement" if be_d else "stress" if be_s else "none"
    return BEGeometryProposal(
        length=length,
        thickness=thickness,
        reason=reason,
        c_used=c,
        extension_above=extension_above,
        extension_below=extension_below,
        required=required,
    )


# ---------------------------------------------------------------------- #
# Helpers consumed by the design pipeline
# ---------------------------------------------------------------------- #
def c_at_demand(
    diagram: InteractionDiagram,
    *,
    Pu: float,
    Mu: float,
) -> float:
    """Neutral-axis depth `c` at the envelope point that matches the demand.

    ACI 318-25 §18.10.6.2 defines `c` as the largest neutral-axis depth
    calculated for the **factored axial force Pu** and the **nominal
    moment strength** consistent with the design displacement. So we read
    the NOMINAL PM diagram at `Pn = Pu` (no phi division — Pu is the axial
    coordinate, Mn is the developed nominal moment). Interpolation:

      1. The `c` vs `Pn` curve is monotonic in `Pn`; interpolate `c` at
         `Pn = Pu`.
      2. Falls back to the balanced point when the demand is zero.

    This matches what an engineer reads from a hand-drawn PM diagram —
    "draw a horizontal line at Pu, read c from the envelope at that
    height" — and is independent of `Mu`. The Mu only enters via the
    PM-envelope ratio check, not via the c-extraction.
    """
    import numpy as np
    if abs(Pu) < 1e-9 and abs(Mu) < 1e-9:
        return float(diagram.balanced_point().c)

    pts = diagram.points
    Pn_arr = np.array([p.Pn for p in pts])
    c_arr = np.array([p.c for p in pts])
    Pn_target = Pu if abs(Pu) > 1e-9 else 0.0

    # Sort by Pn ascending and interpolate c. For Pu outside the diagram
    # range, np.interp clips to the endpoints (which is the conservative
    # answer at the extremes).
    order = np.argsort(Pn_arr)
    return float(np.interp(Pn_target, Pn_arr[order], c_arr[order]))


def a_at_demand(
    diagram: InteractionDiagram,
    *,
    Pu: float,
    Mu: float,
    beta1: float,
) -> float:
    """Whitney stress-block depth `a = β1·c` at the demand point.

    ACI 318-25 §22.2.2.4. Reuses :func:`c_at_demand` and multiplies by
    β1. Useful as a sanity check: `a` must fit inside the section
    (a ≤ h_section) — otherwise the section is over-reinforced in
    flexure-compression and additional concrete area is needed.
    """
    return c_at_demand(diagram, Pu=Pu, Mu=Mu) * beta1


def tension_controlled_at_demand(
    diagram: InteractionDiagram,
    *,
    Pu: float,
    eps_ty: float = 0.002,
) -> tuple[bool, float, float]:
    """Return (is_tension_controlled, eps_t_at_demand, phi_at_demand).

    ACI 318-25 §21.2.2 — tension-controlled when ``eps_t >= eps_ty + 0.003``.
    Interpolates ``eps_t`` and ``phi`` against ``Pn`` at the demand-point
    height. Useful diagnostic that tells the engineer which φ branch
    governs the wall flexure at this Pu.
    """
    import numpy as np
    pts = diagram.points
    Pn_arr = np.array([p.Pn for p in pts])
    eps_arr = np.array([p.eps_t for p in pts])
    phi_arr = np.array([p.phi for p in pts])
    order = np.argsort(Pn_arr)
    Pn_target = Pu if abs(Pu) > 1e-9 else 0.0
    eps_t = float(np.interp(Pn_target, Pn_arr[order], eps_arr[order]))
    phi = float(np.interp(Pn_target, Pn_arr[order], phi_arr[order]))
    tc = eps_t >= eps_ty + 0.003
    return tc, eps_t, phi


def ash_required_be(
    *,
    s: float,
    bc: float,
    Ag_be: float,
    Ach_be: float,
    fc: float,
    fyt: float,
    Pu_over_Ag_fc: float = 0.0,
    nl: int | None = None,
) -> float:
    """Confinement Ash for a BE per §18.10.6.4(g).

    Wraps the column equation (Table 18.7.5.4) with BE-specific gross/core
    areas (Ag_be, Ach_be) and the BE core dimension `bc`. ``nl`` is the
    number of longitudinal bars laterally supported around the BE core
    (feeds expression (c) kn). Used to detail the hoops of the BE
    separately from the wall web.
    """
    return _ash_required_column(
        s=s, bc=bc, Ag=Ag_be, Ach=Ach_be,
        fc=fc, fyt=fyt, Pu_over_Ag_fc=Pu_over_Ag_fc, nl=nl,
    )
