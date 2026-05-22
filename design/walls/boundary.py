"""Special boundary elements. ACI 318-25 §18.10.6."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from design.columns.confinement import ash_required as _ash_required_column
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
    """Displacement-based BE check. §18.10.6.2.

    BE required if delta_u/hw >= 0.005 AND c >= lw / (600 * delta_u/hw).
    """
    drift_ratio = delta_u / hw if hw > 0 else 0.0
    if drift_ratio < 0.005:
        return False
    if 600.0 * drift_ratio <= 0:
        return False
    c_threshold = lw / (600.0 * drift_ratio)
    return c >= c_threshold


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

    # Extension above/below the critical section. §18.10.6.2 demands the
    # larger of lw and Mu/(4 Vu). We approximate by lw (conservative).
    lp = max(lw, hw / 4.0) * extension_factor
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
    """Neutral-axis depth at the PM point closest to the demand.

    Falls back to the balanced point if the demand has zero magnitude.
    """
    if abs(Pu) < 1e-9 and abs(Mu) < 1e-9:
        return diagram.balanced_point().c
    Pn_target = Pu / 0.9 if abs(Pu) > 1e-9 else Pu
    Mn_target = Mu / 0.9 if abs(Mu) > 1e-9 else Mu
    best = min(
        diagram.points,
        key=lambda p: (p.Pn - Pn_target) ** 2 + (p.Mn - Mn_target) ** 2,
    )
    return float(best.c)


def ash_required_be(
    *,
    s: float,
    bc: float,
    Ag_be: float,
    Ach_be: float,
    fc: float,
    fyt: float,
    Pu_over_Ag_fc: float = 0.0,
) -> float:
    """Confinement Ash for a BE per §18.10.6.4(g).

    Wraps the column equation with BE-specific gross/core areas (Ag_be,
    Ach_be) and the BE core dimension `bc`. Used to detail the hoops of
    the BE separately from the wall web.
    """
    return _ash_required_column(
        s=s, bc=bc, Ag=Ag_be, Ach=Ach_be,
        fc=fc, fyt=fyt, Pu_over_Ag_fc=Pu_over_Ag_fc,
    )
