"""Uniaxial P-M interaction diagram for an arbitrary concrete section.

Wraps `pm_curve` and adds the strength caps from §22.4.2.1
(Pn,max = 0.80 Po ties / 0.85 Po spirals).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from design.analysis.section_analysis import (
    PMPoint, PMResult, pm_curve, pn_max, to_capacity,
)
from design.sections.base import Section


@dataclass(frozen=True, slots=True)
class InteractionDiagram:
    points: tuple[PMPoint, ...]
    Po: float
    Pn_max_phi: float
    To: float
    angle_deg: float
    spiral: bool

    @property
    def Pn(self) -> np.ndarray:
        return np.array([p.Pn for p in self.points])

    @property
    def Mn(self) -> np.ndarray:
        return np.array([p.Mn for p in self.points])

    @property
    def phi_Pn(self) -> np.ndarray:
        return np.clip(
            np.array([p.phi_Pn for p in self.points]),
            a_min=None, a_max=self.Pn_max_phi,
        )

    @property
    def phi_Mn(self) -> np.ndarray:
        return np.array([p.phi_Mn for p in self.points])

    def balanced_point(self) -> PMPoint:
        idx = int(np.argmax([p.Mn for p in self.points]))
        return self.points[idx]


def interaction_diagram(
    section: Section,
    *,
    n_points: int = 80,
    angle_deg: float = 0.0,
    spiral: bool = False,
) -> InteractionDiagram:
    """Build the P-M envelope at one neutral-axis orientation.

    Anchors the curve with closed-form endpoints (To, Mn=0) and
    (Po, Mn=0) so the diagram is closed at both ends of the axial
    axis. Intermediate points come from strain compatibility.
    """
    Po, Pn_max_phi = pn_max(section, spiral=spiral)
    To = -to_capacity(section)   # tension is negative in compression+ convention

    pm = pm_curve(
        section,
        n_points=max(n_points - 2, 4),
        angle_deg=angle_deg,
        spiral=spiral,
    )

    phi_tension = 0.90
    phi_compression = 0.75 if spiral else 0.65
    h = section.height()

    to_point = PMPoint(
        c=0.0, a=0.0,
        Pn=To, Mn=0.0,
        phi=phi_tension,
        phi_Pn=phi_tension * To, phi_Mn=0.0,
        eps_t=10.0 * section.concrete.eps_cu,   # arbitrary large tension strain
    )
    po_point = PMPoint(
        c=10.0 * h, a=section.concrete.beta1 * 10.0 * h,
        Pn=Po, Mn=0.0,
        phi=phi_compression,
        phi_Pn=phi_compression * Po, phi_Mn=0.0,
        eps_t=-section.concrete.eps_cu,
    )

    points = (to_point,) + pm.points + (po_point,)
    return InteractionDiagram(
        points=points,
        Po=Po,
        Pn_max_phi=Pn_max_phi,
        To=To,
        angle_deg=angle_deg,
        spiral=spiral,
    )


def demand_inside_envelope(
    diagram: InteractionDiagram,
    *,
    Pu: float,
    Mu: float,
) -> tuple[bool, float]:
    """Return (passed, ratio). Ratio is radial distance to envelope.

    Uses linear interpolation of (phi_Mn, phi_Pn) at the closest pair
    around Pu. Mu and Pu in kN·m, kN. Compression positive.
    """
    Pn_envelope = diagram.phi_Pn
    Mn_envelope = diagram.phi_Mn

    order = np.argsort(Pn_envelope)
    Pn_sorted = Pn_envelope[order]
    Mn_sorted = Mn_envelope[order]
    Mn_cap = float(np.interp(Pu, Pn_sorted, Mn_sorted, left=0.0, right=0.0))
    if Mn_cap <= 0:
        return False, float("inf")
    ratio = Mu / Mn_cap
    return ratio <= 1.0, ratio
