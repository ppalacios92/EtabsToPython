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
    def Mx(self) -> np.ndarray:
        """Nominal moment about global X for every point (kN·m)."""
        return np.array([p.Mx for p in self.points])

    @property
    def My(self) -> np.ndarray:
        """Nominal moment about global Y for every point (kN·m)."""
        return np.array([p.My for p in self.points])

    @property
    def phi_Mx(self) -> np.ndarray:
        """φ-reduced moment about global X (per-point φ; no axial cap)."""
        return np.array([p.phi * p.Mx for p in self.points])

    @property
    def phi_My(self) -> np.ndarray:
        return np.array([p.phi * p.My for p in self.points])

    @property
    def phi_Pn(self) -> np.ndarray:
        return self._capped_phi()[0]

    @property
    def phi_Mn(self) -> np.ndarray:
        return self._capped_phi()[1]

    def _capped_phi(self) -> tuple[np.ndarray, np.ndarray]:
        """Reduced (φ) curve with a clean horizontal top at φPn,max (§22.4.2.1).

        The raw reduced curve rises monotonically in φPn from pure tension up
        to φ·Po; near pure compression the geometric c-sampling yields points
        whose φMn is small and noisy. Naively clipping only φPn to the cap
        (leaving φMn untouched) piles those noisy points at φPn = cap and the
        curve "returns" toward the axis and back — a plotting artifact.

        Here we instead rebuild the upper end as a flat plateau:

            ... ascending points with φPn < cap ...
            (M_corner, cap)   corner: φMn interpolated where the curve hits cap
            (0,        cap)   apex:   closes the top to the axial axis

        Points strictly above the cap are discarded. The corner precedes the
        apex so np.interp(Pu, φPn, φMn) — used by at()/peak()/check_demand/
        at_Pu — returns the ascending φMn for every Pu below the cap.

        The cap is applied ONLY to a reduced design diagram. The probable
        (Mpr) diagram reports φ = 1.0 on every point and must keep its full
        nominal curve, so it is returned uncapped.
        """
        pP = np.array([p.phi_Pn for p in self.points])
        pM = np.array([p.phi_Mn for p in self.points])
        cap = self.Pn_max_phi
        # Probable / unreduced diagram (phi == 1 everywhere): no §22.4.2.1 cap.
        if all(p.phi >= 0.999 for p in self.points):
            return pP, pM
        above = pP > cap
        if not above.any():
            return pP, pM
        i = int(np.argmax(above))          # first index above the cap
        if i == 0:
            return pP, pM                  # degenerate; nothing below to anchor
        lo_P, hi_P = pP[i - 1], pP[i]
        lo_M, hi_M = pM[i - 1], pM[i]
        t = (cap - lo_P) / (hi_P - lo_P) if hi_P != lo_P else 0.0
        m_corner = lo_M + t * (hi_M - lo_M)
        new_P = np.concatenate([pP[:i], [cap, cap]])
        new_M = np.concatenate([pM[:i], [m_corner, 0.0]])
        return new_P, new_M

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
    axis. Intermediate points come from strain compatibility. The
    NOMINAL ``Pn``/``Mn`` arrays span the full curve up to Po; the
    reduced ``phi_Pn``/``phi_Mn`` are clipped with a clean horizontal
    top at φPn,max per §22.4.2.1 (see :meth:`InteractionDiagram._capped_phi`).
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
