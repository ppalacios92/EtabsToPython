"""Bidirectional P-M interaction diagram for a concrete beam.

Beams are flexural-shear elements per ACI 318-25 §9 and §18.6, so the
common workflow uses closed-form Mn (mn_doubly) and Mpr (mpr). This
module restores the **inspection-grade** P-M curve, built from the
same strain-compatibility kernel the columns module uses.

Two senses matter for beams:
    +M: tension on the bottom fiber  (θ = 0   in the column kernel)
    -M: tension on the top fiber     (θ = 180 in the column kernel)

Use cases:
- Inspect Mn vs Nu for a coupled-frame beam with small axial.
- Plot the +M/-M envelopes alongside the closed-form scalars.
- Match the column-style API: ``beam.Mn(direction='pos')``,
  ``beam.phi_Mn.at(Nu=0, direction='pos')``, etc.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np

from design.columns.interaction import InteractionDiagram, interaction_diagram
from design.sections.base import Section


Direction = Literal["pos", "neg"]


@dataclass(frozen=True, slots=True)
class BeamPMCurve:
    """Two P-M curves stacked: positive and negative bending."""
    pos: InteractionDiagram   # tension at bottom fiber
    neg: InteractionDiagram   # tension at top fiber

    @property
    def Po(self) -> float:
        return self.pos.Po

    @property
    def To(self) -> float:
        return self.pos.To

    @property
    def Pn_max_phi(self) -> float:
        return self.pos.Pn_max_phi

    def diagram(self, direction: Direction) -> InteractionDiagram:
        return self.pos if direction == "pos" else self.neg

    def phi_Mn_at(self, *, Nu: float, direction: Direction) -> float:
        """phi*Mn at a given axial Nu (kN) for one bending sense."""
        d = self.diagram(direction)
        order = np.argsort(d.phi_Pn)
        return float(np.interp(Nu, d.phi_Pn[order], d.phi_Mn[order],
                               left=0.0, right=0.0))

    def Mn_at(self, *, Nu: float, direction: Direction) -> float:
        d = self.diagram(direction)
        order = np.argsort(d.Pn)
        return float(np.interp(Nu, d.Pn[order], d.Mn[order],
                               left=0.0, right=0.0))

    def peak(self, *, direction: Direction, reduced: bool = True) -> float:
        d = self.diagram(direction)
        return float(np.max(d.phi_Mn if reduced else d.Mn))

    def check(self, *, Nu: float, Mu_pos: float, Mu_neg: float
              ) -> tuple[bool, float]:
        """Return (passed, ratio) at one station given Nu, Mu_pos, Mu_neg."""
        cap_pos = self.phi_Mn_at(Nu=Nu, direction="pos")
        cap_neg = self.phi_Mn_at(Nu=Nu, direction="neg")
        r_pos = Mu_pos / cap_pos if cap_pos > 0 else float("inf")
        r_neg = Mu_neg / cap_neg if cap_neg > 0 else float("inf")
        ratio = max(r_pos, r_neg)
        return ratio <= 1.0, ratio


def beam_pm_curve(section: Section, *, n_points: int = 60) -> BeamPMCurve:
    """Build the bidirectional P-M curve for a beam section."""
    pos = interaction_diagram(section, n_points=n_points, angle_deg=0.0)
    neg = interaction_diagram(section, n_points=n_points, angle_deg=180.0)
    return BeamPMCurve(pos=pos, neg=neg)
