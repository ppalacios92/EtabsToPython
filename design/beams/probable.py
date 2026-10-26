"""Bidirectional probable strength (Mpr) via the strain-compat kernel.

ACI 318-25 §18.6.5.1 uses fy_pr = 1.25 fy and phi = 1.0 for the probable
moment. The closed-form `mpr(b, d, As, ...)` in ``seismic.py`` ignores
compression steel; this module uses the column kernel
(``probable_interaction_diagram``) which honors every bar's
contribution and is exact for any layout.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np

from design.common.probable import probable_interaction_diagram
from design.sections.base import Section


Direction = Literal["pos", "neg"]


@dataclass(frozen=True, slots=True)
class BeamProbable:
    """Probable moments evaluated at one axial load Nu (typically zero)."""
    Mpr_pos: float
    Mpr_neg: float
    Nu: float


def probable_pm_curve(
    section: Section,
    *,
    direction: Direction = "pos",
    n_points: int = 80,
):
    """Probable P-M diagram at one bending sense (fy_pr = 1.25 fy, phi = 1)."""
    angle = 0.0 if direction == "pos" else 180.0
    return probable_interaction_diagram(
        section, n_points=n_points, angle_deg=angle,
    )


def mpr_at_axial(
    section: Section,
    *,
    direction: Direction = "pos",
    Nu: float = 0.0,
    n_points: int = 80,
) -> float:
    """Interpolate Mpr from the probable diagram at axial load Nu (kN)."""
    d = probable_pm_curve(section, direction=direction, n_points=n_points)
    order = np.argsort(d.Pn)
    return float(np.interp(Nu, d.Pn[order], d.Mn[order], left=0.0, right=0.0))


def mpr_envelope_bidirectional(
    section: Section,
    *,
    Nu: float = 0.0,
    n_points: int = 80,
) -> BeamProbable:
    """Return (Mpr_pos, Mpr_neg) interpolated at the given axial load."""
    Mpr_pos = mpr_at_axial(section, direction="pos", Nu=Nu, n_points=n_points)
    Mpr_neg = mpr_at_axial(section, direction="neg", Nu=Nu, n_points=n_points)
    return BeamProbable(Mpr_pos=Mpr_pos, Mpr_neg=Mpr_neg, Nu=Nu)
