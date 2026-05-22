"""Probable-strength interaction (Mpr) for walls.

Thin wrappers around :mod:`design.common.probable` with explicit names
for in-plane vs out-of-plane analysis, since walls have a preferred
bending plane and that naming makes the design code easier to read.
"""
from __future__ import annotations

from design.columns.interaction import InteractionDiagram
from design.common.probable import probable_interaction_diagram, mpr_envelope
from design.sections.wall import WallSection


def probable_in_plane_diagram(
    section: WallSection,
    *,
    n_points: int = 80,
    fy_factor: float = 1.25,
) -> InteractionDiagram:
    """Probable in-plane PM envelope (fy_pr = 1.25 fy, phi = 1.0)."""
    return probable_interaction_diagram(
        section, n_points=n_points, angle_deg=0.0, fy_factor=fy_factor,
    )


def probable_out_of_plane_diagram(
    section: WallSection,
    *,
    n_points: int = 80,
    fy_factor: float = 1.25,
) -> InteractionDiagram:
    """Probable out-of-plane PM envelope."""
    return probable_interaction_diagram(
        section, n_points=n_points, angle_deg=90.0, fy_factor=fy_factor,
    )


def mpr_in_plane(
    section: WallSection,
    *,
    n_points: int = 80,
    Pu_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return (Mpr_max, Pu_at_Mpr_max) for in-plane bending."""
    return mpr_envelope(section, angle_deg=0.0, n_points=n_points, Pu_range=Pu_range)


def mpr_out_of_plane(
    section: WallSection,
    *,
    n_points: int = 80,
    Pu_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return (Mpr_max, Pu_at_Mpr_max) for out-of-plane bending."""
    return mpr_envelope(section, angle_deg=90.0, n_points=n_points, Pu_range=Pu_range)
