"""P-M-M interaction surface for a concrete column.

Built by sweeping the neutral-axis angle from 0° to 180° (the full
range is captured because the same section produces symmetric curves
on the negative side). For each angle a uniaxial P-M curve is
computed; stacking them yields a surface where for any axial load Pu
we can ask: "what biaxial moment vector (Mx, My) fits in the
envelope?".
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from design.columns.interaction import InteractionDiagram, interaction_diagram
from design.sections.base import Section


@dataclass(frozen=True, slots=True)
class InteractionSurface:
    angles_deg: tuple[float, ...]
    diagrams: tuple[InteractionDiagram, ...]
    spiral: bool

    def at_angle(self, angle_deg: float) -> InteractionDiagram:
        angles = np.asarray(self.angles_deg)
        idx = int(np.argmin(np.abs(angles - angle_deg)))
        return self.diagrams[idx]

    def envelope_at_Pu(self, Pu: float, *, full_circle: bool = True
                       ) -> tuple[np.ndarray, np.ndarray]:
        """For each angle, return (Mx, My) where the envelope crosses Pu.

        With `full_circle=True` (default), the locus is mirrored to cover
        the full 0..360° range so the polygon closes when plotted.
        """
        mx_list, my_list = [], []
        for theta, d in zip(self.angles_deg, self.diagrams):
            order = np.argsort(d.phi_Pn)
            Pn_sorted = d.phi_Pn[order]
            Mn_sorted = d.phi_Mn[order]
            Mn = float(np.interp(Pu, Pn_sorted, Mn_sorted, left=0.0, right=0.0))
            rad = np.deg2rad(theta)
            mx_list.append(Mn * np.cos(rad))
            my_list.append(Mn * np.sin(rad))
        mx = np.asarray(mx_list)
        my = np.asarray(my_list)
        if full_circle:
            mx = np.concatenate([mx, -mx[::-1]])
            my = np.concatenate([my, -my[::-1]])
        return mx, my

    def volume_slices(
        self,
        *,
        n_Pu: int = 30,
        Pu_min: float | None = None,
        Pu_max: float | None = None,
        full_circle: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the interaction volume on a grid of (Pu, theta).

        Returns three arrays of shape (n_Pu, n_angles[*2]):
            Pu_grid[i, j]  axial load at slice i
            Mx_grid[i, j]  Mx coordinate of the envelope on that slice
            My_grid[i, j]  My coordinate of the envelope on that slice

        Stack the rows to draw the surface as a wireframe; cut at a
        fixed i to get a single biaxial contour at Pu_grid[i, 0].
        """
        if Pu_min is None:
            Pu_min = min(d.To for d in self.diagrams)
        if Pu_max is None:
            Pu_max = min(d.Pn_max_phi for d in self.diagrams)

        Pu_axis = np.linspace(Pu_min, Pu_max, n_Pu)
        rows_mx: list[np.ndarray] = []
        rows_my: list[np.ndarray] = []
        for Pu in Pu_axis:
            mx, my = self.envelope_at_Pu(float(Pu), full_circle=full_circle)
            rows_mx.append(mx)
            rows_my.append(my)

        Mx_grid = np.stack(rows_mx, axis=0)
        My_grid = np.stack(rows_my, axis=0)
        Pu_grid = np.broadcast_to(Pu_axis[:, None], Mx_grid.shape).copy()
        return Pu_grid, Mx_grid, My_grid

    def check_demand(self, *, Pu: float, Mux: float, Muy: float) -> tuple[bool, float]:
        """Biaxial check using Bresler reciprocal load eq. §22.4.2.2 alt.

        1/phi_Pn ≈ 1/phi_Pn,x + 1/phi_Pn,y - 1/phi_Po (Bresler).
        Here we adapt: convert (Mux, Muy) to angle and magnitude,
        interpolate the envelope at that angle, then compare.
        """
        Mu_mag = float(np.hypot(Mux, Muy))
        if Mu_mag == 0.0:
            return Pu <= max(d.Pn_max_phi for d in self.diagrams), Pu / max(d.Pn_max_phi for d in self.diagrams)
        theta = np.degrees(np.arctan2(Muy, Mux)) % 180.0
        diagram = self.at_angle(theta)
        order = np.argsort(diagram.phi_Pn)
        Pn_sorted = diagram.phi_Pn[order]
        Mn_sorted = diagram.phi_Mn[order]
        Mn_cap = float(np.interp(Pu, Pn_sorted, Mn_sorted, left=0.0, right=0.0))
        if Mn_cap <= 0.0:
            return False, float("inf")
        ratio = Mu_mag / Mn_cap
        return ratio <= 1.0, ratio


def interaction_surface(
    section: Section,
    *,
    n_angles: int = 19,
    n_points_per_curve: int = 60,
    spiral: bool = False,
    angle_start: float = 0.0,
    angle_end: float = 180.0,
) -> InteractionSurface:
    """Build a PMM surface by sweeping neutral-axis angles."""
    angles = np.linspace(angle_start, angle_end, n_angles)
    diagrams: list[InteractionDiagram] = []
    for theta in angles:
        diagrams.append(
            interaction_diagram(
                section,
                n_points=n_points_per_curve,
                angle_deg=float(theta),
                spiral=spiral,
            )
        )
    return InteractionSurface(
        angles_deg=tuple(float(t) for t in angles),
        diagrams=tuple(diagrams),
        spiral=spiral,
    )
