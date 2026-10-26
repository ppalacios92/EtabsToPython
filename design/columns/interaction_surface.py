"""P-M-M interaction surface for a concrete section.

Built by sweeping the neutral-axis angle and, for each angle, computing a
uniaxial P-M curve. For each point the kernel now reports the TRUE moment
components (Mx, My) about the global axes — so for asymmetric sections
(C/L/T), where the resultant moment is NOT aligned with the neutral-axis
angle, the surface uses the real direction instead of assuming
``(Mx, My) = Mn·(cosθ, sinθ)``.

For doubly-symmetric sections the sweep covers 0..180° and is mirrored to
the full circle. For asymmetric sections the central symmetry no longer
holds, so the sweep covers the full 0..360°.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from design.analysis.section_analysis import pm_at_neutral_axis, rotated_depth
from design.columns.interaction import InteractionDiagram, interaction_diagram
from design.sections.base import Section


def _is_doubly_symmetric(section: Section, *, tol_rel: float = 0.02) -> bool:
    """Cheap test: a section is treated as doubly symmetric if the moment
    transverse to the NA is negligible at both 0° and 90°.

    Evaluates one mid-depth neutral axis at each orientation and compares
    the off-axis moment to the in-axis one. Asymmetric layouts (BE offset,
    C/L/T) fail this and trigger a full 0..360° sweep.
    """
    h = rotated_depth(section, 0.0)
    p0 = pm_at_neutral_axis(section, c=0.4 * h, angle_deg=0.0)
    w = rotated_depth(section, 90.0)
    p90 = pm_at_neutral_axis(section, c=0.4 * w, angle_deg=90.0)
    ref = max(abs(p0.Mx), abs(p90.My), 1.0)
    return abs(p0.My) <= tol_rel * ref and abs(p90.Mx) <= tol_rel * ref


@dataclass(frozen=True, slots=True)
class InteractionSurface:
    angles_deg: tuple[float, ...]
    diagrams: tuple[InteractionDiagram, ...]
    spiral: bool
    symmetric: bool = True

    def at_angle(self, angle_deg: float) -> InteractionDiagram:
        angles = np.asarray(self.angles_deg)
        idx = int(np.argmin(np.abs(angles - angle_deg)))
        return self.diagrams[idx]

    @staticmethod
    def _phi_arrays(d: InteractionDiagram):
        """Per-point (phi_Pn, phi_Mx, phi_My), axial clamped to the §22.4.2.1
        cap, sorted by phi_Pn. Uses raw per-point φ (not the plateau-rebuilt
        uniaxial curve) so the three arrays stay aligned in length."""
        pn = np.array([p.phi * p.Pn for p in d.points])
        mx = np.array([p.phi * p.Mx for p in d.points])
        my = np.array([p.phi * p.My for p in d.points])
        pn = np.minimum(pn, d.Pn_max_phi)        # §22.4.2.1 axial cap
        order = np.argsort(pn)
        return pn[order], mx[order], my[order]

    def envelope_at_Pu(self, Pu: float, *, full_circle: bool = True
                       ) -> tuple[np.ndarray, np.ndarray]:
        """For each NA angle, the TRUE (Mx, My) where the envelope crosses Pu.

        Interpolates the real moment components against the (capped) reduced
        axial. For symmetric sections the 0..180° locus is mirrored through
        the origin to close the full circle; for asymmetric sections the
        diagrams already span 0..360°, so no mirror is applied.
        """
        mx_list, my_list = [], []
        for d in self.diagrams:
            pn, mx, my = self._phi_arrays(d)
            mx_list.append(float(np.interp(Pu, pn, mx, left=0.0, right=0.0)))
            my_list.append(float(np.interp(Pu, pn, my, left=0.0, right=0.0)))
        mx = np.asarray(mx_list)
        my = np.asarray(my_list)
        if full_circle and self.symmetric:
            # Central reflection of the 0..180° locus closes the contour CCW.
            # Use -P (NOT -P[::-1]): the reversed form makes the polygon jump
            # across the origin and self-intersect, breaking point-in-polygon.
            mx = np.concatenate([mx, -mx])
            my = np.concatenate([my, -my])
        return mx, my

    def volume_slices(
        self,
        *,
        n_Pu: int = 30,
        Pu_min: float | None = None,
        Pu_max: float | None = None,
        full_circle: bool = True,
        close_top: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the interaction volume on a grid of (Pu, theta).

        ``close_top`` appends a last row with Mx = My = 0 at ``Pu_max`` so the
        surface is closed by the flat phi*Pn,max plateau (§22.4.2.1) instead of
        ending in an open ring.
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
        if close_top:
            Mx_grid = np.vstack([Mx_grid, np.zeros((1, Mx_grid.shape[1]))])
            My_grid = np.vstack([My_grid, np.zeros((1, My_grid.shape[1]))])
            Pu_axis = np.r_[Pu_axis, Pu_axis[-1]]
        Pu_grid = np.broadcast_to(Pu_axis[:, None], Mx_grid.shape).copy()
        return Pu_grid, Mx_grid, My_grid

    def check_demand(self, *, Pu: float, Mux: float, Muy: float) -> tuple[bool, float]:
        """Biaxial check by point-in-polygon on the TRUE envelope at Pu.

        The demand point (Mux, Muy) is tested against the closed moment
        contour at the axial level Pu. The ratio is the radial distance of
        the demand to the contour boundary along its own direction — correct
        even when the resistant-moment direction differs from the NA angle
        (asymmetric sections), where the old ``θ_demand = θ_NA`` assumption
        failed.
        """
        mx, my = self.envelope_at_Pu(Pu, full_circle=True)
        if mx.size < 3:
            return False, float("inf")
        contour = np.column_stack([mx, my])
        inside = _point_in_polygon(Mux, Muy, contour)
        # radial ratio: |demand| / |boundary crossing along the demand ray|
        r_demand = float(np.hypot(Mux, Muy))
        if r_demand == 0.0:
            cap = max(d.Pn_max_phi for d in self.diagrams)
            return Pu <= cap, (Pu / cap if cap else float("inf"))
        r_bound = _ray_polygon_distance(Mux, Muy, contour)
        ratio = r_demand / r_bound if r_bound > 0 else float("inf")
        return bool(inside), ratio


# ---------------------------------------------------------------------- #
# geometry helpers
# ---------------------------------------------------------------------- #
def _point_in_polygon(px: float, py: float, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and \
                (px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi):
            inside = not inside
        j = i
    return inside


def _ray_polygon_distance(dx: float, dy: float, poly: np.ndarray) -> float:
    """Distance from origin to the polygon boundary along the ray (dx, dy)."""
    r = float(np.hypot(dx, dy))
    ux, uy = dx / r, dy / r
    n = len(poly)
    best = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        denom = ux * ey - uy * ex
        if abs(denom) < 1e-300:
            continue
        t = (x1 * ey - y1 * ex) / denom            # ray param (distance)
        s = (x1 * uy - y1 * ux) / (-denom)         # edge param in [0,1]
        if t >= 0 and 0.0 <= s <= 1.0:
            best = max(best, t)
    return best


def interaction_surface(
    section: Section,
    *,
    n_angles: int = 19,
    n_points_per_curve: int = 60,
    spiral: bool = False,
    angle_start: float = 0.0,
    angle_end: float | None = None,
) -> InteractionSurface:
    """Build a PMM surface by sweeping neutral-axis angles.

    The angular range is chosen automatically: 0..180° for doubly-symmetric
    sections (mirrored to the full circle at query time) and 0..360° for
    asymmetric sections, where central symmetry does not hold.
    """
    symmetric = _is_doubly_symmetric(section)
    if angle_end is None:
        angle_end = 180.0 if symmetric else 360.0
    angles = np.linspace(angle_start, angle_end, n_angles)
    diagrams = [
        interaction_diagram(
            section, n_points=n_points_per_curve,
            angle_deg=float(theta), spiral=spiral,
        )
        for theta in angles
    ]
    return InteractionSurface(
        angles_deg=tuple(float(t) for t in angles),
        diagrams=tuple(diagrams),
        spiral=spiral,
        symmetric=symmetric,
    )
