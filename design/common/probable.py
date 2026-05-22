"""Probable-strength interaction (Mpr) per ACI 318-25 §18.6.5 / §18.7.6.1.

`probable_interaction_diagram` is the same as `interaction_diagram`,
but uses **fy_pr = 1.25 fy** in every bar and reports **phi = 1.0**
(unreduced). Used for capacity-design quantities — i.e. the moment
the column/wall would develop if it actually plastified.

This module is the canonical home of the probable-strength helpers.
``design/columns/probable.py`` and ``design/walls/probable.py`` are
re-export shims pointing here.
"""
from __future__ import annotations
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from design.analysis.section_analysis import (
    PMPoint, pm_at_neutral_axis, pn_max, to_capacity,
)
from design.sections.reinforcement import (
    RebarGroup, RebarLayout,
)
from design.common.materials import Steel

if TYPE_CHECKING:
    from design.columns.interaction import InteractionDiagram
    from design.sections.base import Section


def _scale_layout_fy(rebar: RebarLayout, factor: float) -> RebarLayout:
    """Return a copy of `rebar` with every steel's fy multiplied by `factor`."""
    new_groups = []
    for g in rebar.groups:
        new_steel = Steel(fy=g.steel.fy * factor,
                          Es=g.steel.Es,
                          fu=g.steel.fu,
                          grade=g.steel.grade)
        new_groups.append(RebarGroup(bars=g.bars, steel=new_steel))
    return RebarLayout(groups=tuple(new_groups))


def probable_interaction_diagram(
    section: "Section",
    *,
    n_points: int = 80,
    angle_deg: float = 0.0,
    spiral: bool = False,
    fy_factor: float = 1.25,
) -> "InteractionDiagram":
    """Build the **probable** P-M envelope (fy_pr = 1.25 fy, phi = 1.0).

    The section is rebuilt with a scaled rebar layout that carries
    fy_pr in every bar; the rest of the analysis is identical to
    `interaction_diagram`.
    """
    # Lazy import to avoid circular dependency at module load:
    # design.columns.interaction -> design.columns.__init__ -> column.py
    # which imports from design.columns.probable (shim) -> us.
    from design.columns.interaction import InteractionDiagram

    # Build a section view with scaled fy so pm_at_neutral_axis sees fy_pr
    scaled_rebar = _scale_layout_fy(section.rebar, fy_factor)
    scaled_section = _SectionWithRebar(section, scaled_rebar)

    # Closed-form anchors
    Po_pr, _ = pn_max(scaled_section, spiral=spiral)
    To_pr = -to_capacity(scaled_section)

    h = section.height()
    c_min = 0.001 * h
    c_max = 5.0 * h
    cs = np.geomspace(c_min, c_max, max(n_points - 2, 4))
    pts: list[PMPoint] = []
    for c in cs:
        p = pm_at_neutral_axis(scaled_section, c=float(c),
                               angle_deg=angle_deg, spiral=spiral)
        # Force phi = 1.0 for probable strength
        pts.append(replace(p, phi=1.0, phi_Pn=p.Pn, phi_Mn=p.Mn))

    to_point = PMPoint(c=0.0, a=0.0, Pn=To_pr, Mn=0.0,
                       phi=1.0, phi_Pn=To_pr, phi_Mn=0.0,
                       eps_t=10.0 * section.concrete.eps_cu)
    po_point = PMPoint(c=10.0 * h, a=section.concrete.beta1 * 10.0 * h,
                       Pn=Po_pr, Mn=0.0,
                       phi=1.0, phi_Pn=Po_pr, phi_Mn=0.0,
                       eps_t=-section.concrete.eps_cu)
    points = (to_point,) + tuple(pts) + (po_point,)

    return InteractionDiagram(
        points=points,
        Po=Po_pr,
        Pn_max_phi=0.80 * Po_pr if not spiral else 0.85 * Po_pr,  # no φ
        To=To_pr,
        angle_deg=angle_deg,
        spiral=spiral,
    )


def mpr_envelope(
    section: "Section",
    *,
    angle_deg: float = 0.0,
    spiral: bool = False,
    Pu_range: tuple[float, float] | None = None,
    n_points: int = 80,
) -> tuple[float, float]:
    """Return (Mpr_max, Pu_at_Mpr_max) over a Pu range.

    If `Pu_range` is None, sweeps the whole probable diagram.
    """
    d = probable_interaction_diagram(
        section, n_points=n_points, angle_deg=angle_deg, spiral=spiral,
    )
    Pn = d.Pn
    Mn = d.Mn
    if Pu_range is not None:
        lo, hi = Pu_range
        mask = (Pn >= lo) & (Pn <= hi)
        if mask.any():
            Pn = Pn[mask]
            Mn = Mn[mask]
    i_max = int(np.argmax(Mn))
    return float(Mn[i_max]), float(Pn[i_max])


# ---- internal helper: temporary section with replaced rebar ----
class _SectionWithRebar:
    """View of a Section that reports a different RebarLayout."""

    __slots__ = ("_inner", "rebar", "concrete")

    def __init__(self, inner: Section, rebar: RebarLayout) -> None:
        self._inner = inner
        self.rebar = rebar
        self.concrete = inner.concrete

    def polygons(self):
        return self._inner.polygons()

    def gross_area(self) -> float:
        return self._inner.gross_area()

    def bounding_box(self):
        return self._inner.bounding_box()

    def height(self) -> float:
        return self._inner.height()

    def width(self) -> float:
        return self._inner.width()
