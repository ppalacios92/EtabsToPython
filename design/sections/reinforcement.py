"""Reinforcement layout.

A RebarLayout is just a list of (x, y, area) triplets in section-local
coords. Constructors below build common patterns; nothing stops a user
from passing arbitrary positions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from design.common.materials import Bar, Steel


class RebarArrays(NamedTuple):
    """Vectorized view of a RebarLayout, one row per bar.

    All arrays have the same length (= total number of bars). Steel
    properties are broadcast from each bar's owning group, so a layout
    with mixed grades (e.g. Grade 60 web + Grade 80 boundary elements)
    keeps per-bar fy / Es without flattening to a single material.
    """
    x: np.ndarray       # mm
    y: np.ndarray       # mm
    area: np.ndarray    # mm^2
    fy: np.ndarray      # MPa
    Es: np.ndarray      # MPa


@dataclass(frozen=True, slots=True)
class Rebar:
    x: float
    y: float
    area: float


@dataclass(frozen=True, slots=True)
class RebarGroup:
    """Set of bars sharing one steel grade. Position is per-bar."""
    bars: tuple[Rebar, ...]
    steel: Steel

    @property
    def total_area(self) -> float:
        return sum(r.area for r in self.bars)

    @property
    def n(self) -> int:
        return len(self.bars)


@dataclass(frozen=True, slots=True)
class RebarLayout:
    """Container for all rebar groups in a section."""
    groups: tuple[RebarGroup, ...] = field(default_factory=tuple)

    @property
    def total_area(self) -> float:
        return sum(g.total_area for g in self.groups)

    def iter_bars(self):
        for g in self.groups:
            for r in g.bars:
                yield r, g.steel

    def as_arrays(self) -> RebarArrays:
        """Return per-bar (x, y, area, fy, Es) arrays — broadcast from groups."""
        x, y, a, fy, Es = [], [], [], [], []
        for r, s in self.iter_bars():
            x.append(r.x)
            y.append(r.y)
            a.append(r.area)
            fy.append(s.fy)
            Es.append(s.Es)
        return RebarArrays(
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float),
            area=np.asarray(a, dtype=float),
            fy=np.asarray(fy, dtype=float),
            Es=np.asarray(Es, dtype=float),
        )


# -------- builders --------

def perimeter_bars(
    *,
    b: float,
    h: float,
    cover: float,
    n_x: int,
    n_y: int,
    bar: Bar,
    steel: Steel,
    center: tuple[float, float] = (0.0, 0.0),
) -> RebarGroup:
    """Bars placed evenly along the four edges of a rectangle.

    n_x and n_y count bars along the bottom/top and left/right edges
    respectively; corner bars are shared.
    """
    if n_x < 2 or n_y < 2:
        raise ValueError("n_x and n_y must be >= 2 to seat corner bars.")

    cx, cy = center
    x0 = cx - b / 2 + cover
    x1 = cx + b / 2 - cover
    y0 = cy - h / 2 + cover
    y1 = cy + h / 2 - cover

    xs_h = np.linspace(x0, x1, n_x)
    ys_v = np.linspace(y0, y1, n_y)

    positions: set[tuple[float, float]] = set()
    for x in xs_h:
        positions.add((float(x), y0))
        positions.add((float(x), y1))
    for y in ys_v:
        positions.add((x0, float(y)))
        positions.add((x1, float(y)))

    bars = tuple(Rebar(x=x, y=y, area=bar.area) for (x, y) in sorted(positions))
    return RebarGroup(bars=bars, steel=steel)


def two_layer_bars(
    *,
    b: float,
    h: float,
    cover: float,
    n_top: int,
    n_bottom: int,
    top_bar: Bar,
    bottom_bar: Bar,
    steel: Steel,
    center: tuple[float, float] = (0.0, 0.0),
) -> RebarGroup:
    """Beam-style layout: one layer top, one layer bottom."""
    cx, cy = center
    y_top = cy + h / 2 - cover
    y_bot = cy - h / 2 + cover
    x0 = cx - b / 2 + cover
    x1 = cx + b / 2 - cover

    xs_t = np.linspace(x0, x1, n_top) if n_top > 1 else np.array([cx])
    xs_b = np.linspace(x0, x1, n_bottom) if n_bottom > 1 else np.array([cx])

    bars: list[Rebar] = []
    for x in xs_t:
        bars.append(Rebar(x=float(x), y=y_top, area=top_bar.area))
    for x in xs_b:
        bars.append(Rebar(x=float(x), y=y_bot, area=bottom_bar.area))
    return RebarGroup(bars=tuple(bars), steel=steel)
