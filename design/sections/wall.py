"""Wall section: web + optional boundary elements at each end.

Local axes:
    y — along the wall length lw (top = +lw/2, bot = -lw/2)
    x — across thickness tw
With this convention, in-plane bending corresponds to angle_deg = 0
in the PMM machinery (neutral axis horizontal, sweeping y / lw) and
out-of-plane bending corresponds to angle_deg = 90 (sweeping x / tw).

The section is treated as **immutable** (frozen value-type semantics).
To add or resize boundary elements call `with_boundary_elements(...)`,
which returns a NEW WallSection. The Wall class uses this to evolve
its geometry across design iterations without mutating the original.
"""
from __future__ import annotations
import numpy as np

from design.common.materials import Concrete, Bar, Steel
from design.common.geometry import rectangle
from design.sections.base import Section
from design.sections.reinforcement import (
    Rebar, RebarGroup, RebarLayout, perimeter_bars,
)


class WallSection(Section):
    """Wall section: web + optional boundary elements at each end.

    Local axes:
        x — across thickness tw (web centerline at x=0)
        y — along wall length lw (top at +lw/2, bot at -lw/2)

    Symmetric (barbell) sections have BEs centered on x=0. Asymmetric
    sections (C-shape, L-shape, T-shape) shift one or both BEs along x
    via ``be_top_x_offset`` / ``be_bot_x_offset``.

    Useful offsets (for be_thickness > tw):
        ``+(be_thickness - tw) / 2`` — BE's left face flush with the
          web's left face; BE protrudes to the right.
        ``-(be_thickness - tw) / 2`` — BE's right face flush with the
          web's right face; BE protrudes to the left.
        ``0`` — BE centered on the web (symmetric barbell — default).
    """
    __slots__ = ("lw", "tw", "be_length_bot", "be_length_top",
                 "be_thickness_bot", "be_thickness_top",
                 "be_top_x_offset", "be_bot_x_offset")

    def __init__(
        self,
        *,
        lw: float,
        tw: float,
        concrete: Concrete,
        rebar: RebarLayout,
        be_length_bot: float = 0.0,
        be_length_top: float = 0.0,
        be_thickness_bot: float | None = None,
        be_thickness_top: float | None = None,
        be_top_x_offset: float = 0.0,
        be_bot_x_offset: float = 0.0,
    ) -> None:
        super().__init__(concrete=concrete, rebar=rebar)
        if lw <= 0 or tw <= 0:
            raise ValueError(f"lw and tw must be positive, got lw={lw}, tw={tw}")
        self.lw = lw
        self.tw = tw
        self.be_length_bot = be_length_bot
        self.be_length_top = be_length_top
        self.be_thickness_bot = be_thickness_bot if be_thickness_bot is not None else tw
        self.be_thickness_top = be_thickness_top if be_thickness_top is not None else tw
        self.be_top_x_offset = be_top_x_offset
        self.be_bot_x_offset = be_bot_x_offset

    # ------------------------------------------------------------------ #
    # Required Section ABC methods
    # ------------------------------------------------------------------ #
    def polygons(self) -> list[np.ndarray]:
        polys: list[np.ndarray] = []
        web_lw = self.lw - self.be_length_top - self.be_length_bot
        web_center_y = (self.be_length_bot - self.be_length_top) / 2.0
        polys.append(rectangle(self.tw, web_lw, center=(0.0, web_center_y)))

        if self.be_length_top > 0:
            cy = self.lw / 2 - self.be_length_top / 2
            polys.append(rectangle(
                self.be_thickness_top, self.be_length_top,
                center=(self.be_top_x_offset, cy),
            ))
        if self.be_length_bot > 0:
            cy = -self.lw / 2 + self.be_length_bot / 2
            polys.append(rectangle(
                self.be_thickness_bot, self.be_length_bot,
                center=(self.be_bot_x_offset, cy),
            ))
        return polys

    def gross_area(self) -> float:
        web_lw = self.lw - self.be_length_top - self.be_length_bot
        a = self.tw * web_lw
        a += self.be_thickness_top * self.be_length_top
        a += self.be_thickness_bot * self.be_length_bot
        return a

    def bounding_box(self) -> tuple[float, float, float, float]:
        # For asymmetric (C-shape) walls, the BE may extend further on
        # one side than the other. Compute the true x extent from all
        # three rectangles (web + each BE).
        x_min = -self.tw / 2
        x_max = +self.tw / 2
        if self.be_length_top > 0:
            x_min = min(x_min, self.be_top_x_offset - self.be_thickness_top / 2)
            x_max = max(x_max, self.be_top_x_offset + self.be_thickness_top / 2)
        if self.be_length_bot > 0:
            x_min = min(x_min, self.be_bot_x_offset - self.be_thickness_bot / 2)
            x_max = max(x_max, self.be_bot_x_offset + self.be_thickness_bot / 2)
        return (x_min, -self.lw / 2, x_max, self.lw / 2)

    @property
    def Acv(self) -> float:
        # ACI 318-25 §11.5.4 — gross concrete area resisting in-plane shear,
        # computed using the web thickness and full wall length.
        return self.tw * self.lw

    @property
    def has_boundary_elements(self) -> bool:
        return self.be_length_top > 0 or self.be_length_bot > 0

    # ------------------------------------------------------------------ #
    # Geometric partitioning (web vs BE)
    # ------------------------------------------------------------------ #
    def be_top_rect(self) -> tuple[float, float, float, float] | None:
        """(x_min, y_min, x_max, y_max) of the top BE, or None if absent."""
        if self.be_length_top <= 0:
            return None
        y_max = self.lw / 2
        y_min = y_max - self.be_length_top
        x_half = self.be_thickness_top / 2
        return (self.be_top_x_offset - x_half, y_min,
                self.be_top_x_offset + x_half, y_max)

    def be_bot_rect(self) -> tuple[float, float, float, float] | None:
        if self.be_length_bot <= 0:
            return None
        y_min = -self.lw / 2
        y_max = y_min + self.be_length_bot
        x_half = self.be_thickness_bot / 2
        return (self.be_bot_x_offset - x_half, y_min,
                self.be_bot_x_offset + x_half, y_max)

    @staticmethod
    def _bar_in_rect(rebar: Rebar, rect: tuple[float, float, float, float]) -> bool:
        x_min, y_min, x_max, y_max = rect
        return x_min <= rebar.x <= x_max and y_min <= rebar.y <= y_max

    def _split_groups(
        self,
    ) -> tuple[list[RebarGroup], list[RebarGroup], list[RebarGroup]]:
        """Split the existing rebar layout into (web, be_top, be_bot) groups.

        Bars inside the top BE rect go to BE-top, bars inside the bottom BE
        rect go to BE-bot, the rest stay in the web. Each output group
        preserves the steel grade of the source group.
        """
        rect_top = self.be_top_rect()
        rect_bot = self.be_bot_rect()
        web_groups: list[RebarGroup] = []
        top_groups: list[RebarGroup] = []
        bot_groups: list[RebarGroup] = []
        for g in self.rebar.groups:
            web_bars: list[Rebar] = []
            top_bars: list[Rebar] = []
            bot_bars: list[Rebar] = []
            for r in g.bars:
                if rect_top is not None and self._bar_in_rect(r, rect_top):
                    top_bars.append(r)
                elif rect_bot is not None and self._bar_in_rect(r, rect_bot):
                    bot_bars.append(r)
                else:
                    web_bars.append(r)
            if web_bars:
                web_groups.append(RebarGroup(bars=tuple(web_bars), steel=g.steel))
            if top_bars:
                top_groups.append(RebarGroup(bars=tuple(top_bars), steel=g.steel))
            if bot_bars:
                bot_groups.append(RebarGroup(bars=tuple(bot_bars), steel=g.steel))
        return web_groups, top_groups, bot_groups

    def web_rebar(self) -> RebarLayout:
        web, _, _ = self._split_groups()
        return RebarLayout(groups=tuple(web))

    def be_top_rebar(self) -> RebarLayout | None:
        if self.be_length_top <= 0:
            return None
        _, top, _ = self._split_groups()
        return RebarLayout(groups=tuple(top))

    def be_bot_rebar(self) -> RebarLayout | None:
        if self.be_length_bot <= 0:
            return None
        _, _, bot = self._split_groups()
        return RebarLayout(groups=tuple(bot))

    # ------------------------------------------------------------------ #
    # Convert each BE to a stand-alone rectangular section (Column input)
    # ------------------------------------------------------------------ #
    def be_top_as_rectangular(self) -> "BoundaryElementSection | None":
        if self.be_length_top <= 0:
            return None
        rebar = self.be_top_rebar() or RebarLayout()
        rect = self.be_top_rect()
        assert rect is not None
        x_min, y_min, x_max, y_max = rect
        center_x = (x_min + x_max) / 2.0     # = be_top_x_offset
        center_y = (y_min + y_max) / 2.0
        # Translate BE bars from wall-frame to BE-local frame (centered at 0,0)
        return BoundaryElementSection(
            b=self.be_thickness_top,
            h=self.be_length_top,
            concrete=self.concrete,
            rebar=_translate_rebar(rebar, dx=-center_x, dy=-center_y),
            wall_center_y=center_y,
        )

    def be_bot_as_rectangular(self) -> "BoundaryElementSection | None":
        if self.be_length_bot <= 0:
            return None
        rebar = self.be_bot_rebar() or RebarLayout()
        rect = self.be_bot_rect()
        assert rect is not None
        x_min, y_min, x_max, y_max = rect
        center_x = (x_min + x_max) / 2.0     # = be_bot_x_offset
        center_y = (y_min + y_max) / 2.0
        return BoundaryElementSection(
            b=self.be_thickness_bot,
            h=self.be_length_bot,
            concrete=self.concrete,
            rebar=_translate_rebar(rebar, dx=-center_x, dy=-center_y),
            wall_center_y=center_y,
        )

    # ------------------------------------------------------------------ #
    # Evolve geometry — returns a NEW section
    # ------------------------------------------------------------------ #
    def with_boundary_elements(
        self,
        *,
        top_length: float | None = None,
        bot_length: float | None = None,
        top_thickness: float | None = None,
        bot_thickness: float | None = None,
        top_x_offset: float | None = None,
        bot_x_offset: float | None = None,
        add_be_top_bars: RebarGroup | None = None,
        add_be_bot_bars: RebarGroup | None = None,
    ) -> "WallSection":
        """Return a NEW WallSection with updated BE geometry / rebar.

        Convention: the wall starts at its web. When a new BE is added
        (or grown), any pre-existing bars that fall inside the new BE
        rectangle are REMOVED from the web — the BE bars passed in
        through `add_be_*_bars` are the only bars left in that region.
        This avoids "stacked" bars (web + BE on top of each other).

        Pure geometry changes (no `add_be_*_bars` argument) keep every
        existing bar in place so callers can resize BEs without losing
        their reinforcement.
        """
        new_top_l = top_length if top_length is not None else self.be_length_top
        new_bot_l = bot_length if bot_length is not None else self.be_length_bot
        new_top_t = top_thickness if top_thickness is not None else self.be_thickness_top
        new_bot_t = bot_thickness if bot_thickness is not None else self.be_thickness_bot
        new_top_x = top_x_offset if top_x_offset is not None else self.be_top_x_offset
        new_bot_x = bot_x_offset if bot_x_offset is not None else self.be_bot_x_offset

        # Rectangles where new bars are being added — bars in those go.
        rect_top_new: tuple[float, float, float, float] | None = None
        if add_be_top_bars is not None and new_top_l > 0:
            y_max = self.lw / 2
            y_min = y_max - new_top_l
            x_half = new_top_t / 2
            rect_top_new = (new_top_x - x_half, y_min, new_top_x + x_half, y_max)

        rect_bot_new: tuple[float, float, float, float] | None = None
        if add_be_bot_bars is not None and new_bot_l > 0:
            y_min = -self.lw / 2
            y_max = y_min + new_bot_l
            x_half = new_bot_t / 2
            rect_bot_new = (new_bot_x - x_half, y_min, new_bot_x + x_half, y_max)

        # Filter pre-existing bars out of the new BE rectangles.
        filtered_groups: list[RebarGroup] = []
        for g in self.rebar.groups:
            kept: list[Rebar] = []
            for r in g.bars:
                in_top = rect_top_new is not None and self._bar_in_rect(r, rect_top_new)
                in_bot = rect_bot_new is not None and self._bar_in_rect(r, rect_bot_new)
                if not (in_top or in_bot):
                    kept.append(r)
            if kept:
                filtered_groups.append(RebarGroup(bars=tuple(kept), steel=g.steel))

        if add_be_top_bars is not None:
            filtered_groups.append(add_be_top_bars)
        if add_be_bot_bars is not None:
            filtered_groups.append(add_be_bot_bars)

        return WallSection(
            lw=self.lw, tw=self.tw,
            concrete=self.concrete,
            rebar=RebarLayout(groups=tuple(filtered_groups)),
            be_length_top=new_top_l,
            be_length_bot=new_bot_l,
            be_thickness_top=new_top_t,
            be_thickness_bot=new_bot_t,
            be_top_x_offset=new_top_x,
            be_bot_x_offset=new_bot_x,
        )

    def __repr__(self) -> str:
        be = ""
        if self.has_boundary_elements:
            be = (f", BE top={self.be_length_top:.0f}×{self.be_thickness_top:.0f}, "
                  f"bot={self.be_length_bot:.0f}×{self.be_thickness_bot:.0f}")
        return f"WallSection(lw={self.lw:.0f}, tw={self.tw:.0f}{be})"


class BoundaryElementSection(Section):
    """Stand-alone rectangular view of a single boundary element.

    Created by `WallSection.be_top_as_rectangular()` /
    `WallSection.be_bot_as_rectangular()`. Behaves like a
    RectangularSection so the columns.Column machinery can consume it
    directly. Coordinates are local to the BE centroid; the original
    y-offset within the wall is kept in `wall_center_y` for plotting.
    """
    __slots__ = ("b", "h", "wall_center_y")

    def __init__(
        self,
        *,
        b: float,
        h: float,
        concrete: Concrete,
        rebar: RebarLayout,
        wall_center_y: float = 0.0,
    ) -> None:
        super().__init__(concrete=concrete, rebar=rebar)
        self.b = b
        self.h = h
        self.wall_center_y = wall_center_y

    def polygons(self) -> list[np.ndarray]:
        return [rectangle(self.b, self.h)]

    def gross_area(self) -> float:
        return self.b * self.h

    def bounding_box(self) -> tuple[float, float, float, float]:
        return (-self.b / 2, -self.h / 2, self.b / 2, self.h / 2)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _translate_rebar(rebar: RebarLayout, *, dx: float = 0.0, dy: float = 0.0) -> RebarLayout:
    """Return a copy of `rebar` with every bar translated by (dx, dy)."""
    new_groups = []
    for g in rebar.groups:
        new_bars = tuple(Rebar(x=r.x + dx, y=r.y + dy, area=r.area) for r in g.bars)
        new_groups.append(RebarGroup(bars=new_bars, steel=g.steel))
    return RebarLayout(groups=tuple(new_groups))


def be_perimeter_bars(
    *,
    be_thickness: float,
    be_length: float,
    center_y: float,
    cover: float,
    n_x: int,
    n_y: int,
    bar: Bar,
    steel: Steel,
    center_x: float = 0.0,
) -> RebarGroup:
    """Perimeter bar layout for a boundary element rectangle.

    `center_y` is the BE centroid in the wall's local coordinates
    (positive for top BE, negative for bottom). `center_x` shifts the
    BE in the thickness direction — 0 for a symmetric barbell, nonzero
    for asymmetric (C-shape) walls. Returned bars are in wall
    coordinates, ready to be added to a WallSection's RebarLayout.
    """
    return perimeter_bars(
        b=be_thickness, h=be_length, cover=cover,
        n_x=n_x, n_y=n_y, bar=bar, steel=steel,
        center=(center_x, center_y),
    )
