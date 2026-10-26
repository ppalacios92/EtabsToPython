"""Rectangular concrete section."""
from __future__ import annotations
import numpy as np

from design.common.materials import Concrete
from design.common.geometry import rectangle
from design.sections.base import Section
from design.sections.reinforcement import RebarLayout


class RectangularSection(Section):
    __slots__ = ("b", "h", "clear_cover")

    def __init__(self, *, b: float, h: float,
                 concrete: Concrete, rebar: RebarLayout,
                 clear_cover: float = 25.0) -> None:
        """Rectangular reinforced-concrete section.

        ``clear_cover`` is the distance from the OUTSIDE face of the section
        to the OUTSIDE edge of the transverse reinforcement (ACI 318-25
        Tabla 18.7.5.4 definition for bc). Used by Column.design(), .check(),
        and the transverse-steel quantity computation. Default: 25 mm.
        """
        super().__init__(concrete=concrete, rebar=rebar)
        if b <= 0 or h <= 0:
            raise ValueError(f"b and h must be positive, got b={b}, h={h}")
        if clear_cover < 0:
            raise ValueError(f"clear_cover must be ≥ 0, got {clear_cover}")
        self.b = b
        self.h = h
        self.clear_cover = clear_cover

    def polygons(self) -> list[np.ndarray]:
        return [rectangle(self.b, self.h)]

    def gross_area(self) -> float:
        return self.b * self.h

    def bounding_box(self) -> tuple[float, float, float, float]:
        return (-self.b / 2, -self.h / 2, self.b / 2, self.h / 2)
