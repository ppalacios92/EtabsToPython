"""Rectangular concrete section."""
from __future__ import annotations
import numpy as np

from design.common.materials import Concrete
from design.common.geometry import rectangle
from design.sections.base import Section
from design.sections.reinforcement import RebarLayout


class RectangularSection(Section):
    __slots__ = ("b", "h")

    def __init__(self, *, b: float, h: float,
                 concrete: Concrete, rebar: RebarLayout) -> None:
        super().__init__(concrete=concrete, rebar=rebar)
        if b <= 0 or h <= 0:
            raise ValueError(f"b and h must be positive, got b={b}, h={h}")
        self.b = b
        self.h = h

    def polygons(self) -> list[np.ndarray]:
        return [rectangle(self.b, self.h)]

    def gross_area(self) -> float:
        return self.b * self.h

    def bounding_box(self) -> tuple[float, float, float, float]:
        return (-self.b / 2, -self.h / 2, self.b / 2, self.h / 2)
