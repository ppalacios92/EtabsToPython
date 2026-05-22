"""Section abstract base.

A Section owns its geometry (one or more concrete polygons in local
coordinates), its concrete material, and a RebarLayout. The analysis
engine consumes those three pieces — it never asks the section for
anything else.

Local axes convention (used by every concrete Section subclass):
    x — horizontal, positive to the right
    y — vertical, positive upward
    Origin at the geometric centroid of the gross section unless noted.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from design.common.materials import Concrete
from design.sections.reinforcement import RebarLayout


class Section(ABC):
    __slots__ = ("concrete", "rebar")

    def __init__(self, *, concrete: Concrete, rebar: RebarLayout) -> None:
        self.concrete = concrete
        self.rebar = rebar

    @abstractmethod
    def polygons(self) -> list[np.ndarray]:
        """Return concrete polygons in local coords. Each (N, 2) CCW."""

    @abstractmethod
    def gross_area(self) -> float: ...

    @abstractmethod
    def bounding_box(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) of the gross section."""

    def y_top(self) -> float:
        return self.bounding_box()[3]

    def y_bottom(self) -> float:
        return self.bounding_box()[1]

    def height(self) -> float:
        x0, y0, x1, y1 = self.bounding_box()
        return y1 - y0

    def width(self) -> float:
        x0, y0, x1, y1 = self.bounding_box()
        return x1 - x0
