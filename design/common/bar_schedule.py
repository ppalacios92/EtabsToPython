"""User-preferred bar-diameter lists used by element.design().

The lists are intentionally short — these are the diameters Patricio
actually keeps in his designs. Override them at any time by assigning
new lists to the attributes.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(slots=True)
class BarSchedule:
    longitudinal: list[float] = field(
        default_factory=lambda: [12, 16, 20, 22, 25, 28, 32]
    )
    hoops: list[float] = field(
        default_factory=lambda: [10, 12, 16]
    )

    def __repr__(self) -> str:
        return (
            f"BarSchedule(longitudinal={self.longitudinal}, hoops={self.hoops})"
        )
