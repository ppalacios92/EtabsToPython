"""Wall piers. ACI 318-25 §18.10.8."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


PierClass = Literal["column-like", "wall-like"]


@dataclass(frozen=True, slots=True)
class WallPierClassification:
    hw_over_lw: float
    lw_over_bw: float
    classification: PierClass
    design_path: str
    notes: tuple[str, ...]


def classify_wall_pier(*, hw: float, lw: float, bw: float) -> WallPierClassification:
    """Decide whether a wall pier is treated as a column or as a wall.

    §18.10.8.1: when lw/bw <= 2.5 -> design as SMF column (§18.7).
    When lw/bw > 2.5 -> design as wall with §18.10.8.1(a)-(e) tweaks.
    """
    hw_over_lw = hw / lw if lw > 0 else float("inf")
    lw_over_bw = lw / bw if bw > 0 else float("inf")
    notes: list[str] = []

    if lw_over_bw <= 2.5:
        notes.append("lw/bw <= 2.5: design as SMF column per §18.7.")
        return WallPierClassification(
            hw_over_lw=hw_over_lw,
            lw_over_bw=lw_over_bw,
            classification="column-like",
            design_path="§18.7 SMF column",
            notes=tuple(notes),
        )
    notes.append("lw/bw > 2.5: design as wall with §18.10.8 modifications.")
    notes.append("Hoop spacing sv <= 150 mm (§18.10.8.1(d)).")
    notes.append("Extend transverse reinforcement >= 300 mm above/below clear height (§18.10.8.1(e)).")
    return WallPierClassification(
        hw_over_lw=hw_over_lw,
        lw_over_bw=lw_over_bw,
        classification="wall-like",
        design_path="§18.10.4 / §18.10.8",
        notes=tuple(notes),
    )


class WallPier:
    __slots__ = ("hw", "lw", "bw")

    def __init__(self, *, hw: float, lw: float, bw: float) -> None:
        self.hw = hw
        self.lw = lw
        self.bw = bw

    def classify(self) -> WallPierClassification:
        return classify_wall_pier(hw=self.hw, lw=self.lw, bw=self.bw)
