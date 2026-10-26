"""Result-view proxies for Beam (Mn, Pn, phi_Mn, phi_Pn).

Each proxy holds a reference to its owning Beam and the name of the
underlying field in the InteractionDiagram (e.g. 'Mn', 'phi_Mn'). It
supports:

    beam.Mn(direction='pos')                 -> full curve as np.ndarray
    beam.Mn.at(Nu=0, direction='pos')        -> scalar interpolated at Nu
    beam.Mn.peak(direction='pos')            -> max(Mn) on that direction
    beam.Mn.peak()                           -> max over both directions

Indexing is supported:
    beam.Mn['pos']                           == beam.Mn(direction='pos')
    beam.Mn['pos', 0]                        == beam.Mn.at(direction='pos', Nu=0)
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from design.beams.beam import Beam


Direction = Literal["pos", "neg"]


class _BeamFieldView:
    """Generic proxy over a scalar field of InteractionDiagram."""

    __slots__ = ("_beam", "_field")

    def __init__(self, beam: "Beam", field: str) -> None:
        self._beam = beam
        self._field = field

    def __call__(self, *, direction: Direction = "pos") -> np.ndarray:
        diagram = self._beam.pm.diagram(direction)
        return getattr(diagram, self._field)

    def at(self, *, Nu: float, direction: Direction = "pos") -> float:
        diagram = self._beam.pm.diagram(direction)
        x = diagram.phi_Pn if self._field.startswith("phi_") else diagram.Pn
        y = getattr(diagram, self._field)
        order = np.argsort(x)
        return float(np.interp(Nu, x[order], y[order], left=0.0, right=0.0))

    def peak(self, *, direction: Direction | None = None) -> float:
        if direction is None:
            return max(
                float(np.max(getattr(self._beam.pm.diagram("pos"), self._field))),
                float(np.max(getattr(self._beam.pm.diagram("neg"), self._field))),
            )
        return float(np.max(getattr(self._beam.pm.diagram(direction), self._field)))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) == 2:
                direction, Nu = key
                return self.at(Nu=Nu, direction=direction)
            raise TypeError("Use ['pos'|'neg'] or ['pos'|'neg', Nu].")
        return self(direction=key)

    def __repr__(self) -> str:
        return f"<{self._field}View beam={self._beam.label!r}>"
