"""Result-view proxies for Column (Mn, Pn, phi_Mn, phi_Pn).

Each proxy holds a reference to its owning Column and the name of the
underlying field in the InteractionDiagram (e.g. 'Mn', 'phi_Mn'). It
supports three call shapes:

    col.Mn(angle=0)                     -> full curve as np.ndarray
    col.Mn.at(Pu=0, angle=0)            -> scalar interpolated at Pu, theta
    col.Mn.peak(angle=0)                -> max(Mn) at theta  (balanced moment)
    col.Mn.peak()                       -> max(Mn) over all theta

Indexing is also supported:
    col.Mn[0]                           == col.Mn(angle=0)
    col.Mn[0, 2500]                     == col.Mn.at(angle=0, Pu=2500)
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from design.columns.column import Column


class _DiagramFieldView:
    """Generic proxy over a scalar field of InteractionDiagram (Mn, Pn, phi_Mn, phi_Pn)."""

    __slots__ = ("_column", "_field")

    def __init__(self, column: "Column", field: str) -> None:
        self._column = column
        self._field = field

    # ---- curve access ----
    def __call__(self, *, angle: float = 0.0) -> np.ndarray:
        diagram = self._column.surface[angle]
        return getattr(diagram, self._field)

    # ---- scalar interpolation ----
    def at(self, *, Pu: float, angle: float = 0.0) -> float:
        """Interpolate the field at a given Pu on the curve at `angle`.

        The underlying parameter is the neutral-axis depth c, but for
        engineering use it is more natural to interpolate against Pn
        (or phi_Pn for the reduced field).
        """
        diagram = self._column.surface[angle]
        # Pn axis to interpolate against
        if self._field.startswith("phi_"):
            x = diagram.phi_Pn
        else:
            x = diagram.Pn
        y = getattr(diagram, self._field)
        order = np.argsort(x)
        return float(np.interp(Pu, x[order], y[order], left=0.0, right=0.0))

    # ---- reductions ----
    def peak(self, *, angle: float | None = None) -> float:
        """Maximum of the field. If angle is None, sweeps every angle."""
        if angle is None:
            best = -np.inf
            for theta in self._column.surface.angles:
                v = float(np.max(getattr(self._column.surface[theta], self._field)))
                if v > best:
                    best = v
            return best
        return float(np.max(getattr(self._column.surface[angle], self._field)))

    # ---- indexing ----
    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) == 2:
                angle, Pu = key
                return self.at(Pu=Pu, angle=angle)
            raise TypeError("Use [angle] or [angle, Pu].")
        return self(angle=float(key))

    def __repr__(self) -> str:
        return f"<{self._field}View column={self._column.label!r}>"
