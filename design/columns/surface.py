"""User-facing PMM surface, owned by Column.

This is a thin facade over the lower-level InteractionSurface. It keeps
all PMM-surface operations bound to one object that hangs from the
column, so callers never import InteractionSurface directly.

API: col.surface[theta]                           -> InteractionDiagram
     col.surface.angles                           -> np.ndarray
     col.surface.at_Pu(Pu)                        -> (Mx, My) arrays
     col.surface.volume(n_Pu=25)                  -> (Pu, Mx, My) grids
     col.surface.check(Pu, Mux, Muy)              -> (passed, ratio)
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from design.columns.interaction import InteractionDiagram
    from design.columns.interaction_surface import InteractionSurface


class Surface:
    """Mapping-like view of the interaction surface owned by a Column."""

    __slots__ = ("_inner",)

    def __init__(self, inner: "InteractionSurface") -> None:
        self._inner = inner

    # ---- mapping-like access ----
    def __getitem__(self, theta_deg: float) -> "InteractionDiagram":
        return self._inner.at_angle(float(theta_deg))

    def __contains__(self, theta_deg: float) -> bool:
        return float(theta_deg) in self._inner.angles_deg

    def __iter__(self):
        return iter(self._inner.angles_deg)

    def __len__(self) -> int:
        return len(self._inner.angles_deg)

    @property
    def angles(self) -> np.ndarray:
        return np.asarray(self._inner.angles_deg)

    @property
    def diagrams(self) -> tuple:
        return self._inner.diagrams

    # ---- queries ----
    def at_Pu(self, Pu: float, *, full_circle: bool = True
              ) -> tuple[np.ndarray, np.ndarray]:
        """Biaxial envelope (Mx, My) at a fixed axial load."""
        return self._inner.envelope_at_Pu(float(Pu), full_circle=full_circle)

    def volume(self, *, n_Pu: int = 25,
               Pu_min: float | None = None,
               Pu_max: float | None = None,
               full_circle: bool = True
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the volume on a grid (Pu, theta). Returns (Pu, Mx, My) grids."""
        return self._inner.volume_slices(
            n_Pu=n_Pu, Pu_min=Pu_min, Pu_max=Pu_max, full_circle=full_circle,
        )

    def check(self, *, Pu: float, Mux: float, Muy: float
              ) -> tuple[bool, float]:
        """3D biaxial check. Returns (passed, ratio)."""
        return self._inner.check_demand(Pu=Pu, Mux=Mux, Muy=Muy)

    def __repr__(self) -> str:
        return f"<Surface n_angles={len(self)} theta=[{self.angles.min():.0f}..{self.angles.max():.0f}]°>"
