"""Plot helpers for PM and PMM diagrams.

matplotlib is imported lazily so the rest of the design package keeps
its numpy-only footprint.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from design.columns.interaction import InteractionDiagram
from design.columns.interaction_surface import InteractionSurface


def plot_pm_diagram(
    diagram: InteractionDiagram,
    *,
    ax=None,
    show_nominal: bool = True,
    show_reduced: bool = True,
    nominal_kwargs: Optional[dict] = None,
    reduced_kwargs: Optional[dict] = None,
):
    """2D plot of one P-M curve. Returns the Axes object."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    nk = {"color": "k", "lw": 1.3, "label": "Nominal"}
    nk.update(nominal_kwargs or {})
    rk = {"color": "tab:blue", "lw": 1.6, "label": r"$\phi\,M_n\,/\,\phi\,P_n$"}
    rk.update(reduced_kwargs or {})

    if show_nominal:
        ax.plot(diagram.Mn, diagram.Pn, **nk)
        ax.plot(-diagram.Mn, diagram.Pn, color=nk["color"], lw=nk["lw"])
    if show_reduced:
        ax.plot(diagram.phi_Mn, diagram.phi_Pn, **rk)
        ax.plot(-diagram.phi_Mn, diagram.phi_Pn, color=rk["color"], lw=rk["lw"])

    ax.axhline(diagram.Pn_max_phi, color="red", ls="--", lw=0.8,
               label=r"$\phi\,P_{n,\max}$")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Moment (kN·m)")
    ax.set_ylabel("Axial load (kN, + compression)")
    ax.set_title(f"P-M @ θ = {diagram.angle_deg:.0f}°")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_pmm_slice(
    surface: InteractionSurface,
    *,
    Pu: float,
    ax=None,
    demand: Optional[tuple[float, float]] = None,
    fill: bool = True,
):
    """2D contour of admissible (Mx, My) at a fixed Pu.

    `demand`: optional (Mux, Muy) point drawn on top.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    Mx, My = surface.envelope_at_Pu(Pu, full_circle=True)
    if fill:
        ax.fill(Mx, My, alpha=0.15, color="tab:blue")
    ax.plot(Mx, My, color="tab:blue", lw=1.5,
            label=f"Envelope @ Pu = {Pu:.0f} kN")

    if demand is not None:
        Mux, Muy = demand
        ax.plot(Mux, Muy, marker="o", color="red", ms=8, label="Demand")

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$M_x$ (kN·m)")
    ax.set_ylabel(r"$M_y$ (kN·m)")
    ax.set_title(f"Biaxial slice at Pu = {Pu:.0f} kN")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_pmm_volume(
    surface: InteractionSurface,
    *,
    ax=None,
    n_Pu: int = 25,
    Pu_min: float | None = None,
    Pu_max: float | None = None,
    surface_alpha: float = 0.25,
    wireframe: bool = True,
    cmap: str = "viridis",
):
    """3D plot of the interaction volume. Returns the 3D Axes."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)

    if ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

    Pu_grid, Mx_grid, My_grid = surface.volume_slices(
        n_Pu=n_Pu, Pu_min=Pu_min, Pu_max=Pu_max, full_circle=True,
    )

    if wireframe:
        ax.plot_wireframe(Mx_grid, My_grid, Pu_grid,
                          rstride=2, cstride=2, lw=0.5, color="tab:blue", alpha=0.6)
    else:
        ax.plot_surface(Mx_grid, My_grid, Pu_grid,
                        cmap=cmap, alpha=surface_alpha, edgecolor="none")

    ax.set_xlabel(r"$M_x$ (kN·m)")
    ax.set_ylabel(r"$M_y$ (kN·m)")
    ax.set_zlabel("Pu (kN)")
    ax.set_title("P-M-M interaction volume")
    return ax


def plot_pmm_with_demand(
    surface: InteractionSurface,
    *,
    Pu: float,
    Mux: float,
    Muy: float,
    ax=None,
    n_Pu: int = 25,
):
    """3D volume + demand point + Pu-slice ring."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

    plot_pmm_volume(surface, ax=ax, n_Pu=n_Pu)

    Mx_ring, My_ring = surface.envelope_at_Pu(Pu, full_circle=True)
    Pu_ring = np.full_like(Mx_ring, Pu)
    ax.plot(Mx_ring, My_ring, Pu_ring, color="tab:red", lw=1.8,
            label=f"Slice @ Pu = {Pu:.0f}")
    ax.scatter([Mux], [Muy], [Pu], color="black", s=60, label="Demand")
    ax.legend(loc="upper right", fontsize=9)
    return ax
