"""Column plot proxy.

Bound to a Column instance. Reads col.units and col.plot.style for
all visualization. matplotlib is imported lazily.

API: col.plot.section()
     col.plot.pm(angle=0)
     col.plot.pm_compare(angles=[0, 45, 90])
     col.plot.pmm_volume(n_Pu=25)
     col.plot.pmm_slice(Pu, demand=None)
     col.plot.pmm_with_demand(Pu, Mux, Muy)
     col.plot.demand(demands)
     col.plot.dashboard()
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

from design.columns.style import PlotStyle

if TYPE_CHECKING:
    from design.columns.column import Column


class _ColumnPlotter:
    __slots__ = ("_column", "style")

    def __init__(self, column: "Column") -> None:
        self._column = column
        self.style = PlotStyle()

    # ---- geometry ----
    def section(self, *, ax=None, show_bar_labels: bool = False,
                bar_scale: float | None = None, **kw):
        import matplotlib.pyplot as plt
        from design.plotting.sections import plot_section
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_2d)
        return plot_section(
            self._column.section,
            ax=ax,
            show_bar_labels=show_bar_labels,
            bar_scale=bar_scale if bar_scale is not None else s.bar_scale,
            concrete_color=s.color_concrete,
            rebar_color=s.color_rebar,
            **kw,
        )

    # ---- P-M curves ----
    def pm(self, *, angle: float = 0.0, ax=None, **kw):
        import matplotlib.pyplot as plt
        from design.plotting.interaction import plot_pm_diagram
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_2d)
        diagram = self._column.surface[angle]
        units = self._column.units
        ax = plot_pm_diagram(
            diagram,
            ax=ax,
            nominal_kwargs={"color": s.color_nominal,
                            "lw": s.linewidth,
                            "label": "Nominal"},
            reduced_kwargs={"color": s.color_reduced,
                            "lw": s.linewidth,
                            "label": r"$\phi\,M_n / \phi\,P_n$"},
            **kw,
        )
        _rescale_axes(ax, units, x="moment", y="force")
        ax.set_title(f"P-M at θ={angle:.0f}°  ({units.name})")
        return ax

    def pm_compare(self, *, angles=(0.0, 45.0, 90.0), ax=None, **kw):
        import matplotlib.pyplot as plt
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_2d)
        units = self._column.units
        for theta in angles:
            d = self._column.surface[theta]
            line, = ax.plot(d.phi_Mn * units.moment_factor,
                            d.phi_Pn * units.force_factor,
                            lw=s.linewidth, label=f"θ = {theta:.0f}°")
            ax.plot(-d.phi_Mn * units.moment_factor,
                    d.phi_Pn * units.force_factor,
                    lw=s.linewidth, color=line.get_color())
        ax.axhline(self._column.phi_Pn_max * units.force_factor,
                   color=s.color_envelope_cap,
                   lw=s.linewidth_cap, ls=s.linestyle_cap,
                   label=r"$\phi P_{n,\max}$")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(f"Moment ({units.force}·{units.length})")
        ax.set_ylabel(f"Axial ({units.force}, + compression)")
        ax.set_title(f"P-M envelopes  ({units.name})")
        ax.grid(True, alpha=s.grid_alpha)
        ax.legend(fontsize=9)
        return ax

    # ---- PMM volume ----
    def pmm_volume(self, *, ax=None, n_Pu: int = 25, wireframe: bool = True, **kw):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        s = self.style
        if ax is None:
            fig = plt.figure(figsize=s.figsize_3d)
            ax = fig.add_subplot(111, projection="3d")
        units = self._column.units
        Pu, Mx, My = self._column.surface.volume(n_Pu=n_Pu)
        Pu = Pu * units.force_factor
        Mx = Mx * units.moment_factor
        My = My * units.moment_factor
        if wireframe:
            ax.plot_wireframe(Mx, My, Pu, rstride=2, cstride=2,
                              lw=0.5, color=s.color_reduced, alpha=0.6)
        else:
            ax.plot_surface(Mx, My, Pu, cmap="viridis",
                            alpha=s.fill_alpha, edgecolor="none")
        ax.set_xlabel(f"Mx ({units.force}·{units.length})")
        ax.set_ylabel(f"My ({units.force}·{units.length})")
        ax.set_zlabel(f"Pu ({units.force})")
        ax.set_title(f"P-M-M volume  ({units.name})")
        return ax

    def pmm_slice(self, *, Pu: float, demand: Optional[tuple[float, float]] = None,
                  ax=None, fill: bool = True, **kw):
        import matplotlib.pyplot as plt
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_2d)
        units = self._column.units
        Mx, My = self._column.surface.at_Pu(Pu)
        Mx = Mx * units.moment_factor
        My = My * units.moment_factor
        if fill:
            ax.fill(Mx, My, alpha=s.fill_alpha, color=s.color_reduced)
        ax.plot(Mx, My, color=s.color_reduced, lw=s.linewidth,
                label=f"Envelope @ Pu = {Pu*units.force_factor:.0f} {units.force}")
        if demand is not None:
            mux, muy = demand
            ax.plot(mux * units.moment_factor, muy * units.moment_factor,
                    "o", color=s.color_demand, ms=8, label="Demand")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(f"Mx ({units.force}·{units.length})")
        ax.set_ylabel(f"My ({units.force}·{units.length})")
        ax.set_title(f"Biaxial slice at Pu = {Pu*units.force_factor:.0f} {units.force}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=s.grid_alpha)
        return ax

    def pmm_with_demand(self, *, Pu: float, Mux: float, Muy: float,
                        ax=None, n_Pu: int = 25):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        s = self.style
        if ax is None:
            fig = plt.figure(figsize=s.figsize_3d)
            ax = fig.add_subplot(111, projection="3d")
        units = self._column.units
        self.pmm_volume(ax=ax, n_Pu=n_Pu)
        Mx, My = self._column.surface.at_Pu(Pu)
        Pu_ring = np.full_like(Mx, Pu * units.force_factor)
        ax.plot(Mx * units.moment_factor, My * units.moment_factor, Pu_ring,
                color=s.color_demand, lw=1.8,
                label=f"Slice @ Pu = {Pu*units.force_factor:.0f} {units.force}")
        ax.scatter([Mux * units.moment_factor],
                   [Muy * units.moment_factor],
                   [Pu * units.force_factor],
                   color="black", s=60, label="Demand")
        ax.legend(loc="upper right", fontsize=9)
        return ax

    # ---- compound ----
    def demand(self, demands, *, ax=None):
        import matplotlib.pyplot as plt
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_2d)
        chk = self._column.check(demands)
        self.pmm_slice(Pu=demands.Pu, demand=(demands.Mux, demands.Muy), ax=ax)
        ax.set_title(
            f"Demand check  |  ratio PMM = {chk.ratio_pmm:.3f}  "
            f"|  passed = {chk.passed}"
        )
        return ax

    def dashboard(self, *, figsize=None, demands=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        s = self.style
        fig = plt.figure(figsize=figsize or s.figsize_dashboard)
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3, projection="3d")
        ax4 = fig.add_subplot(2, 2, 4)
        self.section(ax=ax1)
        self.pm(angle=0, ax=ax2)
        self.pmm_volume(ax=ax3)
        if demands is not None:
            self.demand(demands, ax=ax4)
        else:
            self.pm(angle=90, ax=ax4)
            ax4.set_title("P-M at θ=90°  (pass demands to dashboard for biaxial slice)")
        fig.tight_layout()
        return fig


def _rescale_axes(ax, units, *, x: str, y: str) -> None:
    """Rescale axis tick labels by multiplicative factor (internal -> display).

    Internally everything is kN, kN·m. Here we relabel.
    """
    fac_x = _factor_for(units, x)
    fac_y = _factor_for(units, y)
    if fac_x != 1.0:
        ax.xaxis.set_major_formatter(_mul_formatter(fac_x))
    if fac_y != 1.0:
        ax.yaxis.set_major_formatter(_mul_formatter(fac_y))
    ax.set_xlabel({"moment": f"Moment ({units.force}·{units.length})",
                   "force":  f"Force ({units.force})"}[x])
    ax.set_ylabel({"moment": f"Moment ({units.force}·{units.length})",
                   "force":  f"Force ({units.force}, + compression)"}[y])


def _factor_for(units, kind: str) -> float:
    return {
        "moment": units.moment_factor,
        "force":  units.force_factor,
        "length": units.length_factor,
    }[kind]


def _mul_formatter(factor: float):
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda x, pos: f"{x * factor:.0f}")
