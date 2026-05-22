"""Wall plot proxy. Bound to a Wall instance, reads wall.units.

API: wall.plot.section()
     wall.plot.pm(plane='in-plane' | 'out-of-plane')
     wall.plot.pm_compare()
     wall.plot.pmm_volume(n_Pu=25)
     wall.plot.pmm_slice(Pu, demand=(Mu, Mu_out))
     wall.plot.demand(demands)
     wall.plot.be_section(which='top' | 'bot')
     wall.plot.dashboard(demands=None)
     wall.plot.iteration_history(results)
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

from design.columns.style import PlotStyle

if TYPE_CHECKING:
    from design.walls.wall import Wall, WallDemands, WallDesignResults


class _WallPlotter:
    __slots__ = ("_wall", "style")

    def __init__(self, wall: "Wall") -> None:
        self._wall = wall
        self.style = PlotStyle()

    # ---- geometry ----------------------------------------------------- #
    def section(self, *, ax=None, show_bar_labels: bool = False,
                bar_scale: float | None = None):
        import matplotlib.pyplot as plt
        from design.plotting.sections import plot_section
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_2d)
        plot_section(
            self._wall.section, ax=ax,
            show_bar_labels=show_bar_labels,
            bar_scale=bar_scale if bar_scale is not None else s.bar_scale,
            concrete_color=s.color_concrete,
            rebar_color=s.color_rebar,
        )
        ax.set_title(f"Wall section — {self._wall.label}")
        return ax

    def be_section(self, *, which: str = "top", ax=None):
        be = self._wall.be_top if which == "top" else self._wall.be_bot
        if be is None:
            raise ValueError(f"Wall has no {which} boundary element.")
        return be.plot.section(ax=ax)

    # ---- PM curves ---------------------------------------------------- #
    def pm(self, *, plane: str = "in-plane", ax=None):
        import matplotlib.pyplot as plt
        from design.plotting.interaction import plot_pm_diagram
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_2d)
        angle = 0.0 if plane == "in-plane" else 90.0
        diagram = self._wall.surface[angle]
        units = self._wall.units
        plot_pm_diagram(
            diagram, ax=ax,
            nominal_kwargs={"color": s.color_nominal,
                            "lw": s.linewidth, "label": "Nominal"},
            reduced_kwargs={"color": s.color_reduced,
                            "lw": s.linewidth,
                            "label": r"$\phi M_n / \phi P_n$"},
        )
        ax.set_title(f"P-M ({plane})  ({units.name})")
        return ax

    def pm_compare(self, *, ax=None):
        import matplotlib.pyplot as plt
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_2d)
        u = self._wall.units
        for ang, lbl in [(0.0, "in-plane"), (90.0, "out-of-plane")]:
            d = self._wall.surface[ang]
            line, = ax.plot(d.phi_Mn * u.moment_factor,
                            d.phi_Pn * u.force_factor,
                            lw=s.linewidth, label=lbl)
            ax.plot(-d.phi_Mn * u.moment_factor,
                    d.phi_Pn * u.force_factor,
                    lw=s.linewidth, color=line.get_color())
        ax.axhline(self._wall.phi_Pn_max * u.force_factor,
                   color=s.color_envelope_cap,
                   lw=s.linewidth_cap, ls=s.linestyle_cap,
                   label=r"$\phi P_{n,\max}$")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(f"Moment ({u.force}·{u.length})")
        ax.set_ylabel(f"Axial ({u.force}, + compression)")
        ax.set_title(f"P-M envelopes  ({u.name})")
        ax.grid(True, alpha=s.grid_alpha)
        ax.legend()
        return ax

    # ---- PMM ---------------------------------------------------------- #
    def pmm_volume(self, *, ax=None, n_Pu: int = 25, wireframe: bool = True):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        s = self.style
        if ax is None:
            fig = plt.figure(figsize=s.figsize_3d)
            ax = fig.add_subplot(111, projection="3d")
        u = self._wall.units
        Pu, Mx, My = self._wall.surface.volume(n_Pu=n_Pu)
        Pu = Pu * u.force_factor
        Mx = Mx * u.moment_factor
        My = My * u.moment_factor
        if wireframe:
            ax.plot_wireframe(Mx, My, Pu, rstride=2, cstride=2,
                              lw=0.5, color=s.color_reduced, alpha=0.6)
        else:
            ax.plot_surface(Mx, My, Pu, cmap="viridis",
                            alpha=s.fill_alpha, edgecolor="none")
        ax.set_xlabel(f"M in-plane ({u.force}·{u.length})")
        ax.set_ylabel(f"M out-of-plane ({u.force}·{u.length})")
        ax.set_zlabel(f"Pu ({u.force})")
        ax.set_title(f"P-M-M volume  ({u.name})")
        return ax

    def pmm_slice(self, *, Pu: float,
                  demand: Optional[tuple[float, float]] = None, ax=None):
        import matplotlib.pyplot as plt
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_2d)
        u = self._wall.units
        Mx, My = self._wall.surface.at_Pu(Pu)
        ax.fill(Mx * u.moment_factor, My * u.moment_factor,
                alpha=s.fill_alpha, color=s.color_reduced)
        ax.plot(Mx * u.moment_factor, My * u.moment_factor,
                color=s.color_reduced, lw=s.linewidth,
                label=f"Envelope @ Pu = {Pu*u.force_factor:.0f} {u.force}")
        if demand is not None:
            mx, my = demand
            ax.plot(mx * u.moment_factor, my * u.moment_factor,
                    "o", color=s.color_demand, ms=8, label="Demand")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(f"M in-plane ({u.force}·{u.length})")
        ax.set_ylabel(f"M out-of-plane ({u.force}·{u.length})")
        ax.set_title(f"Biaxial slice at Pu = {Pu*u.force_factor:.0f} {u.force}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
        ax.grid(True, alpha=s.grid_alpha)
        return ax

    def demand(self, demands: "WallDemands", *, ax=None):
        chk = self._wall.check(demands)
        ax = self.pmm_slice(Pu=demands.Pu,
                            demand=(demands.Mu, demands.Mu_out), ax=ax)
        ax.set_title(
            f"Demand check  |  ratio PM = {chk.ratio_pm:.3f}  "
            f"|  ratio V = {chk.ratio_shear:.3f}  |  passed = {chk.passed}"
        )
        return ax

    # ---- dashboard ---------------------------------------------------- #
    def dashboard(self, *, demands: "WallDemands | None" = None, figsize=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        s = self.style
        fig = plt.figure(figsize=figsize or s.figsize_dashboard)
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3, projection="3d")
        ax4 = fig.add_subplot(2, 2, 4)
        self.section(ax=ax1)
        self.pm(plane="in-plane", ax=ax2)
        self.pmm_volume(ax=ax3)
        if demands is not None:
            self.demand(demands, ax=ax4)
        else:
            self.pm(plane="out-of-plane", ax=ax4)
        fig.tight_layout()
        return fig

    # ---- iteration history ------------------------------------------- #
    def iteration_history(self, results: "WallDesignResults", *, ax=None):
        """Show how BE length evolved across iterations."""
        import matplotlib.pyplot as plt
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_2d)
        u = self._wall.units
        top_lengths = [w.section.be_length_top * u.length_factor
                       for w in results.history]
        bot_lengths = [w.section.be_length_bot * u.length_factor
                       for w in results.history]
        x = list(range(len(results.history)))
        ax.plot(x, top_lengths, "-o", color=s.color_nominal, label="BE top")
        ax.plot(x, bot_lengths, "-s", color=s.color_reduced, label="BE bot")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"BE length ({u.length})")
        ax.set_title(
            f"Iteration history  |  iterations = {results.iterations}  "
            f"|  converged = {results.converged}"
        )
        ax.grid(True, alpha=s.grid_alpha)
        ax.legend()
        return ax
