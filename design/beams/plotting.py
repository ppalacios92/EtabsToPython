"""Beam plot proxy.

Bound to a Beam instance. Reads beam.units and beam.plot.style for
all visualization. matplotlib is imported lazily.

API:
    beam.plot.section()                          — section + bars
    beam.plot.moment_envelope(demands)           — Mu+/Mu- per station vs phi*Mn
    beam.plot.shear_envelope(demands)            — Vu per station vs phi*Vn / Ve
    beam.plot.demand(demands)                    — alias for moment_envelope
    beam.plot.dashboard(demands=None)            — 2x2 grid

Beams have no PMM diagram — they are flexural-shear elements per
ACI 318-25 §9 and §18.6.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

from design.beams.style import BeamPlotStyle

if TYPE_CHECKING:
    from design.beams.beam import Beam, BeamDemands


class _BeamPlotter:
    __slots__ = ("_beam", "style")

    def __init__(self, beam: "Beam") -> None:
        self._beam = beam
        self.style = BeamPlotStyle()

    # ---- geometry ----
    def section(self, *, ax=None, show_bar_labels: bool = False,
                bar_scale: float | None = None, **kw):
        import matplotlib.pyplot as plt
        from design.plotting.sections import plot_section
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_2d)
        return plot_section(
            self._beam.section,
            ax=ax,
            show_bar_labels=show_bar_labels,
            bar_scale=bar_scale if bar_scale is not None else s.bar_scale,
            concrete_color=s.color_concrete,
            rebar_color=s.color_rebar,
            **kw,
        )

    # ---- moment demand bar chart ----
    def moment_envelope(self, demands: "BeamDemands", *, ax=None, **kw):
        import matplotlib.pyplot as plt
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_wide)
        units = self._beam.units
        cap = self._beam.capacity()

        stations = ["i", "mid", "j"]
        Mu_pos = np.array([demands.Mu_pos_i, demands.Mu_pos_mid, demands.Mu_pos_j])
        Mu_neg = np.array([demands.Mu_neg_i, demands.Mu_neg_mid, demands.Mu_neg_j])

        x = np.arange(len(stations))
        w = 0.35
        ax.bar(x - w / 2, Mu_pos * units.moment_factor, w,
               label=r"$M_u^+$", color=s.color_pos, alpha=0.7)
        ax.bar(x + w / 2, Mu_neg * units.moment_factor, w,
               label=r"$M_u^-$", color=s.color_neg, alpha=0.7)
        ax.axhline(cap.phi_Mn_pos * units.moment_factor,
                   color=s.color_pos, lw=s.linewidth_cap, ls=s.linestyle_cap,
                   label=r"$\phi M_n^+$")
        ax.axhline(cap.phi_Mn_neg * units.moment_factor,
                   color=s.color_neg, lw=s.linewidth_cap, ls=s.linestyle_cap,
                   label=r"$\phi M_n^-$")
        ax.set_xticks(x)
        ax.set_xticklabels(["face i", "mid", "face j"])
        ax.set_ylabel(f"Moment ({units.force}·{units.length})")
        ax.set_title(f"Moment demand per station  ({units.name})")
        ax.grid(True, alpha=s.grid_alpha, axis="y")
        ax.legend(fontsize=9)
        return ax

    # ---- shear ----
    def shear_envelope(self, demands: "BeamDemands", *, ax=None, **kw):
        import matplotlib.pyplot as plt
        s = self.style
        if ax is None:
            _, ax = plt.subplots(figsize=s.figsize_wide)
        units = self._beam.units
        cap = self._beam.capacity()

        stations = ["i", "mid", "j"]
        Vu = np.array([abs(demands.Vu_i), abs(demands.Vu_mid), abs(demands.Vu_j)])
        x = np.arange(len(stations))

        ax.bar(x, Vu * units.force_factor, 0.5,
               color=s.color_demand, alpha=0.7, label=r"$V_u$")
        ax.axhline(cap.phi_Vn * units.force_factor,
                   color="k", lw=s.linewidth_cap, ls=s.linestyle_cap,
                   label=r"$\phi V_n$")
        if cap.Mn_pr_pos is not None and cap.Mn_pr_neg is not None and self._beam.ln:
            Ve = (max(cap.Mn_pr_pos, cap.Mn_pr_neg) * 2) / self._beam.ln + self._beam.Vg
            ax.axhline(Ve * units.force_factor, color="tab:purple",
                       lw=s.linewidth_cap, ls=":",
                       label=r"$V_e$ (capacity)")
        ax.set_xticks(x)
        ax.set_xticklabels(["face i", "mid", "face j"])
        ax.set_ylabel(f"Shear ({units.force})")
        ax.set_title(f"Shear demand per station  ({units.name})")
        ax.grid(True, alpha=s.grid_alpha, axis="y")
        ax.legend(fontsize=9)
        return ax

    # ---- compound ----
    def demand(self, demands: "BeamDemands", *, ax=None):
        return self.moment_envelope(demands, ax=ax)

    def dashboard(self, *, figsize=None, demands: Optional["BeamDemands"] = None):
        import matplotlib.pyplot as plt
        s = self.style
        fig = plt.figure(figsize=figsize or s.figsize_dashboard)
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        self.section(ax=ax1)
        # Top-right: moment capacity bars (phi_Mn+ and phi_Mn-)
        cap = self._beam.capacity()
        u = self._beam.units
        ax2.bar([0], [cap.phi_Mn_pos * u.moment_factor], 0.5,
                color=s.color_pos, alpha=0.7, label=r"$\phi M_n^+$")
        ax2.bar([1], [cap.phi_Mn_neg * u.moment_factor], 0.5,
                color=s.color_neg, alpha=0.7, label=r"$\phi M_n^-$")
        if cap.Mn_pr_pos is not None:
            ax2.bar([0], [cap.Mn_pr_pos * u.moment_factor], 0.5,
                    color=s.color_pos, alpha=0.25, label=r"$M_{pr}^+$")
            ax2.bar([1], [cap.Mn_pr_neg * u.moment_factor], 0.5,
                    color=s.color_neg, alpha=0.25, label=r"$M_{pr}^-$")
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["+M", "-M"])
        ax2.set_ylabel(f"Moment ({u.force}·{u.length})")
        ax2.set_title("Beam capacity")
        ax2.grid(True, alpha=s.grid_alpha, axis="y")
        ax2.legend(fontsize=9)
        if demands is not None:
            self.moment_envelope(demands, ax=ax3)
            self.shear_envelope(demands, ax=ax4)
        else:
            ax3.set_axis_off()
            ax4.set_axis_off()
            ax3.text(0.5, 0.5, "pass demands to dashboard()",
                     ha="center", va="center", transform=ax3.transAxes)
        fig.tight_layout()
        return fig
