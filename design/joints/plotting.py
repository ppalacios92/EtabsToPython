"""Plotting for :class:`BeamColumnJoint`.

Bound lazily through ``joint.plot`` (an instance of :class:`_JointPlotter`),
mirroring ``beam.plot`` / ``column.plot`` / ``wall.plot``. A joint owns no
reinforcement cage of its own (the confinement steel lives in the column),
so the plots focus on the two things a joint *does* own:

    * the shear demand/capacity balance in each principal direction, and
    * the plan geometry of the joint core (column footprint + framing beams).
"""
from __future__ import annotations

from typing import Optional


class _JointPlotter:
    """Matplotlib helper bound to a single :class:`BeamColumnJoint`."""

    def __init__(self, joint) -> None:
        self._joint = joint

    # ------------------------------------------------------------------ #
    # Shear demand vs capacity, per principal direction
    # ------------------------------------------------------------------ #
    def demand(self, *, ax=None):
        """Grouped bars of ``Vu`` vs ``phi*Vn`` for the X and Y directions."""
        import matplotlib.pyplot as plt

        j = self._joint
        chk = j.check()
        axes = [("X", chk.x), ("Y", chk.y)]

        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.0))

        x = range(len(axes))
        width = 0.38
        vu = [c.Vu for _, c in axes]
        phivn = [c.phi_Vn for _, c in axes]
        ax.bar([i - width / 2 for i in x], vu, width,
               color="tab:red", alpha=0.75, label=r"$V_u$")
        ax.bar([i + width / 2 for i in x], phivn, width,
               color="tab:green", alpha=0.75, label=r"$\phi V_n$")
        for i, (_, c) in enumerate(axes):
            ax.annotate(f"r={c.ratio:.2f}",
                        (i, max(c.Vu, c.phi_Vn)),
                        ha="center", va="bottom", fontsize=9)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"dir {name}" for name, _ in axes])
        ax.set_ylabel("Shear (kN)")
        ax.set_title(f"Joint shear — {j.label} ({j.joint_type()})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        return ax

    # ------------------------------------------------------------------ #
    # Plan view of the joint core
    # ------------------------------------------------------------------ #
    def section(self, *, ax=None):
        """Plan schematic: column footprint + framing-beam footprints."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        j = self._joint
        col = j._col
        bc = col.section.width()    # x dimension (mm)
        hc = col.section.height()   # y dimension (mm)

        if ax is None:
            _, ax = plt.subplots(figsize=(5.0, 5.0))

        # Column core (centred at origin).
        ax.add_patch(Rectangle((-bc / 2, -hc / 2), bc, hc,
                               facecolor="0.85", edgecolor="k", lw=1.5,
                               label="column"))

        ext = max(bc, hc) * 0.9  # how far the beams stick out
        # Beams framing in X (left/right) -> horizontal bands of height = bw.
        for fb, sign in ((j.beam_left, -1), (j.beam_right, +1)):
            if fb is None:
                continue
            bw = fb.beam.section.width()
            ax.add_patch(Rectangle((sign * bc / 2 if sign > 0 else -bc / 2 - ext,
                                    -bw / 2), ext, bw,
                                   facecolor="tab:blue", alpha=0.35,
                                   edgecolor="tab:blue"))
        # Beams framing in Y (up/down) -> vertical bands of width = bw.
        for fb, sign in ((j.beam_up, +1), (j.beam_down, -1)):
            if fb is None:
                continue
            bw = fb.beam.section.width()
            ax.add_patch(Rectangle((-bw / 2,
                                    sign * hc / 2 if sign > 0 else -hc / 2 - ext),
                                   bw, ext,
                                   facecolor="tab:orange", alpha=0.35,
                                   edgecolor="tab:orange"))

        cap = j.capacity()
        ax.set_title(f"Joint core — {j.label}\n"
                     f"Aj(X)={cap.x.Aj:.0f}  Aj(Y)={cap.y.Aj:.0f} mm²")
        lim = (max(bc, hc) / 2 + ext) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.grid(True, alpha=0.3)
        return ax

    # ------------------------------------------------------------------ #
    # Both
    # ------------------------------------------------------------------ #
    def dashboard(self, *, figsize: Optional[tuple] = None):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize or (10.5, 4.6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        self.section(ax=ax1)
        self.demand(ax=ax2)
        fig.tight_layout()
        return fig
