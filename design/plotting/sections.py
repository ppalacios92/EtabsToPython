"""Section geometry plots (concrete polygons + rebar positions)."""
from __future__ import annotations
from typing import Optional

from design.sections.base import Section


def plot_section(
    section: Section,
    *,
    ax=None,
    show_bar_labels: bool = False,
    bar_scale: float = 1.0,
    concrete_color: str = "#D8D8D8",
    rebar_color: str = "tab:blue",
):
    """Draw concrete polygons + rebar bars in section-local coords."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    from matplotlib.collections import PatchCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # concrete polygons
    patches = [Polygon(p, closed=True) for p in section.polygons()]
    coll = PatchCollection(patches, facecolor=concrete_color,
                           edgecolor="black", lw=1.0)
    ax.add_collection(coll)

    # bars
    for i, (rebar, steel) in enumerate(section.rebar.iter_bars()):
        radius = (rebar.area / 3.14159) ** 0.5 * bar_scale
        c = Circle((rebar.x, rebar.y), radius=radius,
                   facecolor=rebar_color, edgecolor="black", lw=0.5, zorder=5)
        ax.add_patch(c)
        if show_bar_labels:
            ax.annotate(str(i), (rebar.x, rebar.y),
                        fontsize=6, ha="center", va="center",
                        color="white", zorder=6)

    x0, y0, x1, y1 = section.bounding_box()
    pad = max(x1 - x0, y1 - y0) * 0.08
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, alpha=0.3)

    # title with totals
    Ast = section.rebar.total_area
    n = sum(g.n for g in section.rebar.groups)
    Ag = section.gross_area()
    rho = Ast / Ag if Ag > 0 else 0.0
    ax.set_title(
        f"Section  |  Ag = {Ag:,.0f} mm²   "
        f"n_bars = {n}   As_total = {Ast:,.0f} mm²   ρ = {rho*100:.2f}%"
    )
    return ax
