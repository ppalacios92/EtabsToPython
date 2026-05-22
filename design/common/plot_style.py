"""Unified plot-style settings for Beam / Column / Wall plotters.

Every element exposes ``element.style: PlotStyle``. Plotters read from
``self.element.style``. Per-call kwargs always override the persistent
style.

The ``columns/style.py``, ``beams/style.py`` and ``walls/style.py``
modules are thin re-export shims pointing here.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class PlotStyle:
    """Persistent plot-style settings shared across all element plotters."""

    # Curve colors
    color_nominal: str = "k"
    color_reduced: str = "tab:blue"
    color_demand: str = "tab:red"

    # Beam-specific (positive vs negative envelope)
    color_pos: str = "tab:blue"
    color_neg: str = "tab:red"

    # Section colors
    color_concrete: str = "#D8D8D8"
    color_rebar: str = "tab:blue"
    color_envelope_cap: str = "red"

    # Line styles
    linewidth: float = 1.5
    linewidth_cap: float = 0.8
    linestyle_cap: str = "--"

    # Aesthetics
    fill_alpha: float = 0.15
    bar_scale: float = 1.0
    grid_alpha: float = 0.3

    # Figure sizes
    figsize_2d: tuple[float, float] = (6.0, 6.0)
    figsize_wide: tuple[float, float] = (10.0, 4.0)
    figsize_3d: tuple[float, float] = (8.0, 7.0)
    figsize_dashboard: tuple[float, float] = (14.0, 10.0)
