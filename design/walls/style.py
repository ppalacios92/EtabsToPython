"""Wall plot style — alias of ``design.common.plot_style.PlotStyle``.

The canonical PlotStyle lives in ``design.common.plot_style``. This module
exists so that ``from design.walls.style import PlotStyle`` works the same
as ``from design.columns.style`` and ``from design.beams.style``.
``WallPlotStyle`` is provided as a back-compat alias.
"""
from design.common.plot_style import PlotStyle

WallPlotStyle = PlotStyle

__all__ = ["PlotStyle", "WallPlotStyle"]
