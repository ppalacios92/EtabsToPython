"""Re-export — canonical PlotStyle lives in :mod:`design.common.plot_style`.

``BeamPlotStyle`` is kept as a backwards-compatible alias of
:class:`design.common.plot_style.PlotStyle`. New code should use
``PlotStyle`` directly.
"""
from design.common.plot_style import PlotStyle

# Backwards-compatible alias
BeamPlotStyle = PlotStyle

__all__ = ["PlotStyle", "BeamPlotStyle"]
