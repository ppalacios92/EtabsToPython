# API reference

Auto-generated documentation for the public API. The `design` package pages are
generated from docstrings by [mkdocstrings]; the ETABS layer is documented by hand
(it depends on Windows COM and is not importable on the docs CI).

## ETABS layer

- [**etabstopython**](etabstopython.md) — `EtabsModel`, the model plotters, and the
  numerical helper tools.

## Design layer

| Page | Module(s) | Highlights |
|---|---|---|
| [design.common](common.md) | contracts, materials, units, factors, shear, spacing, geometry, … | the contract, unit system, ACI formulas |
| [design.sections](sections.md) | reinforcement, rectangular, wall | `RectangularSection`, `WallSection`, rebar builders |
| [design.beams](beams.md) | beam, optimize, flexure, seismic, interaction | `Beam` + flexure/shear/seismic |
| [design.columns](columns.md) | column, interaction, interaction_surface, surface, optimize | `Column` + P-M-M surface |
| [design.walls](walls.md) | wall, wall_pier, boundary, coupling_beam | `Wall` + boundary elements + coupling beams |
| [design.joints](joints.md) | joint, provisions | `BeamColumnJoint` + §18.8 provisions |
| [design.analysis](analysis.md) | section_analysis | strain-compatibility P-M kernel |
| [design.plotting](plotting.md) | interaction, sections | standalone diagram functions |

## The public surface in one import

```python
from design import (
    Concrete, Steel, Bar,                       # materials
    UnitSystem, units_from, to_internal,        # units
    BarSchedule, PlotStyle,                      # detailing + style
    Element, Demands, Capacity, Check,           # contract
    DesignProposal, DesignResults, wall_demands_to_be,
    beta1, phi_axial_flexure, phi_shear,         # factors
    lambda_concrete, omega_v_wall,
    RebarGroup, RebarLayout,                     # rebar
    RectangularSection, WallSection,             # sections
    Beam, Column, Wall,                          # elements
    BeamColumnJoint, FramingBeam,
)
```

[mkdocstrings]: https://mkdocstrings.github.io/
