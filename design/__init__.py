"""Concrete design modules per ACI 318-25.

Internal unit system: MPa, mm, kN, kN·m. All public APIs assume these
units. See ``design/AGENTS.md`` for the full contract.

Bidirectional contract for every element class (Beam, Column, Wall):
    element.run()                  -> self  (lazy + cache)
    element.capacity()             -> XCapacity
    element.check(demands)         -> XCheck
    element.design(demands=None)   -> XDesignResults
    element.evolve(proposal)       -> new element (immutable)
    element.apply(proposal)        -> alias of evolve()
    element.summary() / .report()
    element.plot.*

The protocol that formalizes this contract lives in
``design.common.contracts.Element`` and is verified by
``tests/test_contract.py``.
"""
# ---- Canonical re-exports from common ----
from design.common.materials import Concrete, Steel, Bar
from design.common.units import UnitSystem, units_from, to_internal
from design.common.bar_schedule import BarSchedule
from design.common.plot_style import PlotStyle
from design.common.contracts import (
    Element, Demands, Capacity, Check, DesignProposal, DesignResults,
    wall_demands_to_be,
)
from design.common.factors import (
    beta1, phi_axial_flexure, phi_shear, lambda_concrete,
    OMEGA_0_COLUMN_DEFAULT, OMEGA_V_WALL_DEFAULT, omega_v_wall,
)

# ---- Sections ----
from design.sections.reinforcement import RebarGroup, RebarLayout
from design.sections.rectangular import RectangularSection
from design.sections.wall import WallSection

# ---- Elements ----
from design.beams.beam import Beam
from design.columns.column import Column
from design.walls.wall import Wall

__all__ = [
    # Materials
    "Concrete", "Steel", "Bar",
    # Units
    "UnitSystem", "units_from", "to_internal",
    # Bar schedule + plot style
    "BarSchedule", "PlotStyle",
    # Contract
    "Element", "Demands", "Capacity", "Check",
    "DesignProposal", "DesignResults", "wall_demands_to_be",
    # Factors
    "beta1", "phi_axial_flexure", "phi_shear", "lambda_concrete",
    "OMEGA_0_COLUMN_DEFAULT", "OMEGA_V_WALL_DEFAULT", "omega_v_wall",
    # Sections
    "RebarGroup", "RebarLayout", "RectangularSection", "WallSection",
    # Elements
    "Beam", "Column", "Wall",
]
