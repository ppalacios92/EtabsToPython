"""Shared building blocks for ``design/`` — materials, factors, shear,
spacing, plot style, optimize base and contract Protocols.

Imports from this package should be preferred over the per-element
shims (``design.columns.style``, ``design.columns.probable``, etc.).
"""

from design.common.materials import Concrete, Steel, Bar
from design.common.factors import (
    beta1,
    phi_axial_flexure,
    phi_shear,
    phi_bearing,
    phi_compression_strut_tie,
    lambda_concrete,
    OMEGA_0_COLUMN_DEFAULT,
    OMEGA_V_WALL_DEFAULT,
    omega_v_wall,
)
from design.common.codes import ACI
from design.common.units import UnitSystem, units_from, to_internal, DEFAULT_UNITS

# --- shear ---
from design.common.shear import (
    vc_simplified,
    vc_detailed,
    vs_capacity,
    vs_max,
    av_s_required,
    av_s_minimum,
    vc_zero_seismic,
    alpha_c,
    vn_wall,
    vn_wall_max,
    ve_in_plane_capacity_wall,
)

# --- spacing / confinement ---
from design.common.spacing import (
    lo_length_beam,
    lo_length_column,
    s_max_seismic_smf_beam,
    s_max_seismic_outside_beam,
    s_max_confined_column,
    s_max_middle_zone_column,
    hx_check,
    ash_required,
)

# --- plotting / optimize / contracts ---
from design.common.plot_style import PlotStyle
from design.common.optimize import OptimizeAlternative, format_optimize_table
from design.common.contracts import (
    Demands,
    Capacity,
    Check,
    DesignProposal,
    DesignResults,
    Element,
    wall_demands_to_be,
)

# --- probable strength ---
from design.common.probable import (
    probable_interaction_diagram,
    mpr_envelope,
)


__all__ = [
    # materials
    "Concrete", "Steel", "Bar",
    # factors
    "beta1", "phi_axial_flexure", "phi_shear", "phi_bearing",
    "phi_compression_strut_tie", "lambda_concrete",
    "OMEGA_0_COLUMN_DEFAULT", "OMEGA_V_WALL_DEFAULT", "omega_v_wall",
    # codes / units
    "ACI", "UnitSystem", "units_from", "to_internal", "DEFAULT_UNITS",
    # shear
    "vc_simplified", "vc_detailed", "vs_capacity", "vs_max",
    "av_s_required", "av_s_minimum", "vc_zero_seismic",
    "alpha_c", "vn_wall", "vn_wall_max", "ve_in_plane_capacity_wall",
    # spacing
    "lo_length_beam", "lo_length_column",
    "s_max_seismic_smf_beam", "s_max_seismic_outside_beam",
    "s_max_confined_column", "s_max_middle_zone_column",
    "hx_check", "ash_required",
    # plot / optimize / contracts
    "PlotStyle", "OptimizeAlternative", "format_optimize_table",
    "Demands", "Capacity", "Check", "DesignProposal", "DesignResults",
    "Element", "wall_demands_to_be",
    # probable
    "probable_interaction_diagram", "mpr_envelope",
]
