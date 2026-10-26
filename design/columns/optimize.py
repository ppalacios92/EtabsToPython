"""Transverse-reinforcement optimization for concrete columns.

Generates feasible (db_hoop, n_legs_x, n_legs_y) combinations from the
column's `bar_schedule.hoops` and the physical bar-count limit per face,
runs `column.design()` for each, computes the **steel quantity** (kg/m³
of concrete) over a 1 m slice of the confined region, and returns a
ranked list. The lightest combination that still passes the code checks
is the optimum.

The quantity formula (transverse only):

    legs_per_layer  = n_legs_x · b_core + n_legs_y · h_core      [mm]
    layers_per_m    = 1000 / spacing_confined                     [-]
    length_per_m    = legs_per_layer · layers_per_m               [mm]
    volume_steel    = length_per_m · a_per_leg                     [mm³]
    mass_steel      = volume_steel · steel.density · 1e-9          [kg]
    volume_concrete = Ag · 1000 · 1e-9                              [m³]
    rho_transverse  = mass_steel / volume_concrete                  [kg/m³]

The longitudinal-steel quantity is reported alongside but does not vary
across alternatives (the longitudinal layout is fixed by the section).

The element-specific ``OptimizeAlternative`` keeps the legacy public
fields (``db_hoop``, ``n_legs_x``, ``n_legs_y``) AND exposes the unified
``detailing`` dict from :mod:`design.common.optimize` so that the
shared :func:`format_optimize_table` can render it without per-element
knowledge.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from math import pi
from typing import TYPE_CHECKING

from design.common.optimize import format_optimize_table as _common_format_table

if TYPE_CHECKING:
    from design.columns.column import Column, ColumnDesignProposal, ColumnDemands


@dataclass(frozen=True, slots=True)
class OptimizeAlternative:
    """Column-specific optimize alternative.

    Compatible with :class:`design.common.optimize.OptimizeAlternative`
    (same field set + a ``detailing`` dict). The legacy ``db_hoop``,
    ``n_legs_x``, ``n_legs_y`` attributes are kept on the dataclass so
    existing callers keep working.
    """
    proposal: "ColumnDesignProposal"
    db_hoop: float
    n_legs_x: int
    n_legs_y: int
    rho_transverse: float       # kg/m³ of concrete (transverse only, lo region)
    rho_longitudinal: float     # kg/m³ of concrete (longitudinal only)
    rho_total: float            # kg/m³ of concrete
    feasible: bool              # passes Ash + hx + physical-bar gates
    is_baseline: bool = False   # True if this is the design() envelope
    is_provided: bool = False   # True if this is the as-input detailing
    notes: tuple[str, ...] = ()

    @property
    def detailing(self) -> dict:
        """Unified detailing dict (interop with common.optimize)."""
        return {
            "db_hoop": self.db_hoop,
            "n_legs_x": self.n_legs_x,
            "n_legs_y": self.n_legs_y,
        }


def _column_detail_fmt(alt: OptimizeAlternative) -> str:
    """Human-friendly detailing column for :func:`format_optimize_table`."""
    return (
        f"phi{alt.db_hoop:.0f} ({alt.n_legs_x}x, {alt.n_legs_y}y)"
    )


def format_optimize_table(alternatives, *, top_n: int = 12) -> str:
    """Pretty-print column optimize alternatives.

    Thin wrapper around :func:`design.common.optimize.format_optimize_table`
    that supplies the column-specific detail formatter. The implementation
    of the table layout lives in ``common.optimize``.
    """
    return _common_format_table(
        alternatives,
        top_n=top_n,
        detail_fmt=_column_detail_fmt,
    )


def transverse_steel_quantity(
    column: "Column",
    *,
    db_hoop: float,
    n_legs_x: int,
    n_legs_y: int,
    spacing: float,
    clear_cover: float | None = None,
) -> float:
    """Mass of transverse steel per m³ of concrete in the lo region (kg/m³).

    Uses the column.section + column.bar_schedule + materials.density of
    the first longitudinal group as a proxy for the hoop density (hoops
    are usually the same grade).

    ``clear_cover`` defaults to ``column.section.clear_cover``.
    """
    section = column.section
    if clear_cover is None:
        clear_cover = getattr(section, "clear_cover", 25.0)
    b_out = section.width()
    h_out = section.height()
    Ag = section.gross_area()                          # mm²
    bc = max(b_out - 2 * clear_cover, 100.0)  # mm
    hc = max(h_out - 2 * clear_cover, 100.0)

    a_per_leg = pi * db_hoop ** 2 / 4.0                # mm²
    legs_per_layer = n_legs_x * bc + n_legs_y * hc     # mm per hoop level
    layers_per_m = 1000.0 / spacing                    # hoops in 1 m
    length_per_m = legs_per_layer * layers_per_m       # mm of bar per m of column
    volume_steel = length_per_m * a_per_leg            # mm³

    density = column.section.rebar.groups[0].steel.density if column.section.rebar.groups else 7850.0
    mass_steel = volume_steel * density * 1e-9         # kg

    volume_concrete = Ag * 1000.0 * 1e-9               # m³  (1 m of column)
    if volume_concrete <= 0:
        return 0.0
    return mass_steel / volume_concrete


def longitudinal_steel_quantity(column: "Column") -> float:
    """Mass of longitudinal steel per m³ of concrete (kg/m³)."""
    section = column.section
    Ag = section.gross_area()
    Ast = section.rebar.total_area                     # mm²
    density = section.rebar.groups[0].steel.density if section.rebar.groups else 7850.0
    # Long bar runs full length → per m of column: volume = Ast · 1000 mm³
    mass_steel = Ast * 1000.0 * density * 1e-9         # kg/m of column
    volume_concrete = Ag * 1000.0 * 1e-9               # m³ per m of column
    return mass_steel / volume_concrete if volume_concrete > 0 else 0.0


def rho_g_check(column: "Column") -> tuple[float, bool, str | None]:
    """Validate the longitudinal reinforcement ratio ρg = Ast/Ag.

    ACI 318-25 limits:
        seismic SMF columns (§18.7.4.1): 0.01 <= ρg <= 0.06
        non-seismic columns (§10.6.1):   0.01 <= ρg <= 0.08

    Returns ``(rho_g, ok, note)`` where ``note`` is ``None`` when within
    range, otherwise a human-readable violation message citing the clause.
    The bound is a property of the longitudinal layout only, so it is the
    same across all transverse-detailing alternatives.
    """
    Ag = column.section.gross_area()
    Ast = column.section.rebar.total_area
    rho_g = Ast / Ag if Ag > 0 else 0.0
    rho_max = 0.06 if column.seismic else 0.08
    clause = "Sec.18.7.4.1" if column.seismic else "Sec.10.6.1"
    if rho_g < 0.01:
        return rho_g, False, (
            f"rho_g = {rho_g*100:.2f}% < 1.0% minimum ({clause})."
        )
    if rho_g > rho_max:
        return rho_g, False, (
            f"rho_g = {rho_g*100:.2f}% > {rho_max*100:.0f}% maximum ({clause})."
        )
    return rho_g, True, None


def run_optimize(
    column: "Column",
    demands: "ColumnDemands | None" = None,
    *,
    db_hoop_list: list[float] | None = None,
    clear_cover: float = 25.0,
    include_infeasible: bool = False,
    sort_by: str = "rho_total",       # 'rho_total' | 'rho_transverse'
) -> list[OptimizeAlternative]:
    """Explore detailing alternatives **around the design() envelope**.

    Flow:
        1. Run `column.design(demands)` and take its envelope as baseline.
        2. Enumerate `(db, n_legs_x, n_legs_y)` from `bar_schedule.hoops`
           and the physical bar-count limit per face.
        3. Run `column.design(demands, db_hoop=..., n_legs_x=..., n_legs_y=...)`
           with each combo.
        4. Keep only the alternatives that pass the code (Av_provided ≥
           Ash and hx ≤ 350 mm). Always include the baseline.
        5. Rank by `sort_by` ascending — lightest first.

    The output is what to actually build. The baseline is marked
    `is_baseline=True` so you can spot the original design() choice in
    the table.

    All steel quantities (`rho_transverse`, `rho_longitudinal`,
    `rho_total`) are in **kg/m³ of concrete**, measured over a 1 m slice
    of the confined region.

    Pass `include_infeasible=True` to also see the combinations that did
    not cover Ash or violated hx — useful for debugging.
    """
    if db_hoop_list is None:
        db_hoop_list = list(column.bar_schedule.hoops)

    rho_long = longitudinal_steel_quantity(column)

    # ---- ρg gate (§18.7.4.1 seismic / §10.6.1) — same for every alt ----
    _rho_g, rho_g_ok, rho_g_note = rho_g_check(column)
    rho_g_extra = (rho_g_note,) if rho_g_note else ()

    # ---- 1. Baseline ----
    baseline_results = column.design(
        demands, clear_cover=clear_cover,
    )
    baseline = baseline_results.envelope
    rho_t_base = transverse_steel_quantity(
        column,
        db_hoop=baseline.db_hoop,
        n_legs_x=baseline.n_legs_x,
        n_legs_y=baseline.n_legs_y,
        spacing=baseline.spacing_confined,
        clear_cover=clear_cover,
    )
    baseline_ash_ok = (baseline.av_provided_x >= baseline.ash_required_x and
                       baseline.av_provided_y >= baseline.ash_required_y)
    alternatives: list[OptimizeAlternative] = [OptimizeAlternative(
        proposal=baseline,
        db_hoop=baseline.db_hoop,
        n_legs_x=baseline.n_legs_x,
        n_legs_y=baseline.n_legs_y,
        rho_transverse=rho_t_base,
        rho_longitudinal=rho_long,
        rho_total=rho_t_base + rho_long,
        feasible=baseline_ash_ok and baseline.hx_ok and rho_g_ok,
        is_baseline=True,
        notes=rho_g_extra + baseline.notes,
    )]

    baseline_key = (baseline.db_hoop, baseline.n_legs_x, baseline.n_legs_y)

    # ---- 2. Explore around the baseline ----
    max_x, max_y = column._count_longitudinal_per_face()
    for db in sorted(db_hoop_list):
        for nx, ny in product(range(2, max_x + 1), range(2, max_y + 1)):
            if (db, nx, ny) == baseline_key:
                continue
            results = column.design(
                demands,
                db_hoop=db, n_legs_x=nx, n_legs_y=ny,
                clear_cover=clear_cover,
            )
            p = results.envelope
            ash_ok = (p.av_provided_x >= p.ash_required_x and
                      p.av_provided_y >= p.ash_required_y)
            feasible = ash_ok and p.hx_ok and rho_g_ok
            if not feasible and not include_infeasible:
                continue
            rho_t = transverse_steel_quantity(
                column,
                db_hoop=p.db_hoop, n_legs_x=p.n_legs_x, n_legs_y=p.n_legs_y,
                spacing=p.spacing_confined,
                clear_cover=clear_cover,
            )
            alternatives.append(OptimizeAlternative(
                proposal=p,
                db_hoop=p.db_hoop, n_legs_x=p.n_legs_x, n_legs_y=p.n_legs_y,
                rho_transverse=rho_t,
                rho_longitudinal=rho_long,
                rho_total=rho_t + rho_long,
                feasible=feasible,
                is_baseline=False,
                notes=rho_g_extra + p.notes,
            ))

    alternatives.sort(key=lambda a: getattr(a, sort_by))
    return alternatives
