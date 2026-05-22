"""Reinforcement optimization for structural walls.

Explores feasible combinations of:

  * **web** vertical + horizontal distributed bars (db_web, spacing, layers),
  * **BE longitudinal** bars (db_be, n_y_be in the perimeter cage),

and computes the resulting steel quantities (kg/m³ of concrete) so the
user can pick the lightest detailing that still satisfies §18.10.2 /
§18.10.6.

The optimizer does NOT vary the BE *geometry* (length, thickness, x_offset)
— those are inputs you fix when building the wall. To explore BE
geometry, build several walls and compare; to explore BE *confinement*
(hoops, hx, lo), call ``wall.be_top.run_optimize()`` on the boundary
element column.

Public API:

    run_optimize(wall, demands=None, **kw) -> list[OptimizeAlternative]
    format_optimize_table(alts, *, top_n=12) -> str

This module mirrors ``design.columns.optimize`` and
``design.beams.optimize`` so the workflow is uniform across elements.
"""
from __future__ import annotations
from math import pi
from typing import TYPE_CHECKING

from design.common.optimize import (
    OptimizeAlternative,
    format_optimize_table as _common_format_table,
)
from design.walls.distributed import (
    web_rho_provided, web_bar_spacing_max, two_curtain_required,
)

if TYPE_CHECKING:
    from design.walls.wall import Wall, WallDemands


__all__ = [
    "OptimizeAlternative",
    "run_optimize",
    "format_optimize_table",
    "transverse_steel_quantity",
    "longitudinal_steel_quantity",
    "be_longitudinal_quantity",
]


# ====================================================================== #
# Steel-quantity helpers (all return kg/m³ of concrete on a 1 m slice)
# ====================================================================== #
def _density(wall: "Wall") -> float:
    section = wall.section
    return (
        section.rebar.groups[0].steel.density
        if section.rebar.groups else 7850.0
    )


def transverse_steel_quantity(
    wall: "Wall",
    *,
    db_web: float,
    spacing: float,
    layers: int,
) -> float:
    """Mass of web horizontal (transverse) bars per m³ of concrete.

    On a 1 m slice of the wall:
        n_bars_per_m   = 1000 / spacing
        length_per_bar = lw  (one horizontal bar spans the wall length)
        volume         = layers * n_bars * lw * a_per_bar
    """
    section = wall.section
    Ag = section.gross_area()
    if Ag <= 0 or spacing <= 0:
        return 0.0
    a_per_bar = pi * db_web ** 2 / 4.0
    n_bars_per_m = 1000.0 / spacing
    volume_steel = layers * n_bars_per_m * section.lw * a_per_bar
    mass = volume_steel * _density(wall) * 1e-9
    volume_concrete = Ag * 1000.0 * 1e-9
    return mass / volume_concrete


def longitudinal_steel_quantity(
    wall: "Wall",
    *,
    db_web: float | None = None,
    spacing: float | None = None,
    layers: int | None = None,
) -> float:
    """Mass of web vertical (longitudinal) bars per m³ of concrete.

    Walls keep equal horizontal and vertical distributed ratios in the
    web by convention (`rho_l == rho_t`), so the formula mirrors
    :func:`transverse_steel_quantity` but the bar length is `hw_slice =
    1 m` (one vertical bar per spacing along the length, running full
    height — for the 1 m slice the bar length is 1000 mm).

    If ``db_web`` / ``spacing`` / ``layers`` are not provided, falls
    back to summing the actual vertical bars already in
    ``wall.section.rebar`` that fall inside the web (i.e. excluding the
    BE bars). That is the "as-built" longitudinal quantity.
    """
    section = wall.section
    Ag = section.gross_area()
    if Ag <= 0:
        return 0.0

    if db_web is None or spacing is None or layers is None:
        # As-built: sum the bars that are in the web (split off the BE
        # contribution by reusing ``web_rebar()``).
        web_rebar = section.web_rebar()
        Ast_web = web_rebar.total_area if web_rebar is not None else 0.0
    else:
        # Closed-form from (db_web, spacing, layers): along the wall
        # length, one bar per ``spacing`` per face times ``layers`` curtains.
        if spacing <= 0:
            return 0.0
        a_per_bar = pi * db_web ** 2 / 4.0
        n_bars_per_face = section.lw / spacing
        # Walls almost always have at least 2 faces (one on each side
        # of the web). ``layers`` controls per-face curtains.
        Ast_web = 2 * layers * n_bars_per_face * a_per_bar

    mass = Ast_web * 1000.0 * _density(wall) * 1e-9     # kg per m of wall
    volume_concrete = Ag * 1000.0 * 1e-9                # m³ per m of wall
    return mass / volume_concrete


def be_longitudinal_quantity(
    wall: "Wall",
    *,
    db_be: float,
    n_x_be: int,
    n_y_be: int,
) -> float:
    """Mass of BE longitudinal bars per m³ of concrete.

    Sums the perimeter cages of both BEs (top and bottom). For a
    perimeter pattern with ``n_x_be`` bars across the thickness and
    ``n_y_be`` along the length, the number of unique bars per BE is

        n_per_BE = 2 * n_x_be + 2 * (n_y_be - 2)
                 = 2 * (n_x_be + n_y_be - 2)

    (corner bars are shared). Bars run the full wall height, so on a
    1 m slice the contribution per bar is ``1000 * a_per_bar``.
    """
    section = wall.section
    Ag = section.gross_area()
    if Ag <= 0:
        return 0.0
    a_per_bar = pi * db_be ** 2 / 4.0
    n_per_BE = 2 * (n_x_be + n_y_be - 2) if (n_x_be >= 2 and n_y_be >= 2) else 0
    total_bars = 0
    if section.be_length_top > 0:
        total_bars += n_per_BE
    if section.be_length_bot > 0:
        total_bars += n_per_BE
    if total_bars == 0:
        return 0.0
    mass = total_bars * 1000.0 * a_per_bar * _density(wall) * 1e-9
    volume_concrete = Ag * 1000.0 * 1e-9
    return mass / volume_concrete


# ====================================================================== #
# Optimize loop
# ====================================================================== #
def run_optimize(
    wall: "Wall",
    demands: "WallDemands | None" = None,
    *,
    db_web_list: list[float] | None = None,
    spacing_list: list[float] | None = None,
    layers_options: tuple[int, ...] = (1, 2),
    db_be_list: list[float] | None = None,
    n_y_be_list: list[int] | None = None,
    include_infeasible: bool = False,
    sort_by: str = "rho_total",
) -> list[OptimizeAlternative]:
    """Enumerate web and BE detailing alternatives.

    Each alternative carries a ``detailing`` dict with keys
    ``db_web``, ``spacing``, ``layers``, ``db_be``, ``n_y_be`` (the BE
    keys are absent when the wall has no boundary element).

    The baseline (``wall.design(demands).envelope``) is always present
    with ``is_baseline=True``. The as-built provided detailing is also
    inserted with ``is_provided=True`` so the user can see where their
    own choice ranks.

    Quantities returned (all in kg/m³ of concrete):

      * ``rho_transverse``  = web horizontal bars
      * ``rho_longitudinal`` = web vertical bars + BE longitudinal bars
      * ``rho_total``       = sum

    Parameters
    ----------
    db_web_list :
        Candidate web bar diameters in mm. Defaults to the small
        diameters from ``wall.bar_schedule.hoops`` (<= 16 mm).
    spacing_list :
        Candidate web bar spacings in mm. Default: 100..250 mm in 25 mm
        steps.
    layers_options :
        Candidate curtain counts (1 or 2). Two curtains are forced when
        §18.10.2.2 requires them.
    db_be_list :
        Candidate BE longitudinal bar diameters in mm. Defaults to the
        ``wall.bar_schedule.longitudinal`` filtered to <= 28 mm. Ignored
        when the wall has no BE.
    n_y_be_list :
        Candidate ``n_y_be`` values (bars along the BE length).
        Defaults to ``[4, 5, 6, 7]``. Ignored when the wall has no BE.
    include_infeasible :
        Keep alternatives that violate ``rho >= rho_min``, ``spacing <=
        s_max`` or the BE longitudinal floor (~1% per §18.10.6.4(b)).
    sort_by :
        ``"rho_total"`` (default), ``"rho_transverse"`` or
        ``"rho_longitudinal"``.
    """
    section = wall.section
    has_be = section.has_boundary_elements
    s_max = web_bar_spacing_max(lw=section.lw, tw=section.tw)
    hw_over_lw = wall.hw / section.lw if section.lw > 0 else float("inf")
    n_x_be = getattr(wall, "n_x_be", 3)

    # Default candidate grids
    if db_web_list is None:
        db_web_list = [d for d in wall.bar_schedule.hoops if d <= 16.0] or [10.0, 12.0]
    if spacing_list is None:
        spacing_list = [100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0]
    if db_be_list is None:
        db_be_list = [d for d in wall.bar_schedule.longitudinal if d <= 28.0]
        if not db_be_list:
            db_be_list = [20.0, 22.0, 25.0]
    if n_y_be_list is None:
        n_y_be_list = [4, 5, 6, 7]

    # ---- baseline from wall.design ----
    base_results = wall.design(demands)
    base = base_results.envelope
    rho_t_required = base.rho_t_required
    rho_l_required = base.rho_l_required

    # Two-curtain requirement
    Vu = demands.Vu if demands is not None else 0.0
    dc_required = two_curtain_required(
        Vu=Vu, Acv=section.Acv, fc=section.concrete.fc,
        lam=section.concrete.lam, hw_over_lw=hw_over_lw, tw=section.tw,
    )

    def _be_feasible(db_be: float, n_y_be: int) -> bool:
        """Approximate BE longitudinal floor: rho_be >= 1% (§18.10.6.4(b))."""
        if not has_be:
            return True
        if n_x_be < 3 or n_y_be < 3:
            return False
        n_bars = 2 * (n_x_be + n_y_be - 2)
        a_bar = pi * db_be ** 2 / 4.0
        As_be = n_bars * a_bar
        be_area = section.be_length_top * section.be_thickness_top
        return be_area <= 0 or (As_be / be_area) >= 0.0099   # ~1%

    def _make_alt(*, db_web, spacing, layers, db_be, n_y_be,
                  is_baseline=False, is_provided=False,
                  override_feasibility=None):
        rho_t = transverse_steel_quantity(
            wall, db_web=db_web, spacing=spacing, layers=layers,
        )
        rho_l_web = longitudinal_steel_quantity(
            wall, db_web=db_web, spacing=spacing, layers=layers,
        )
        rho_l_be = (
            be_longitudinal_quantity(
                wall, db_be=db_be, n_x_be=n_x_be, n_y_be=n_y_be,
            ) if has_be else 0.0
        )
        rho_l = rho_l_web + rho_l_be
        rho_web_provided = web_rho_provided(
            tw=section.tw, db=db_web, spacing=spacing, layers=layers,
        )
        ok_rho = rho_web_provided >= rho_t_required - 1e-9
        ok_spacing = spacing <= s_max + 1e-6
        ok_layers = (not dc_required) or (layers >= 2)
        ok_be = _be_feasible(db_be, n_y_be)
        feasible = ok_rho and ok_spacing and ok_layers and ok_be
        if override_feasibility is not None:
            feasible = override_feasibility
        detailing = {
            "db_web": db_web, "spacing": spacing, "layers": layers,
        }
        if has_be:
            detailing["db_be"] = db_be
            detailing["n_y_be"] = n_y_be
        return OptimizeAlternative(
            proposal=base, detailing=detailing,
            rho_transverse=rho_t,
            rho_longitudinal=rho_l,
            rho_total=rho_t + rho_l,
            feasible=feasible,
            is_baseline=is_baseline,
            is_provided=is_provided,
            notes=base.notes if is_baseline else (),
        )

    alternatives: list[OptimizeAlternative] = []

    # ---- baseline ----
    # Read the as-built BE longitudinal detail from the wall (set by
    # Wall.rectangular() or by Wall.__init__). This guarantees that the
    # baseline row in the table matches what the user actually built —
    # so it shows as feasible unless the demand truly exceeds it.
    base_db_be = getattr(wall, "be_db_default", 22.0)
    base_n_y_be = getattr(wall, "be_n_y_default", 5)
    alternatives.append(_make_alt(
        db_web=base.web_bar_db, spacing=base.web_bar_spacing,
        layers=base.web_bar_layers,
        db_be=base_db_be, n_y_be=base_n_y_be,
        is_baseline=True,
    ))
    baseline_key = (base.web_bar_db, base.web_bar_spacing, base.web_bar_layers,
                    base_db_be, base_n_y_be)

    # ---- enumerate web × BE ----
    seen: set = {baseline_key}
    for db_web in sorted(db_web_list):
        for spacing in sorted(spacing_list):
            for layers in layers_options:
                # Iterate BE options when applicable, otherwise once with dummies
                be_iter = (
                    [(d, n) for d in sorted(db_be_list) for n in sorted(n_y_be_list)]
                    if has_be else [(base_db_be, n_y_be_list[0])]
                )
                for db_be, n_y_be in be_iter:
                    key = (db_web, spacing, layers, db_be, n_y_be)
                    if key in seen:
                        continue
                    seen.add(key)
                    alt = _make_alt(
                        db_web=db_web, spacing=spacing, layers=layers,
                        db_be=db_be, n_y_be=n_y_be,
                    )
                    if (not alt.feasible) and (not include_infeasible):
                        continue
                    alternatives.append(alt)

    alternatives.sort(key=lambda a: getattr(a, sort_by))
    return alternatives


# ====================================================================== #
# Pretty-print
# ====================================================================== #
def _fmt_wall_detail(alt: OptimizeAlternative) -> str:
    """Detail string in the style of the beams / columns tables.

    Layout:
        web  ø10@200x2c | BE  4φ22 perim.
    or, when the wall has no boundary element:
        web  ø10@200x2c
    """
    d = alt.detailing
    web = f"phi{int(d['db_web'])}@{int(d['spacing'])}x{int(d['layers'])}c"
    if "db_be" in d:
        n_total = 2 * 2 + 2 * (int(d['n_y_be']) - 2)   # n_x_be=2 assumption for label
        # Use the actual n_x_be when available via wall's attribute — but we
        # don't have wall here; print just n_y_be so user can read it.
        be = f"BE {int(d['n_y_be'])}xphi{int(d['db_be'])}"
        return f"web {web} | {be}"
    return f"web {web}"


def format_optimize_table(
    alternatives: list[OptimizeAlternative],
    *,
    top_n: int = 12,
) -> str:
    """Pretty-print a wall optimize ranking.

    Wrapper around :func:`design.common.optimize.format_optimize_table`
    with a wall-specific detail formatter. Same call signature as the
    beams / columns equivalents.

    Example::

        alts = wall.run_optimize(demands)
        print(format_optimize_table(alts, top_n=10))
    """
    return _common_format_table(
        alternatives, top_n=top_n, detail_fmt=_fmt_wall_detail,
    )
