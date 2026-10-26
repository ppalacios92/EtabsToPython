"""Transverse + longitudinal reinforcement optimization for beams.

Generates feasible (db_top, n_top, db_bot, n_bot, db_stirrup, n_legs)
combinations from the beam's bar_schedule, runs beam.design() for each,
and ranks by total steel quantity in kg/m³ of concrete.

Quantity formulas (mass DENSITY, kg steel / m³ concrete — a weight metric,
NOT the volumetric ratio rho_s of §25.7.3, which beams do not have):
    longitudinal: (As_top + As_bot) * 1000 mm = volume per metre of beam
    transverse:   closed-hoop centerline perimeter
                  2*[(bw - 2*cover - db_e) + (h - 2*cover - db_e)] plus
                  vertical interior legs (h - 2*cover - db_e) each, times
                  a_per_leg times (1000/spacing), per metre in the lo region.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from itertools import product
from math import pi
from typing import TYPE_CHECKING

from design.common.factors import phi_shear
from design.common.shear import vc_simplified, vs_capacity

if TYPE_CHECKING:
    from design.beams.beam import Beam, BeamDemands, BeamDesignProposal

# Constructive lower bound for hoop spacing (mm) and the candidate grid the
# optimizer sweeps. Candidates are intersected with [S_MIN, spacing_confined]
# (the §18.6.4.4 cap realized for each db/n_legs combo) so the optimizer can
# propose intermediate spacings instead of collapsing onto the cap.
S_MIN_CONSTRUCTIVE = 50.0
SPACING_CANDIDATES = (75.0, 100.0, 125.0, 150.0)


@dataclass(frozen=True, slots=True)
class OptimizeAlternative:
    proposal: "BeamDesignProposal"
    db_top: float
    n_top:  int
    db_bot: float
    n_bot:  int
    db_stirrup: float
    n_legs: int
    rho_transverse:  float       # kg/m³ of concrete (lo region)
    rho_longitudinal: float      # kg/m³ of concrete (full span)
    rho_total:       float
    feasible:        bool
    is_baseline:     bool = False
    is_provided:     bool = False
    notes: tuple[str, ...] = ()


def transverse_steel_quantity(
    beam: "Beam",
    *,
    db_stirrup: float,
    n_legs: int,
    spacing: float,
) -> float:
    """Transverse-steel MASS DENSITY in the lo region (kg of steel / m³ of
    concrete).

    NOTE: this is a *mass density* metric for ranking detailing weight, NOT
    the formal volumetric ratio rho_s of §25.7.3 — beams do not have a rho_s
    (that governs spirals/columns). The area-based analogue of the confinement
    demand for a beam is Av/(s*bw); this function only weighs the steel.
    """
    section = beam.section
    bw = section.width()
    h  = section.height()
    Ag = section.gross_area()

    # Centerline of the closed hoop: it runs along the bar axis, inset from
    # each face by (cover + db_e/2), so each side spans (dim - 2*cover - db_e).
    db_e = db_stirrup                                   # hoop bar diameter
    perim = 2.0 * ((bw - 2.0 * beam.cover - db_e)
                   + (h - 2.0 * beam.cover - db_e))
    a_per_leg = pi * db_stirrup ** 2 / 4.0
    # Interior legs (crossties) are VERTICAL: length ~ h - 2*cover - db_e,
    # NOT bw (they tie top to bottom across the depth).
    extra_legs = max(0, n_legs - 2)
    length_per_hoop = perim + extra_legs * (h - 2.0 * beam.cover - db_e)
    layers_per_m = 1000.0 / spacing
    volume_steel = length_per_hoop * a_per_leg * layers_per_m   # mm³ per m of beam

    density = (section.rebar.groups[0].steel.density
               if section.rebar.groups else 7850.0)
    mass = volume_steel * density * 1e-9                         # kg

    volume_concrete = Ag * 1000.0 * 1e-9                         # m³
    return mass / volume_concrete if volume_concrete > 0 else 0.0


def longitudinal_steel_quantity(
    beam: "Beam",
    *,
    As_top: float | None = None,
    As_bot: float | None = None,
) -> float:
    """Mass of longitudinal steel per m³ of concrete (kg/m³).

    If As_top / As_bot are None, uses the current section.
    """
    section = beam.section
    if As_top is None or As_bot is None:
        As_total = section.rebar.total_area
    else:
        As_total = As_top + As_bot
    Ag = section.gross_area()
    density = (section.rebar.groups[0].steel.density
               if section.rebar.groups else 7850.0)
    mass = As_total * 1000.0 * density * 1e-9       # kg per m of beam
    volume_concrete = Ag * 1000.0 * 1e-9            # m³ per m
    return mass / volume_concrete if volume_concrete > 0 else 0.0


def _spacing_grid(p: "BeamDesignProposal", spacing_list: list[float]
                  ) -> list[float]:
    """Candidate confined-zone spacings for a realized (db, n_legs) combo.

    The grid is ``spacing_list`` intersected with
    ``[S_MIN_CONSTRUCTIVE, p.spacing_confined]`` (the §18.6.4.4 cap that
    design() already realized for this combo). The design-derived spacing is
    always kept so the optimizer never loses the as-designed point.
    """
    s_cap = p.spacing_confined
    grid = {round(s, 6) for s in spacing_list
            if S_MIN_CONSTRUCTIVE - 1e-6 <= s <= s_cap + 1e-6}
    grid.add(round(s_cap, 6))                # never drop the design point
    return sorted(grid)


def _proposal_at_spacing(beam: "Beam", p: "BeamDesignProposal",
                         spacing: float) -> "BeamDesignProposal":
    """Clone ``p`` at a different hoop spacing, recomputing s-dependent fields.

    Av (db, n_legs) is unchanged, so only spacing_confined, phi_Vn and the
    Ve/phiVn ratio move. §18.6.4.4 cap is respected by `_spacing_grid`.
    """
    d, _, _, _ = beam._depths()
    bw = beam.section.width()
    c = beam.section.concrete
    Vc = vc_simplified(b=bw, d=d, concrete=c)
    Vs = vs_capacity(Av=p.av_provided, fyt=beam.fyt, d=d, s=spacing)
    phi_Vn = phi_shear() * (Vc + Vs)
    ratio = (p.Ve_capacity / phi_Vn
             if (p.Ve_capacity is not None and phi_Vn > 0) else None)
    return replace(p, spacing_confined=spacing,
                   phi_Vn=phi_Vn, ratio_Ve_over_phiVn=ratio)


def run_optimize(
    beam: "Beam",
    demands: "BeamDemands | None" = None,
    *,
    db_long_list: list[float] | None = None,
    db_stirrup_list: list[float] | None = None,
    n_legs_list: list[int] | None = None,
    spacing_list: list[float] | None = None,
    include_provided: bool = True,
    include_infeasible: bool = False,
    sort_by: str = "rho_total",
) -> list[OptimizeAlternative]:
    """Explore detailing alternatives around the design() baseline.

    Flow:
      1. Run beam.design(demands) — baseline (picker chooses everything).
      2. Enumerate (db_top, db_bot, db_stirrup, n_legs) explicitly.
         For each stirrup diameter, also try {2, 3, 4, ...} vertical
         legs (extra crossties parallel to y-axis). `n_legs` candidates
         are filtered to those that geometrically fit the web width.
      3. Run beam.design() forcing each combo via the picker overrides,
         then expand each accepted combo over `spacing_list` (intersected
         with [S_MIN_CONSTRUCTIVE, §18.6.4.4 cap]) so intermediate spacings
         are proposed instead of collapsing onto the cap.
      4. Feasibility = As provided >= required AND av/s provided >=
         required AND continuity rules. Drop infeasible (unless
         include_infeasible=True).
      5. Rank by `sort_by` ascending — lightest first.
    """
    if db_long_list is None:
        db_long_list = list(beam.bar_schedule.longitudinal)
    if db_stirrup_list is None:
        db_stirrup_list = list(beam.bar_schedule.hoops)
    if n_legs_list is None:
        n_legs_list = [2, 3, 4]
    if spacing_list is None:
        spacing_list = list(SPACING_CANDIDATES)

    # FIX V3(a): drop leg counts that cannot physically fit across the width
    # (the largest hoop bar gives the most permissive geometric limit). The
    # horizontal leg spacing limit (~350 mm, analogous to hx §25.7.2.3) caps
    # n_legs at 1 + floor((bw - 2*cover - db_e) / 350). Without this the sweep
    # generated 5–6-leg cages that no detail could support in a narrow web.
    db_e_max = max(db_stirrup_list) if db_stirrup_list else beam.db_stirrup
    n_legs_max = beam._max_legs(db_e_max)
    n_legs_list = [n for n in n_legs_list if n <= n_legs_max] or [2]

    def _feasible(p: "BeamDesignProposal") -> bool:
        ok_long = (p.As_top_continuous >= max(p.As_top_i_required,
                                              p.As_top_mid_required,
                                              p.As_top_j_required) and
                   p.As_bot_continuous >= max(p.As_bot_i_required,
                                              p.As_bot_mid_required,
                                              p.As_bot_j_required) and
                   p.two_bars_top_ok and p.two_bars_bot_ok)
        if not ok_long:
            return False
        if p.av_s_required is not None and p.av_s_required > 0:
            av_needed = p.av_s_required * p.spacing_confined
            if p.av_provided < av_needed - 1e-6:
                return False
        return True

    # ---- Provided (the as-input beam, BEFORE design()) ----
    provided_alt: OptimizeAlternative | None = None
    if include_provided:
        provided_alt = _alternative_from_current_state(
            beam, demands, feasibility_fn=_feasible,
        )

    # ---- Baseline ----
    baseline_results = beam.design(demands)
    base = baseline_results.envelope
    rho_long_base = longitudinal_steel_quantity(
        beam, As_top=base.As_top_continuous, As_bot=base.As_bot_continuous,
    )
    rho_trans_base = transverse_steel_quantity(
        beam, db_stirrup=base.db_stirrup, n_legs=base.n_legs,
        spacing=base.spacing_confined,
    )
    alternatives: list[OptimizeAlternative] = [OptimizeAlternative(
        proposal=base,
        db_top=base.db_top, n_top=base.n_top,
        db_bot=base.db_bot, n_bot=base.n_bot,
        db_stirrup=base.db_stirrup, n_legs=base.n_legs,
        rho_transverse=rho_trans_base,
        rho_longitudinal=rho_long_base,
        rho_total=rho_long_base + rho_trans_base,
        feasible=_feasible(base),
        is_baseline=True,
        notes=base.notes,
    )]
    # Deduplicate by RESULT (not by input): the picker may fall back to
    # the max diameter when an input doesn't fit, so multiple inputs can
    # map to the same realized detailing. Spacing is part of the key so the
    # spacing sweep (FIX V3b) yields distinct entries.
    seen: set = {
        (base.db_top, base.n_top,
         base.db_bot, base.n_bot,
         base.db_stirrup, base.n_legs,
         round(base.spacing_confined, 6))
    }

    # ---- Explore: db_top × db_bot × db_stirrup × n_legs × spacing ----
    # design() fixes the spacing at the §18.6.4.4 cap for each combo; FIX V3b
    # then expands each accepted combo over the candidate spacing grid so the
    # optimizer can trade a tighter spacing against fewer legs / smaller bars
    # instead of collapsing onto one or two spacings.
    for db_top, db_bot, db_stp, n_legs in product(
        sorted(db_long_list), sorted(db_long_list),
        sorted(db_stirrup_list), sorted(n_legs_list),
    ):
        results = beam.design(
            demands,
            db_top=db_top, db_bot=db_bot,
            db_stirrup=db_stp, n_legs=n_legs,
        )
        p_base = results.envelope
        rho_long = longitudinal_steel_quantity(
            beam, As_top=p_base.As_top_continuous,
            As_bot=p_base.As_bot_continuous,
        )
        for s in _spacing_grid(p_base, spacing_list):
            p = _proposal_at_spacing(beam, p_base, s)
            key = (p.db_top, p.n_top, p.db_bot, p.n_bot,
                   p.db_stirrup, p.n_legs, round(p.spacing_confined, 6))
            if key in seen:
                continue
            seen.add(key)
            feasible = _feasible(p)
            if not feasible and not include_infeasible:
                continue
            rho_trans = transverse_steel_quantity(
                beam, db_stirrup=p.db_stirrup, n_legs=p.n_legs,
                spacing=p.spacing_confined,
            )
            alternatives.append(OptimizeAlternative(
                proposal=p,
                db_top=p.db_top, n_top=p.n_top,
                db_bot=p.db_bot, n_bot=p.n_bot,
                db_stirrup=p.db_stirrup, n_legs=p.n_legs,
                rho_transverse=rho_trans,
                rho_longitudinal=rho_long,
                rho_total=rho_long + rho_trans,
                feasible=feasible,
                is_baseline=False,
                notes=p.notes,
            ))

    # ---- Insert PROVIDED if requested and not already in the set ----
    if provided_alt is not None:
        prov_s = round(provided_alt.proposal.spacing_confined, 6)
        prov_key = (provided_alt.db_top, provided_alt.n_top,
                    provided_alt.db_bot, provided_alt.n_bot,
                    provided_alt.db_stirrup, provided_alt.n_legs)
        # Prefer an existing entry at the SAME spacing; only collapse onto a
        # different-spacing variant of the same cage if no exact match exists.
        exact_i = next(
            (i for i, a in enumerate(alternatives)
             if (a.db_top, a.n_top, a.db_bot, a.n_bot,
                 a.db_stirrup, a.n_legs) == prov_key
             and round(a.proposal.spacing_confined, 6) == prov_s),
            None,
        )
        already_listed = False
        for i, a in enumerate(alternatives):
            same_cage = (a.db_top, a.n_top, a.db_bot, a.n_bot,
                         a.db_stirrup, a.n_legs) == prov_key
            same_spacing = round(a.proposal.spacing_confined, 6) == prov_s
            # If an exact (cage+spacing) match exists, only tag that one.
            if same_cage and (same_spacing if exact_i is not None else True):
                # Mark the existing entry as is_provided too
                alternatives[i] = OptimizeAlternative(
                    proposal=a.proposal,
                    db_top=a.db_top, n_top=a.n_top,
                    db_bot=a.db_bot, n_bot=a.n_bot,
                    db_stirrup=a.db_stirrup, n_legs=a.n_legs,
                    rho_transverse=a.rho_transverse,
                    rho_longitudinal=a.rho_longitudinal,
                    rho_total=a.rho_total,
                    feasible=a.feasible,
                    is_baseline=a.is_baseline,
                    is_provided=True,
                    notes=a.notes,
                )
                already_listed = True
                break
        if not already_listed:
            # Always include the PROVIDED entry when requested — even
            # if infeasible — so the user sees the verdict of their input.
            alternatives.append(provided_alt)

    alternatives.sort(key=lambda a: getattr(a, sort_by))
    return alternatives


def _alternative_from_current_state(
    beam: "Beam",
    demands: "BeamDemands | None",
    *,
    feasibility_fn,
) -> OptimizeAlternative:
    """Build an OptimizeAlternative reflecting the CURRENT beam state.

    Delegates to `Beam.current_proposal()` for the BeamDesignProposal,
    then wraps it with the steel-quantity ranking fields.
    """
    prop = beam.current_proposal(demands)
    rho_long = longitudinal_steel_quantity(
        beam, As_top=prop.As_top_continuous, As_bot=prop.As_bot_continuous,
    )
    rho_trans = transverse_steel_quantity(
        beam, db_stirrup=prop.db_stirrup, n_legs=prop.n_legs,
        spacing=prop.spacing_confined,
    )
    return OptimizeAlternative(
        proposal=prop,
        db_top=prop.db_top, n_top=prop.n_top,
        db_bot=prop.db_bot, n_bot=prop.n_bot,
        db_stirrup=prop.db_stirrup, n_legs=prop.n_legs,
        rho_transverse=rho_trans,
        rho_longitudinal=rho_long,
        rho_total=rho_long + rho_trans,
        feasible=feasibility_fn(prop),
        is_baseline=False,
        is_provided=True,
        notes=(),
    )


def format_optimize_table(alternatives: list[OptimizeAlternative],
                          *, top_n: int = 12) -> str:
    """Pretty-print the optimize output in the columns-module style.

    Shows the top `top_n` entries ranked by `rho_total`. PROVIDED and
    baseline entries are ALWAYS shown, even when ranked outside top_n
    (appended after the top with a separator).

    All steel quantities in kg/m³ of concrete.
    """
    lines = []
    header = (f"{'rank':>4}  {'top':>9}  {'bot':>9}  {'stirrup':>14}  "
              f"{'rho_t':>8}  {'rho_l':>8}  {'rho_tot':>8}  {'tag':>20}")
    units_line = (f"{'':>4}  {'':>9}  {'':>9}  {'':>14}  "
                  f"{'(kg/m3)':>8}  {'(kg/m3)':>8}  {'(kg/m3)':>8}  {'':>20}")
    lines.append(header)
    lines.append(units_line)
    lines.append("-" * len(header))

    def _row(rank: int, a: OptimizeAlternative) -> str:
        top = f"{a.n_top}phi{a.db_top:.0f}"
        bot = f"{a.n_bot}phi{a.db_bot:.0f}"
        stp = f"phi{a.db_stirrup:.0f} x {a.n_legs}"
        tags = []
        if a.is_baseline: tags.append("baseline")
        if a.is_provided: tags.append("provided")
        if not a.feasible: tags.append("infeasible")
        tag = ", ".join(tags) if tags else ""
        return (f"{rank:>4}  {top:>9}  {bot:>9}  {stp:>14}  "
                f"{a.rho_transverse:>8.1f}  {a.rho_longitudinal:>8.1f}  "
                f"{a.rho_total:>8.1f}  {tag:>20}")

    top_alts = alternatives[:top_n]
    top_indices = set(range(min(top_n, len(alternatives))))
    for i, a in enumerate(top_alts):
        lines.append(_row(i + 1, a))

    # Always show PROVIDED and baseline, even if ranked outside top_n
    extras = [
        (i, a) for i, a in enumerate(alternatives)
        if i not in top_indices and (a.is_provided or a.is_baseline)
    ]
    if extras:
        lines.append("." * len(header))
        for orig_rank, a in extras:
            lines.append(_row(orig_rank + 1, a))
    return "\n".join(lines)
