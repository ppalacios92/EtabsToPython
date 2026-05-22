"""Transverse + longitudinal reinforcement optimization for beams.

Generates feasible (db_top, n_top, db_bot, n_bot, db_stirrup, n_legs)
combinations from the beam's bar_schedule, runs beam.design() for each,
and ranks by total steel quantity in kg/m³ of concrete.

Quantity formulas:
    longitudinal: (As_top + As_bot) * 1000 mm = volume per metre of beam
    transverse:   (n_legs * (bw + h - ...) * a_per_leg * (1000/spacing))
                  approximated as 2*(bw + h - 4*cover) * a_per_leg / spacing
                  per metre of beam in the lo region.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from math import pi
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from design.beams.beam import Beam, BeamDemands, BeamDesignProposal


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
    """Mass of transverse steel per m³ of concrete in the lo region (kg/m³)."""
    section = beam.section
    bw = section.width()
    h  = section.height()
    Ag = section.gross_area()

    # Single closed stirrup perimeter, approx
    perim = 2.0 * (bw + h - 4.0 * beam.cover)
    a_per_leg = pi * db_stirrup ** 2 / 4.0
    # Extra interior legs add ~bw of length each
    extra_legs = max(0, n_legs - 2)
    length_per_hoop = perim + extra_legs * (bw - 2.0 * beam.cover)
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


def run_optimize(
    beam: "Beam",
    demands: "BeamDemands | None" = None,
    *,
    db_long_list: list[float] | None = None,
    db_stirrup_list: list[float] | None = None,
    n_legs_list: list[int] | None = None,
    include_provided: bool = True,
    include_infeasible: bool = False,
    sort_by: str = "rho_total",
) -> list[OptimizeAlternative]:
    """Explore detailing alternatives around the design() baseline.

    Flow:
      1. Run beam.design(demands) — baseline (picker chooses everything).
      2. Enumerate (db_top, db_bot, db_stirrup, n_legs) explicitly.
         For each stirrup diameter, also try {2, 3, 4, ...} vertical
         legs (extra crossties parallel to y-axis).
      3. Run beam.design() forcing each combo via the picker overrides.
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
    # map to the same realized detailing.
    seen: set = {
        (base.db_top, base.n_top,
         base.db_bot, base.n_bot,
         base.db_stirrup, base.n_legs)
    }

    # ---- Explore: db_top × db_bot × db_stirrup × n_legs ----
    for db_top, db_bot, db_stp, n_legs in product(
        sorted(db_long_list), sorted(db_long_list),
        sorted(db_stirrup_list), sorted(n_legs_list),
    ):
        results = beam.design(
            demands,
            db_top=db_top, db_bot=db_bot,
            db_stirrup=db_stp, n_legs=n_legs,
        )
        p = results.envelope
        key = (p.db_top, p.n_top, p.db_bot, p.n_bot, p.db_stirrup, p.n_legs)
        if key in seen:
            continue
        seen.add(key)
        feasible = _feasible(p)
        if not feasible and not include_infeasible:
            continue
        rho_long = longitudinal_steel_quantity(
            beam, As_top=p.As_top_continuous, As_bot=p.As_bot_continuous,
        )
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
        prov_key = (provided_alt.db_top, provided_alt.n_top,
                    provided_alt.db_bot, provided_alt.n_bot,
                    provided_alt.db_stirrup, provided_alt.n_legs)
        already_listed = False
        for i, a in enumerate(alternatives):
            if (a.db_top, a.n_top, a.db_bot, a.n_bot,
                a.db_stirrup, a.n_legs) == prov_key:
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
        top = f"{a.n_top}φ{a.db_top:.0f}"
        bot = f"{a.n_bot}φ{a.db_bot:.0f}"
        stp = f"φ{a.db_stirrup:.0f} x {a.n_legs}"
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
