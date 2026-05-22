"""Concrete column — self-contained, lazy, with chained access.

A Column wraps a Section (geometry + materials + rebar) plus the
detailing parameters (lu, k, hoop spacing, …) and offers:

    col.run(...)                  -> compute Po, To, surface, curves
    col.surface[theta]            -> uniaxial diagram at that angle
    col.surface.at_Pu(Pu)         -> biaxial envelope (Mx, My)
    col.surface.volume(n_Pu=25)   -> (Pu, Mx, My) grids for 3D
    col.surface.check(...)        -> biaxial demand check
    col.Mn / col.phi_Mn / col.Pn / col.phi_Pn   -> result-view proxies
    col.capacity()                -> legacy ColumnCapacity (back-compat)
    col.check(demands)            -> ColumnCheck
    col.design(demands=None, ...) -> ColumnDesignProposal — minimum
                                     reinforcement if demands is None.
    col.evolve(proposal)          -> NEW Column with that detailing applied
    col.apply(proposal)           -> alias for evolve
    col.set_units(code_or_name)   -> presentation units (ETABS codes 1..16)
    col.plot.*                    -> bound plotter
    col.summary() / col.report()  -> formatted views

Internal units (per design/AGENTS.md):
    Stress   MPa
    Length   mm   (INCLUDES lu — NOT metres)
    Force    kN
    Moment   kN·m
    Area     mm²

ACI 318-25 sections used:
    §22.4.2.1  Pn,max cap for ties/spirals
    §21.2.1    phi_shear
    §21.2.2    phi(eps_t)  — strength reduction factor
    §22.2      cross-section strain compatibility / Whitney block
    §22.5.5.1  Vc simplified
    §22.5.10   Vs
    §18.7.5.1  lo length of confined region
    §18.7.5.2  hx limit (≤350 mm) on supported longitudinal bars
    §18.7.5.3  spacing in confined region
    §18.7.5.4  Ash transverse reinforcement
    §18.7.5.5  spacing outside the confined region
    §18.7.6.1.1 Omega_0 cap on capacity-design shear
    §18.7.6.2.1 Vc=0 in plastic-hinge zone (light axial + seismic-dominated)
"""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil, pi
from typing import Optional
import warnings

from design.common.factors import phi_shear, OMEGA_0_COLUMN_DEFAULT
from design.common.materials import Bar, Concrete, Steel
from design.common.units import UnitSystem, units_from, DEFAULT_UNITS
from design.common.shear import (
    vc_simplified, vs_capacity, av_s_required, vc_zero_seismic,
)
from design.common.spacing import (
    ash_required, s_max_confined_column, lo_length_column,
    s_max_middle_zone_column, hx_check,
)
from design.sections.base import Section
from design.sections.rectangular import RectangularSection
from design.sections.reinforcement import RebarLayout, perimeter_bars
from design.columns.bar_schedule import BarSchedule
from design.columns.interaction import InteractionDiagram, interaction_diagram
from design.columns.interaction_surface import (
    InteractionSurface, interaction_surface,
)
from design.columns.plotting import _ColumnPlotter
from design.columns.probable import mpr_envelope
from design.columns.proxies import _DiagramFieldView
from design.columns.surface import Surface


# ---------------------------------------------------------------- #
# Demand / Capacity / Check / Design dataclasses
# ---------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ColumnDemands:
    Pu: float = 0.0          # axial (kN, + compression)
    Mux: float = 0.0         # moment about local X (kN·m)
    Muy: float = 0.0         # moment about local Y (kN·m)
    Vux: float = 0.0
    Vuy: float = 0.0


@dataclass(frozen=True, slots=True)
class ColumnCapacity:
    Po: float
    phi_Pn_max: float
    To: float
    surface: Surface
    diagram_x: InteractionDiagram
    diagram_y: InteractionDiagram
    phi_Vn: float


@dataclass(frozen=True, slots=True)
class ColumnCheck:
    capacity: ColumnCapacity
    demands: ColumnDemands
    ratio_pmm: float
    ratio_shear: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ColumnDesignProposal:
    """Detailing proposal for ONE design mode.

    Sign / direction convention:
      - Mpr_x   : probable moment around the local X axis (strong axis if h>b)
      - Mpr_y   : probable moment around the local Y axis (weak axis if h>b)
      - Ve_y    : design shear in Y direction (caused by Mpr_x). Resisted by legs ∥ X.
      - Ve_x    : design shear in X direction (caused by Mpr_y). Resisted by legs ∥ Y.
      - phi_Vn_x: column shear capacity in X — uses bw=h, d=0.9·b, Av = legs ∥ Y.
      - phi_Vn_y: column shear capacity in Y — uses bw=b, d=0.9·h, Av = legs ∥ X.

    `clear_cover` (passed to design) is the cover to the OUTSIDE EDGE of the
    transverse reinforcement — same definition the ACI Tabla 18.7.5.4 uses
    for bc and Ach. ``cover`` is accepted as an alias.
    """
    mode: str                     # 'minimum' | 'demand' | 'capacity' | 'envelope'

    # Required confinement — §18.7.5.4 with directional bc
    ash_required_x: float
    ash_required_y: float
    # Chosen detailing
    db_hoop: float
    n_legs_x: int
    n_legs_y: int
    av_provided_x: float
    av_provided_y: float
    # Spacings
    spacing_confined: float       # §18.7.5.3
    spacing_middle: float         # §18.7.5.5
    lo: float                     # §18.7.5.1
    # Geometry checks
    hx_x_face: float              # §18.7.5.2
    hx_y_face: float
    hx_ok: bool
    # Drivers used
    Pu_used: float | None = None          # kN, axial that fed eq (iii)
    Pu_over_Ag_fc: float | None = None
    eq_iii_active: bool = False
    # Capacity-design quantities — properties of section + lu (per direction)
    Mpr_x: float | None = None
    Mpr_y: float | None = None
    Pu_for_Mpr_x: float | None = None
    Pu_for_Mpr_y: float | None = None
    Ve_x_capacity: float | None = None
    Ve_y_capacity: float | None = None
    Ve_x_used: float | None = None
    Ve_y_used: float | None = None
    amplification_x: float | None = None
    amplification_y: float | None = None
    # Shear capacity of the *proposed* transverse reinforcement (per direction)
    phi_Vn_x: float | None = None
    phi_Vn_y: float | None = None
    av_s_required_x: float | None = None
    av_s_required_y: float | None = None
    ratio_Ve_over_phiVn_x: float | None = None
    ratio_Ve_over_phiVn_y: float | None = None
    # §18.7.6.2.1 — Vc=0 in lo (light axial + seismic-dominated)
    vc_zero_x: bool = False
    vc_zero_y: bool = False
    # Diagnostics
    notes: tuple[str, ...] = ()

    # ---- Legacy aliases (back-compat) ----
    @property
    def Mpr_top(self) -> float | None:
        return self.Mpr_x

    @property
    def Mpr_bot(self) -> float | None:
        return self.Mpr_x

    @property
    def Pu_for_Mpr(self) -> float | None:
        return self.Pu_for_Mpr_x

    @property
    def Ve_capacity(self) -> float | None:
        return self.Ve_y_capacity

    @property
    def Ve_used(self) -> float | None:
        return self.Ve_y_used

    @property
    def amplification_vs_demand(self) -> float | None:
        return self.amplification_y

    @property
    def Vu_used(self) -> float | None:
        return self.Ve_y_used

    @property
    def av_s_required(self) -> float | None:
        x, y = self.av_s_required_x, self.av_s_required_y
        if x is None and y is None:
            return None
        return max(x or 0.0, y or 0.0)

    @property
    def av_s_provided(self) -> float | None:
        if self.spacing_confined <= 0:
            return None
        a_leg = pi * self.db_hoop ** 2 / 4.0
        per_y = self.n_legs_x * a_leg / self.spacing_confined   # for Vy → legs ∥ X
        per_x = self.n_legs_y * a_leg / self.spacing_confined   # for Vx → legs ∥ Y
        return max(per_x, per_y)

    @property
    def ratio_shear(self) -> float | None:
        x, y = self.ratio_Ve_over_phiVn_x, self.ratio_Ve_over_phiVn_y
        if x is None and y is None:
            return None
        return max(x or 0.0, y or 0.0)


@dataclass(frozen=True, slots=True)
class ColumnDesignResults:
    """Container of all three design modes for one column.

    Access:
        results.minimum   — code minimum (§18.7.5.4 i, ii)
        results.demand    — using demands (adds eq iii if Pu/Ag·fc'>0.3)
        results.capacity  — Mpr-based capacity design (§18.7.6)
        results.envelope  — most demanding of the three
    """
    minimum:  ColumnDesignProposal
    demand:   ColumnDesignProposal | None
    capacity: ColumnDesignProposal
    envelope: ColumnDesignProposal

    def __iter__(self):
        yield self.minimum
        if self.demand is not None:
            yield self.demand
        yield self.capacity
        yield self.envelope

    def as_dict(self) -> dict:
        d = {"minimum": self.minimum,
             "capacity": self.capacity,
             "envelope": self.envelope}
        if self.demand is not None:
            d["demand"] = self.demand
        return d


# ---------------------------------------------------------------- #
# Column class
# ---------------------------------------------------------------- #
class Column:
    """Reinforced concrete column, autocontained.

    Internal numerics (per design/AGENTS.md):
        stress   MPa
        length   mm   (INCLUDES lu)
        force    kN
        moment   kN·m
        area     mm²

    Use ``col.set_units(code)`` to change PRESENTATION units; the
    formulas always stay in the internal system.
    """

    def __init__(
        self,
        *,
        section: Section,
        lu: float,                       # unbraced length (mm — see AGENTS.md)
        k: float = 1.0,
        spiral: bool = False,
        Av: float = 0.0,                 # transverse area used by capacity()
        fyt: float = 420.0,
        s: float | None = None,
        seismic: bool = True,
        hx: float = 200.0,
        h_min: float | None = None,
        transverse_bar_diameter: float = 10.0,
        units: int | str | UnitSystem | None = None,
        bar_schedule: BarSchedule | None = None,
        label: str = "Column",
        run_settings: dict | None = None,
    ) -> None:
        # ---- inputs ----
        self.section = section
        # ``lu`` is canonically in mm (design/AGENTS.md §1). The
        # ``Column.rectangular()`` factory keeps a metres-shortcut for
        # back-compat; the bare constructor expects mm. Anyone passing
        # ``lu < 30`` here is almost certainly still in metres, so warn
        # and convert.
        if lu is not None and lu < 30.0:
            warnings.warn(
                f"Column(lu={lu}) looks like metres; the canonical unit is "
                f"mm. Converting (lu * 1000). Pass mm explicitly to silence.",
                DeprecationWarning,
                stacklevel=2,
            )
            lu = lu * 1000.0
        self.lu = lu                     # mm
        self.k = k
        self.spiral = spiral
        self.Av = Av
        self.fyt = fyt
        self.s = s
        self.seismic = seismic
        self.hx = hx
        self.h_min = h_min if h_min is not None else section.width()
        self.transverse_bar_diameter = transverse_bar_diameter
        self.label = label
        # Directional leg counts — set by Column.rectangular() or evolve().
        # None means "not specified yet — design() will propose them".
        self.n_legs_x: int | None = None
        self.n_legs_y: int | None = None

        # ---- presentation units ----
        if units is None:
            self.units = DEFAULT_UNITS
        elif isinstance(units, UnitSystem):
            self.units = units
        else:
            self.units = units_from(units)

        # ---- bar schedule (defaults if not provided) ----
        self.bar_schedule = bar_schedule if bar_schedule is not None else BarSchedule()

        # ---- run settings (persistent kwargs for run()) ----
        self.run_settings: dict = {"n_angles": 37, "n_points": 60}
        if run_settings:
            self.run_settings.update(run_settings)

        # ---- caches (filled by run()) ----
        self._surface: Optional[Surface] = None
        self._inner_surface: Optional[InteractionSurface] = None
        self._Po: Optional[float] = None
        self._To: Optional[float] = None
        self._phi_Pn_max: Optional[float] = None
        self._phi_Vn: Optional[float] = None

        # ---- plotter (lazy) ----
        self._plotter: Optional[_ColumnPlotter] = None

        # ---- last design results (NOT mutated by design()) ----
        # Kept as a slot for legacy code that reads it; populating it is
        # the caller's responsibility now.
        self.design_results: Optional["ColumnDesignResults"] = None

        # ---- omega_0 cap for capacity-design shear (§18.7.6.1.1) ----
        self.omega_0: float = OMEGA_0_COLUMN_DEFAULT

    # =============================================================== #
    # One-shot factory — scalar inputs, builds section + column
    # =============================================================== #
    @classmethod
    def rectangular(
        cls,
        *,
        # --- geometry ---
        b: float,
        h: float,
        clear_cover: float | None = None,
        cover: float | None = None,           # alias for clear_cover
        concrete: Concrete,
        steel: Steel,
        # --- longitudinal (perimeter, uniform diameter) ---
        n_x: int,
        n_y: int,
        db_long: float,
        # --- transverse (optional — leave None to design later) ---
        db_stirrup: float = 10.0,
        n_legs_x: int | None = None,
        n_legs_y: int | None = None,
        s_stirrup: float | None = None,
        # --- member ---
        lu: float,
        k: float = 1.0,
        spiral: bool = False,
        seismic: bool = True,
        hx: float = 200.0,
        h_min: float | None = None,
        steel_transverse: Steel | None = None,
        units: int | str | UnitSystem | None = None,
        bar_schedule: BarSchedule | None = None,
        label: str = "Column",
        run_settings: dict | None = None,
    ) -> "Column":
        """Build a Column from scalar specs in a single call.

        Longitudinal layout is a perimeter pattern with uniform diameter,
        built internally via ``perimeter_bars``. ``clear_cover`` (alias:
        ``cover``) is to the outside edge of the stirrup; bar centroids
        are placed at ``clear_cover + db_stirrup + db_long/2`` from each
        face. The canonical name is ``cover``; ``clear_cover`` is the
        legacy alias kept for back-compat.

        Transverse detailing is OPTIONAL: pass ``db_stirrup, n_legs_x,
        n_legs_y, s_stirrup`` if you have a candidate to check; leave
        them as defaults / None to let ``col.design(demands)`` propose
        the detailing.

        ``lu`` is canonically in **mm** (per ``design/AGENTS.md``). As a
        convenience this factory accepts metres too: any value below
        30 is assumed to be in metres and converted internally (with a
        ``DeprecationWarning``). Pass mm explicitly to silence the warning.

        Example::

            col = Column.rectangular(
                b=500, h=500, cover=25,
                concrete=c, steel=s,
                n_x=4, n_y=4, db_long=22,
                db_stirrup=10, n_legs_x=3, n_legs_y=3, s_stirrup=100,
                lu=3000, seismic=True, hx=200,
            )
        """
        # ---- cover alias resolution ----
        if cover is not None and clear_cover is not None and cover != clear_cover:
            raise TypeError(
                "Column.rectangular(): pass either `cover` or `clear_cover`, "
                "not both with different values."
            )
        if clear_cover is None:
            clear_cover = cover if cover is not None else 25.0

        # ---- lu unit normalization (metres -> mm) ----
        if lu is not None and lu < 30.0:
            warnings.warn(
                f"Column.rectangular(lu={lu}) looks like metres; the "
                f"canonical unit is mm. Converting (lu * 1000). Pass mm "
                f"explicitly to silence.",
                DeprecationWarning,
                stacklevel=2,
            )
            lu = lu * 1000.0

        # Cover used by perimeter_bars is to the BAR CENTROID
        cover_to_centroid = clear_cover + db_stirrup + db_long / 2.0

        group = perimeter_bars(
            b=b, h=h, cover=cover_to_centroid,
            n_x=n_x, n_y=n_y,
            bar=Bar(diameter=db_long),
            steel=steel,
        )
        section = RectangularSection(
            b=b, h=h, concrete=concrete,
            rebar=RebarLayout(groups=(group,)),
        )

        # Convert (n_legs_x, n_legs_y, db_stirrup, s) into the scalar Av
        # used by the legacy phi_Vn path. min() is the conservative pick.
        a_per_leg = pi * db_stirrup ** 2 / 4.0
        Av = 0.0
        if n_legs_x is not None and n_legs_y is not None:
            Av = min(n_legs_x, n_legs_y) * a_per_leg

        fyt_val = (steel_transverse or steel).fy

        # Build the Column directly with mm-lu (bypass the __init__
        # deprecation warning since we've already converted).
        col = cls.__new__(cls)
        Column.__init__(
            col,
            section=section,
            # __init__'s lu<30 shortcut won't trip because we already
            # converted above.
            lu=lu, k=k, spiral=spiral,
            Av=Av, fyt=fyt_val, s=s_stirrup,
            seismic=seismic, hx=hx, h_min=h_min,
            transverse_bar_diameter=db_stirrup,
            units=units, bar_schedule=bar_schedule, label=label,
            run_settings=run_settings,
        )
        col.n_legs_x = n_legs_x
        col.n_legs_y = n_legs_y
        return col

    # =============================================================== #
    # Configuration helpers
    # =============================================================== #
    @property
    def lu_m(self) -> float:
        """Legacy view of ``lu`` in metres. The canonical store is mm."""
        return self.lu / 1000.0

    def set_units(self, code_or_name: int | str) -> "Column":
        """Change presentation units. Returns self for chaining."""
        self.units = units_from(code_or_name)
        return self

    def clear_cache(self) -> None:
        self._surface = None
        self._inner_surface = None
        self._Po = self._To = self._phi_Pn_max = self._phi_Vn = None

    def run_optimize(
        self,
        demands: "ColumnDemands | None" = None,
        *,
        db_hoop_list: list[float] | None = None,
        clear_cover: float = 25.0,
        include_infeasible: bool = False,
        sort_by: str = "rho_total",
    ):
        """Explore detailing alternatives around the design() baseline.

        Returns a list of `OptimizeAlternative` objects (always including
        the baseline marked with `is_baseline=True`), sorted ascending by
        `sort_by` ('rho_total' or 'rho_transverse'). The lightest entry
        is the optimum.

        `clear_cover` is the cover to the OUTSIDE EDGE of the transverse
        reinforcement (the bc definition of ACI 318-25 Table 18.7.5.4).

        All quantities (`rho_transverse`, `rho_longitudinal`, `rho_total`)
        are in kg/m³ of concrete.
        """
        from design.columns.optimize import run_optimize as _opt
        return _opt(
            self, demands,
            db_hoop_list=db_hoop_list,
            clear_cover=clear_cover,
            include_infeasible=include_infeasible,
            sort_by=sort_by,
        )

    def evolve(self, proposal: "ColumnDesignProposal") -> "Column":
        """Return a NEW Column with the transverse detailing of ``proposal``.

        Inmutable per design/AGENTS.md §4. The new column shares the same
        section (which depends only on the longitudinal layout — proposal
        only changes hoops). All other parameters (lu, k, spiral, seismic,
        hx, bar_schedule, units, label, …) are copied.

        After ``evolve``, the new column's ``phi_Vn`` will be recomputed
        lazily through ``run()``.
        """
        a_per_leg = pi * proposal.db_hoop ** 2 / 4.0
        new_Av = min(proposal.n_legs_x, proposal.n_legs_y) * a_per_leg
        new = Column(
            section=self.section,
            lu=self.lu,                  # already mm
            k=self.k,
            spiral=self.spiral,
            Av=new_Av,
            fyt=self.fyt,
            s=proposal.spacing_confined,
            seismic=self.seismic,
            hx=self.hx,
            h_min=self.h_min,
            transverse_bar_diameter=proposal.db_hoop,
            units=self.units,
            bar_schedule=self.bar_schedule,
            label=self.label,
            run_settings=dict(self.run_settings),
        )
        new.n_legs_x = proposal.n_legs_x
        new.n_legs_y = proposal.n_legs_y
        return new

    def apply(self, proposal: "ColumnDesignProposal") -> "Column":
        """Alias for :meth:`evolve` (per design/AGENTS.md §4)."""
        return self.evolve(proposal)

    # =============================================================== #
    # Computation
    # =============================================================== #
    def run(
        self,
        *,
        n_angles: int | None = None,
        n_points: int | None = None,
    ) -> "Column":
        """Build the PMM surface and cache scalars. Returns self."""
        n_a = n_angles if n_angles is not None else self.run_settings.get("n_angles", 37)
        n_p = n_points if n_points is not None else self.run_settings.get("n_points", 60)

        self._inner_surface = interaction_surface(
            self.section,
            n_angles=n_a,
            n_points_per_curve=n_p,
            spiral=self.spiral,
        )
        self._surface = Surface(self._inner_surface)

        # Cache the §22.4.2.1 constants from the 0°-curve
        d0 = self._inner_surface.at_angle(0.0)
        self._Po = d0.Po
        self._To = d0.To
        self._phi_Pn_max = d0.Pn_max_phi

        # Shear capacity (simple Vc + Vs)  — §22.5.5.1 + §22.5.10
        d = 0.9 * self.section.height()
        b = self.section.width()
        Vc = vc_simplified(b=b, d=d, concrete=self.section.concrete)
        Vs = 0.0
        if self.Av > 0 and self.s and self.s > 0:
            Vs = vs_capacity(Av=self.Av, fyt=self.fyt, d=d, s=self.s)
        self._phi_Vn = phi_shear() * (Vc + Vs)

        return self

    def _ensure_run(self) -> None:
        if self._surface is None:
            self.run()

    # =============================================================== #
    # Surface, scalars, and proxies (all lazy)
    # =============================================================== #
    @property
    def surface(self) -> Surface:
        self._ensure_run()
        return self._surface  # type: ignore[return-value]

    @property
    def Po(self) -> float:
        self._ensure_run()
        return self._Po  # type: ignore[return-value]

    @property
    def To(self) -> float:
        self._ensure_run()
        return self._To  # type: ignore[return-value]

    @property
    def phi_Pn_max(self) -> float:
        self._ensure_run()
        return self._phi_Pn_max  # type: ignore[return-value]

    @property
    def phi_Vn(self) -> float:
        self._ensure_run()
        return self._phi_Vn  # type: ignore[return-value]

    @property
    def Mn(self) -> _DiagramFieldView:
        return _DiagramFieldView(self, "Mn")

    @property
    def phi_Mn(self) -> _DiagramFieldView:
        return _DiagramFieldView(self, "phi_Mn")

    @property
    def Pn(self) -> _DiagramFieldView:
        return _DiagramFieldView(self, "Pn")

    @property
    def phi_Pn(self) -> _DiagramFieldView:
        return _DiagramFieldView(self, "phi_Pn")

    @property
    def plot(self) -> _ColumnPlotter:
        if self._plotter is None:
            self._plotter = _ColumnPlotter(self)
        return self._plotter

    # =============================================================== #
    # Legacy / back-compat capacity()
    # =============================================================== #
    def capacity(
        self,
        *,
        biaxial: bool = True,
        n_angles: int = 19,
        n_points_per_curve: int = 60,
    ) -> ColumnCapacity:
        """Back-compat shape — same as before but now uses the cached surface."""
        if self._surface is None or n_angles != self.run_settings.get("n_angles"):
            self.run_settings["n_angles"] = n_angles
            self.run_settings["n_points"] = n_points_per_curve
            self.run(n_angles=n_angles, n_points=n_points_per_curve)
        return ColumnCapacity(
            Po=self.Po,
            phi_Pn_max=self.phi_Pn_max,
            To=self.To,
            surface=self.surface,
            diagram_x=self.surface[0.0],
            diagram_y=self.surface[90.0],
            phi_Vn=self.phi_Vn,
        )

    # =============================================================== #
    # check(demands)
    # =============================================================== #
    def check(self, demands: ColumnDemands) -> ColumnCheck:
        self._ensure_run()
        passed_pmm, ratio_pmm = self.surface.check(
            Pu=demands.Pu, Mux=demands.Mux, Muy=demands.Muy,
        )
        Vu_mag = (demands.Vux ** 2 + demands.Vuy ** 2) ** 0.5
        ratio_shear = Vu_mag / self.phi_Vn if self.phi_Vn > 0 else float("inf")
        passed = passed_pmm and (ratio_shear <= 1.0)
        cap = ColumnCapacity(
            Po=self.Po, phi_Pn_max=self.phi_Pn_max, To=self.To,
            surface=self.surface,
            diagram_x=self.surface[0.0],
            diagram_y=self.surface[90.0],
            phi_Vn=self.phi_Vn,
        )
        return ColumnCheck(
            capacity=cap, demands=demands,
            ratio_pmm=ratio_pmm,
            ratio_shear=ratio_shear,
            passed=passed,
        )

    # =============================================================== #
    # design — runs the three modes and returns them all
    # =============================================================== #
    def design(
        self,
        demands: ColumnDemands | None = None,
        *,
        db_hoop: float | None = None,
        n_legs_x: int | None = None,
        n_legs_y: int | None = None,
        db_long_min: float = 20.0,
        clear_cover: float | None = None,
        cover: float | None = None,             # alias for clear_cover
    ) -> ColumnDesignResults:
        """Run the three design modes and return them all.

        Modes computed:
          - minimum   : ACI §18.7.5.4 eqs (i), (ii) only — no axial input
          - demand    : adds eq (iii) using demands.Pu (None if no demands)
          - capacity  : adds eq (iii) using Pu_for_Mpr; computes Mpr and
                        Ve = 2·Mpr / lu per §18.7.6.1. Evaluates Vc=0
                        in lo per §18.7.6.2.1.
          - envelope  : pointwise worst of the above

        Does NOT mutate ``self.design_results`` (per design/AGENTS.md §4).
        The caller is responsible for stashing the return value if it
        wants to keep it around.

        ``cover`` is accepted as alias of ``clear_cover``.
        """
        # ---- cover alias resolution ----
        if cover is not None and clear_cover is not None and cover != clear_cover:
            raise TypeError(
                "Column.design(): pass either `cover` or `clear_cover`, "
                "not both with different values."
            )
        if clear_cover is None:
            clear_cover = cover if cover is not None else 25.0

        self._ensure_run()
        common = dict(
            db_hoop=db_hoop, n_legs_x=n_legs_x, n_legs_y=n_legs_y,
            db_long_min=db_long_min, clear_cover=clear_cover,
        )

        prop_min = self._design_one(mode="minimum", demands=None, **common)
        prop_dem = (
            self._design_one(mode="demand", demands=demands, **common)
            if demands is not None else None
        )
        prop_cap = self._design_one(
            mode="capacity", demands=demands, **common,
        )
        prop_env = self._design_envelope(prop_min, prop_dem, prop_cap)

        results = ColumnDesignResults(
            minimum=prop_min, demand=prop_dem,
            capacity=prop_cap, envelope=prop_env,
        )
        # NOTE: per design/AGENTS.md §4, design() must NOT mutate
        # self.design_results. The caller stashes the return value.
        return results

    # ---- internal: build one proposal for a given mode ----
    def _design_one(
        self,
        *,
        mode: str,
        demands: ColumnDemands | None,
        db_hoop: float | None,
        n_legs_x: int | None,
        n_legs_y: int | None,
        db_long_min: float,
        clear_cover: float,
    ) -> ColumnDesignProposal:
        c = self.section.concrete
        notes: list[str] = []

        # ---- spacing in confined zone (§18.7.5.3) ----
        s_try = self.s if self.s is not None else 100.0
        if self.seismic:
            grade = self.section.rebar.groups[0].steel.grade if self.section.rebar.groups else 60
            s_try = min(s_try, s_max_confined_column(
                h_min=self.h_min, db_long_min=db_long_min,
                hx=self.hx, grade=grade,
            ))
            notes.append("Spacing in lo capped by §18.7.5.3.")

        # ---- core geometry ----
        # bc is measured to the OUTSIDE EDGE of the hoop (ACI 318-25 Tabla 18.7.5.4),
        # so we use clear_cover (not cover-to-center).
        Ag = self.section.gross_area()
        b_out = self.section.width()
        h_out = self.section.height()
        bc = max(b_out - 2 * clear_cover, 100.0)
        hc = max(h_out - 2 * clear_cover, 100.0)
        Ach = bc * hc

        # ---- pick Pu_used per mode (for §18.7.5.4 eq iii activation) ----
        # ACI R18.7.5.4 says Pu = the factored axial from any load combo
        # that includes earthquake effects. In modes where we don't have
        # explicit demands, we use the axial at which the plastic hinge
        # forms (Pu_for_Mpr), not Po — Po is conservative to the point of
        # being unrealistic (the column never reaches pure compression
        # while it is plastifying in flexure).
        Pu_used: float | None = None
        if mode == "demand" and demands is not None:
            Pu_used = demands.Pu
        elif mode == "capacity":
            # Use the larger Pu among the two Mpr formation states.
            Mpr_pre_x, Pu_pre_x = mpr_envelope(self.section, angle_deg=0.0, spiral=self.spiral)
            Mpr_pre_y, Pu_pre_y = mpr_envelope(self.section, angle_deg=90.0, spiral=self.spiral)
            Pu_used = max(Pu_pre_x, Pu_pre_y)
        # 'minimum' → leave Pu_used = None

        Pu_ratio = 0.0
        if Pu_used is not None and Ag * c.fc > 0:
            Pu_ratio = Pu_used * 1000.0 / (Ag * c.fc)

        # ---- Ash required (§18.7.5.4) ----
        ash_req_x = ash_required(s=s_try, bc=bc, Ag=Ag, Ach=Ach,
                                 fc=c.fc, fyt=self.fyt, Pu_over_Ag_fc=Pu_ratio)
        ash_req_y = ash_required(s=s_try, bc=hc, Ag=Ag, Ach=Ach,
                                 fc=c.fc, fyt=self.fyt, Pu_over_Ag_fc=Pu_ratio)

        # ---- preliminary directional shear demand (Av/s required) ----
        # We compute it BEFORE the hoop pick so the picker sees both
        # confinement (Ash) and capacity-design shear (Av/s) demands.
        # For Vx: bw = h (perpendicular to X), d = 0.9·b
        # For Vy: bw = b (perpendicular to Y), d = 0.9·h
        d_x_pre = 0.9 * b_out
        d_y_pre = 0.9 * h_out
        Vc_x_pre = vc_simplified(b=h_out, d=d_x_pre, concrete=c)
        Vc_y_pre = vc_simplified(b=b_out, d=d_y_pre, concrete=c)
        av_s_req_x_for_pick: float | None = None
        av_s_req_y_for_pick: float | None = None

        if mode == "demand" and demands is not None:
            av_s_req_x_for_pick = self._av_s_required_for_direction(
                Vu=abs(demands.Vux), Vc=Vc_x_pre, d=d_x_pre,
            )
            av_s_req_y_for_pick = self._av_s_required_for_direction(
                Vu=abs(demands.Vuy), Vc=Vc_y_pre, d=d_y_pre,
            )
        elif mode == "capacity":
            # Use uncapped Ve_capacity here (Ωo·Vu cap is applied later).
            Mpr_x_tmp, _ = mpr_envelope(self.section, angle_deg=0.0, spiral=self.spiral)
            Mpr_y_tmp, _ = mpr_envelope(self.section, angle_deg=90.0, spiral=self.spiral)
            Ve_y_tmp = 2.0 * Mpr_x_tmp / (self.lu / 1000.0)
            Ve_x_tmp = 2.0 * Mpr_y_tmp / (self.lu / 1000.0)
            av_s_req_x_for_pick = self._av_s_required_for_direction(
                Vu=Ve_x_tmp, Vc=Vc_x_pre, d=d_x_pre,
            )
            av_s_req_y_for_pick = self._av_s_required_for_direction(
                Vu=Ve_y_tmp, Vc=Vc_y_pre, d=d_y_pre,
            )

        # ---- pick db_hoop and n_legs (combines Ash + Av/s + hx) ----
        chosen_db, n_x, n_y = self._pick_hoop_and_legs(
            db_hoop=db_hoop, n_legs_x=n_legs_x, n_legs_y=n_legs_y,
            ash_req_x=ash_req_x, ash_req_y=ash_req_y,
            av_s_req_x=av_s_req_x_for_pick, av_s_req_y=av_s_req_y_for_pick,
            bc=bc, hc=hc,
            s=s_try, notes=notes,
        )
        a_per_leg = pi * chosen_db ** 2 / 4.0
        av_prov_x = n_x * a_per_leg
        av_prov_y = n_y * a_per_leg

        if av_prov_x < ash_req_x:
            notes.append(f"n_legs_x={n_x} gives {av_prov_x:.0f} mm² < {ash_req_x:.0f} required.")
        if av_prov_y < ash_req_y:
            notes.append(f"n_legs_y={n_y} gives {av_prov_y:.0f} mm² < {ash_req_y:.0f} required.")

        # ---- hx checks (§18.7.5.2) ----
        hx_x_face, ok_x = hx_check(b_core=bc, n_legs=n_x)
        hx_y_face, ok_y = hx_check(b_core=hc, n_legs=n_y)
        hx_ok = ok_x and ok_y
        if not hx_ok:
            notes.append(
                f"hx > 350 mm on at least one face "
                f"(X face: {hx_x_face:.0f}, Y face: {hx_y_face:.0f})."
            )

        grade = self.section.rebar.groups[0].steel.grade if self.section.rebar.groups else 60
        s_middle = s_max_middle_zone_column(db_long_min=db_long_min, grade=grade)
        lo = lo_length_column(h_member=max(b_out, h_out), lu_clear_mm=self.lu)

        # ---- Mpr in BOTH directions — properties of section + lu (every mode) ----
        Mpr_x, Pu_at_x = mpr_envelope(self.section, angle_deg=0.0, spiral=self.spiral)
        Mpr_y, Pu_at_y = mpr_envelope(self.section, angle_deg=90.0, spiral=self.spiral)

        # Capacity-design shears (§18.7.6.1).  lu is mm, Mpr is kN·m, so
        # we convert lu to metres for Ve [kN].
        lu_m = self.lu / 1000.0
        Ve_y_capacity = 2.0 * Mpr_x / lu_m   # Vy comes from Mx; resisted by legs ∥ X
        Ve_x_capacity = 2.0 * Mpr_y / lu_m   # Vx comes from My; resisted by legs ∥ Y

        # ---- Vc per direction (§22.5.5.1) ----
        phi_v = phi_shear()
        d_x = 0.9 * b_out    # depth in X direction (for Vx)
        d_y = 0.9 * h_out    # depth in Y direction (for Vy)
        Vc_x = vc_simplified(b=h_out, d=d_x, concrete=c)   # bw=h → resists Vx
        Vc_y = vc_simplified(b=b_out, d=d_y, concrete=c)   # bw=b → resists Vy

        # ---- §18.7.6.2.1 — Vc = 0 in lo when light axial + seismic-dominated ----
        # For the column we treat the demand shear as "seismic" when the
        # load case is a capacity-design scenario (Ve_capacity) or when
        # demands are provided; the simplification is V_seismic == V_total
        # in those cases (the worst case). The strict check uses Pu_used.
        vc_zero_x = False
        vc_zero_y = False
        if self.seismic and mode in ("demand", "capacity"):
            V_total_x = (Ve_x_used_pre := (
                Ve_x_capacity if mode == "capacity" else (abs(demands.Vux) if demands else 0.0)
            ))
            V_total_y = (Ve_y_used_pre := (
                Ve_y_capacity if mode == "capacity" else (abs(demands.Vuy) if demands else 0.0)
            ))
            # In the seismic load combo we conservatively take the
            # capacity / demand shear as ~100% seismic (V_seismic == V_total).
            # That captures the §18.7.6.2.1 spirit: if the seismic share is
            # ≥ 50% AND the axial is light, Vc → 0 in lo.
            Pu_for_check = Pu_used if Pu_used is not None else 0.0
            vc_zero_x = vc_zero_seismic(
                V_seismic=V_total_x, V_total=V_total_x,
                Pu=Pu_for_check, Ag=Ag, fc=c.fc,
            )
            vc_zero_y = vc_zero_seismic(
                V_seismic=V_total_y, V_total=V_total_y,
                Pu=Pu_for_check, Ag=Ag, fc=c.fc,
            )
            if vc_zero_x:
                Vc_x = 0.0
                notes.append(
                    "§18.7.6.2.1 — Vc=0 in lo (X): Pu < Ag·fc/20 and seismic-dominated shear."
                )
            if vc_zero_y:
                Vc_y = 0.0
                notes.append(
                    "§18.7.6.2.1 — Vc=0 in lo (Y): Pu < Ag·fc/20 and seismic-dominated shear."
                )

        # Vs provided by the *proposed* hoops, per direction
        Vs_x_prov = vs_capacity(Av=av_prov_y, fyt=self.fyt, d=d_x, s=s_try)  # legs ∥ Y resist Vx
        Vs_y_prov = vs_capacity(Av=av_prov_x, fyt=self.fyt, d=d_y, s=s_try)  # legs ∥ X resist Vy
        phi_Vn_x = phi_v * (Vc_x + Vs_x_prov)
        phi_Vn_y = phi_v * (Vc_y + Vs_y_prov)

        # ---- Ve_used per direction, capped by Ωo·Vu if demands exist ----
        Ve_x_used: float | None = None
        Ve_y_used: float | None = None
        amplif_x: float | None = None
        amplif_y: float | None = None

        if mode == "capacity":
            Ve_x_used = Ve_x_capacity
            Ve_y_used = Ve_y_capacity
            if demands is not None:
                Vu_x = abs(demands.Vux)
                Vu_y = abs(demands.Vuy)
                if Vu_x > 0:
                    cap_x = self.omega_0 * Vu_x
                    if cap_x < Ve_x_used:
                        notes.append(
                            f"Ve_x {Ve_x_used:.0f} kN capped by Ωo·Vux = {cap_x:.0f} kN (§18.7.6.1.1)."
                        )
                        Ve_x_used = cap_x
                    amplif_x = Ve_x_used / Vu_x
                if Vu_y > 0:
                    cap_y = self.omega_0 * Vu_y
                    if cap_y < Ve_y_used:
                        notes.append(
                            f"Ve_y {Ve_y_used:.0f} kN capped by Ωo·Vuy = {cap_y:.0f} kN (§18.7.6.1.1)."
                        )
                        Ve_y_used = cap_y
                    amplif_y = Ve_y_used / Vu_y

        # ---- Av/s required per direction (only for modes that use shear demand) ----
        av_s_req_x: float | None = None
        av_s_req_y: float | None = None
        if mode == "demand" and demands is not None:
            av_s_req_x = self._av_s_required_for_direction(
                Vu=abs(demands.Vux), Vc=Vc_x, d=d_x,
            )
            av_s_req_y = self._av_s_required_for_direction(
                Vu=abs(demands.Vuy), Vc=Vc_y, d=d_y,
            )
        elif mode == "capacity":
            av_s_req_x = self._av_s_required_for_direction(
                Vu=Ve_x_used or 0.0, Vc=Vc_x, d=d_x,
            )
            av_s_req_y = self._av_s_required_for_direction(
                Vu=Ve_y_used or 0.0, Vc=Vc_y, d=d_y,
            )

        # ---- ratio Ve_capacity / φVn per direction (uncapped Ve) ----
        ratio_Ve_phiVn_x = Ve_x_capacity / phi_Vn_x if phi_Vn_x > 0 else None
        ratio_Ve_phiVn_y = Ve_y_capacity / phi_Vn_y if phi_Vn_y > 0 else None

        return ColumnDesignProposal(
            mode=mode,
            ash_required_x=ash_req_x, ash_required_y=ash_req_y,
            db_hoop=chosen_db, n_legs_x=n_x, n_legs_y=n_y,
            av_provided_x=av_prov_x, av_provided_y=av_prov_y,
            spacing_confined=s_try, spacing_middle=s_middle, lo=lo,
            hx_x_face=hx_x_face, hx_y_face=hx_y_face, hx_ok=hx_ok,
            Pu_used=Pu_used, Pu_over_Ag_fc=Pu_ratio,
            eq_iii_active=(Pu_ratio > 0.3),
            Mpr_x=Mpr_x, Mpr_y=Mpr_y,
            Pu_for_Mpr_x=Pu_at_x, Pu_for_Mpr_y=Pu_at_y,
            Ve_x_capacity=Ve_x_capacity,
            Ve_y_capacity=Ve_y_capacity,
            Ve_x_used=Ve_x_used,
            Ve_y_used=Ve_y_used,
            amplification_x=amplif_x,
            amplification_y=amplif_y,
            phi_Vn_x=phi_Vn_x, phi_Vn_y=phi_Vn_y,
            av_s_required_x=av_s_req_x,
            av_s_required_y=av_s_req_y,
            ratio_Ve_over_phiVn_x=ratio_Ve_phiVn_x,
            ratio_Ve_over_phiVn_y=ratio_Ve_phiVn_y,
            vc_zero_x=vc_zero_x,
            vc_zero_y=vc_zero_y,
            notes=tuple(notes),
        )

    # ---- internal: envelope of three modes ----
    @staticmethod
    def _design_envelope(prop_min, prop_dem, prop_cap) -> ColumnDesignProposal:
        modes = [prop_min] + ([prop_dem] if prop_dem else []) + [prop_cap]
        # Pick the mode whose proposed detailing is the most demanding —
        # use total transverse steel "density" (db² × legs / spacing) as
        # the discriminator. This guarantees that when two modes show
        # equal Ash_required (eq iii dormant) the one that *also* needed
        # extra legs for capacity shear wins.
        def steel_index(p):
            return p.db_hoop ** 2 * (p.n_legs_x + p.n_legs_y) / max(p.spacing_confined, 1e-9)
        ash_worst = max(modes, key=steel_index)
        all_notes = tuple(f"[{p.mode}] {n}" for p in modes for n in p.notes)
        return ColumnDesignProposal(
            mode="envelope",
            ash_required_x=ash_worst.ash_required_x,
            ash_required_y=ash_worst.ash_required_y,
            db_hoop=ash_worst.db_hoop,
            n_legs_x=ash_worst.n_legs_x,
            n_legs_y=ash_worst.n_legs_y,
            av_provided_x=ash_worst.av_provided_x,
            av_provided_y=ash_worst.av_provided_y,
            spacing_confined=ash_worst.spacing_confined,
            spacing_middle=ash_worst.spacing_middle,
            lo=ash_worst.lo,
            hx_x_face=ash_worst.hx_x_face,
            hx_y_face=ash_worst.hx_y_face,
            hx_ok=all(p.hx_ok for p in modes),
            Pu_used=ash_worst.Pu_used,
            Pu_over_Ag_fc=ash_worst.Pu_over_Ag_fc,
            eq_iii_active=any(p.eq_iii_active for p in modes),
            Mpr_x=prop_cap.Mpr_x,
            Mpr_y=prop_cap.Mpr_y,
            Pu_for_Mpr_x=prop_cap.Pu_for_Mpr_x,
            Pu_for_Mpr_y=prop_cap.Pu_for_Mpr_y,
            Ve_x_capacity=prop_cap.Ve_x_capacity,
            Ve_y_capacity=prop_cap.Ve_y_capacity,
            Ve_x_used=prop_cap.Ve_x_used,
            Ve_y_used=prop_cap.Ve_y_used,
            amplification_x=prop_cap.amplification_x,
            amplification_y=prop_cap.amplification_y,
            phi_Vn_x=ash_worst.phi_Vn_x,
            phi_Vn_y=ash_worst.phi_Vn_y,
            av_s_required_x=max((p.av_s_required_x or 0.0) for p in modes) or None,
            av_s_required_y=max((p.av_s_required_y or 0.0) for p in modes) or None,
            ratio_Ve_over_phiVn_x=ash_worst.ratio_Ve_over_phiVn_x,
            ratio_Ve_over_phiVn_y=ash_worst.ratio_Ve_over_phiVn_y,
            vc_zero_x=any(p.vc_zero_x for p in modes),
            vc_zero_y=any(p.vc_zero_y for p in modes),
            notes=all_notes,
        )

    # ---- internal: Av/s required given a Vu, Vc, d in ONE direction (§22.5) ----
    def _av_s_required_for_direction(self, *, Vu: float, Vc: float, d: float
                                     ) -> float | None:
        """Av/s in mm²/mm required to resist Vu given Vc and d.

        Thin wrapper around :func:`design.common.shear.av_s_required` —
        passes ``self.fyt`` and the default ``phi_shear()`` factor (§21.2.1).

        Vu and Vc in kN, d in mm. Returns:
            - None if Vu <= 0
            - 0.0  if Vc alone (with phi) covers Vu
            - Av/s [mm²/mm] otherwise
        """
        return av_s_required(Vu=Vu, Vc=Vc, fyt=self.fyt, d=d)

    # Back-compat: scalar version (peraltes = altura, conservador)
    def _av_s_required_for(self, Vu: float) -> float | None:
        d = 0.9 * self.section.height()
        b = self.section.width()
        Vc = vc_simplified(b=b, d=d, concrete=self.section.concrete)
        return self._av_s_required_for_direction(Vu=Vu, Vc=Vc, d=d)

    # ---- internal: count longitudinal bars per face ----
    def _count_longitudinal_per_face(
        self, *, layer_tol: float = 5.0,
    ) -> tuple[int, int]:
        """Return (max_legs_x, max_legs_y) — the physical upper bound on
        the number of hoop legs in each direction.

        We pick the extreme layer of bars (top-most, bottom-most,
        left-most, right-most) and count how many bars sit on it
        (within `layer_tol` mm). Bars are placed a cover-distance from
        the outer face, not on it — so we operate on actual bar
        coordinates, not on the bounding box.

        Legs ∥ X span left↔right faces → bounded by bars in the
        extreme vertical layers.
        Legs ∥ Y span top↔bottom faces → bounded by bars in the
        extreme horizontal layers.
        """
        xs: list[float] = []
        ys: list[float] = []
        for r, _ in self.section.rebar.iter_bars():
            xs.append(r.x)
            ys.append(r.y)
        if not xs:
            return 2, 2

        y_top = max(ys); y_bot = min(ys)
        x_left = min(xs); x_right = max(xs)

        n_top   = sum(1 for y in ys if abs(y - y_top)   <= layer_tol)
        n_bot   = sum(1 for y in ys if abs(y - y_bot)   <= layer_tol)
        n_left  = sum(1 for x in xs if abs(x - x_left)  <= layer_tol)
        n_right = sum(1 for x in xs if abs(x - x_right) <= layer_tol)

        max_legs_x = max(2, min(n_left, n_right)) if (n_left and n_right) else 2
        max_legs_y = max(2, min(n_top, n_bot)) if (n_top and n_bot) else 2
        return max_legs_x, max_legs_y

    # ---- internal: hoop + leg picker driven by bar_schedule.hoops ----
    def _pick_hoop_and_legs(
        self,
        *,
        db_hoop: float | None,
        n_legs_x: int | None,
        n_legs_y: int | None,
        ash_req_x: float,
        ash_req_y: float,
        av_s_req_x: float | None,
        av_s_req_y: float | None,
        bc: float,
        hc: float,
        s: float,
        notes: list[str],
    ) -> tuple[float, int, int]:
        """For each diameter, picks the minimum n_legs that satisfies
        confinement (§18.7.5.4 Ash) AND capacity-design shear (§22.5 Av/s)
        AND §18.7.5.2 (hx ≤ 350 mm), respecting the physical bar-count limit.
        """
        max_x, max_y = self._count_longitudinal_per_face()

        def n_required_for(db: float) -> tuple[int, int]:
            """Minimum legs for (X, Y) by Ash + capacity shear, both gates."""
            a_per_leg = pi * db ** 2 / 4.0
            # Confinement (§18.7.5.4)
            n_x_conf = max(2, ceil(ash_req_x / a_per_leg))
            n_y_conf = max(2, ceil(ash_req_y / a_per_leg))
            # Shear capacity: legs ∥ X resist Vy → av_s_req_y;
            #                 legs ∥ Y resist Vx → av_s_req_x
            n_x_shear = (max(2, ceil((av_s_req_y * s) / a_per_leg))
                         if av_s_req_y and av_s_req_y > 0 else 2)
            n_y_shear = (max(2, ceil((av_s_req_x * s) / a_per_leg))
                         if av_s_req_x and av_s_req_x > 0 else 2)
            return max(n_x_conf, n_x_shear), max(n_y_conf, n_y_shear)

        def bump_to_hx(n: int, core_dim: float, n_max: int) -> int:
            """Increase n until hx ≤ 350 or hit physical limit."""
            while n <= n_max:
                _, ok = hx_check(b_core=core_dim, n_legs=n)
                if ok:
                    return n
                n += 1
            return n  # > n_max → infeasible at this diameter

        if db_hoop is not None:
            chosen_db = db_hoop
            n_x_req, n_y_req = n_required_for(chosen_db)
            n_x = n_legs_x if n_legs_x is not None else n_x_req
            n_y = n_legs_y if n_legs_y is not None else n_y_req
            if n_legs_x is None:
                n_x = bump_to_hx(n_x, bc, max_x)
                if n_x > max_x:
                    notes.append(
                        f"With φ{chosen_db:.0f}, no n_legs_x ≤ {max_x} cleared hx≤350 mm."
                    )
                    n_x = min(n_x_req, max_x)
            if n_legs_y is None:
                n_y = bump_to_hx(n_y, hc, max_y)
                if n_y > max_y:
                    notes.append(
                        f"With φ{chosen_db:.0f}, no n_legs_y ≤ {max_y} cleared hx≤350 mm."
                    )
                    n_y = min(n_y_req, max_y)
            return chosen_db, n_x, n_y

        # Auto-pick: smallest db that, after bumping legs to satisfy hx,
        # still fits within the physical limit.
        for db in sorted(self.bar_schedule.hoops):
            n_x_req, n_y_req = n_required_for(db)
            n_x = n_legs_x if n_legs_x is not None else bump_to_hx(n_x_req, bc, max_x)
            n_y = n_legs_y if n_legs_y is not None else bump_to_hx(n_y_req, hc, max_y)
            if n_x > max_x or n_y > max_y:
                continue
            return db, n_x, n_y

        # Fallback: the largest hoop available, capped to physical limit.
        db = max(self.bar_schedule.hoops)
        n_x_req, n_y_req = n_required_for(db)
        n_x = n_legs_x if n_legs_x is not None else min(n_x_req, max_x)
        n_y = n_legs_y if n_legs_y is not None else min(n_y_req, max_y)
        notes.append(
            f"No hoop in bar_schedule satisfied both physical-bar limit "
            f"(max_x={max_x}, max_y={max_y}) and hx≤350; used max φ{db:.0f}."
        )
        return db, n_x, n_y

    # =============================================================== #
    # Reports
    # =============================================================== #
    def summary(self) -> None:
        """Print a brief summary using current presentation units."""
        u = self.units
        s = self.section
        rebar = s.rebar
        n_bars = sum(g.n for g in rebar.groups)
        Ast = rebar.total_area
        Ag = s.gross_area()
        rho = Ast / Ag * 100 if Ag > 0 else 0.0

        print(f"=== {self.label} ===  (units: {u.name})")
        print(f"Section : {s.width()*u.length_factor:.0f}×{s.height()*u.length_factor:.0f} {u.length}")
        print(f"Materials: fc = {s.concrete.fc:.0f} MPa, fy = {rebar.groups[0].steel.fy:.0f} MPa")
        print(f"Rebar   : {n_bars} bars, As = {Ast*u.area_factor:.1f} {u.length}², ρ = {rho:.2f}%")
        if self._surface is None:
            print("State   : not run yet — call col.run() to compute the PMM surface.")
            return
        print(f"Po      : {self.Po*u.force_factor:.0f} {u.force}")
        print(f"To      : {self.To*u.force_factor:.0f} {u.force}")
        print(f"φPn,max : {self.phi_Pn_max*u.force_factor:.0f} {u.force}")
        print(f"φVn     : {self.phi_Vn*u.force_factor:.0f} {u.force}")
        print(f"max φMn @ θ=0°  : {max(self.surface[0.0].phi_Mn)*u.moment_factor:.1f} {u.force}·{u.length}")
        print(f"max φMn @ θ=90° : {max(self.surface[90.0].phi_Mn)*u.moment_factor:.1f} {u.force}·{u.length}")

    def report(self) -> dict:
        """Return a structured dict with current state (display units applied)."""
        self._ensure_run()
        u = self.units
        s = self.section
        return {
            "label": self.label,
            "units": u.name,
            "section": {
                "b": s.width() * u.length_factor,
                "h": s.height() * u.length_factor,
                "Ag": s.gross_area() * u.area_factor,
            },
            "materials": {
                "fc_MPa": s.concrete.fc,
                "fy_MPa": s.rebar.groups[0].steel.fy if s.rebar.groups else None,
            },
            "rebar": {
                "n_bars": sum(g.n for g in s.rebar.groups),
                "As": s.rebar.total_area * u.area_factor,
                "rho": s.rebar.total_area / s.gross_area() if s.gross_area() > 0 else 0.0,
            },
            "capacity": {
                "Po": self.Po * u.force_factor,
                "To": self.To * u.force_factor,
                "phi_Pn_max": self.phi_Pn_max * u.force_factor,
                "phi_Vn": self.phi_Vn * u.force_factor,
                "max_phi_Mn_x": float(max(self.surface[0.0].phi_Mn)) * u.moment_factor,
                "max_phi_Mn_y": float(max(self.surface[90.0].phi_Mn)) * u.moment_factor,
            },
            "params": {
                "lu_mm": self.lu,
                "lu_m": self.lu_m,
                "k": self.k,
                "spiral": self.spiral,
                "seismic": self.seismic,
                "hx_mm": self.hx,
                "fyt_MPa": self.fyt,
                "s_mm": self.s,
            },
        }

    def __repr__(self) -> str:
        s = self.section
        return (
            f"Column(label={self.label!r}, section={s.width():.0f}×{s.height():.0f} mm, "
            f"fc={s.concrete.fc:.0f} MPa, units={self.units.name}, "
            f"run={'yes' if self._surface is not None else 'no'})"
        )
