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

from design.common.factors import phi_shear, OMEGA_0_COLUMN_DEFAULT
from design.common.materials import Bar, Concrete, Steel
from design.common.units import UnitSystem, units_from, to_internal, DEFAULT_UNITS
from design.common.element_base import ElementBase
from design.common.shear import (
    vc_one_way, vs_capacity, av_s_required, av_s_minimum, vc_zero_seismic,
)
from design.common.spacing import (
    ash_required, s_max_confined_column, lo_length_column,
    s_max_middle_zone_column, hx_check, rho_s_spiral_required,
)
from design.sections.base import Section
from design.sections.rectangular import RectangularSection
from design.sections.reinforcement import RebarLayout, perimeter_bars
from design.common.bar_schedule import BarSchedule
from design.columns.interaction import InteractionDiagram, interaction_diagram
from design.columns.interaction_surface import (
    InteractionSurface, interaction_surface,
)
from design.columns.plotting import _ColumnPlotter
from design.common.probable import mpr_envelope
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
    """Full check of the column against demands.

    ``passed`` is True only if PMM + shear + confinement + hx + s_max all pass.
    """
    capacity: ColumnCapacity
    demands: ColumnDemands
    # Top-level summary
    ratio_pmm: float
    ratio_shear: float           # max of all shear ratios (demand & capacity)
    passed: bool
    # Shear — directional, demand vs capacity
    phi_Vn_x: float = 0.0
    phi_Vn_y: float = 0.0
    ratio_shear_x: float = 0.0   # max(Vu_x, Ve_x_cap) / phi_Vn_x
    ratio_shear_y: float = 0.0
    ratio_Vu_x: float = 0.0      # |Vux| / phi_Vn_x
    ratio_Vu_y: float = 0.0
    ratio_Ve_x: float = 0.0      # Ve_x_capacity / phi_Vn_x
    ratio_Ve_y: float = 0.0
    Ve_x_capacity: float = 0.0
    Ve_y_capacity: float = 0.0
    Mpr_x: float = 0.0
    Mpr_y: float = 0.0
    # Confinement (Ash §18.7.5.4) — directional
    Av_x_provided: float = 0.0
    Av_y_provided: float = 0.0
    Ash_x_demand: float = 0.0    # with demands.Pu in eq.iii
    Ash_y_demand: float = 0.0
    Ash_x_capacity: float = 0.0  # with Pu_for_Mpr in eq.iii
    Ash_y_capacity: float = 0.0
    ratio_ash_x: float = 0.0     # max(demand, capacity) / Av_provided
    ratio_ash_y: float = 0.0
    ash_ok: bool = True
    # Geometry — hx ≤ 350 (§18.7.5.2)
    hx_x_face: float = 0.0
    hx_y_face: float = 0.0
    hx_ok: bool = True
    # Spacing — §18.7.5.3 confined  + §18.7.5.5 middle
    s_provided: float = 0.0
    s_max_confined: float = 0.0
    s_max_middle: float = 0.0
    s_confined_ok: bool = True
    # Steel quantities (kg/m³ of concrete) — populated by check()
    rho_long: float = 0.0
    rho_trans: float = 0.0
    rho_total: float = 0.0
    # Diagnostic notes — one per failed check
    notes: tuple[str, ...] = ()

    def summary(self, units: "UnitSystem | None" = None) -> None:
        """Print a structured report of the check.

        ``units`` (optional, a :class:`UnitSystem`) drives the presentation
        of lengths (hx, s) and areas (Ash). The ``Column`` threads
        ``self.units`` so the check honors ``set_units(...)``. When omitted,
        falls back to the package default (mm / mm^2).
        """
        u = units if units is not None else DEFAULT_UNITS
        tag = lambda b: "PASS" if b else "FAIL"
        print(f"=== Column check ===")
        print(f"  PMM            : ratio = {self.ratio_pmm:.3f}    [{tag(self.ratio_pmm <= 1.0)}]")
        print(f"  Shear X        : ratio = {self.ratio_shear_x:.3f}    "
              f"(Vu={self.ratio_Vu_x:.3f}, Ve={self.ratio_Ve_x:.3f})    "
              f"[{tag(self.ratio_shear_x <= 1.0)}]")
        print(f"  Shear Y        : ratio = {self.ratio_shear_y:.3f}    "
              f"(Vu={self.ratio_Vu_y:.3f}, Ve={self.ratio_Ve_y:.3f})    "
              f"[{tag(self.ratio_shear_y <= 1.0)}]")
        print(f"  Ash X (short)  : ratio = {self.ratio_ash_x:.3f}    "
              f"prov={u.fmt_area(self.Av_x_provided, unit=False)} / "
              f"req(D={u.fmt_area(self.Ash_x_demand, unit=False)}, "
              f"C={u.fmt_area(self.Ash_x_capacity)})    "
              f"[{tag(self.ratio_ash_x <= 1.0)}]")
        print(f"  Ash Y (long)   : ratio = {self.ratio_ash_y:.3f}    "
              f"prov={u.fmt_area(self.Av_y_provided, unit=False)} / "
              f"req(D={u.fmt_area(self.Ash_y_demand, unit=False)}, "
              f"C={u.fmt_area(self.Ash_y_capacity)})    "
              f"[{tag(self.ratio_ash_y <= 1.0)}]")
        print(f"  hx             : x={u.fmt_length(self.hx_x_face, unit=False)}, "
              f"y={u.fmt_length(self.hx_y_face)}    "
              f"[{tag(self.hx_ok)}]")
        print(f"  s (confined)   : prov={u.fmt_length(self.s_provided, unit=False)}, "
              f"max={u.fmt_length(self.s_max_confined)}    "
              f"[{tag(self.s_confined_ok)}]")
        print(f"  ---")
        print(f"  passed         : {self.passed}")
        if self.rho_total > 0:
            print(f"  ---")
            print(f"  Steel quantities (kg/m^3 of concrete):")
            print(f"    rho_long  = {self.rho_long :7.2f}")
            print(f"    rho_trans = {self.rho_trans:7.2f}")
            print(f"    rho_total = {self.rho_total:7.2f}")
        if self.notes:
            print(f"  Notes:")
            for n in self.notes:
                print(f"    - {n}")


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
    # Spiral confinement — volumetric ratio (§25.7.3.3 / Table 18.7.5.4 d/e/f).
    # Only populated when the column is spiral; None for tied columns.
    rho_s_required: float | None = None
    rho_s_provided: float | None = None
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
    # Demand ratios (|Vu|/phi_Vn) — only meaningful in mode='demand'
    ratio_Vu_over_phiVn_x: float | None = None
    ratio_Vu_over_phiVn_y: float | None = None
    # PMM check against the surface (only meaningful when demands given)
    ratio_pmm: float | None = None
    # §18.7.6.2.1 — Vc=0 in lo (light axial + seismic-dominated)
    vc_zero_x: bool = False
    vc_zero_y: bool = False
    # ---- Middle zone (§18.7.5.5 — only §22.5 shear applies) ----
    middle_db_hoop: float | None = None
    middle_n_legs_x: int | None = None
    middle_n_legs_y: int | None = None
    middle_spacing: float | None = None
    middle_av_provided_x: float | None = None
    middle_av_provided_y: float | None = None
    middle_av_s_required_x: float | None = None
    middle_av_s_required_y: float | None = None
    middle_phi_Vn_x: float | None = None
    middle_phi_Vn_y: float | None = None
    middle_ratio_Ve_over_phiVn_x: float | None = None
    middle_ratio_Ve_over_phiVn_y: float | None = None
    # Picker trace — diameter-by-diameter decision for each zone
    picker_trace_confined: tuple[str, ...] = ()
    picker_trace_middle: tuple[str, ...] = ()
    # Diagnostics
    notes: tuple[str, ...] = ()

    # ---- Legacy aliases (back-compat) ----
    # NOTE: a column proposal has no top/bot distinction — it has x/y axes.
    # The old ``Mpr_top``/``Mpr_bot`` aliases both returned ``Mpr_x`` (a bug);
    # use ``Mpr_x`` / ``Mpr_y`` explicitly instead. The dead aliases
    # (``amplification_vs_demand``, ``Vu_used``, ``av_s_provided``) were removed.
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
    def av_s_required(self) -> float | None:
        x, y = self.av_s_required_x, self.av_s_required_y
        if x is None and y is None:
            return None
        return max(x or 0.0, y or 0.0)

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
class Column(ElementBase):
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
        Av: float = 0.0,                 # legacy scalar — kept for back-compat
        Av_x: float | None = None,       # transverse area of legs ∥ X (short legs)
        Av_y: float | None = None,       # transverse area of legs ∥ Y (long legs)
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
        # The bare constructor is INTERNAL-only: lu/hx/etc. are mm, fyt is
        # MPa, Av is mm². Unit-aware input goes through Column.rectangular().
        self.section = section
        self.lu = lu                     # mm
        self.k = k
        self.spiral = spiral
        # Directional transverse areas (mm²) — Av_x = legs ∥ X (short),
        # Av_y = legs ∥ Y (long). If only the legacy scalar ``Av`` is given,
        # both directions default to it.
        if Av_x is None and Av_y is None:
            self.Av_x = Av
            self.Av_y = Av
        else:
            self.Av_x = Av_x if Av_x is not None else Av
            self.Av_y = Av_y if Av_y is not None else Av
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
        self._phi_Vn_x: Optional[float] = None
        self._phi_Vn_y: Optional[float] = None

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

        All length inputs (``b, h, cover, db_long, db_stirrup, s_stirrup,
        lu, hx, h_min``) are canonical **mm** — coherent with the common/
        equations. ``units`` is PRESENTATION ONLY: it sets how
        summary()/report()/plots DISPLAY results and how ``demands()`` reads
        forces/moments; it NEVER reinterprets the geometry. Change it any time
        with ``col.set_units(...)``. Material strengths (``concrete.fc``,
        ``steel.fy``) are always MPa.

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

        # ---- `units` is PRESENTATION ONLY ----
        # Every scalar input is canonical (mm, MPa, kN, kN·m), coherent with the
        # common/ equations. `units` never reinterprets the geometry — it only
        # governs how summary()/report()/plots DISPLAY results and how demands()
        # reads forces/moments. Change it any time with col.set_units(...).
        u = (units if isinstance(units, UnitSystem)
             else units_from(units) if units is not None
             else DEFAULT_UNITS)
        # b, h, clear_cover, db_long, db_stirrup, s_stirrup, lu, hx, h_min are
        # all canonical mm — no unit reinterpretation here.

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
            clear_cover=clear_cover,
        )

        # Convert (n_legs_x, n_legs_y, db_stirrup, s) into the scalar Av
        # Directional Av — Av_x = legs ∥ X (short), Av_y = legs ∥ Y (long).
        a_per_leg = pi * db_stirrup ** 2 / 4.0
        Av_x_val = (n_legs_x * a_per_leg) if n_legs_x is not None else 0.0
        Av_y_val = (n_legs_y * a_per_leg) if n_legs_y is not None else 0.0

        fyt_val = (steel_transverse or steel).fy

        # All scalars are now internal (mm/mm²); build directly.
        col = cls.__new__(cls)
        Column.__init__(
            col,
            section=section,
            lu=lu, k=k, spiral=spiral,
            Av_x=Av_x_val, Av_y=Av_y_val,
            fyt=fyt_val, s=s_stirrup,
            seismic=seismic, hx=hx, h_min=h_min,
            transverse_bar_diameter=db_stirrup,
            units=u, bar_schedule=bar_schedule, label=label,
            run_settings=run_settings,
        )
        col.n_legs_x = n_legs_x
        col.n_legs_y = n_legs_y
        return col

    # =============================================================== #
    # Configuration helpers
    # =============================================================== #
    def demands(
        self,
        *,
        Pu: float = 0.0,
        Mux: float = 0.0,
        Muy: float = 0.0,
        Vux: float = 0.0,
        Vuy: float = 0.0,
        units: int | str | UnitSystem | None = None,
    ) -> "ColumnDemands":
        """Build a :class:`ColumnDemands` from values in display units.

        Forces (``Pu, Vux, Vuy``) and moments (``Mux, Muy``) are read in
        ``units`` (defaults to this column's ``self.units``) and converted to
        the internal kN / kN·m. Pass ``units=`` to override for one call.
        """
        u = (units if isinstance(units, UnitSystem)
             else units_from(units) if units is not None
             else self.units)
        return ColumnDemands(
            Pu=to_internal("force", Pu, units=u),
            Mux=to_internal("moment", Mux, units=u),
            Muy=to_internal("moment", Muy, units=u),
            Vux=to_internal("force", Vux, units=u),
            Vuy=to_internal("force", Vuy, units=u),
        )

    def clear_cache(self) -> None:
        self._surface = None
        self._inner_surface = None
        self._Po = self._To = self._phi_Pn_max = self._phi_Vn = None
        self._phi_Vn_x = self._phi_Vn_y = None

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
        new_Av_x = proposal.n_legs_x * a_per_leg     # legs ∥ X (short)
        new_Av_y = proposal.n_legs_y * a_per_leg     # legs ∥ Y (long)
        new = Column(
            section=self.section,
            lu=self.lu,                  # already mm
            k=self.k,
            spiral=self.spiral,
            Av_x=new_Av_x,
            Av_y=new_Av_y,
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

        # Shear capacity — directional  (§22.5.5.1 + §22.5.10)
        # Convention:
        #   Vx (corte en X) → bw = h, d ≈ 0.9·b,  resisted by legs ∥ Y (Av_y)
        #   Vy (corte en Y) → bw = b, d ≈ 0.9·h,  resisted by legs ∥ X (Av_x)
        b_out = self.section.width()
        h_out = self.section.height()
        d_x = 0.9 * b_out
        d_y = 0.9 * h_out
        c   = self.section.concrete
        phi = phi_shear()

        Vc_x = self._vc("x")
        Vc_y = self._vc("y")
        Vs_x = (vs_capacity(Av=self.Av_y, fyt=self.fyt, d=d_x, s=self.s)
                if (self.Av_y and self.s and self.s > 0) else 0.0)
        Vs_y = (vs_capacity(Av=self.Av_x, fyt=self.fyt, d=d_y, s=self.s)
                if (self.Av_x and self.s and self.s > 0) else 0.0)
        self._phi_Vn_x = phi * (Vc_x + Vs_x)
        self._phi_Vn_y = phi * (Vc_y + Vs_y)
        # Legacy scalar = the smaller (conservative)
        self._phi_Vn = min(self._phi_Vn_x, self._phi_Vn_y)

        return self

    def _vc(
        self,
        direction: str,
        *,
        Av: float | None = None,
        s: float | None = None,
    ) -> float:
        """Table 22.5.5.1 ``Vc`` for one column shear direction.

        The tension face is the bars with the largest coordinate along the
        shear direction.  ``Av`` is the area of legs resisting that shear;
        callers sizing a proposal pass its provided transverse area and
        spacing explicitly.
        """
        b_out = self.section.width()
        h_out = self.section.height()
        if direction == "x":
            bw, d, coordinate = h_out, 0.9 * b_out, "x"
            Av_used = self.Av_y if Av is None else Av
        elif direction == "y":
            bw, d, coordinate = b_out, 0.9 * h_out, "y"
            Av_used = self.Av_x if Av is None else Av
        else:
            raise ValueError(f"direction must be 'x' or 'y', got {direction!r}")

        bars = list(self.section.rebar.iter_bars())
        if bars:
            face_coordinate = max(getattr(bar, coordinate) for bar, _ in bars)
            As_tension = sum(
                bar.area for bar, _ in bars
                if abs(getattr(bar, coordinate) - face_coordinate) <= 1.0e-9
            )
        else:
            As_tension = 0.0
        rho_w = As_tension / (bw * d) if bw > 0.0 and d > 0.0 else 0.0
        s_used = self.s if s is None else s
        av_ge_av_min = bool(
            Av_used > 0.0 and s_used is not None and s_used > 0.0
            and Av_used / s_used >= av_s_minimum(
                b=bw, concrete=self.section.concrete, fyt=self.fyt,
            )
        )
        return vc_one_way(
            b=bw, d=d, rho_w=rho_w, concrete=self.section.concrete,
            av_ge_av_min=av_ge_av_min, Nu=0.0,
            Ag=self.section.gross_area(),
        )

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
        """Legacy scalar = min(phi_Vn_x, phi_Vn_y).  Prefer the directional
        properties ``phi_Vn_x`` / ``phi_Vn_y`` for asymmetric layouts."""
        self._ensure_run()
        return self._phi_Vn  # type: ignore[return-value]

    @property
    def phi_Vn_x(self) -> float:
        """phi*Vn in direction X — resisted by legs ∥ Y (Av_y)."""
        self._ensure_run()
        return self._phi_Vn_x  # type: ignore[return-value]

    @property
    def phi_Vn_y(self) -> float:
        """phi*Vn in direction Y — resisted by legs ∥ X (Av_x)."""
        self._ensure_run()
        return self._phi_Vn_y  # type: ignore[return-value]

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
    def check(self, demands: ColumnDemands,
              *, clear_cover: float | None = None) -> ColumnCheck:
        """Full code check — PMM + shear + confinement + hx + s_max.

        ``passed`` is True only if EVERY gate passes:
          PMM ≤ 1, shear ≤ 1 (using max of |Vu| and Ve_capacity), Ash ≥ req
          for BOTH demand-Pu and capacity-Pu scenarios, hx ≤ 350 (§18.7.5.2),
          s ≤ s_max in lo (§18.7.5.3).

        ``clear_cover`` is the cover to the OUTSIDE of the hoop (§18.7.5.4
        table), used to compute bc and Ach. If omitted, falls back to
        ``self.section.clear_cover``.
        """
        if clear_cover is None:
            clear_cover = getattr(self.section, "clear_cover", 25.0)
        self._ensure_run()
        notes: list[str] = []
        c   = self.section.concrete
        fyt = self.fyt
        phi_v = phi_shear()

        # ---- 1) PMM ----
        passed_pmm, ratio_pmm = self.surface.check(
            Pu=demands.Pu, Mux=demands.Mux, Muy=demands.Muy,
        )
        if not passed_pmm:
            notes.append(f"PMM ratio = {ratio_pmm:.3f} > 1 (Pu+Mux+Muy outside surface).")

        # ---- 2) Geometry common to several gates ----
        b_out = self.section.width()
        h_out = self.section.height()
        Ag    = b_out * h_out
        bc_short = max(b_out - 2 * clear_cover, 100.0)
        bc_long  = max(h_out - 2 * clear_cover, 100.0)
        Ach = bc_short * bc_long

        # ---- 3) Mpr + Ve capacity (§18.7.6.1) ----
        Mpr_x, Pu_at_x = mpr_envelope(self.section, angle_deg=0.0, spiral=self.spiral)
        Mpr_y, Pu_at_y = mpr_envelope(self.section, angle_deg=90.0, spiral=self.spiral)
        lu_m = self.lu / 1000.0
        Ve_x_cap = 2.0 * Mpr_y / lu_m
        Ve_y_cap = 2.0 * Mpr_x / lu_m

        # ---- 4) Shear — Vu vs Ve, directional ----
        phi_Vn_x = self._phi_Vn_x or 0.0
        phi_Vn_y = self._phi_Vn_y or 0.0
        ratio_Vu_x = abs(demands.Vux) / phi_Vn_x if phi_Vn_x > 0 else float("inf")
        ratio_Vu_y = abs(demands.Vuy) / phi_Vn_y if phi_Vn_y > 0 else float("inf")
        ratio_Ve_x = Ve_x_cap / phi_Vn_x if phi_Vn_x > 0 else float("inf")
        ratio_Ve_y = Ve_y_cap / phi_Vn_y if phi_Vn_y > 0 else float("inf")
        ratio_shear_x = max(ratio_Vu_x, ratio_Ve_x)
        ratio_shear_y = max(ratio_Vu_y, ratio_Ve_y)
        ratio_shear   = max(ratio_shear_x, ratio_shear_y)
        if ratio_shear_x > 1.0:
            notes.append(f"Shear X: ratio = {ratio_shear_x:.3f} > 1.")
        if ratio_shear_y > 1.0:
            notes.append(f"Shear Y: ratio = {ratio_shear_y:.3f} > 1.")

        # ---- 5) Confinement Ash (§18.7.5.4) — eval in two Pu scenarios ----
        s_prov = self.s or 100.0
        nl = self._count_perimeter_bars()   # kn = nl/(nl-2), Eq. 18.7.5.4b
        def _ash_for(bc, Pu_kN):
            Pu_ratio = max(0.0, Pu_kN) * 1000.0 / (Ag * c.fc) if Ag * c.fc > 0 else 0.0
            return ash_required(s=s_prov, bc=bc, Ag=Ag, Ach=Ach,
                                fc=c.fc, fyt=fyt, Pu_over_Ag_fc=Pu_ratio, nl=nl)
        Pu_capacity = max(Pu_at_x, Pu_at_y)
        Ash_x_demand   = _ash_for(bc_short, demands.Pu)
        Ash_y_demand   = _ash_for(bc_long,  demands.Pu)
        Ash_x_capacity = _ash_for(bc_short, Pu_capacity)
        Ash_y_capacity = _ash_for(bc_long,  Pu_capacity)
        Av_x_prov = self.Av_x or 0.0
        Av_y_prov = self.Av_y or 0.0
        Ash_x_worst = max(Ash_x_demand, Ash_x_capacity)
        Ash_y_worst = max(Ash_y_demand, Ash_y_capacity)
        ratio_ash_x = Ash_x_worst / Av_x_prov if Av_x_prov > 0 else float("inf")
        ratio_ash_y = Ash_y_worst / Av_y_prov if Av_y_prov > 0 else float("inf")
        ash_ok = (ratio_ash_x <= 1.0) and (ratio_ash_y <= 1.0)
        if not ash_ok:
            notes.append(
                f"Ash insufficient -- X: prov {Av_x_prov:.0f} / req {Ash_x_worst:.0f}, "
                f"Y: prov {Av_y_prov:.0f} / req {Ash_y_worst:.0f} mm^2."
            )

        # ---- 6) hx ≤ 350 (§18.7.5.2) ----
        n_x = self.n_legs_x or 2
        n_y = self.n_legs_y or 2
        hx_x_face, hx_x_ok = hx_check(b_core=bc_short, n_legs=n_x)
        hx_y_face, hx_y_ok = hx_check(b_core=bc_long,  n_legs=n_y)
        hx_ok = hx_x_ok and hx_y_ok
        if not hx_ok:
            notes.append(
                f"hx > 350 mm -- X face: {hx_x_face:.0f}, Y face: {hx_y_face:.0f}."
            )

        # ---- 7) s ≤ s_max in lo (§18.7.5.3) and middle (§18.7.5.5) ----
        # Pick a representative db_long from the section (smallest bar).
        db_long_min = 20.0
        try:
            diams = [2.0 * (r.area / pi) ** 0.5
                     for g in self.section.rebar.groups for r in g.bars]
            if diams:
                db_long_min = min(diams)
        except Exception:
            pass
        grade = self.section.rebar.groups[0].steel.grade if self.section.rebar.groups else 60
        # Use the EFFECTIVE hx from the real layout (max of the two faces),
        # not the constructor default, for so in §18.7.5.3.
        hx_eff = max(hx_x_face, hx_y_face)
        s_max_confined = s_max_confined_column(
            h_min=self.h_min, db_long_min=db_long_min, hx=hx_eff, grade=grade,
        )
        s_max_middle = s_max_middle_zone_column(db_long_min=db_long_min, grade=grade)
        s_confined_ok = s_prov <= s_max_confined + 1e-6
        if not s_confined_ok:
            notes.append(
                f"s in lo {s_prov:.0f} > s_max_confined {s_max_confined:.0f} mm (Sec.18.7.5.3)."
            )

        # ---- 7b) Longitudinal ratio ρg (§18.7.4.1 seismic / §10.6.1) ----
        from design.columns.optimize import rho_g_check
        _rho_g, rho_g_ok, rho_g_note = rho_g_check(self)
        if rho_g_note:
            notes.append(rho_g_note)

        # ---- 8) Overall passed ----
        passed = (passed_pmm
                  and (ratio_shear <= 1.0)
                  and ash_ok and hx_ok and s_confined_ok and rho_g_ok)

        # ---- Steel quantities (kg/m³) ----
        from design.columns.optimize import (
            longitudinal_steel_quantity, transverse_steel_quantity,
        )
        rho_long_val = longitudinal_steel_quantity(self)
        rho_trans_val = 0.0
        if (self.transverse_bar_diameter and self.n_legs_x and self.n_legs_y
                and self.s and self.s > 0):
            rho_trans_val = transverse_steel_quantity(
                self,
                db_hoop=self.transverse_bar_diameter,
                n_legs_x=self.n_legs_x,
                n_legs_y=self.n_legs_y,
                spacing=self.s,
            )

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
            phi_Vn_x=phi_Vn_x, phi_Vn_y=phi_Vn_y,
            ratio_shear_x=ratio_shear_x, ratio_shear_y=ratio_shear_y,
            ratio_Vu_x=ratio_Vu_x, ratio_Vu_y=ratio_Vu_y,
            ratio_Ve_x=ratio_Ve_x, ratio_Ve_y=ratio_Ve_y,
            Ve_x_capacity=Ve_x_cap, Ve_y_capacity=Ve_y_cap,
            Mpr_x=Mpr_x, Mpr_y=Mpr_y,
            Av_x_provided=Av_x_prov, Av_y_provided=Av_y_prov,
            Ash_x_demand=Ash_x_demand, Ash_y_demand=Ash_y_demand,
            Ash_x_capacity=Ash_x_capacity, Ash_y_capacity=Ash_y_capacity,
            ratio_ash_x=ratio_ash_x, ratio_ash_y=ratio_ash_y,
            ash_ok=ash_ok,
            hx_x_face=hx_x_face, hx_y_face=hx_y_face, hx_ok=hx_ok,
            s_provided=s_prov,
            s_max_confined=s_max_confined,
            s_max_middle=s_max_middle,
            s_confined_ok=s_confined_ok,
            rho_long=rho_long_val,
            rho_trans=rho_trans_val,
            rho_total=rho_long_val + rho_trans_val,
            notes=tuple(notes),
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
            clear_cover = (cover if cover is not None
                           else getattr(self.section, "clear_cover", 25.0))

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

        # ---- core geometry ----
        # bc is measured to the OUTSIDE EDGE of the hoop (ACI 318-25 Tabla 18.7.5.4),
        # so we use clear_cover (not cover-to-center).
        Ag = self.section.gross_area()
        b_out = self.section.width()
        h_out = self.section.height()
        bc = max(b_out - 2 * clear_cover, 100.0)
        hc = max(h_out - 2 * clear_cover, 100.0)
        Ach = bc * hc

        # ---- effective hx from the REAL layout (§18.7.5.2/.3) ----
        # so = 100 + (350 - hx)/3 in §18.7.5.3 depends on the ACTUAL hx, not
        # the constructor default. The hx that will be installed is the one
        # at the leg count the design will use: the user-fixed n_legs if
        # given, else the smallest n that brings hx <= 350 (what the picker
        # bumps to). We take the larger of the two directional hx (governing,
        # i.e. the smallest so).
        grade = self.section.rebar.groups[0].steel.grade if self.section.rebar.groups else 60
        max_x, max_y = self._count_longitudinal_per_face()
        n_x_eff = self._n_legs_for_hx(bc, fixed=n_legs_x, n_max=max_x)
        n_y_eff = self._n_legs_for_hx(hc, fixed=n_legs_y, n_max=max_y)
        hx_eff_x = hx_check(b_core=bc, n_legs=n_x_eff)[0]
        hx_eff_y = hx_check(b_core=hc, n_legs=n_y_eff)[0]
        hx_eff = max(hx_eff_x, hx_eff_y)

        # ---- spacing in confined zone (§18.7.5.3) — uses real hx ----
        s_try = self.s if self.s is not None else 100.0
        if self.seismic:
            s_try = min(s_try, s_max_confined_column(
                h_min=self.h_min, db_long_min=db_long_min,
                hx=hx_eff, grade=grade,
            ))
            notes.append("Spacing in lo capped by Sec.18.7.5.3.")

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
        # nl = laterally-supported perimeter longitudinal bars; feeds
        # kn = nl/(nl-2) in Eq. 18.7.5.4b for expression (c).
        nl = self._count_perimeter_bars()
        ash_req_x = ash_required(s=s_try, bc=bc, Ag=Ag, Ach=Ach,
                                 fc=c.fc, fyt=self.fyt, Pu_over_Ag_fc=Pu_ratio,
                                 nl=nl)
        ash_req_y = ash_required(s=s_try, bc=hc, Ag=Ag, Ach=Ach,
                                 fc=c.fc, fyt=self.fyt, Pu_over_Ag_fc=Pu_ratio,
                                 nl=nl)

        # ---- preliminary directional shear demand (Av/s required) ----
        # We compute it BEFORE the hoop pick so the picker sees both
        # confinement (Ash) and capacity-design shear (Av/s) demands.
        # For Vx: bw = h (perpendicular to X), d = 0.9·b
        # For Vy: bw = b (perpendicular to Y), d = 0.9·h
        d_x_pre = 0.9 * b_out
        d_y_pre = 0.9 * h_out
        Vc_x_pre = self._vc("x")
        Vc_y_pre = self._vc("y")
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
        chosen_db, n_x, n_y, trace_conf = self._pick_hoop_and_legs(
            db_hoop=db_hoop, n_legs_x=n_legs_x, n_legs_y=n_legs_y,
            ash_req_x=ash_req_x, ash_req_y=ash_req_y,
            av_s_req_x=av_s_req_x_for_pick, av_s_req_y=av_s_req_y_for_pick,
            bc=bc, hc=hc,
            s=s_try, notes=notes,
            check_hx=True, check_ash=True,
        )
        a_per_leg = pi * chosen_db ** 2 / 4.0
        av_prov_x = n_x * a_per_leg
        av_prov_y = n_y * a_per_leg

        if av_prov_x < ash_req_x:
            notes.append(f"n_legs_x={n_x} gives {av_prov_x:.0f} mm^2 < {ash_req_x:.0f} required.")
        if av_prov_y < ash_req_y:
            notes.append(f"n_legs_y={n_y} gives {av_prov_y:.0f} mm^2 < {ash_req_y:.0f} required.")

        # ---- hx checks (§18.7.5.2) ----
        hx_x_face, ok_x = hx_check(b_core=bc, n_legs=n_x)
        hx_y_face, ok_y = hx_check(b_core=hc, n_legs=n_y)
        hx_ok = ok_x and ok_y
        if not hx_ok:
            notes.append(
                f"hx > 350 mm on at least one face "
                f"(X face: {hx_x_face:.0f}, Y face: {hx_y_face:.0f})."
            )

        s_middle = s_max_middle_zone_column(db_long_min=db_long_min, grade=grade)
        lo = lo_length_column(h_member=max(b_out, h_out), lu_clear_mm=self.lu)

        # ---- Spiral confinement (§25.7.3.3 / Table 18.7.5.4 d/e/f) ----
        # A tied column never sizes a spiral; only compute ρ_s when spiral.
        rho_s_req: float | None = None
        rho_s_prov: float | None = None
        if self.spiral:
            rho_s_req = rho_s_spiral_required(
                Ag=Ag, Ach=Ach, fc=c.fc, fyt=self.fyt,
                Pu_over_Ag_fc=Pu_ratio,
            )
            # Provided volumetric ratio ρ_s = 4·Asp/(dc·s):
            #   Asp = area of one spiral bar (use the chosen hoop bar)
            #   dc  = core diameter to the spiral centerline (inscribed circle
            #         in the rectangular core: smaller core dim, less db_hoop)
            #   s   = pitch (confined-zone spacing)
            Asp = pi * chosen_db ** 2 / 4.0
            dc = max(min(bc, hc) - chosen_db, 1.0)
            if s_try > 0:
                rho_s_prov = 4.0 * Asp / (dc * s_try)
            if rho_s_prov is not None and rho_s_prov < rho_s_req:
                notes.append(
                    f"Spiral rho_s provided {rho_s_prov:.4f} < required "
                    f"{rho_s_req:.4f} (Sec.25.7.3.3) -- tighten pitch or "
                    f"increase spiral bar."
                )

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
        Vc_x = self._vc("x", Av=av_prov_y, s=s_try)  # bw=h → resists Vx
        Vc_y = self._vc("y", Av=av_prov_x, s=s_try)  # bw=b → resists Vy

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
                    "Sec.18.7.6.2.1 -- Vc=0 in lo (X): Pu < Ag*fc/20 and seismic-dominated shear."
                )
            if vc_zero_y:
                Vc_y = 0.0
                notes.append(
                    "Sec.18.7.6.2.1 -- Vc=0 in lo (Y): Pu < Ag*fc/20 and seismic-dominated shear."
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
                            f"Ve_x {Ve_x_used:.0f} kN capped by Omega_o*Vux = {cap_x:.0f} kN (Sec.18.7.6.1.1)."
                        )
                        Ve_x_used = cap_x
                    amplif_x = Ve_x_used / Vu_x
                if Vu_y > 0:
                    cap_y = self.omega_0 * Vu_y
                    if cap_y < Ve_y_used:
                        notes.append(
                            f"Ve_y {Ve_y_used:.0f} kN capped by Omega_o*Vuy = {cap_y:.0f} kN (Sec.18.7.6.1.1)."
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

        # ---- ratio Vu / φVn per direction (demand check, only if demands) ----
        ratio_Vu_phiVn_x: float | None = None
        ratio_Vu_phiVn_y: float | None = None
        ratio_pmm_val: float | None = None
        if demands is not None:
            if phi_Vn_x > 0:
                ratio_Vu_phiVn_x = abs(demands.Vux) / phi_Vn_x
            if phi_Vn_y > 0:
                ratio_Vu_phiVn_y = abs(demands.Vuy) / phi_Vn_y
            # PMM check against the surface
            _, ratio_pmm_val = self.surface.check(
                Pu=demands.Pu, Mux=demands.Mux, Muy=demands.Muy,
            )

        # ============================================================ #
        # MIDDLE ZONE — §18.7.5.5  (only §22.5 shear gates the design)
        #   • Ash         → NOT required (only in lo)
        #   • hx ≤ 350    → NOT required (only in lo)
        #   • Vc = 0 lo   → NOT applicable (Vc reappears here)
        #   • s_max       → §18.7.5.5 (≤ 6·db_long, ≤ 150 mm)
        # ============================================================ #
        # The shear demand is the SAME as in lo (Ve_x_used / Ve_y_used,
        # or |Vu| in demand mode), but Vc is restored (no §18.7.6.2.1).
        Vc_x_mid = self._vc("x")
        Vc_y_mid = self._vc("y")
        av_s_req_x_mid: float | None = None
        av_s_req_y_mid: float | None = None
        if mode == "demand" and demands is not None:
            av_s_req_x_mid = self._av_s_required_for_direction(
                Vu=abs(demands.Vux), Vc=Vc_x_mid, d=d_x,
            )
            av_s_req_y_mid = self._av_s_required_for_direction(
                Vu=abs(demands.Vuy), Vc=Vc_y_mid, d=d_y,
            )
        elif mode == "capacity":
            av_s_req_x_mid = self._av_s_required_for_direction(
                Vu=Ve_x_used or 0.0, Vc=Vc_x_mid, d=d_x,
            )
            av_s_req_y_mid = self._av_s_required_for_direction(
                Vu=Ve_y_used or 0.0, Vc=Vc_y_mid, d=d_y,
            )

        chosen_db_mid, n_x_mid, n_y_mid, trace_mid = self._pick_hoop_and_legs(
            db_hoop=db_hoop, n_legs_x=n_legs_x, n_legs_y=n_legs_y,
            ash_req_x=0.0, ash_req_y=0.0,
            av_s_req_x=av_s_req_x_mid, av_s_req_y=av_s_req_y_mid,
            bc=bc, hc=hc,
            s=s_middle, notes=notes,
            check_hx=False, check_ash=False,
        )
        a_per_leg_mid = pi * chosen_db_mid ** 2 / 4.0
        av_prov_x_mid = n_x_mid * a_per_leg_mid
        av_prov_y_mid = n_y_mid * a_per_leg_mid
        Vc_x_mid = self._vc("x", Av=av_prov_y_mid, s=s_middle)
        Vc_y_mid = self._vc("y", Av=av_prov_x_mid, s=s_middle)
        Vs_x_prov_mid = vs_capacity(Av=av_prov_y_mid, fyt=self.fyt, d=d_x, s=s_middle)
        Vs_y_prov_mid = vs_capacity(Av=av_prov_x_mid, fyt=self.fyt, d=d_y, s=s_middle)
        phi_Vn_x_mid = phi_v * (Vc_x_mid + Vs_x_prov_mid)
        phi_Vn_y_mid = phi_v * (Vc_y_mid + Vs_y_prov_mid)
        # Ratios in the middle zone — use Ve_used if capacity, else |Vu|
        if mode == "capacity":
            Vu_x_mid = Ve_x_used or 0.0
            Vu_y_mid = Ve_y_used or 0.0
        elif mode == "demand" and demands is not None:
            Vu_x_mid = abs(demands.Vux)
            Vu_y_mid = abs(demands.Vuy)
        else:
            Vu_x_mid = Vu_y_mid = 0.0
        ratio_Vu_phiVn_x_mid = (Vu_x_mid / phi_Vn_x_mid
                                if phi_Vn_x_mid > 0 and Vu_x_mid > 0 else None)
        ratio_Vu_phiVn_y_mid = (Vu_y_mid / phi_Vn_y_mid
                                if phi_Vn_y_mid > 0 and Vu_y_mid > 0 else None)

        return ColumnDesignProposal(
            mode=mode,
            ash_required_x=ash_req_x, ash_required_y=ash_req_y,
            db_hoop=chosen_db, n_legs_x=n_x, n_legs_y=n_y,
            av_provided_x=av_prov_x, av_provided_y=av_prov_y,
            spacing_confined=s_try, spacing_middle=s_middle, lo=lo,
            hx_x_face=hx_x_face, hx_y_face=hx_y_face, hx_ok=hx_ok,
            Pu_used=Pu_used, Pu_over_Ag_fc=Pu_ratio,
            eq_iii_active=(Pu_ratio > 0.3),
            rho_s_required=rho_s_req, rho_s_provided=rho_s_prov,
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
            ratio_Vu_over_phiVn_x=ratio_Vu_phiVn_x,
            ratio_Vu_over_phiVn_y=ratio_Vu_phiVn_y,
            ratio_pmm=ratio_pmm_val,
            vc_zero_x=vc_zero_x,
            vc_zero_y=vc_zero_y,
            middle_db_hoop=chosen_db_mid,
            middle_n_legs_x=n_x_mid,
            middle_n_legs_y=n_y_mid,
            middle_spacing=s_middle,
            middle_av_provided_x=av_prov_x_mid,
            middle_av_provided_y=av_prov_y_mid,
            middle_av_s_required_x=av_s_req_x_mid,
            middle_av_s_required_y=av_s_req_y_mid,
            middle_phi_Vn_x=phi_Vn_x_mid,
            middle_phi_Vn_y=phi_Vn_y_mid,
            middle_ratio_Ve_over_phiVn_x=ratio_Vu_phiVn_x_mid,
            middle_ratio_Ve_over_phiVn_y=ratio_Vu_phiVn_y_mid,
            picker_trace_confined=trace_conf,
            picker_trace_middle=trace_mid,
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
        def steel_index_mid(p):
            if p.middle_db_hoop is None or p.middle_spacing is None:
                return 0.0
            return (p.middle_db_hoop ** 2
                    * (p.middle_n_legs_x + p.middle_n_legs_y)
                    / max(p.middle_spacing, 1e-9))
        ash_worst = max(modes, key=steel_index)
        mid_worst = max(modes, key=steel_index_mid)
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
            rho_s_required=max(
                (p.rho_s_required for p in modes if p.rho_s_required is not None),
                default=None,
            ),
            rho_s_provided=ash_worst.rho_s_provided,
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
            ratio_Vu_over_phiVn_x=ash_worst.ratio_Vu_over_phiVn_x,
            ratio_Vu_over_phiVn_y=ash_worst.ratio_Vu_over_phiVn_y,
            ratio_pmm=ash_worst.ratio_pmm,
            vc_zero_x=any(p.vc_zero_x for p in modes),
            vc_zero_y=any(p.vc_zero_y for p in modes),
            # Middle zone — pick the worst by same steel-intensity criterion
            middle_db_hoop=mid_worst.middle_db_hoop,
            middle_n_legs_x=mid_worst.middle_n_legs_x,
            middle_n_legs_y=mid_worst.middle_n_legs_y,
            middle_spacing=mid_worst.middle_spacing,
            middle_av_provided_x=mid_worst.middle_av_provided_x,
            middle_av_provided_y=mid_worst.middle_av_provided_y,
            middle_av_s_required_x=max(
                (p.middle_av_s_required_x or 0.0) for p in modes
            ) or None,
            middle_av_s_required_y=max(
                (p.middle_av_s_required_y or 0.0) for p in modes
            ) or None,
            middle_phi_Vn_x=mid_worst.middle_phi_Vn_x,
            middle_phi_Vn_y=mid_worst.middle_phi_Vn_y,
            middle_ratio_Ve_over_phiVn_x=mid_worst.middle_ratio_Ve_over_phiVn_x,
            middle_ratio_Ve_over_phiVn_y=mid_worst.middle_ratio_Ve_over_phiVn_y,
            picker_trace_confined=ash_worst.picker_trace_confined,
            picker_trace_middle=mid_worst.picker_trace_middle,
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
        Vc = self._vc("y")
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

    # ---- internal: count laterally-supported perimeter longitudinal bars ----
    def _count_perimeter_bars(self, *, layer_tol: float = 5.0) -> int:
        """Number of longitudinal bars on the core perimeter (= ``nl``).

        Feeds ``kn = nl/(nl-2)`` in ACI 318-25 Eq. 18.7.5.4b (Ash
        expression (c)). A bar is on the perimeter when it sits on the
        outermost ring in either direction (its x is at the left/right
        extreme OR its y is at the top/bottom extreme, within
        ``layer_tol`` mm). For the standard ``perimeter_bars`` layout this
        returns every longitudinal bar; for layouts with interior bars it
        counts only the supported outer ring, which is the conservative
        (smaller ``kn``) and code-correct choice.
        """
        xs: list[float] = []
        ys: list[float] = []
        for r, _ in self.section.rebar.iter_bars():
            xs.append(r.x)
            ys.append(r.y)
        if not xs:
            return 0
        x_left, x_right = min(xs), max(xs)
        y_bot, y_top = min(ys), max(ys)
        nl = 0
        for x, y in zip(xs, ys):
            on_perim = (
                abs(x - x_left) <= layer_tol or abs(x - x_right) <= layer_tol
                or abs(y - y_bot) <= layer_tol or abs(y - y_top) <= layer_tol
            )
            if on_perim:
                nl += 1
        return nl

    # ---- internal: leg count that governs hx in one direction (§18.7.5.2) ----
    @staticmethod
    def _n_legs_for_hx(core_dim: float, *, fixed: int | None, n_max: int) -> int:
        """Smallest leg count that makes hx <= 350 in this direction.

        If ``fixed`` is given (user-forced n_legs), it is returned as-is.
        Otherwise we mirror the picker's ``bump_to_hx``: increase legs from
        2 until hx <= 350 or the physical bar limit ``n_max`` is reached.
        Used to derive the EFFECTIVE hx that feeds so = 100 + (350-hx)/3
        in §18.7.5.3, so the spacing cap reflects the real layout.
        """
        if fixed is not None:
            return max(2, fixed)
        n = 2
        while n <= n_max:
            if hx_check(b_core=core_dim, n_legs=n)[1]:
                return n
            n += 1
        return max(2, n_max)

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
        check_hx: bool = True,
        check_ash: bool = True,
    ) -> tuple[float, int, int, tuple[str, ...]]:
        """For each diameter, pick the minimum n_legs that satisfies
        the active gates (§18.7.5.4 Ash, §22.5 Av/s, §18.7.5.2 hx),
        respecting the physical bar-count limit.

        Gates can be disabled with ``check_hx`` / ``check_ash`` — used by
        the middle-zone design where neither §18.7.5.4 nor §18.7.5.2 apply.

        Returns
        -------
        (db_chosen, n_x, n_y, trace)
            ``trace`` is a tuple of human-readable strings, one per diameter
            tested, explaining why each one was accepted or rejected.
        """
        max_x, max_y = self._count_longitudinal_per_face()
        trace: list[str] = []

        def n_required_for(db: float) -> tuple[int, int, str]:
            """Min legs for (X, Y) and which gate ruled each direction."""
            a_per_leg = pi * db ** 2 / 4.0
            # Confinement (§18.7.5.4)
            if check_ash:
                n_x_conf = max(2, ceil(ash_req_x / a_per_leg))
                n_y_conf = max(2, ceil(ash_req_y / a_per_leg))
            else:
                n_x_conf = n_y_conf = 2
            # Shear capacity: legs ∥ X resist Vy → av_s_req_y;
            #                 legs ∥ Y resist Vx → av_s_req_x
            n_x_shear = (max(2, ceil((av_s_req_y * s) / a_per_leg))
                         if av_s_req_y and av_s_req_y > 0 else 2)
            n_y_shear = (max(2, ceil((av_s_req_x * s) / a_per_leg))
                         if av_s_req_x and av_s_req_x > 0 else 2)
            gov_x = "Ash" if n_x_conf > n_x_shear else ("Av/s" if n_x_shear > 2 else "min(2)")
            gov_y = "Ash" if n_y_conf > n_y_shear else ("Av/s" if n_y_shear > 2 else "min(2)")
            return (max(n_x_conf, n_x_shear), max(n_y_conf, n_y_shear),
                    f"X<-{gov_x}, Y<-{gov_y}")

        def bump_to_hx(n: int, core_dim: float, n_max: int) -> int:
            """Increase n until hx ≤ 350 or hit physical limit."""
            if not check_hx:
                return n
            while n <= n_max:
                _, ok = hx_check(b_core=core_dim, n_legs=n)
                if ok:
                    return n
                n += 1
            return n  # > n_max → infeasible at this diameter

        # ---- USER-FORCED diameter ----
        if db_hoop is not None:
            chosen_db = db_hoop
            n_x_req, n_y_req, gov_str = n_required_for(chosen_db)
            n_x = n_legs_x if n_legs_x is not None else n_x_req
            n_y = n_legs_y if n_legs_y is not None else n_y_req
            if n_legs_x is None:
                n_x = bump_to_hx(n_x, bc, max_x)
                if n_x > max_x:
                    notes.append(
                        f"With phi{chosen_db:.0f}, no n_legs_x <= {max_x} cleared hx<=350 mm."
                    )
                    n_x = min(n_x_req, max_x)
            if n_legs_y is None:
                n_y = bump_to_hx(n_y, hc, max_y)
                if n_y > max_y:
                    notes.append(
                        f"With phi{chosen_db:.0f}, no n_legs_y <= {max_y} cleared hx<=350 mm."
                    )
                    n_y = min(n_y_req, max_y)
            trace.append(f"phi{chosen_db:.0f} [FORCED by user] gov: {gov_str} "
                         f"-> legs=({n_x},{n_y})")
            return chosen_db, n_x, n_y, tuple(trace)

        # ---- AUTO-PICK: try each diameter in ascending order ----
        # Acceptance criterion: Ash + Vs satisfied within the physical
        # bar-count limit. hx is a TOPOLOGICAL gate (longitudinal layout),
        # not a hoop gate — if it fails for every diameter, we still pick
        # the smallest one that satisfies Ash + Vs and emit a clear note.
        for db in sorted(self.bar_schedule.hoops):
            n_x_req, n_y_req, gov_str = n_required_for(db)
            n_x = n_legs_x if n_legs_x is not None else n_x_req
            n_y = n_legs_y if n_legs_y is not None else n_y_req

            # Physical fit by Ash+Vs ALONE — this is the only hard gate.
            if n_x > max_x or n_y > max_y:
                reasons = []
                if n_x > max_x:
                    reasons.append(f"Ash/Vs req {n_x}>max_x={max_x} ({gov_str.split(', ')[0]})")
                if n_y > max_y:
                    reasons.append(f"Ash/Vs req {n_y}>max_y={max_y} ({gov_str.split(', ')[1]})")
                trace.append(f"phi{db:.0f}  x rejected -- " + "; ".join(reasons))
                continue

            # Try to bump for hx (soft gate). If it cannot satisfy hx within
            # max_*, KEEP the Ash/Vs-feasible n and flag it — do NOT skip.
            n_x_hx = bump_to_hx(n_x, bc, max_x) if n_legs_x is None else n_x
            n_y_hx = bump_to_hx(n_y, hc, max_y) if n_legs_y is None else n_y

            hx_x_ok = (n_x_hx <= max_x)
            hx_y_ok = (n_y_hx <= max_y)
            if hx_x_ok:
                n_x = n_x_hx
            if hx_y_ok:
                n_y = n_y_hx

            if hx_x_ok and hx_y_ok:
                trace.append(f"phi{db:.0f}  legs=({n_x},{n_y})  gov: {gov_str}  ok ACCEPTED")
            else:
                tags = []
                if not hx_x_ok: tags.append(f"hx_x violated (need n>{max_x})")
                if not hx_y_ok: tags.append(f"hx_y violated (need n>{max_y})")
                trace.append(f"phi{db:.0f}  legs=({n_x},{n_y})  gov: {gov_str}  "
                             f"ok accepted (Ash+Vs ok) -- " + "; ".join(tags))
                notes.append(
                    f"Sec.18.7.5.2 violated with phi{db:.0f}: "
                    f"longitudinal layout cannot satisfy hx<=350. "
                    f"Add bars on the long face(s) to reduce hx."
                )
            return db, n_x, n_y, tuple(trace)

        # ---- FALLBACK: nothing satisfied Ash/Vs within physical limit ----
        db = max(self.bar_schedule.hoops)
        n_x_req, n_y_req, gov_str = n_required_for(db)
        n_x = n_legs_x if n_legs_x is not None else min(n_x_req, max_x)
        n_y = n_legs_y if n_legs_y is not None else min(n_y_req, max_y)
        notes.append(
            f"Ash/Vs cannot be satisfied within the physical bar limit "
            f"(max_x={max_x}, max_y={max_y}) with any hoop in {sorted(self.bar_schedule.hoops)}. "
            f"Used max phi{db:.0f} capped to physical max."
        )
        trace.append(f"-> FALLBACK: phi{db:.0f}  legs=({n_x},{n_y})  "
                     f"(Ash/Vs under-met)")
        return db, n_x, n_y, tuple(trace)

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
        print(f"Section : {u.fmt_length(s.width(), unit=False)}x{u.fmt_length(s.height())}")
        print(f"Materials: fc = {s.concrete.fc:.0f} MPa, fy = {rebar.groups[0].steel.fy:.0f} MPa")
        # ---- longitudinal bar diameters present in the section ----
        from collections import Counter
        from math import pi as _pi
        diam_count: Counter = Counter()
        for g in rebar.groups:
            for r in g.bars:
                # area = pi/4 * d^2  →  d = 2·sqrt(area/pi)
                d_mm = 2.0 * (r.area / _pi) ** 0.5
                diam_count[round(d_mm)] += 1
        if len(diam_count) == 1:
            d, n = next(iter(diam_count.items()))
            bar_str = f"{n} bars - phi {d} mm"
        else:
            # multiple diameters: list each
            parts = [f"{n} phi{d}" for d, n in sorted(diam_count.items())]
            bar_str = f"{n_bars} bars (" + " + ".join(parts) + ")"
        print(f"Rebar   : {bar_str}, As = {u.fmt_area(Ast)}, rho = {rho:.2f}%")
        if self._surface is None:
            print("State   : not run yet -- call col.run() to compute the PMM surface.")
            return
        print(f"Po      : {u.fmt_force(self.Po)}")
        print(f"To      : {u.fmt_force(self.To)}")
        print(f"phiPn,max : {u.fmt_force(self.phi_Pn_max)}")
        print(f"phiVn (x/y): {u.fmt_force(self.phi_Vn_x, unit=False)} / "
              f"{u.fmt_force(self.phi_Vn_y)}")
        print(f"max phiMn @ theta=0deg  : {u.fmt_moment(max(self.surface[0.0].phi_Mn))}")
        print(f"max phiMn @ theta=90deg : {u.fmt_moment(max(self.surface[90.0].phi_Mn))}")

        # Steel quantities — shown when hoops are defined
        from design.columns.optimize import (
            longitudinal_steel_quantity, transverse_steel_quantity,
        )
        rho_l = longitudinal_steel_quantity(self)
        rho_t = 0.0
        if (self.transverse_bar_diameter and self.n_legs_x and self.n_legs_y
                and self.s and self.s > 0):
            rho_t = transverse_steel_quantity(
                self,
                db_hoop=self.transverse_bar_diameter,
                n_legs_x=self.n_legs_x,
                n_legs_y=self.n_legs_y,
                spacing=self.s,
            )
            print(f"Hoops   : phi{self.transverse_bar_diameter:.0f}  "
                  f"legs=({self.n_legs_x},{self.n_legs_y})  s={u.fmt_length(self.s)}")
        print(f"Steel quantities (kg/m^3 of concrete):")
        print(f"  rho_long  = {rho_l:.2f}")
        print(f"  rho_trans = {rho_t:.2f}")
        print(f"  rho_total = {rho_l + rho_t:.2f}")

    def print_final(
        self,
        proposal: "ColumnDesignProposal",
        demands: ColumnDemands | None = None,
    ) -> None:
        """Print the FINAL design in one consolidated block (English).

        Combines section + materials + longitudinal layout + hoops (lo &
        middle from proposal) + directional capacities + steel quantities,
        and, if ``demands`` is given, the full ColumnCheck ratios.
        """
        from design.columns.optimize import (
            transverse_steel_quantity, longitudinal_steel_quantity,
        )
        self._ensure_run()
        u = self.units
        F = u.force_factor
        Fu = u.force
        Lu = u.length
        Mu = u.moment_factor
        s = self.section
        rebar = s.rebar
        n_bars = sum(g.n for g in rebar.groups)
        Ast = rebar.total_area
        Ag = s.gross_area()
        rho_geo = Ast / Ag * 100 if Ag > 0 else 0.0
        steel = rebar.groups[0].steel if rebar.groups else None
        fc = s.concrete.fc
        fy = steel.fy if steel else 0.0

        # Hoop quantities (lo)
        rho_long  = longitudinal_steel_quantity(self)
        rho_trans = transverse_steel_quantity(
            self,
            db_hoop=proposal.db_hoop,
            n_legs_x=proposal.n_legs_x,
            n_legs_y=proposal.n_legs_y,
            spacing=proposal.spacing_confined,
        )
        rho_total = rho_long + rho_trans

        bar_w = 76
        print()
        print("=" * bar_w)
        print(f"  COLUMN -- FINAL DESIGN   |   {self.label}   |   units: {u.name}")
        print("=" * bar_w)
        print(f"  Section          : {u.fmt_length(s.width(), unit=False)} x {u.fmt_length(s.height())}   "
              f"clear_cover = {u.fmt_length(getattr(s, 'clear_cover', 25.0))}")
        print(f"  Materials        : fc' = {fc:.0f} MPa    fy = {fy:.0f} MPa")
        print(f"  Longitudinal     : {n_bars} bars   "
              f"As = {u.fmt_area(Ast)}   rho_geom = {rho_geo:.2f} %")
        print("-" * bar_w)
        print(f"  Hoops -- lo (Sec.18.7.5):")
        print(f"    phi{proposal.db_hoop:.0f}   "
              f"legs = (short={proposal.n_legs_x}, long={proposal.n_legs_y})   "
              f"s = {u.fmt_length(proposal.spacing_confined)}   "
              f"lo = {u.fmt_length(proposal.lo)}")
        print(f"    Av provided      : short = {u.fmt_area(proposal.av_provided_x, unit=False)},   "
              f"long = {u.fmt_area(proposal.av_provided_y)}")
        print(f"    Ash required     : short = {u.fmt_area(proposal.ash_required_x, unit=False)},   "
              f"long = {u.fmt_area(proposal.ash_required_y)}")
        if self.spiral and proposal.rho_s_required is not None:
            prov = (f"{proposal.rho_s_provided:.4f}"
                    if proposal.rho_s_provided is not None else "n/a (not modeled)")
            ok = (proposal.rho_s_provided is not None
                  and proposal.rho_s_provided >= proposal.rho_s_required)
            tagrs = "PASS" if ok else ("FAIL" if proposal.rho_s_provided is not None else "--")
            print(f"    Spiral rho_s     : req = {proposal.rho_s_required:.4f}   "
                  f"prov = {prov}   [{tagrs}]   (Sec.25.7.3.3)")
        if proposal.middle_db_hoop is not None:
            print(f"  Hoops -- middle (Sec.18.7.5.5):")
            print(f"    phi{proposal.middle_db_hoop:.0f}   "
                  f"legs = (short={proposal.middle_n_legs_x}, long={proposal.middle_n_legs_y})   "
                  f"s = {u.fmt_length(proposal.middle_spacing)}")
            print(f"    Av provided      : short = {u.fmt_area(proposal.middle_av_provided_x, unit=False)},   "
                  f"long = {u.fmt_area(proposal.middle_av_provided_y)}")
        print("-" * bar_w)
        print(f"  Capacities:")
        print(f"    Po              = {u.fmt_force(self.Po)}")
        print(f"    To              = {u.fmt_force(self.To)}")
        print(f"    phi_Pn_max      = {u.fmt_force(self.phi_Pn_max)}")
        print(f"    phi_Vn (x / y)  = {u.fmt_force(self.phi_Vn_x, unit=False)} / "
              f"{u.fmt_force(self.phi_Vn_y)}")
        print(f"    max phi_Mn @ 0  = {u.fmt_moment(max(self.surface[0.0].phi_Mn))}")
        print(f"    max phi_Mn @ 90 = {u.fmt_moment(max(self.surface[90.0].phi_Mn))}")
        print(f"    Mpr (x / y)     = {u.fmt_moment(proposal.Mpr_x or 0, unit=False)} / "
              f"{u.fmt_moment(proposal.Mpr_y or 0)}")
        print(f"    Ve cap (x / y)  = {u.fmt_force(proposal.Ve_x_capacity or 0, unit=False)} / "
              f"{u.fmt_force(proposal.Ve_y_capacity or 0)}")
        print("-" * bar_w)
        print(f"  Steel quantities (kg/m^3 of concrete):")
        print(f"    rho_long        = {rho_long :8.2f}")
        print(f"    rho_trans (lo)  = {rho_trans:8.2f}")
        print(f"    rho_total       = {rho_total:8.2f}")

        if demands is not None:
            print("-" * bar_w)
            print(f"  Code check vs demands:")
            chk = self.check(demands)
            tag = lambda b: "PASS" if b else "FAIL"
            print(f"    PMM             : ratio = {chk.ratio_pmm:.3f}   "
                  f"[{tag(chk.ratio_pmm <= 1.0)}]")
            print(f"    Shear X         : ratio = {chk.ratio_shear_x:.3f}   "
                  f"(Vu={chk.ratio_Vu_x:.3f}, Ve={chk.ratio_Ve_x:.3f})   "
                  f"[{tag(chk.ratio_shear_x <= 1.0)}]")
            print(f"    Shear Y         : ratio = {chk.ratio_shear_y:.3f}   "
                  f"(Vu={chk.ratio_Vu_y:.3f}, Ve={chk.ratio_Ve_y:.3f})   "
                  f"[{tag(chk.ratio_shear_y <= 1.0)}]")
            print(f"    Ash short       : ratio = {chk.ratio_ash_x:.3f}   "
                  f"[{tag(chk.ratio_ash_x <= 1.0)}]")
            print(f"    Ash long        : ratio = {chk.ratio_ash_y:.3f}   "
                  f"[{tag(chk.ratio_ash_y <= 1.0)}]")
            print(f"    hx              : x = {u.fmt_length(chk.hx_x_face, unit=False)}, "
                  f"y = {u.fmt_length(chk.hx_y_face)}   [{tag(chk.hx_ok)}]")
            print(f"    s (confined)    : prov = {u.fmt_length(chk.s_provided, unit=False)},   "
                  f"max = {u.fmt_length(chk.s_max_confined)}   [{tag(chk.s_confined_ok)}]")
            print("  " + "-" * (bar_w - 4))
            verdict = "PASS" if chk.passed else "FAIL"
            print(f"  OVERALL          : {verdict}")
            if chk.notes:
                print(f"  Notes:")
                for n in chk.notes:
                    print(f"    - {n}")
        print("=" * bar_w)

    def print_design(
        self,
        results: "ColumnDesignResults",
        *,
        modes: tuple[str, ...] = ("demand", "capacity", "envelope"),
        show_trace: bool = True,
    ) -> None:
        """Print a compact, structured report of the design results.

        For each requested mode, prints two sub-blocks (confined ``lo`` and
        ``middle``) and — when ``show_trace=True`` — the picker's
        diameter-by-diameter decision log for both zones. Notes are printed
        at the very end, grouped by mode.
        """
        u = self.units
        # Header
        print()
        print("=" * 74)
        print(f"  COLUMN DESIGN -- {self.label}    (units: {u.name})")
        print(f"  Section {u.fmt_length(self.section.width(), unit=False)}x"
              f"{u.fmt_length(self.section.height())}    "
              f"lu = {u.fmt_length(self.lu)}    "
              f"seismic = {self.seismic}")
        mx, my = self._count_longitudinal_per_face()
        print(f"  Max legs (from longitudinal bars):  ||X = {mx}    ||Y = {my}")
        print(f"  Hoop schedule: {sorted(self.bar_schedule.hoops)}")
        print("=" * 74)

        all_notes: list[tuple[str, str]] = []  # (mode, note)

        def _zone_lines_lo(p) -> list[str]:
            lines: list[str] = []
            lines.append(
                f"    phi{p.db_hoop:.0f}  legs=({p.n_legs_x},{p.n_legs_y})  "
                f"s = {u.fmt_length(p.spacing_confined)}    "
                f"lo = {u.fmt_length(p.lo)}"
            )
            lines.append(
                f"    Ash    req ({u.fmt_area(p.ash_required_x, unit=False)}, "
                f"{u.fmt_area(p.ash_required_y, unit=False)})  "
                f"prov ({u.fmt_area(p.av_provided_x, unit=False)}, "
                f"{u.fmt_area(p.av_provided_y)})"
            )
            if p.rho_s_required is not None:
                prov = (f"{p.rho_s_provided:.4f}"
                        if p.rho_s_provided is not None else "n/a")
                lines.append(
                    f"    rho_s  req {p.rho_s_required:.4f}  prov {prov}  "
                    f"(Sec.25.7.3.3 spiral)"
                )
            if p.av_s_required_x is not None or p.av_s_required_y is not None:
                lines.append(
                    f"    Av/s   req ({p.av_s_required_x or 0:.3f}, "
                    f"{p.av_s_required_y or 0:.3f}) mm^2/mm"
                )
            lines.append(
                f"    hx     x = {u.fmt_length(p.hx_x_face, unit=False)}, "
                f"y = {u.fmt_length(p.hx_y_face)}   "
                f"hx_ok = {p.hx_ok}"
            )
            if p.phi_Vn_x is not None and p.phi_Vn_y is not None:
                lines.append(
                    f"    phiVn  x = {u.fmt_force(p.phi_Vn_x, unit=False)}, "
                    f"y = {u.fmt_force(p.phi_Vn_y)}"
                )
            if p.Mpr_x is not None and p.Mpr_y is not None:
                lines.append(
                    f"    Mpr    x = {u.fmt_moment(p.Mpr_x, unit=False)}, "
                    f"y = {u.fmt_moment(p.Mpr_y)}"
                )
            if p.Ve_x_capacity is not None:
                lines.append(
                    f"    Ve_cap x = {u.fmt_force(p.Ve_x_capacity, unit=False)}, "
                    f"y = {u.fmt_force(p.Ve_y_capacity)}"
                )
            if p.Ve_x_used is not None:
                cap_str = ("  (capped by Omega_o*Vu)"
                           if p.amplification_x is not None else "")
                lines.append(
                    f"    Ve_use x = {u.fmt_force(p.Ve_x_used, unit=False)}, "
                    f"y = {u.fmt_force(p.Ve_y_used)}{cap_str}"
                )
            if p.ratio_Ve_over_phiVn_x is not None:
                lines.append(
                    f"    ratio  Ve/phiVn  x = {p.ratio_Ve_over_phiVn_x:.2f}, "
                    f"y = {p.ratio_Ve_over_phiVn_y:.2f}    (capacity)"
                )
            if p.ratio_Vu_over_phiVn_x is not None:
                lines.append(
                    f"    ratio  Vu/phiVn  x = {p.ratio_Vu_over_phiVn_x:.2f}, "
                    f"y = {p.ratio_Vu_over_phiVn_y:.2f}    (demand)"
                )
            if p.ratio_pmm is not None:
                tag = "PASS" if p.ratio_pmm <= 1.0 else "FAIL"
                lines.append(
                    f"    ratio  PMM       = {p.ratio_pmm:.2f}    [{tag}]"
                )
            if p.vc_zero_x or p.vc_zero_y:
                lines.append(
                    f"    Sec.18.7.6.2.1  Vc=0:  x = {p.vc_zero_x}, y = {p.vc_zero_y}"
                )
            return lines

        def _zone_lines_mid(p) -> list[str]:
            if p.middle_db_hoop is None:
                return ["    (not computed)"]
            lines: list[str] = []
            lines.append(
                f"    phi{p.middle_db_hoop:.0f}  "
                f"legs=({p.middle_n_legs_x},{p.middle_n_legs_y})  "
                f"s = {u.fmt_length(p.middle_spacing)}"
            )
            if (p.middle_av_s_required_x is not None
                    or p.middle_av_s_required_y is not None):
                lines.append(
                    f"    Av/s   req ({p.middle_av_s_required_x or 0:.3f}, "
                    f"{p.middle_av_s_required_y or 0:.3f}) mm^2/mm"
                )
            lines.append(
                f"    Av     prov ({u.fmt_area(p.middle_av_provided_x, unit=False)}, "
                f"{u.fmt_area(p.middle_av_provided_y)})"
            )
            lines.append(
                f"    phiVn  x = {u.fmt_force(p.middle_phi_Vn_x, unit=False)}, "
                f"y = {u.fmt_force(p.middle_phi_Vn_y)}"
            )
            if (p.middle_ratio_Ve_over_phiVn_x is not None
                    and p.middle_ratio_Ve_over_phiVn_y is not None):
                lines.append(
                    f"    ratio  Vu/phiVn  x = {p.middle_ratio_Ve_over_phiVn_x:.2f}, "
                    f"y = {p.middle_ratio_Ve_over_phiVn_y:.2f}"
                )
            return lines

        for m in modes:
            p = getattr(results, m, None)
            if p is None:
                continue
            print()
            tag = f" [ {m.upper():^8} ] "
            print(f"{tag}".center(74, "-"))
            print("  lo  (confined zone, Sec.18.7.5):")
            for line in _zone_lines_lo(p):
                print(line)
            print("  middle  (Sec.18.7.5.5 -- Av/s only):")
            for line in _zone_lines_mid(p):
                print(line)
            if show_trace:
                if p.picker_trace_confined:
                    print("  picker trace [lo]:")
                    for t in p.picker_trace_confined:
                        print(f"    - {t}")
                if p.picker_trace_middle:
                    print("  picker trace [middle]:")
                    for t in p.picker_trace_middle:
                        print(f"    - {t}")
            # Steel quantities for this mode (uses confined-zone hoops)
            from design.columns.optimize import (
                longitudinal_steel_quantity, transverse_steel_quantity,
            )
            rho_l_m = longitudinal_steel_quantity(self)
            rho_t_m = transverse_steel_quantity(
                self,
                db_hoop=p.db_hoop,
                n_legs_x=p.n_legs_x,
                n_legs_y=p.n_legs_y,
                spacing=p.spacing_confined,
            )
            print(f"  steel quantities (kg/m^3):  "
                  f"rho_long = {rho_l_m:.1f}    "
                  f"rho_trans = {rho_t_m:.1f}    "
                  f"rho_total = {rho_l_m + rho_t_m:.1f}")
            # Skip envelope notes — they would duplicate per-mode notes
            # because _design_envelope already prefixes them with [mode].
            if m == "envelope":
                continue
            for n in p.notes:
                all_notes.append((m, n))

        if all_notes:
            print()
            print("-" * 74)
            print("  Notes:")
            for m, n in all_notes:
                print(f"    [{m}] {n}")
        print("=" * 74)

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
                "k": self.k,
                "spiral": self.spiral,
                "seismic": self.seismic,
                "hx_mm": self.hx,
                "fyt_MPa": self.fyt,
                "s_mm": self.s,
            },
            "confinement": self._confinement_report_dict(),
        }

    def _confinement_report_dict(self) -> dict:
        """Confinement summary for report(): nl and (spiral) code-minimum ρ_s.

        ``rho_s_required_min`` uses §25.7.3.3 expressions (d)/(e) at zero
        axial (the deterministic code minimum, independent of demands);
        demand/capacity-driven ρ_s (expression (f)) is reported per-mode by
        ``design()``/``print_design``. None for tied columns.
        """
        s = self.section
        Ag = s.gross_area()
        cover = getattr(s, "clear_cover", 25.0)
        bc = max(s.width() - 2 * cover, 100.0)
        hc = max(s.height() - 2 * cover, 100.0)
        Ach = bc * hc
        d: dict = {"nl": self._count_perimeter_bars()}
        if self.spiral:
            d["rho_s_required_min"] = rho_s_spiral_required(
                Ag=Ag, Ach=Ach, fc=s.concrete.fc, fyt=self.fyt,
            )
        return d

    def __repr__(self) -> str:
        s = self.section
        u = self.units
        return (
            f"Column(label={self.label!r}, "
            f"section={u.fmt_length(s.width(), unit=False)}x{u.fmt_length(s.height())}, "
            f"fc={s.concrete.fc:.0f} MPa, units={u.name}, "
            f"run={'yes' if self._surface is not None else 'no'})"
        )
