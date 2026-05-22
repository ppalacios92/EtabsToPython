"""Concrete wall — composition of web + (optional) boundary elements.

The Wall wraps a `WallSection` and exposes the bidirectional contract
shared by every element in this package:

    wall.capacity()             -> WallCapacity
    wall.check(demands)         -> WallCheck
    wall.design(demands, ...)   -> WallDesignResults

The wall is **immutable**. Any change to geometry or detailing produces
a NEW `Wall` instance through `wall.evolve(proposal)`. The original
wall stays untouched, so the design history is fully inspectable.

Boundary elements are *real* `Column` objects (`wall.be_top`,
`wall.be_bot`). The columns module is reused as the BE detailing engine
— Ash, hoops, hx, lo, Mpr, Ve are all delegated. The web is detailed
separately (rho_l, rho_t, two-curtain).

ACI 318-25 references:
    §11.5.4   Acv definition (in-plane shear)
    §11.6     Minimum reinforcement (rho_l, rho_t)
    §11.7     Spacing / two-curtain limits
    §18.10.2  Distributed reinforcement of special walls
    §18.10.3  Capacity-design shear (Ve, omega_v)
    §18.10.4  Vn of the wall (alpha_c)
    §18.10.6  Special boundary elements
    §18.10.7  Coupling beams
    §18.10.8  Wall piers
    §22.5     Vc detailed
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from math import ceil, pi
from typing import Optional

from design.common.contracts import wall_demands_to_be
from design.common.factors import phi_shear
from design.common.materials import Bar, Concrete, Steel
from design.common.plot_style import PlotStyle
from design.common.shear import (
    alpha_c, vn_wall, vn_wall_max, ve_in_plane_capacity_wall,
)
from design.common.units import UnitSystem, units_from, DEFAULT_UNITS
from design.sections.wall import (
    WallSection, BoundaryElementSection, be_perimeter_bars,
)
from design.sections.reinforcement import RebarGroup, RebarLayout, perimeter_bars
from design.columns.column import Column, ColumnDesignProposal
from design.columns.bar_schedule import BarSchedule
from design.columns.interaction import InteractionDiagram
from design.columns.interaction_surface import (
    InteractionSurface, interaction_surface,
)
from design.columns.surface import Surface
from design.columns.proxies import _DiagramFieldView

from design.walls.shear import (
    omega_v as default_omega_v,
    av_s_required_wall,
)
from design.walls.boundary import (
    boundary_element_required_displacement,
    boundary_element_required_stress,
    boundary_extension_length,
    be_thickness_minimum,
    propose_be_geometry, c_at_demand,
    BEGeometryProposal,
)
from design.walls.distributed import (
    rho_min_distributed, web_bar_spacing_max, two_curtain_required,
    web_bars_for_rho, web_rho_provided,
)
from design.walls.probable import mpr_in_plane


# ====================================================================== #
# Dataclasses
# ====================================================================== #
@dataclass(frozen=True, slots=True)
class WallDemands:
    Pu: float = 0.0           # axial (kN, + compression)
    Mu: float = 0.0           # in-plane moment (kN·m)
    Mu_out: float = 0.0       # out-of-plane moment (kN·m)
    Vu: float = 0.0           # in-plane shear (kN)
    Vu_out: float = 0.0       # out-of-plane shear (kN)
    delta_u: float = 0.0      # inelastic top displacement (mm) — §18.10.6.2
    sigma_max: float = 0.0    # extreme compressive fiber stress (MPa) — §18.10.6.3


@dataclass(frozen=True, slots=True)
class WallCapacity:
    Po: float
    To: float
    phi_Pn_max: float
    surface: Surface
    diagram: InteractionDiagram                # in-plane (angle 0)
    diagram_out: Optional[InteractionDiagram]  # out-of-plane (angle 90)
    phi_Vn: float
    Vn_max: float
    rho_t_provided: float
    rho_l_provided: float
    rho_l_min: float
    rho_t_min: float
    double_curtain_required: bool


@dataclass(frozen=True, slots=True)
class WallCheck:
    capacity: WallCapacity
    demands: WallDemands
    ratio_pm: float
    ratio_shear: float
    ratio_shear_out: Optional[float]
    be_required_disp: bool
    be_required_stress: bool
    be_length_required: float
    be_thickness_min: float
    distributed_rho_ok: bool
    double_curtain_ok: bool
    c_at_demand: float
    passed: bool


@dataclass(frozen=True, slots=True)
class WallDesignProposal:
    """Detailing proposal for ONE design mode."""
    mode: str                                # 'minimum' | 'demand' | 'capacity' | 'envelope'

    # ---- Web distributed reinforcement (§18.10.2 / §11.6 / §11.7) ----
    rho_l_required: float
    rho_t_required: float
    rho_l_provided: float
    rho_t_provided: float
    double_curtain_required: bool
    web_bar_db: float
    web_bar_spacing: float
    web_bar_layers: int

    # ---- BE decision (§18.10.6) ----
    be_required: bool
    be_required_disp: bool
    be_required_stress: bool
    be_length_proposed: float
    be_thickness_proposed: float
    be_extension_above: float
    be_extension_below: float
    c_used: float

    # ---- BE confinement (delegated to columns.Column) ----
    be_top_proposal: Optional[ColumnDesignProposal] = None
    be_bot_proposal: Optional[ColumnDesignProposal] = None

    # ---- Capacity-design shear (§18.10.3) ----
    Mpr_in_plane: Optional[float] = None
    Pu_for_Mpr: Optional[float] = None
    omega_v: Optional[float] = None
    Ve_capacity: Optional[float] = None
    Ve_used: Optional[float] = None
    amplification: Optional[float] = None

    # ---- Shear of the proposed wall (§18.10.4 / §22.5) ----
    alpha_c_factor: float = 0.0
    phi_Vn: float = 0.0
    phi_Vn_max: float = 0.0
    av_s_required: Optional[float] = None
    ratio_Ve_over_phiVn: Optional[float] = None

    # ---- Demand context ----
    Pu_used: Optional[float] = None
    Pu_over_Ag_fc: Optional[float] = None

    # ---- Geometry handed to evolve() ----
    proposed_section: Optional[WallSection] = None

    # ---- Diagnostics ----
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WallDesignResults:
    minimum:    WallDesignProposal
    demand:     Optional[WallDesignProposal]
    capacity:   WallDesignProposal
    envelope:   WallDesignProposal
    original_wall: "Wall"
    final_wall:    "Wall"
    history:    tuple["Wall", ...] = ()
    iterations: int = 1
    converged:  bool = True


# ====================================================================== #
# Wall class
# ====================================================================== #
class Wall:
    """Concrete wall (in-plane + out-of-plane).

    Immutable composition of:
        - `section` : WallSection (web + optional BE)
        - `be_top`  : Column or None — BE at +lw/2 end
        - `be_bot`  : Column or None — BE at -lw/2 end

    Internal numerics: MPa, mm, kN, kN·m. Display units configurable
    via `wall.set_units(code_or_name)`.
    """

    __slots__ = (
        "section", "hw", "lu",
        "rho_t", "rho_l", "fyt",
        "seismic", "omega_v_factor", "hx",
        "transverse_bar_diameter", "web_bar_diameter",
        "n_x_be",                                       # bars across BE thickness — §18.10.6.4(b)
        "be_db_default", "be_n_y_default",              # as-built BE bar size + count (for optimize)
        "label", "bar_schedule", "units", "run_settings", "style",
        "_surface", "_inner_surface",
        "_Po", "_To", "_phi_Pn_max", "_phi_Vn",
        "_be_top", "_be_bot",
        "_plotter", "design_results",
    )

    def __init__(
        self,
        *,
        section: WallSection,
        hw: float,
        lu: float | None = None,
        rho_t: float = 0.0025,
        rho_l: float = 0.0025,
        fyt: float = 420.0,
        seismic: bool = True,
        omega_v_factor: float = 1.5,
        hx: float = 200.0,
        transverse_bar_diameter: float = 10.0,
        web_bar_diameter: float = 10.0,
        n_x_be: int = 3,
        be_db_default: float = 22.0,
        be_n_y_default: int = 5,
        units: int | str | UnitSystem | None = None,
        bar_schedule: BarSchedule | None = None,
        label: str = "Wall",
        run_settings: dict | None = None,
    ) -> None:
        self.section = section
        self.hw = hw
        self.lu = lu if lu is not None else hw
        self.rho_t = rho_t
        self.rho_l = rho_l
        self.fyt = fyt
        self.seismic = seismic
        self.omega_v_factor = omega_v_factor
        self.hx = hx
        self.transverse_bar_diameter = transverse_bar_diameter
        self.web_bar_diameter = web_bar_diameter
        # ACI 318-25 §18.10.6.4(b) — column-style BE detailing. The minimum
        # is 3 (one interior bar on each long face that can be tied as a
        # crosstie). For barbell walls (be_thickness >> tw) use 4 or more.
        if n_x_be < 3:
            raise ValueError(
                f"n_x_be={n_x_be} violates ACI 318-25 §18.10.6.4(b) — "
                "boundary elements need column-style detailing (n_x_be >= 3)."
            )
        self.n_x_be = n_x_be
        # As-built BE longitudinal detailing — kept on the wall so the
        # optimizer can use it as the baseline reference (the bars
        # themselves live in section.rebar).
        self.be_db_default = float(be_db_default)
        self.be_n_y_default = int(be_n_y_default)
        self.label = label
        self.bar_schedule = bar_schedule if bar_schedule is not None else BarSchedule()

        if units is None:
            self.units = DEFAULT_UNITS
        elif isinstance(units, UnitSystem):
            self.units = units
        else:
            self.units = units_from(units)

        self.run_settings: dict = {"n_angles": 19, "n_points": 60}
        if run_settings:
            self.run_settings.update(run_settings)

        # plotting style (shared with all element plotters via common.PlotStyle)
        self.style: PlotStyle = PlotStyle()

        # caches
        self._surface: Optional[Surface] = None
        self._inner_surface: Optional[InteractionSurface] = None
        self._Po: Optional[float] = None
        self._To: Optional[float] = None
        self._phi_Pn_max: Optional[float] = None
        self._phi_Vn: Optional[float] = None
        self._be_top: Optional[Column] = None
        self._be_bot: Optional[Column] = None
        self._plotter = None
        self.design_results: Optional[WallDesignResults] = None

    # ================================================================== #
    # One-shot factory — scalar inputs, builds section + wall
    # ================================================================== #
    @classmethod
    def rectangular(
        cls,
        *,
        # --- geometry (mm) ---
        lw: float,
        tw: float,
        hw: float,
        lu: float | None = None,
        # --- materials ---
        concrete: Concrete,
        steel: Steel,
        # --- web vertical reinforcement (perimeter through the web) ---
        n_y_web: int = 8,
        db_web: float = 10.0,
        cover: float = 25.0,
        # --- boundary elements (optional; mm) ---
        be_top_length: float = 0.0,
        be_top_thickness: float | None = None,
        be_bot_length: float = 0.0,
        be_bot_thickness: float | None = None,
        # asymmetric (C-shape / T-shape) BE alignment:
        #   'center' — symmetric barbell (default).
        #   'left'   — BE protrudes only to the LEFT (BE's right face
        #              flush with the web's right face).
        #   'right'  — BE protrudes only to the RIGHT (BE's left face
        #              flush with the web's left face).
        # Apply to both BEs at once via ``be_align`` OR per-end via
        # ``be_top_align`` / ``be_bot_align``. Per-end values win.
        be_align: str | None = None,
        be_top_align: str | None = None,
        be_bot_align: str | None = None,
        n_x_be: int = 3,                       # bars across BE thickness — §18.10.6.4(b)
        n_y_be: int = 4,                       # bars along BE length
        db_be: float = 20.0,
        # --- transverse / seismic ---
        db_stirrup: float = 10.0,
        fyt: float | None = None,
        seismic: bool = True,
        omega_v_factor: float = 1.5,
        hx: float = 200.0,
        # --- distributed-reinforcement defaults ---
        rho_t: float = 0.0025,
        rho_l: float = 0.0025,
        # --- presentation ---
        units: int | str | UnitSystem | None = None,
        bar_schedule: "BarSchedule | None" = None,
        label: str = "Wall",
        run_settings: dict | None = None,
    ) -> "Wall":
        """Build a Wall in a single call from scalar specs.

        Parity with ``Beam.rectangular()`` and ``Column.rectangular()``.
        Canonical units per ``design/AGENTS.md`` §1: mm everywhere.

        ``hw`` / ``lu`` accept metres as a back-compat convenience (any
        value < 30 is treated as metres and converted to mm with a
        DeprecationWarning); pass mm explicitly to silence.

        Web reinforcement is laid out as a perimeter cage of ``n_y_web``
        bars per face (left/right) with vertical bars at the long faces.
        BE reinforcement (when ``be_*_length > 0``) is a 2×``n_y_be``
        perimeter rectangle at each end. ``be_*_thickness`` defaults to
        ``tw`` if not provided.

        Example:

            wall = Wall.rectangular(
                lw=4000, tw=300, hw=12000, lu=3000,
                concrete=Concrete(fc=35), steel=Steel(fy=420),
                n_y_web=10, db_web=12,
                be_top_length=600, n_y_be=4, db_be=22,
                be_bot_length=600,
                cover=40, seismic=True,
            )
        """
        # --- legacy metres -> mm with DeprecationWarning ---
        import warnings
        def _coerce_mm(val: float | None, name: str) -> float | None:
            if val is None:
                return None
            if 0 < val < 30:
                warnings.warn(
                    f"Wall.rectangular({name}={val}) looks like metres; "
                    f"converting to {val * 1000:.0f} mm. Pass mm directly "
                    "to silence this warning.",
                    DeprecationWarning, stacklevel=3,
                )
                return float(val) * 1000.0
            return float(val)

        hw_mm = _coerce_mm(hw, "hw")
        lu_mm = _coerce_mm(lu, "lu") if lu is not None else hw_mm

        # --- thickness defaults for BE ---
        bt_top = be_top_thickness if be_top_thickness is not None else tw
        bt_bot = be_bot_thickness if be_bot_thickness is not None else tw

        # --- alignment -> x-offset (mm) ---
        # Convention: a positive offset moves the BE toward +x.
        #   align='left'  -> BE protrudes only to the LEFT  -> offset = -(be_t - tw)/2  (negative)
        #   align='right' -> BE protrudes only to the RIGHT -> offset = +(be_t - tw)/2  (positive)
        #   align='center' (default) -> offset = 0 (symmetric barbell).
        def _offset_from_align(align: str | None, be_t: float) -> float:
            tag = align if align is not None else "center"
            tag = tag.lower()
            if tag in ("center", "centered", "c"):
                return 0.0
            if be_t <= tw:
                # BE not wider than web — alignment has no visible effect.
                return 0.0
            delta = (be_t - tw) / 2.0
            if tag in ("left", "l"):
                return -delta
            if tag in ("right", "r"):
                return +delta
            raise ValueError(
                f"Unknown be align={align!r}; expected "
                "'left' | 'center' | 'right'."
            )
        top_align = be_top_align if be_top_align is not None else be_align
        bot_align = be_bot_align if be_bot_align is not None else be_align
        top_x = _offset_from_align(top_align, bt_top)
        bot_x = _offset_from_align(bot_align, bt_bot)

        # --- web vertical bars: perimeter cage through the web portion ---
        # n_x=2 (one row per long face), n_y = n_y_web bars along the wall
        # length (in mm). cover is to the bar centroid, computed from clear
        # cover + stirrup + db_web/2.
        cover_centroid_web = cover + db_stirrup + db_web / 2.0
        web_lw = lw - be_top_length - be_bot_length
        web_center_y = (be_bot_length - be_top_length) / 2.0
        web_group = perimeter_bars(
            b=tw, h=web_lw, cover=cover_centroid_web,
            n_x=2, n_y=n_y_web,
            bar=Bar(diameter=db_web), steel=steel,
            center=(0.0, web_center_y),
        )

        groups: list[RebarGroup] = [web_group]

        # --- BE perimeter cages (if requested) ---
        # ACI 318-25 §18.10.6.4(b) requires column-style detailing in the
        # BE. n_x_be is the number of longitudinal bars across the BE
        # thickness — minimum 3 (default) to allow at least one interior
        # bar that can be tied as a crosstie. Many engineers use n_x_be=4
        # in barbell walls (be_thickness > tw).
        cover_centroid_be = cover + db_stirrup + db_be / 2.0
        if be_top_length > 0:
            cy_top = lw / 2 - be_top_length / 2
            top_be = be_perimeter_bars(
                be_thickness=bt_top, be_length=be_top_length,
                center_x=top_x, center_y=cy_top,
                cover=cover_centroid_be,
                n_x=n_x_be, n_y=n_y_be,
                bar=Bar(diameter=db_be), steel=steel,
            )
            groups.append(top_be)

        if be_bot_length > 0:
            cy_bot = -lw / 2 + be_bot_length / 2
            bot_be = be_perimeter_bars(
                be_thickness=bt_bot, be_length=be_bot_length,
                center_x=bot_x, center_y=cy_bot,
                cover=cover_centroid_be,
                n_x=n_x_be, n_y=n_y_be,
                bar=Bar(diameter=db_be), steel=steel,
            )
            groups.append(bot_be)

        section = WallSection(
            lw=lw, tw=tw, concrete=concrete,
            rebar=RebarLayout(groups=tuple(groups)),
            be_length_top=be_top_length,
            be_length_bot=be_bot_length,
            be_thickness_top=bt_top,
            be_thickness_bot=bt_bot,
            be_top_x_offset=top_x,
            be_bot_x_offset=bot_x,
        )

        fyt_val = fyt if fyt is not None else steel.fy

        return cls(
            section=section,
            hw=hw_mm,
            lu=lu_mm,
            rho_t=rho_t, rho_l=rho_l,
            fyt=fyt_val,
            seismic=seismic,
            omega_v_factor=omega_v_factor,
            hx=hx,
            transverse_bar_diameter=db_stirrup,
            web_bar_diameter=db_web,
            n_x_be=n_x_be,
            be_db_default=db_be,
            be_n_y_default=n_y_be,
            units=units,
            bar_schedule=bar_schedule,
            label=label,
            run_settings=run_settings,
        )

    # ================================================================== #
    # Configuration helpers
    # ================================================================== #
    def set_units(self, code_or_name: int | str) -> "Wall":
        self.units = units_from(code_or_name)
        return self

    @property
    def hw_m(self) -> float:
        """Wall height in metres — legacy view (canonical unit is mm)."""
        return self.hw / 1000.0

    @property
    def lu_m(self) -> float:
        """Unbraced length in metres — legacy view (canonical unit is mm)."""
        return self.lu / 1000.0

    def run_optimize(
        self,
        demands: "WallDemands | None" = None,
        *,
        db_web_list: list[float] | None = None,
        spacing_list: list[float] | None = None,
        layers_options: tuple[int, ...] = (1, 2),
        include_infeasible: bool = False,
        sort_by: str = "rho_total",
    ):
        """Explore web detailing alternatives around the ``design()`` envelope.

        Returns a list of ``OptimizeAlternative`` ranked by ``sort_by``
        (default ``rho_total`` — kg/m³ of concrete). The baseline (the
        envelope of ``wall.design(demands)``) is always included and
        marked with ``is_baseline=True``.

        Parity with ``Column.run_optimize()`` and
        ``Beam.run_optimize()`` — see ``design.walls.optimize.run_optimize``
        for details.
        """
        from design.walls.optimize import run_optimize as _opt
        return _opt(
            self, demands,
            db_web_list=db_web_list,
            spacing_list=spacing_list,
            layers_options=layers_options,
            include_infeasible=include_infeasible,
            sort_by=sort_by,
        )

    def clear_cache(self) -> None:
        self._surface = None
        self._inner_surface = None
        self._Po = self._To = self._phi_Pn_max = self._phi_Vn = None
        self._be_top = self._be_bot = None

    # ================================================================== #
    # Lazy compute
    # ================================================================== #
    def run(
        self,
        *,
        n_angles: int | None = None,
        n_points: int | None = None,
    ) -> "Wall":
        n_a = n_angles if n_angles is not None else self.run_settings.get("n_angles", 19)
        n_p = n_points if n_points is not None else self.run_settings.get("n_points", 60)

        self._inner_surface = interaction_surface(
            self.section, n_angles=n_a, n_points_per_curve=n_p, spiral=False,
        )
        self._surface = Surface(self._inner_surface)

        d0 = self._inner_surface.at_angle(0.0)
        self._Po = d0.Po
        self._To = d0.To
        self._phi_Pn_max = d0.Pn_max_phi

        # In-plane shear capacity
        s = self.section
        hw_over_lw = self.hw / s.lw if s.lw > 0 else float("inf")
        Vn = vn_wall(
            Acv=s.Acv, fc=s.concrete.fc,
            rho_t=self.rho_t, fyt=self.fyt,
            hw_over_lw=hw_over_lw, lam=s.concrete.lam,
        )
        Vn_cap = vn_wall_max(Acv=s.Acv, fc=s.concrete.fc)
        self._phi_Vn = phi_shear() * min(Vn, Vn_cap)
        return self

    def _ensure_run(self) -> None:
        if self._surface is None:
            self.run()

    # ================================================================== #
    # Surface, scalars, proxies
    # ================================================================== #
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
    def Vn_max(self) -> float:
        return vn_wall_max(Acv=self.section.Acv, fc=self.section.concrete.fc)

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

    # ================================================================== #
    # Boundary elements as Column instances
    # ================================================================== #
    @property
    def has_boundary_elements(self) -> bool:
        return self.section.has_boundary_elements

    @property
    def be_top(self) -> Optional[Column]:
        if self.section.be_length_top <= 0:
            return None
        if self._be_top is None:
            self._be_top = self._build_be_column("top")
        return self._be_top

    @property
    def be_bot(self) -> Optional[Column]:
        if self.section.be_length_bot <= 0:
            return None
        if self._be_bot is None:
            self._be_bot = self._build_be_column("bot")
        return self._be_bot

    def _build_be_column(self, which: str) -> Column:
        be_section = (
            self.section.be_top_as_rectangular() if which == "top"
            else self.section.be_bot_as_rectangular()
        )
        assert be_section is not None
        return Column(
            section=be_section,
            lu=self.lu,                              # mm (Fase 1A migrates Column to mm)
            seismic=self.seismic,
            fyt=self.fyt,
            hx=self.hx,
            transverse_bar_diameter=self.transverse_bar_diameter,
            bar_schedule=self.bar_schedule,
            units=self.units,
            label=f"{self.label}-BE-{which}",
        )

    @property
    def plot(self):
        if self._plotter is None:
            from design.walls.plotting import _WallPlotter
            self._plotter = _WallPlotter(self)
        return self._plotter

    # ================================================================== #
    # Back-compat capacity()
    # ================================================================== #
    def capacity(
        self,
        *,
        biaxial: bool = False,
        n_angles: int = 19,
        n_points: int = 60,
    ) -> WallCapacity:
        if (self._surface is None
                or n_angles != self.run_settings.get("n_angles")):
            self.run_settings["n_angles"] = n_angles
            self.run_settings["n_points"] = n_points
            self.run(n_angles=n_angles, n_points=n_points)

        s = self.section
        Vu_kN = 0.0  # capacity() has no demands — use 0 for the rho-min branch
        rho_l_min, rho_t_min = rho_min_distributed(
            Vu=Vu_kN, Acv=s.Acv, fc=s.concrete.fc, lam=s.concrete.lam,
        )
        hw_over_lw = self.hw / s.lw if s.lw > 0 else float("inf")
        dc = two_curtain_required(
            Vu=Vu_kN, Acv=s.Acv, fc=s.concrete.fc, lam=s.concrete.lam,
            hw_over_lw=hw_over_lw, tw=s.tw,
        )

        diagram_in = self.surface[0.0]
        diagram_out = self.surface[90.0] if biaxial else None

        return WallCapacity(
            Po=self.Po, To=self.To, phi_Pn_max=self.phi_Pn_max,
            surface=self.surface,
            diagram=diagram_in,
            diagram_out=diagram_out,
            phi_Vn=self.phi_Vn,
            Vn_max=self.Vn_max,
            rho_t_provided=self.rho_t,
            rho_l_provided=self.rho_l,
            rho_l_min=rho_l_min,
            rho_t_min=rho_t_min,
            double_curtain_required=dc,
        )

    # ================================================================== #
    # check(demands)
    # ================================================================== #
    def check(self, demands: WallDemands) -> WallCheck:
        biaxial = abs(demands.Mu_out) > 0
        cap = self.capacity(biaxial=biaxial)
        s = self.section

        # PM ratio
        if biaxial:
            _, ratio_pm = self.surface.check(
                Pu=demands.Pu, Mux=demands.Mu, Muy=demands.Mu_out,
            )
        else:
            from design.columns.interaction import demand_inside_envelope
            _, ratio_pm = demand_inside_envelope(
                cap.diagram, Pu=demands.Pu, Mu=demands.Mu,
            )

        # In-plane shear ratio
        ratio_v = (demands.Vu / cap.phi_Vn) if cap.phi_Vn > 0 else float("inf")
        ratio_v_out = None  # out-of-plane shear: conservative, not modeled here

        # BE checks based on c at demand
        c_dem = c_at_demand(cap.diagram, Pu=demands.Pu, Mu=demands.Mu)
        be_d = boundary_element_required_displacement(
            c=c_dem, lw=s.lw, delta_u=demands.delta_u, hw=self.hw,
        )
        be_s = boundary_element_required_stress(
            sigma_max_compressive=demands.sigma_max, fc=s.concrete.fc,
        )
        be_len = boundary_extension_length(c=c_dem, lw=s.lw)
        hw_over_lw = self.hw / s.lw if s.lw > 0 else float("inf")
        be_t = be_thickness_minimum(
            lu=self.lu, hw_over_lw=hw_over_lw, c=c_dem, lw=s.lw,
        )

        distributed_ok = (
            self.rho_l >= cap.rho_l_min and self.rho_t >= cap.rho_t_min
        )
        dc_required = two_curtain_required(
            Vu=demands.Vu, Acv=s.Acv, fc=s.concrete.fc, lam=s.concrete.lam,
            hw_over_lw=hw_over_lw, tw=s.tw,
        )
        # double-curtain detail decision lives outside this class; we just
        # warn if the wall is too thin for it to be possible
        double_curtain_ok = (not dc_required) or s.tw >= 200.0

        be_present = self.has_boundary_elements
        be_ok = (not (be_d or be_s)) or be_present

        passed = (
            ratio_pm <= 1.0 and ratio_v <= 1.0
            and distributed_ok and double_curtain_ok and be_ok
        )

        return WallCheck(
            capacity=cap, demands=demands,
            ratio_pm=ratio_pm,
            ratio_shear=ratio_v,
            ratio_shear_out=ratio_v_out,
            be_required_disp=be_d,
            be_required_stress=be_s,
            be_length_required=be_len,
            be_thickness_min=be_t,
            distributed_rho_ok=distributed_ok,
            double_curtain_ok=double_curtain_ok,
            c_at_demand=c_dem,
            passed=passed,
        )

    # ================================================================== #
    # design — three modes, optionally iterates to update geometry
    # ================================================================== #
    def design(
        self,
        demands: WallDemands | None = None,
        *,
        auto_update: bool = True,
        max_iter: int = 5,
        tolerance: float = 25.0,
        clear_cover: float = 25.0,
    ) -> WallDesignResults:
        """Three-mode design with optional BE auto-update.

        Modes:
            minimum   ACI floor: rho_min, two-curtain, no BE forced
            demand    Uses demands.Vu for Av/s; demands for §18.10.6 checks
            capacity  Mpr-based Ve; Ash for BE computed at Pu_for_Mpr

        If `auto_update` is True and a BE is required but missing
        (or too short), the wall is rebuilt with the proposed BE and the
        whole design re-runs. The loop converges when the BE length
        stabilizes within `tolerance` (default 25 mm).
        """
        history: list[Wall] = [self]
        current = self
        last_proposal: Optional[WallDesignProposal] = None
        converged = True
        iterations = 1

        for it in range(max_iter):
            iterations = it + 1
            results = current._design_modes(
                demands=demands, clear_cover=clear_cover,
            )
            last_results = results
            envelope = results.envelope
            last_proposal = envelope

            if (not auto_update) or (demands is None):
                break

            new_section = envelope.proposed_section
            if new_section is None:
                break

            # Convergence: BE length stable to tolerance
            cur_top = current.section.be_length_top
            cur_bot = current.section.be_length_bot
            new_top = new_section.be_length_top
            new_bot = new_section.be_length_bot
            if (abs(new_top - cur_top) <= tolerance
                    and abs(new_bot - cur_bot) <= tolerance):
                break

            current = current._with_section(new_section)
            history.append(current)
        else:
            converged = False

        final_wall = history[-1]
        # Re-run on the final wall to capture its final design proposal
        final_results = final_wall._design_modes(
            demands=demands, clear_cover=clear_cover,
        )

        out = WallDesignResults(
            minimum=final_results.minimum,
            demand=final_results.demand,
            capacity=final_results.capacity,
            envelope=final_results.envelope,
            original_wall=self,
            final_wall=final_wall,
            history=tuple(history),
            iterations=iterations,
            converged=converged,
        )
        # Per AGENTS.md §4: design() does NOT mutate self.design_results.
        # The caller stores the return value if it needs to keep it.
        return out

    def _design_modes(
        self,
        *,
        demands: WallDemands | None,
        clear_cover: float,
    ) -> WallDesignResults:
        """Build minimum / demand / capacity / envelope on THIS wall (no iteration)."""
        prop_min = self._design_one(mode="minimum", demands=None,
                                    clear_cover=clear_cover)
        prop_dem = (
            self._design_one(mode="demand", demands=demands, clear_cover=clear_cover)
            if demands is not None else None
        )
        prop_cap = self._design_one(mode="capacity", demands=demands,
                                    clear_cover=clear_cover)
        prop_env = self._design_envelope(prop_min, prop_dem, prop_cap)

        return WallDesignResults(
            minimum=prop_min, demand=prop_dem,
            capacity=prop_cap, envelope=prop_env,
            original_wall=self, final_wall=self,
            history=(self,), iterations=1, converged=True,
        )

    # ---- one mode ---------------------------------------------------- #
    def _design_one(
        self,
        *,
        mode: str,
        demands: WallDemands | None,
        clear_cover: float,
    ) -> WallDesignProposal:
        s = self.section
        c = s.concrete
        notes: list[str] = []

        Vu = demands.Vu if demands is not None else 0.0
        hw_over_lw = self.hw / s.lw if s.lw > 0 else float("inf")
        d_eff = 0.8 * s.lw

        # ---- Distributed reinforcement (§18.10.2 / §11.6) ----
        rho_l_min, rho_t_min = rho_min_distributed(
            Vu=Vu, Acv=s.Acv, fc=c.fc, lam=c.lam,
        )
        # Required rho_t to cover Vu (the simplified path of §18.10.4.1).
        # Vc_term is the concrete contribution of Vn_wall (rho_t=0).
        # See common.shear.vn_wall — equals a_c * lambda * sqrt(fc) * Acv / 1000.
        a_c = alpha_c(hw_over_lw)
        Vc_term = vn_wall(
            Acv=s.Acv, fc=c.fc, rho_t=0.0, fyt=self.fyt,
            hw_over_lw=hw_over_lw, lam=c.lam,
        )
        rho_t_req_for_v = max(
            0.0,
            (Vu / phi_shear() - Vc_term) / (self.fyt * s.Acv / 1000.0),
        )
        rho_t_req = max(rho_t_min, rho_t_req_for_v)
        rho_l_req = rho_l_min

        layers = 2 if two_curtain_required(
            Vu=Vu, Acv=s.Acv, fc=c.fc, lam=c.lam,
            hw_over_lw=hw_over_lw, tw=s.tw,
        ) else 1
        spacing_max = web_bar_spacing_max(lw=s.lw, tw=s.tw)
        spacing_proposed = web_bars_for_rho(
            rho=rho_t_req, tw=s.tw, db=self.web_bar_diameter, layers=layers,
        )
        spacing_final = min(spacing_max, spacing_proposed)
        rho_t_prov = web_rho_provided(
            tw=s.tw, db=self.web_bar_diameter, spacing=spacing_final, layers=layers,
        )
        rho_l_prov = rho_t_prov  # in walls long/transv minima are equal

        # ---- Capacity / shear (§18.10.4) ----
        Vn_provided = vn_wall(
            Acv=s.Acv, fc=c.fc, rho_t=rho_t_prov, fyt=self.fyt,
            hw_over_lw=hw_over_lw, lam=c.lam,
        )
        Vn_cap = vn_wall_max(Acv=s.Acv, fc=c.fc)
        phi_Vn = phi_shear() * min(Vn_provided, Vn_cap)
        phi_Vn_max = phi_shear() * Vn_cap

        # ---- Mpr / Ve (§18.10.3) ----
        Mpr_ip = Pu_for_Mpr = None
        Ve_cap_kN = Ve_used = amplif = None
        omega = self.omega_v_factor

        if mode == "capacity":
            Mpr_ip, Pu_for_Mpr = mpr_in_plane(s)
            Mu_dem = demands.Mu if demands is not None else None
            Vu_dem = demands.Vu if demands is not None else None
            Ve_cap_kN = ve_in_plane_capacity_wall(
                Mpr=Mpr_ip, Mu=Mu_dem, Vu=Vu_dem, omega_v_factor=omega,
            )
            Ve_used = Ve_cap_kN
            if Vu_dem and Vu_dem > 0:
                amplif = Ve_used / Vu_dem
            if Ve_used > phi_Vn_max:
                notes.append(
                    f"Ve_used {Ve_used:.0f} kN exceeds phi*Vn_max {phi_Vn_max:.0f} kN "
                    f"(§18.10.4.4) — wall is shear-undersized."
                )

        # ---- Av/s required for the shear demand ----
        Vu_for_avs = (
            Ve_used if mode == "capacity"
            else (Vu if mode == "demand" else None)
        )
        av_s_req = av_s_required_wall(
            Vu=Vu_for_avs if Vu_for_avs else 0.0,
            Vc=Vc_term, fyt=self.fyt, d_eff=d_eff,
        )
        ratio_Ve_phi = (
            Ve_used / phi_Vn if (Ve_used is not None and phi_Vn > 0) else None
        )

        # ---- BE checks (§18.10.6) ----
        diagram = self.surface[0.0]
        if demands is not None:
            c_dem = c_at_demand(diagram, Pu=demands.Pu, Mu=demands.Mu)
            be_proposal = propose_be_geometry(
                c=c_dem, lw=s.lw, tw=s.tw, hw=self.hw, lu=self.lu,
                delta_u=demands.delta_u,
                sigma_max_compressive=demands.sigma_max,
                fc=c.fc,
            )
        else:
            # Without demands we cannot evaluate §18.10.6.2 — use balanced c
            c_dem = diagram.balanced_point().c
            be_proposal = BEGeometryProposal(
                length=0.0, thickness=0.0,
                reason="none", c_used=c_dem,
                extension_above=0.0, extension_below=0.0,
                required=False,
            )

        # ---- Build proposed section if BE is needed and current is short ----
        proposed_section: Optional[WallSection] = None
        if be_proposal.required and mode in ("demand", "capacity"):
            need_top = be_proposal.length > s.be_length_top + 1.0
            need_bot = be_proposal.length > s.be_length_bot + 1.0
            if need_top or need_bot:
                proposed_section = self._build_be_section(
                    be_proposal, clear_cover=clear_cover,
                )
                notes.append(
                    f"[{mode}] Proposed BE length = {be_proposal.length:.0f} mm "
                    f"(reason: {be_proposal.reason}, c = {c_dem:.0f} mm)."
                )

        # ---- Detail each BE as a column proposal (if BE exists) ----
        be_top_prop = be_bot_prop = None
        col_demands = wall_demands_to_be(demands) if demands is not None else None
        if self.be_top is not None:
            be_top_prop = self.be_top.design(
                col_demands, clear_cover=clear_cover,
            ).envelope
        if self.be_bot is not None:
            be_bot_prop = self.be_bot.design(
                col_demands, clear_cover=clear_cover,
            ).envelope

        Pu_used = demands.Pu if (mode == "demand" and demands is not None) else (
            Pu_for_Mpr if mode == "capacity" else None
        )
        Pu_ratio = (
            (Pu_used * 1000.0) / (s.gross_area() * c.fc)
            if (Pu_used is not None and s.gross_area() > 0) else None
        )

        return WallDesignProposal(
            mode=mode,
            rho_l_required=rho_l_req,
            rho_t_required=rho_t_req,
            rho_l_provided=rho_l_prov,
            rho_t_provided=rho_t_prov,
            double_curtain_required=(layers == 2),
            web_bar_db=self.web_bar_diameter,
            web_bar_spacing=spacing_final,
            web_bar_layers=layers,
            be_required=be_proposal.required,
            be_required_disp=(be_proposal.reason == "displacement"),
            be_required_stress=(be_proposal.reason == "stress"),
            be_length_proposed=be_proposal.length if be_proposal.required else 0.0,
            be_thickness_proposed=be_proposal.thickness if be_proposal.required else 0.0,
            be_extension_above=be_proposal.extension_above,
            be_extension_below=be_proposal.extension_below,
            c_used=c_dem,
            be_top_proposal=be_top_prop,
            be_bot_proposal=be_bot_prop,
            Mpr_in_plane=Mpr_ip,
            Pu_for_Mpr=Pu_for_Mpr,
            omega_v=omega if mode == "capacity" else None,
            Ve_capacity=Ve_cap_kN,
            Ve_used=Ve_used,
            amplification=amplif,
            alpha_c_factor=a_c,
            phi_Vn=phi_Vn,
            phi_Vn_max=phi_Vn_max,
            av_s_required=av_s_req,
            ratio_Ve_over_phiVn=ratio_Ve_phi,
            Pu_used=Pu_used,
            Pu_over_Ag_fc=Pu_ratio,
            proposed_section=proposed_section,
            notes=tuple(notes),
        )

    # ---- envelope of three modes ------------------------------------- #
    @staticmethod
    def _design_envelope(
        prop_min: WallDesignProposal,
        prop_dem: Optional[WallDesignProposal],
        prop_cap: WallDesignProposal,
    ) -> WallDesignProposal:
        modes = [prop_min] + ([prop_dem] if prop_dem else []) + [prop_cap]

        # Worst of: BE required (any), longest BE length, max rho_t_required
        be_required = any(p.be_required for p in modes)
        be_length = max(p.be_length_proposed for p in modes)
        be_thickness = max(p.be_thickness_proposed for p in modes)
        rho_t_req = max(p.rho_t_required for p in modes)
        rho_l_req = max(p.rho_l_required for p in modes)

        # Pull proposed section from whichever mode actually proposed one
        proposed_section = next(
            (p.proposed_section for p in modes if p.proposed_section is not None),
            None,
        )

        # Notes: prefix with mode tag
        all_notes = tuple(f"[{p.mode}] {n}" for p in modes for n in p.notes)

        # Layers / spacing / phi_Vn come from the capacity mode (most demanding)
        ref = prop_cap

        return WallDesignProposal(
            mode="envelope",
            rho_l_required=rho_l_req,
            rho_t_required=rho_t_req,
            rho_l_provided=ref.rho_l_provided,
            rho_t_provided=ref.rho_t_provided,
            double_curtain_required=any(p.double_curtain_required for p in modes),
            web_bar_db=ref.web_bar_db,
            web_bar_spacing=ref.web_bar_spacing,
            web_bar_layers=ref.web_bar_layers,
            be_required=be_required,
            be_required_disp=any(p.be_required_disp for p in modes),
            be_required_stress=any(p.be_required_stress for p in modes),
            be_length_proposed=be_length,
            be_thickness_proposed=be_thickness,
            be_extension_above=max(p.be_extension_above for p in modes),
            be_extension_below=max(p.be_extension_below for p in modes),
            c_used=ref.c_used,
            be_top_proposal=ref.be_top_proposal,
            be_bot_proposal=ref.be_bot_proposal,
            Mpr_in_plane=prop_cap.Mpr_in_plane,
            Pu_for_Mpr=prop_cap.Pu_for_Mpr,
            omega_v=prop_cap.omega_v,
            Ve_capacity=prop_cap.Ve_capacity,
            Ve_used=prop_cap.Ve_used,
            amplification=prop_cap.amplification,
            alpha_c_factor=ref.alpha_c_factor,
            phi_Vn=ref.phi_Vn,
            phi_Vn_max=ref.phi_Vn_max,
            av_s_required=max(
                (p.av_s_required or 0.0) for p in modes
            ) or None,
            ratio_Ve_over_phiVn=ref.ratio_Ve_over_phiVn,
            Pu_used=ref.Pu_used,
            Pu_over_Ag_fc=ref.Pu_over_Ag_fc,
            proposed_section=proposed_section,
            notes=all_notes,
        )

    # ---- build a new section with BE per the proposal ---------------- #
    def _build_be_section(
        self,
        proposal: BEGeometryProposal,
        *,
        clear_cover: float,
        n_x_be: int | None = None,
    ) -> WallSection:
        """Return a new WallSection with BE added per `proposal`.

        Bars in the BE region come from the bar_schedule's longitudinal
        list, laid out as an n_x_be × N_y perimeter rectangle. ACI 318-25
        §18.10.6.4(b) calls for column-style detailing in the BE, so the
        minimum is n_x_be=3 (default) to allow at least one interior bar
        on each long face that can be tied as a crosstie. Use n_x_be=4
        for barbell walls (be_thickness ≫ tw) where more interior bars
        are needed for confinement.

        The web bars are not modified — the existing layout stays
        continuous through the BE region.
        """
        # Default n_x_be from the wall instance attribute
        if n_x_be is None:
            n_x_be = self.n_x_be
        # Pick BE longitudinal bars: use the largest available <= 25 mm,
        # with rho_BE ~ 1% as a baseline (§18.10.6.4(b)).
        be_db = min(
            (d for d in self.bar_schedule.longitudinal if d <= 25.0),
            default=self.bar_schedule.longitudinal[-1],
        )
        a_bar = pi * be_db ** 2 / 4.0
        be_area_target = 0.01 * proposal.length * proposal.thickness  # mm² (~1%)
        # Bars per long edge — distributed along the BE length so that
        # n_x_be * n_y * a_bar >= be_area_target (perimeter pattern).
        n_y = max(4, ceil(be_area_target / (n_x_be * a_bar)))

        steel = (
            self.section.rebar.groups[0].steel
            if self.section.rebar.groups else None
        )
        if steel is None:
            from design.common.materials import Steel
            steel = Steel(fy=self.fyt)

        bar = Bar(diameter=be_db)

        # Where to place BEs: top/bot of the wall
        s = self.section
        new_section = s
        # Preserve the wall's existing BE x-offset alignment when growing
        # the BE during auto-update (so a C-shape wall keeps its C shape).
        top_x = s.be_top_x_offset
        bot_x = s.be_bot_x_offset
        if proposal.length > s.be_length_top + 1.0:
            cy_top = s.lw / 2 - proposal.length / 2
            top_group = be_perimeter_bars(
                be_thickness=proposal.thickness,
                be_length=proposal.length,
                center_x=top_x, center_y=cy_top,
                cover=clear_cover,
                n_x=n_x_be, n_y=n_y, bar=bar, steel=steel,
            )
            new_section = new_section.with_boundary_elements(
                top_length=proposal.length,
                top_thickness=proposal.thickness,
                top_x_offset=top_x,
                add_be_top_bars=top_group,
            )
        if proposal.length > s.be_length_bot + 1.0:
            cy_bot = -s.lw / 2 + proposal.length / 2
            bot_group = be_perimeter_bars(
                be_thickness=proposal.thickness,
                be_length=proposal.length,
                center_x=bot_x, center_y=cy_bot,
                cover=clear_cover,
                n_x=n_x_be, n_y=n_y, bar=bar, steel=steel,
            )
            new_section = new_section.with_boundary_elements(
                bot_length=proposal.length,
                bot_thickness=proposal.thickness,
                bot_x_offset=bot_x,
                add_be_bot_bars=bot_group,
            )
        return new_section

    # ================================================================== #
    # Evolve / apply — return a NEW Wall
    # ================================================================== #
    def evolve(self, proposal: WallDesignProposal) -> "Wall":
        """Return a NEW Wall built from `proposal`.

        Updates:
            - rho_t / rho_l with the provided values
            - section if `proposal.proposed_section` is set
            - bar diameters (web + transverse) stay the same unless the
              proposal explicitly changed them
        """
        new_section = proposal.proposed_section or self.section
        new = self._with_section(new_section)
        new.rho_t = proposal.rho_t_provided or self.rho_t
        new.rho_l = proposal.rho_l_provided or self.rho_l
        new.web_bar_diameter = proposal.web_bar_db or self.web_bar_diameter
        # Propagate BE column proposals to the new wall's BE columns
        if proposal.be_top_proposal is not None and new.be_top is not None:
            new.be_top.apply(proposal.be_top_proposal)
        if proposal.be_bot_proposal is not None and new.be_bot is not None:
            new.be_bot.apply(proposal.be_bot_proposal)
        return new

    def apply(self, proposal: WallDesignProposal) -> "Wall":
        """Back-compat alias for `evolve()` — returns a NEW Wall."""
        return self.evolve(proposal)

    def _with_section(self, new_section: WallSection) -> "Wall":
        """Return a copy of self with a different section. No design cache."""
        return Wall(
            section=new_section,
            hw=self.hw, lu=self.lu,
            rho_t=self.rho_t, rho_l=self.rho_l,
            fyt=self.fyt, seismic=self.seismic,
            omega_v_factor=self.omega_v_factor, hx=self.hx,
            transverse_bar_diameter=self.transverse_bar_diameter,
            web_bar_diameter=self.web_bar_diameter,
            n_x_be=self.n_x_be,
            be_db_default=self.be_db_default,
            be_n_y_default=self.be_n_y_default,
            units=self.units,
            bar_schedule=self.bar_schedule,
            label=self.label,
            run_settings=dict(self.run_settings),
        )

    # ================================================================== #
    # Reports
    # ================================================================== #
    def summary(self) -> None:
        u = self.units
        s = self.section
        print(f"=== {self.label} ===  (units: {u.name})")
        print(f"Section : lw = {s.lw*u.length_factor:.0f} {u.length}, "
              f"tw = {s.tw*u.length_factor:.0f} {u.length}")
        print(f"Heights : hw = {self.hw*u.length_factor:.0f} {u.length}, "
              f"lu = {self.lu*u.length_factor:.0f} {u.length}")
        print(f"Material: fc = {s.concrete.fc:.0f} MPa, fyt = {self.fyt:.0f} MPa")
        print(f"Web rho : rho_l = {self.rho_l:.4f}, rho_t = {self.rho_t:.4f}")
        if self.has_boundary_elements:
            print(f"BE top  : {s.be_thickness_top:.0f} x {s.be_length_top:.0f} mm")
            print(f"BE bot  : {s.be_thickness_bot:.0f} x {s.be_length_bot:.0f} mm")
        else:
            print("BE      : none")
        if self._surface is None:
            print("State   : not run yet — call wall.run() to compute the PMM surface.")
            return
        print(f"Po      : {self.Po*u.force_factor:.0f} {u.force}")
        print(f"To      : {self.To*u.force_factor:.0f} {u.force}")
        print(f"phi*Vn  : {self.phi_Vn*u.force_factor:.0f} {u.force}  "
              f"(cap = {self.Vn_max*u.force_factor:.0f} {u.force})")
        print(f"max phi*Mn (in-plane)    : "
              f"{max(self.surface[0.0].phi_Mn)*u.moment_factor:.1f} {u.force}·{u.length}")
        print(f"max phi*Mn (out-of-plane): "
              f"{max(self.surface[90.0].phi_Mn)*u.moment_factor:.1f} {u.force}·{u.length}")

    def report(self) -> dict:
        self._ensure_run()
        u = self.units
        s = self.section
        return {
            "label": self.label,
            "units": u.name,
            "section": {
                "lw": s.lw * u.length_factor,
                "tw": s.tw * u.length_factor,
                "Ag": s.gross_area() * u.area_factor,
                "has_BE": self.has_boundary_elements,
                "be_top_length": s.be_length_top * u.length_factor,
                "be_bot_length": s.be_length_bot * u.length_factor,
            },
            "material": {"fc_MPa": s.concrete.fc, "fyt_MPa": self.fyt},
            "rho": {"l": self.rho_l, "t": self.rho_t},
            "capacity": {
                "Po": self.Po * u.force_factor,
                "To": self.To * u.force_factor,
                "phi_Vn": self.phi_Vn * u.force_factor,
                "Vn_max": self.Vn_max * u.force_factor,
                "max_phi_Mn_in_plane": float(max(self.surface[0.0].phi_Mn)) * u.moment_factor,
                "max_phi_Mn_out_plane": float(max(self.surface[90.0].phi_Mn)) * u.moment_factor,
            },
        }

    def __repr__(self) -> str:
        be = ""
        if self.has_boundary_elements:
            be = (f", BE top={self.section.be_length_top:.0f}, "
                  f"BE bot={self.section.be_length_bot:.0f}")
        return (
            f"Wall(label={self.label!r}, lw={self.section.lw:.0f}, "
            f"tw={self.section.tw:.0f}{be}, hw={self.hw:.0f}, "
            f"run={'yes' if self._surface is not None else 'no'})"
        )
