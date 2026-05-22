"""Concrete beam — self-contained, with chained access.

A Beam wraps a Section (geometry + materials + rebar) plus the detailing
parameters (ln, cover, stirrup spacing, ...) and offers:

    beam.run()                     -> lazy compute caches; returns self
    beam.capacity()                -> BeamCapacity (phi_Mn+/-, phi_Vn, Mpr+/-)
    beam.check(demands)            -> BeamCheck   (ratios per station)
    beam.design(demands=None, ...) -> BeamDesignResults (three modes)
    beam.evolve(proposal)          -> NEW Beam with the proposed detailing
    beam.apply(proposal)           -> alias of evolve()
    beam.set_units(code_or_name)   -> presentation units (ETABS codes 1..16)
    beam.plot.*                    -> bound plotter
    beam.summary() / beam.report() -> formatted views

Beams resist flexure and shear only — they do NOT need a P-M
interaction diagram. ACI 318-25 §9 and §18.6 treat them as flexural
elements; M-N coupling lives in Cap. 10 (columns) and §22.4.

Mn is computed in closed form using `mn_doubly` (Whitney block plus
compression-steel correction). Mpr per §18.6.5.1 uses fy_pr = 1.25 fy
and phi = 1.0.

Internal units (per ``design/AGENTS.md``):
    Stress  MPa,  Length  mm,  Force  kN,  Moment  kN·m,  Area  mm².

``ln`` is stored in **mm**. ``Beam.rectangular(ln=...)`` accepts metres
as a convenience (heuristic ``ln < 30``) and converts with a
``DeprecationWarning``. ``beam.ln_m`` is a read-only view in metres.

ACI 318-25 sections used:
    §9.3.1     h_min for beams (Table 9.3.1.1)
    §9.6.1.2   minimum tension reinforcement (rho_min)
    §9.6.3.4   minimum shear reinforcement (av_s_min)
    §18.6.3.1  rho_min, rho_max for SMF
    §18.6.3.2  at least 2 bars continuous top and bottom
    §18.6.3.3  phi_Mn_pos at joint face >= 1/2 phi_Mn_neg at that face
    §18.6.3.4  phi_Mn anywhere >= 1/4 max phi_Mn at face of either joint
    §18.6.4.1  length of confined region lo = 2 h
    §18.6.4.4  hoop spacing in confined region
    §18.6.4.6  hoop spacing outside confined region
    §18.6.5.1  Ve = (Mpr_i + Mpr_j) / ln + Vg
    §18.6.5.2  Vc = 0 in lo when seismic dominates and Pu is small
    §21.2.2    phi for axial+flexural sections (transition zone)
    §22.2      strain compatibility / Whitney block
    §22.5      one-way shear (Vc + Vs)
    §25.2      bar spacing limits
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
from math import ceil, pi
from typing import Optional

import numpy as np

from design.common.bar_schedule import BarSchedule
from design.common.factors import phi_shear, phi_axial_flexure
from design.common.materials import Bar, Concrete, Steel
from design.common.shear import (
    vc_simplified, vs_capacity, av_s_required as _av_s_required_common,
    av_s_minimum, vc_zero_seismic,
)
from design.common.spacing import (
    s_max_seismic_smf_beam as s_max_seismic_smf,
    s_max_seismic_outside_beam as s_max_seismic_outside,
)
from design.common.units import UnitSystem, units_from, DEFAULT_UNITS
from design.sections.base import Section
from design.sections.rectangular import RectangularSection
from design.sections.reinforcement import Rebar, RebarGroup, RebarLayout
from design.beams.flexure import (
    as_required_singly, mn_doubly,
    rho_min_beam, rho_balanced, rho_max_beam, rho_max_seismic,
    is_tension_controlled,
)
from design.beams.seismic import (
    mpr as mpr_closed_form, v_seismic_from_mpr,
)
from design.beams.plotting import _BeamPlotter


# ---------------------------------------------------------------- #
# Dataclasses
# ---------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BeamDemands:
    """Beam demands at three stations along the member.

    Stations along the longitudinal axis:
        i   — face of the left joint
        mid — middle of clear span
        j   — face of the right joint

    Each station carries:
        Mu_pos > 0  — tension on the bottom fiber (gravity-like).
        Mu_neg > 0  — magnitude, tension on the top fiber.
        Vu          — factored shear at that station (worst-combo envelope).

    Optional gravity shear (per station, from a gravity-only combo —
    typically 1.2D + 1.6L). Used in capacity mode (§18.6.5.1):
        Vg_i, Vg_mid, Vg_j

    If left as None, the beam's `Vg` attribute is used as fallback
    (Vg_i = Vg_j = beam.Vg, Vg_mid = 0  — uniformly loaded assumption).
    For uniformly loaded SMF beams, Vg_i = -Vg_j = wu·ln/2 and
    Vg_mid = 0; only the magnitudes matter for shear design.
    """
    Mu_pos_i:   float = 0.0
    Mu_neg_i:   float = 0.0
    Vu_i:       float = 0.0
    Mu_pos_mid: float = 0.0
    Mu_neg_mid: float = 0.0
    Vu_mid:     float = 0.0
    Mu_pos_j:   float = 0.0
    Mu_neg_j:   float = 0.0
    Vu_j:       float = 0.0
    Vg_i:       Optional[float] = None
    Vg_mid:     Optional[float] = None
    Vg_j:       Optional[float] = None

    @property
    def Vu_max(self) -> float:
        return max(abs(self.Vu_i), abs(self.Vu_mid), abs(self.Vu_j))


@dataclass(frozen=True, slots=True)
class BeamCapacity:
    """Continuous reinforcement capacity — one value per direction.

    Canonical names per ``design/AGENTS.md`` §2:
        ``Mpr_pos`` / ``Mpr_neg``   — probable moment (§18.6.5.1).

    Legacy aliases ``Mn_pr_pos`` / ``Mn_pr_neg`` are exposed as
    read-only properties.
    """
    phi_Mn_pos: float            # kN·m
    phi_Mn_neg: float
    phi_Vn:     float            # kN
    Mpr_pos:    Optional[float] = None
    Mpr_neg:    Optional[float] = None
    rho:        float = 0.0      # As_bot / (bw·d)  (tension under +M)
    rho_p:      float = 0.0      # As_top / (bw·d)  (compression under +M)
    rho_min:    float = 0.0      # §9.6.1.2
    rho_max:    float = 0.0      # 0.5 · rho_balanced (this module)
    rho_balanced: float = 0.0    # ACI §22.2 balanced ratio
    rho_max_aci_smf: float = 0.0 # §18.6.3.1 (informative)
    tension_controlled: bool = True

    # ---- Legacy aliases (back-compat) ----
    @property
    def Mn_pr_pos(self) -> Optional[float]:
        """Legacy alias for :attr:`Mpr_pos` — prefer ``Mpr_pos``."""
        return self.Mpr_pos

    @property
    def Mn_pr_neg(self) -> Optional[float]:
        """Legacy alias for :attr:`Mpr_neg` — prefer ``Mpr_neg``."""
        return self.Mpr_neg


@dataclass(frozen=True, slots=True)
class BeamCheck:
    capacity: BeamCapacity
    demands:  BeamDemands
    # ratios per station per direction
    ratio_M_pos_i:   float
    ratio_M_neg_i:   float
    ratio_M_pos_mid: float
    ratio_M_neg_mid: float
    ratio_M_pos_j:   float
    ratio_M_neg_j:   float
    ratio_M_overall: float
    ratio_V_i:       float
    ratio_V_mid:     float
    ratio_V_j:       float
    ratio_V_overall: float
    rho_ok:          bool
    tc_ok:           bool
    smf_continuity_ok: bool
    passed:          bool


@dataclass(frozen=True, slots=True)
class BeamDesignProposal:
    """Detailing proposal for ONE design mode.

    All quantities in internal units (mm, mm², MPa, kN, kN·m).
    """
    mode: str                   # 'minimum' | 'demand' | 'capacity' | 'envelope'

    # Required As per station (envelope of the demand at each station)
    As_top_i_required:   float
    As_top_mid_required: float
    As_top_j_required:   float
    As_bot_i_required:   float
    As_bot_mid_required: float
    As_bot_j_required:   float

    # Continuous reinforcement actually adopted (constant along the beam)
    As_top_continuous: float
    As_bot_continuous: float
    db_top: float
    n_top:  int
    db_bot: float
    n_bot:  int

    # Transverse
    db_stirrup: float
    n_legs:     int
    av_provided: float
    spacing_confined: float      # §18.6.4.4
    spacing_middle:   float      # §18.6.4.6
    lo: float                    # §18.6.4.1   = 2 h

    # Shear demand by station + §18.6.5.2 flag
    Vu_i_used:        Optional[float] = None
    Vu_mid_used:      Optional[float] = None
    Vu_j_used:        Optional[float] = None
    av_s_required:    Optional[float] = None
    vc_zero_active:   bool = False

    # Capacity-design quantities
    Mpr_pos:  Optional[float] = None
    Mpr_neg:  Optional[float] = None
    Ve_capacity: Optional[float] = None
    phi_Vn:      Optional[float] = None
    ratio_Ve_over_phiVn: Optional[float] = None

    # §18.6.3 SMF continuity ratios
    ratio_pos_over_neg_at_i: Optional[float] = None
    ratio_pos_over_neg_at_j: Optional[float] = None
    ratio_min_over_max_anywhere: Optional[float] = None
    two_bars_top_ok: bool = True
    two_bars_bot_ok: bool = True

    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BeamDesignResults:
    """Container of the three design modes."""
    minimum:  BeamDesignProposal
    demand:   Optional[BeamDesignProposal]
    capacity: BeamDesignProposal
    envelope: BeamDesignProposal

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
# Beam class
# ---------------------------------------------------------------- #
class Beam:
    """Reinforced concrete beam, autocontained.

    Internal numerics: MPa, mm, kN, kN·m.
    Use beam.set_units(code) to change PRESENTATION units.

    The beam takes its longitudinal steel from `section.rebar`. Stirrup
    steel uses the same grade by default (override `steel_transverse` if
    you need a different one). There is no separate `fyt` scalar.
    """

    def __init__(
        self,
        *,
        section: Section,
        ln: float,                          # clear span (mm) — see class docstring
        cover: float = 40.0,                # mm to outside of stirrup
        Av: float = 0.0,
        s: float | None = None,             # initial stirrup spacing (mm)
        seismic: bool = True,
        db_long_min: float = 16.0,          # smallest longitudinal db (for s_max)
        db_stirrup: float = 10.0,           # initial stirrup diameter
        Vg: float = 0.0,                    # gravity shear at face of support (kN)
        steel_transverse: Steel | None = None,
        units: int | str | UnitSystem | None = None,
        bar_schedule: BarSchedule | None = None,
        label: str = "Beam",
    ) -> None:
        self.section = section
        # ``ln`` is canonical in mm. The Beam.rectangular() factory converts
        # m -> mm with a DeprecationWarning. Direct __init__ callers should
        # already pass mm (no auto-detect here to keep the contract clear).
        self.ln = float(ln)
        self.cover = cover
        self.Av = Av
        self.s = s
        self.seismic = seismic
        self.db_long_min = db_long_min
        self.db_stirrup = db_stirrup
        self.Vg = Vg
        self.label = label

        # Steel: primary comes from the layout. Transverse defaults to it.
        primary = (section.rebar.groups[0].steel
                   if section.rebar.groups else Steel(fy=420.0))
        self._steel_transverse = steel_transverse or primary

        if units is None:
            self.units = DEFAULT_UNITS
        elif isinstance(units, UnitSystem):
            self.units = units
        else:
            self.units = units_from(units)

        self.bar_schedule = bar_schedule if bar_schedule is not None else BarSchedule()

        self._plotter: Optional[_BeamPlotter] = None
        self.design_results: Optional[BeamDesignResults] = None

        # Lazy caches — populated by .run()
        self._phi_Mn_pos: Optional[float] = None
        self._phi_Mn_neg: Optional[float] = None
        self._phi_Vn_cache: Optional[float] = None
        self._mpr_pos_cache: Optional[float] = None
        self._mpr_neg_cache: Optional[float] = None

    # =============================================================== #
    # Length helpers
    # =============================================================== #
    @property
    def ln_m(self) -> float:
        """Clear span in **metres** (legacy view over the canonical mm value)."""
        return self.ln / 1000.0

    @property
    def _ln_m(self) -> float:
        """Clear span in metres — internal helper for kN·m / m = kN math."""
        return self.ln / 1000.0

    # =============================================================== #
    # One-shot factory from scalar specs
    # =============================================================== #
    @classmethod
    def rectangular(
        cls,
        *,
        bw: float, h: float, cover: float,
        concrete: Concrete, steel: Steel,
        # ---- longitudinal ----
        n_top: int, db_top: float,
        n_bot: int, db_bot: float,
        # ---- stirrups (PROVIDED detailing) ----
        db_stirrup: float = 10.0,
        n_legs: int = 2,                  # vertical legs parallel to Y (resist V_y)
        s_stirrup: float | None = None,   # spacing (mm); also accepts `s`
        # ---- member ----
        ln: float,
        Vg: float = 0.0,
        seismic: bool = True,
        steel_transverse: Steel | None = None,
        units: int | str | UnitSystem | None = None,
        bar_schedule: BarSchedule | None = None,
        label: str = "Beam",
        **kwargs,
    ) -> "Beam":
        """Build a Beam from scalar specs in a single call.

        Cover is the CLEAR cover from the outer concrete face to the
        outside edge of the stirrup. Bar centroids are placed at
        cover + db_stirrup + db/2 from the corresponding face.

        Stirrup arguments:
            db_stirrup  — stirrup diameter (mm)
            n_legs      — number of VERTICAL legs (parallel to Y).
                          A closed perimeter stirrup has 2 vertical
                          legs; each interior crosstie adds one.
                          Only n_legs resists V_y in a beam — the
                          horizontal legs of the perimeter stirrup
                          do not contribute and are implicit (=2).
            s_stirrup   — stirrup spacing (mm). Accepts `s` as alias.

        Av is computed internally as  n_legs · π·db_stirrup²/4.

        Example:
            beam = Beam.rectangular(
                bw=400, h=500, cover=40,
                concrete=concrete, steel=steel,
                n_top=5, db_top=22, n_bot=5, db_bot=22,
                db_stirrup=10, n_legs=3, s_stirrup=100,
                ln=5.0, seismic=True, Vg=80.0,
            )
        """
        # Accept legacy `s` alias
        s_val = s_stirrup if s_stirrup is not None else kwargs.pop("s", None)

        # Heuristic m -> mm conversion (canonical unit is mm per AGENTS.md §1).
        # If ln < 30 we assume the caller passed metres (legacy convention) and
        # convert, emitting a DeprecationWarning. Direct Beam.__init__ callers
        # are expected to pass mm already.
        if ln is not None and ln < 30.0:
            import warnings
            warnings.warn(
                f"Beam.rectangular(ln={ln}) looks like metres; converting to "
                f"{ln * 1000.0:.0f} mm. Pass mm directly to silence this warning.",
                DeprecationWarning, stacklevel=2,
            )
            ln_mm = float(ln) * 1000.0
        else:
            ln_mm = float(ln) if ln is not None else 0.0

        rebar = _build_beam_rebar(
            bw=bw, h=h, cover=cover, db_stirrup=db_stirrup,
            n_top=n_top, db_top=db_top,
            n_bot=n_bot, db_bot=db_bot,
            steel=steel,
        )
        section = RectangularSection(b=bw, h=h, concrete=concrete, rebar=rebar)

        a_per_leg = pi * db_stirrup ** 2 / 4.0
        Av = max(2, n_legs) * a_per_leg

        return cls(
            section=section, ln=ln_mm, cover=cover, db_stirrup=db_stirrup,
            Av=Av, s=s_val,
            Vg=Vg, seismic=seismic, steel_transverse=steel_transverse,
            units=units, bar_schedule=bar_schedule, label=label,
            **kwargs,
        )

    # =============================================================== #
    # Configuration helpers
    # =============================================================== #
    def set_units(self, code_or_name: int | str) -> "Beam":
        self.units = units_from(code_or_name)
        return self

    @property
    def steel(self) -> Steel:
        """Primary longitudinal steel (from the first group of the layout)."""
        if self.section.rebar.groups:
            return self.section.rebar.groups[0].steel
        return Steel(fy=420.0)

    @property
    def fyt(self) -> float:
        """Transverse steel yield strength (defaults to primary fy)."""
        return self._steel_transverse.fy

    # =============================================================== #
    # Mutation: ALWAYS immutable via evolve() (AGENTS.md §4)
    # =============================================================== #
    def evolve(self, proposal: BeamDesignProposal) -> "Beam":
        """Return a NEW Beam with the proposal's detailing applied.

        Per ``design/AGENTS.md`` §4 every element is immutable: ``evolve``
        builds a fresh :class:`Beam` (and a fresh
        :class:`RectangularSection` with new rebar) instead of mutating
        ``self``. ``self.section`` is **not** modified.
        """
        x_min, y_min, x_max, y_max = self.section.bounding_box()
        bw = x_max - x_min
        h = y_max - y_min

        new_rebar = _build_beam_rebar(
            bw=bw, h=h, cover=self.cover,
            db_stirrup=proposal.db_stirrup,
            n_top=proposal.n_top, db_top=proposal.db_top,
            n_bot=proposal.n_bot, db_bot=proposal.db_bot,
            steel=self.steel,
        )
        new_section = RectangularSection(
            b=bw, h=h, concrete=self.section.concrete, rebar=new_rebar,
        )
        a_per_leg = pi * proposal.db_stirrup ** 2 / 4.0
        new_Av = max(2, proposal.n_legs) * a_per_leg

        return Beam(
            section=new_section,
            ln=self.ln, cover=self.cover,
            Av=new_Av, s=proposal.spacing_confined,
            seismic=self.seismic, db_long_min=self.db_long_min,
            db_stirrup=proposal.db_stirrup,
            Vg=self.Vg, steel_transverse=self._steel_transverse,
            units=self.units, bar_schedule=self.bar_schedule, label=self.label,
        )

    def apply(self, proposal: BeamDesignProposal) -> "Beam":
        """Alias of :meth:`evolve` — returns a NEW Beam (immutable)."""
        return self.evolve(proposal)

    # =============================================================== #
    # Lazy run + cached capacity properties
    # =============================================================== #
    def run(self) -> "Beam":
        """Lazy-compute capacity caches and return ``self`` for chaining."""
        self._phi_Mn_pos = self._compute_phi_Mn(direction="pos")
        self._phi_Mn_neg = self._compute_phi_Mn(direction="neg")
        self._phi_Vn_cache = self._compute_phi_Vn()
        if self.seismic:
            self._mpr_pos_cache = self._mpr(direction="pos")
            self._mpr_neg_cache = self._mpr(direction="neg")
        else:
            self._mpr_pos_cache = None
            self._mpr_neg_cache = None
        return self

    def _ensure_run(self) -> None:
        if self._phi_Mn_pos is None:
            self.run()

    @property
    def phi_Mn_pos(self) -> float:
        self._ensure_run()
        return self._phi_Mn_pos

    @property
    def phi_Mn_neg(self) -> float:
        self._ensure_run()
        return self._phi_Mn_neg

    @property
    def phi_Vn(self) -> float:
        self._ensure_run()
        return self._phi_Vn_cache

    @property
    def phi_Mn(self) -> dict:
        """``{'pos': phi_Mn_pos, 'neg': phi_Mn_neg}`` (kN·m)."""
        self._ensure_run()
        return {"pos": self._phi_Mn_pos, "neg": self._phi_Mn_neg}

    @property
    def Mn(self) -> dict:
        """``{'pos': Mn_pos, 'neg': Mn_neg}`` recovered from ``phi*Mn``.

        Beams are flexural elements — no PMM surface — so ``Mn`` is a
        plain dict with two keys, matching how the AGENTS.md contract
        treats beams.
        """
        self._ensure_run()
        # Recover Mn using the actual phi used in _compute_phi_Mn. We
        # recompute the phi for each direction so the back-conversion is
        # consistent with the transition zone (§21.2.2).
        c = self.section.concrete
        s_steel = self.steel
        bw = self.section.width()
        d_pos, d_neg, *_ = self._depths()
        As_bot = self._As_in_layer("bot")
        As_top = self._As_in_layer("top")

        def _mn_from_phi_mn(direction: str, phi_Mn: float) -> float:
            if direction == "pos":
                As, d = As_bot, d_pos
            else:
                As, d = As_top, d_neg
            if As <= 0 or d <= 0:
                return phi_Mn / 0.9
            a = (As * s_steel.fy) / (0.85 * c.fc * bw)
            c_neutral = a / c.beta1 if c.beta1 > 0 else 0.0
            eps_t = (c.eps_cu * (d - c_neutral) / c_neutral
                     if c_neutral > 0 else 0.0)
            phi = phi_axial_flexure(eps_t=eps_t, eps_ty=s_steel.eps_ty)
            return phi_Mn / phi if phi > 0 else phi_Mn / 0.9

        return {
            "pos": _mn_from_phi_mn("pos", self._phi_Mn_pos),
            "neg": _mn_from_phi_mn("neg", self._phi_Mn_neg),
        }

    # =============================================================== #
    # Geometric helpers
    # =============================================================== #
    def _As_in_layer(self, layer: str, tol: float = 10.0) -> float:
        """Sum of bar areas in the top or bottom extreme layer."""
        ys = [(r.y, r.area) for r, _ in self.section.rebar.iter_bars()]
        if not ys:
            return 0.0
        if layer == "top":
            y_ref = max(y for y, _ in ys)
        else:
            y_ref = min(y for y, _ in ys)
        return sum(a for y, a in ys if abs(y - y_ref) <= tol)

    def _infer_layer_bars(self, layer: str, tol: float = 10.0
                          ) -> tuple[float, int]:
        """Infer (db, n) from the bars in the top or bottom extreme layer.

        Assumes a uniform diameter in the layer (no bundled / mixed sizes).
        """
        ys = [(r.y, r.area) for r, _ in self.section.rebar.iter_bars()]
        if not ys:
            return 0.0, 0
        if layer == "top":
            y_ref = max(y for y, _ in ys)
        else:
            y_ref = min(y for y, _ in ys)
        layer_bars = [a for y, a in ys if abs(y - y_ref) <= tol]
        if not layer_bars:
            return 0.0, 0
        a = layer_bars[0]
        db = (4.0 * a / pi) ** 0.5
        return db, len(layer_bars)

    def _current_n_legs(self) -> int:
        """Infer number of stirrup legs from beam.Av and beam.db_stirrup."""
        if self.db_stirrup <= 0:
            return 0
        a_per_leg = pi * self.db_stirrup ** 2 / 4.0
        return max(2, int(round(self.Av / a_per_leg))) if self.Av > 0 else 2

    def _depths(self) -> tuple[float, float, float, float]:
        """Return (d_pos, d_neg, d_prime_pos, d_prime_neg).

        d_pos: distance from TOP fiber to centroid of bottom layer.
        d_neg: distance from BOTTOM fiber to centroid of top layer.
        d_prime_pos: distance from TOP fiber to centroid of top layer.
        d_prime_neg: distance from BOTTOM fiber to centroid of bottom layer.
        """
        x_min, y_min, x_max, y_max = self.section.bounding_box()
        ys = [r.y for r, _ in self.section.rebar.iter_bars()]
        if not ys:
            h = y_max - y_min
            return 0.9 * h, 0.9 * h, 0.1 * h, 0.1 * h
        y_top_bar = max(ys)
        y_bot_bar = min(ys)
        d_pos = y_max - y_bot_bar
        d_neg = y_top_bar - y_min
        d_prime_pos = y_max - y_top_bar
        d_prime_neg = y_bot_bar - y_min
        return d_pos, d_neg, d_prime_pos, d_prime_neg

    # =============================================================== #
    # Capacity / Check
    # =============================================================== #
    def _compute_phi_Mn(self, *, direction: str) -> float:
        """phi*Mn for one bending sense, from mn_doubly closed form.

        Uses ``phi_axial_flexure(eps_t, eps_ty)`` from
        :mod:`design.common.factors`, which handles the §21.2.2
        transition zone instead of a binary 0.90 / 0.65 split.
        """
        c = self.section.concrete
        s_steel = self.steel
        bw = self.section.width()
        As_bot = self._As_in_layer("bot")
        As_top = self._As_in_layer("top")
        d_pos, d_neg, d_p_pos, d_p_neg = self._depths()

        if direction == "pos":
            As, As_p, d, d_p = As_bot, As_top, d_pos, d_p_pos
        else:
            As, As_p, d, d_p = As_top, As_bot, d_neg, d_p_neg

        Mn = mn_doubly(
            b=bw, d=d, d_prime=d_p,
            As=As, As_prime=As_p,
            concrete=c, steel=s_steel,
        )
        # Strain at the tension reinforcement consistent with the Whitney
        # block. The compression-steel correction has a minor effect on
        # eps_t through `c`; ignore it (§21.2.2 evaluation is normally
        # done with the singly-reinforced ``c``).
        if As > 0 and bw > 0 and c.fc > 0:
            a = (As * s_steel.fy) / (0.85 * c.fc * bw)
            c_neutral = a / c.beta1 if c.beta1 > 0 else 0.0
        else:
            c_neutral = 0.0
        eps_t = (c.eps_cu * (d - c_neutral) / c_neutral
                 if c_neutral > 0 else 0.0)
        phi = phi_axial_flexure(eps_t=eps_t, eps_ty=s_steel.eps_ty)
        return phi * Mn

    def _compute_phi_Vn(self) -> float:
        """phi*Vn from Vc + Vs (simplified §22.5) using common helpers."""
        d, _, _, _ = self._depths()
        bw = self.section.width()
        c = self.section.concrete
        Vc = vc_simplified(b=bw, d=d, concrete=c)
        Vs = 0.0
        if self.Av > 0 and self.s and self.s > 0:
            Vs = vs_capacity(Av=self.Av, fyt=self.fyt, d=d, s=self.s)
        return phi_shear() * (Vc + Vs)

    def _mpr(self, *, direction: str) -> float:
        """Mpr closed form for one bending sense — §18.6.5.1."""
        c = self.section.concrete
        s_steel = self.steel
        bw = self.section.width()
        d_pos, d_neg, *_ = self._depths()
        if direction == "pos":
            As, d = self._As_in_layer("bot"), d_pos
        else:
            As, d = self._As_in_layer("top"), d_neg
        return mpr_closed_form(b=bw, d=d, As=As, concrete=c, steel=s_steel)

    def capacity(self) -> BeamCapacity:
        """Build :class:`BeamCapacity` from the current detailing."""
        c = self.section.concrete
        s_steel = self.steel
        bw = self.section.width()
        d_pos, _, _, _ = self._depths()

        # Use cached values when run() has populated them, otherwise
        # compute fresh — both paths converge on the same numbers.
        self._ensure_run()
        phi_Mn_pos = self._phi_Mn_pos
        phi_Mn_neg = self._phi_Mn_neg
        phi_Vn = self._phi_Vn_cache

        As_bot = self._As_in_layer("bot")
        As_top = self._As_in_layer("top")
        rho   = As_bot / (bw * d_pos) if d_pos > 0 else 0.0
        rho_p = As_top / (bw * d_pos) if d_pos > 0 else 0.0
        rho_min = rho_min_beam(c.fc, s_steel.fy)
        rho_b   = rho_balanced(c.fc, s_steel.fy)
        rho_max = rho_max_beam(c.fc, s_steel.fy)
        rho_max_smf = rho_max_seismic(s_steel.fy)

        tc = is_tension_controlled(b=bw, d=d_pos, As=max(As_bot, 1.0),
                                   concrete=c, steel=s_steel)

        Mpr_pos = self._mpr_pos_cache if self.seismic else None
        Mpr_neg = self._mpr_neg_cache if self.seismic else None

        return BeamCapacity(
            phi_Mn_pos=phi_Mn_pos, phi_Mn_neg=phi_Mn_neg,
            phi_Vn=phi_Vn,
            Mpr_pos=Mpr_pos, Mpr_neg=Mpr_neg,
            rho=rho, rho_p=rho_p,
            rho_min=rho_min, rho_max=rho_max,
            rho_balanced=rho_b, rho_max_aci_smf=rho_max_smf,
            tension_controlled=tc,
        )

    def check(self, demands: BeamDemands) -> BeamCheck:
        cap = self.capacity()

        def r_M(M: float, C: float) -> float:
            return M / C if C > 0 else float("inf")

        r_pos_i   = r_M(demands.Mu_pos_i,   cap.phi_Mn_pos)
        r_neg_i   = r_M(demands.Mu_neg_i,   cap.phi_Mn_neg)
        r_pos_mid = r_M(demands.Mu_pos_mid, cap.phi_Mn_pos)
        r_neg_mid = r_M(demands.Mu_neg_mid, cap.phi_Mn_neg)
        r_pos_j   = r_M(demands.Mu_pos_j,   cap.phi_Mn_pos)
        r_neg_j   = r_M(demands.Mu_neg_j,   cap.phi_Mn_neg)
        r_M_overall = max(r_pos_i, r_neg_i, r_pos_mid, r_neg_mid, r_pos_j, r_neg_j)

        r_V_i   = abs(demands.Vu_i)   / cap.phi_Vn if cap.phi_Vn > 0 else float("inf")
        r_V_mid = abs(demands.Vu_mid) / cap.phi_Vn if cap.phi_Vn > 0 else float("inf")
        r_V_j   = abs(demands.Vu_j)   / cap.phi_Vn if cap.phi_Vn > 0 else float("inf")
        r_V_overall = max(r_V_i, r_V_mid, r_V_j)

        rho_ok = cap.rho_min <= max(cap.rho, cap.rho_p) <= cap.rho_max

        # §18.6.3 SMF continuity
        smf_ok = True
        if self.seismic:
            if cap.phi_Mn_neg > 0 and cap.phi_Mn_pos < 0.5 * cap.phi_Mn_neg:
                smf_ok = False
            face_max = max(cap.phi_Mn_pos, cap.phi_Mn_neg)
            if face_max > 0 and min(cap.phi_Mn_pos, cap.phi_Mn_neg) < 0.25 * face_max:
                smf_ok = False

        passed = (r_M_overall <= 1.0 and r_V_overall <= 1.0 and rho_ok and
                  (cap.tension_controlled if self.seismic else True) and
                  smf_ok)
        return BeamCheck(
            capacity=cap, demands=demands,
            ratio_M_pos_i=r_pos_i, ratio_M_neg_i=r_neg_i,
            ratio_M_pos_mid=r_pos_mid, ratio_M_neg_mid=r_neg_mid,
            ratio_M_pos_j=r_pos_j, ratio_M_neg_j=r_neg_j,
            ratio_M_overall=r_M_overall,
            ratio_V_i=r_V_i, ratio_V_mid=r_V_mid, ratio_V_j=r_V_j,
            ratio_V_overall=r_V_overall,
            rho_ok=rho_ok,
            tc_ok=cap.tension_controlled,
            smf_continuity_ok=smf_ok,
            passed=passed,
        )

    # =============================================================== #
    # Design — three modes
    # =============================================================== #
    def design(
        self,
        demands: BeamDemands | None = None,
        *,
        db_top: float | None = None,
        n_top: int | None = None,
        db_bot: float | None = None,
        n_bot: int | None = None,
        db_stirrup: float | None = None,
        n_legs: int | None = None,
    ) -> BeamDesignResults:
        common = dict(
            db_top=db_top, n_top=n_top,
            db_bot=db_bot, n_bot=n_bot,
            db_stirrup=db_stirrup, n_legs=n_legs,
        )
        prop_min = self._design_one(mode="minimum", demands=None, **common)
        prop_dem = (self._design_one(mode="demand", demands=demands, **common)
                    if demands is not None else None)
        prop_cap = self._design_one(mode="capacity", demands=demands, **common)
        prop_env = self._design_envelope(prop_min, prop_dem, prop_cap)
        results = BeamDesignResults(
            minimum=prop_min, demand=prop_dem,
            capacity=prop_cap, envelope=prop_env,
        )
        # NOTE (AGENTS.md §4): design() is non-mutating — the caller is
        # responsible for keeping the returned BeamDesignResults if it
        # wants to print/inspect it later via beam.print_design(results).
        return results

    # ---- helpers ----
    def _max_bars_in_layer(self, db: float, db_stirrup: float,
                           ag_max: float = 19.0) -> int:
        bw = self.section.width()
        span = bw - 2 * (self.cover + db_stirrup) - db
        if span <= 0:
            return 2
        clear = max(db, 25.0, 1.33 * ag_max)
        pitch = db + clear
        return max(2, int(span / pitch) + 1)

    def _pick_longitudinal(self, As_required: float, db_stirrup: float,
                           db_user: float | None, n_user: int | None,
                           ) -> tuple[float, int]:
        if db_user is not None and n_user is not None:
            return db_user, max(2, n_user)
        candidates = sorted(self.bar_schedule.longitudinal)
        if db_user is not None:
            candidates = [db for db in candidates if db == db_user]
        for db in candidates:
            a = pi * db ** 2 / 4.0
            n_min = max(2, ceil(As_required / a)) if As_required > 0 else 2
            n_max = self._max_bars_in_layer(db, db_stirrup)
            if n_min <= n_max:
                return db, n_min
        # Fallback: largest db
        db = max(self.bar_schedule.longitudinal)
        a = pi * db ** 2 / 4.0
        n = max(2, ceil(As_required / a)) if As_required > 0 else 2
        return db, n

    def _pick_stirrup(self, av_s_required: float, s_target: float,
                      db_user: float | None, n_user: int | None,
                      ) -> tuple[float, int, float]:
        """Return (db_stirrup, n_legs, spacing).

        Three cases:
          • user fixes BOTH db and n_legs  →  spacing is computed
            adaptively as min(s_target, av_provided / av_s_required).
            This lets you specify a hoop configuration and have the
            algorithm compute the tightest required spacing.
          • user fixes only db             →  n_legs is computed at
            spacing = s_target (the §18.6.4.4 code maximum).
          • user fixes nothing             →  same as above, picking
            the smallest diameter that fits (n_legs <= 6).
        """
        # Case 1: both fixed — adapt spacing to the shear demand
        if db_user is not None and n_user is not None:
            n_eff = max(2, n_user)
            a = pi * db_user ** 2 / 4.0
            av_prov = n_eff * a
            if av_s_required > 0:
                s_req = av_prov / av_s_required
                s_adapted = min(s_target, s_req)
            else:
                s_adapted = s_target
            return db_user, n_eff, s_adapted

        # Cases 2 and 3: spacing = s_target, pick n_legs
        candidates = sorted(self.bar_schedule.hoops)
        if db_user is not None:
            candidates = [db for db in candidates if db == db_user]
        for db in candidates:
            a = pi * db ** 2 / 4.0
            if av_s_required <= 0:
                return db, 2, s_target
            n_req = max(2, ceil(av_s_required * s_target / a))
            if n_req <= 6:
                return db, n_req, s_target
        # Fallback: largest hoop diameter, even if n_legs > 6
        db = max(self.bar_schedule.hoops)
        a = pi * db ** 2 / 4.0
        n = max(2, ceil(av_s_required * s_target / a)) if av_s_required > 0 else 2
        return db, n, s_target

    def _resolve_vg(self, demands: BeamDemands | None
                    ) -> tuple[float, float, float]:
        """Return (Vg_i, Vg_mid, Vg_j) as positive magnitudes.

        Priority: demands.Vg_*  →  beam.Vg fallback (uniform load assumption:
        Vg_i = Vg_j = beam.Vg, Vg_mid = 0).
        """
        if demands is None:
            return abs(self.Vg), 0.0, abs(self.Vg)
        vg_i   = (abs(demands.Vg_i)   if demands.Vg_i   is not None
                  else abs(self.Vg))
        vg_mid = (abs(demands.Vg_mid) if demands.Vg_mid is not None
                  else 0.0)
        vg_j   = (abs(demands.Vg_j)   if demands.Vg_j   is not None
                  else abs(self.Vg))
        return vg_i, vg_mid, vg_j

    def _av_s_required_for(self, Vu: float, *, vc_zero: bool = False
                           ) -> float | None:
        """Wrapper over :func:`design.common.shear.av_s_required`.

        Computes ``Vc`` (or sets it to zero per §18.6.5.2 / §18.7.6.2.1)
        and delegates to the canonical helper.
        """
        if Vu is None or Vu <= 0:
            return None
        d, _, _, _ = self._depths()
        bw = self.section.width()
        c = self.section.concrete
        Vc = 0.0 if vc_zero else vc_simplified(b=bw, d=d, concrete=c)
        return _av_s_required_common(
            Vu=Vu, Vc=Vc, fyt=self.fyt, d=d, phi=phi_shear(),
        )

    def _design_one(
        self,
        *,
        mode: str,
        demands: BeamDemands | None,
        db_top: float | None,
        n_top:  int | None,
        db_bot: float | None,
        n_bot:  int | None,
        db_stirrup: float | None,
        n_legs: int | None,
    ) -> BeamDesignProposal:
        c = self.section.concrete
        s_steel = self.steel
        bw = self.section.width()
        h = self.section.height()
        d_pos, d_neg, _, _ = self._depths()
        notes: list[str] = []

        # ---- minimum reinforcement (floor for every mode) ----
        rho_min = rho_min_beam(c.fc, s_steel.fy)
        rho_max = rho_max_beam(c.fc, s_steel.fy)
        As_min = rho_min * bw * d_pos
        As_max = rho_max * bw * d_pos

        # ---- per-station As required ----
        if mode in ("demand", "capacity") and demands is not None:
            As_top_i_req   = max(As_min, _as_req(demands.Mu_neg_i,   bw, d_neg, c, s_steel))
            As_top_mid_req = max(As_min, _as_req(demands.Mu_neg_mid, bw, d_neg, c, s_steel))
            As_top_j_req   = max(As_min, _as_req(demands.Mu_neg_j,   bw, d_neg, c, s_steel))
            As_bot_i_req   = max(As_min, _as_req(demands.Mu_pos_i,   bw, d_pos, c, s_steel))
            As_bot_mid_req = max(As_min, _as_req(demands.Mu_pos_mid, bw, d_pos, c, s_steel))
            As_bot_j_req   = max(As_min, _as_req(demands.Mu_pos_j,   bw, d_pos, c, s_steel))
        else:
            As_top_i_req = As_top_mid_req = As_top_j_req = As_min
            As_bot_i_req = As_bot_mid_req = As_bot_j_req = As_min

        # §18.6.3.4: As anywhere >= 1/4 of max As at face for either sense
        if self.seismic:
            As_top_face_max = max(As_top_i_req, As_top_j_req)
            As_bot_face_max = max(As_bot_i_req, As_bot_j_req)
            face_max = max(As_top_face_max, As_bot_face_max)
            min_anywhere = 0.25 * face_max
            for name in ("As_top_mid_req", "As_top_i_req", "As_top_j_req",
                         "As_bot_mid_req", "As_bot_i_req", "As_bot_j_req"):
                pass  # we update individually below
            As_top_mid_req = max(As_top_mid_req, min_anywhere)
            As_top_i_req   = max(As_top_i_req,   min_anywhere)
            As_top_j_req   = max(As_top_j_req,   min_anywhere)
            As_bot_mid_req = max(As_bot_mid_req, min_anywhere)
            As_bot_i_req   = max(As_bot_i_req,   min_anywhere)
            As_bot_j_req   = max(As_bot_j_req,   min_anywhere)

        As_top_cont_req = max(As_top_i_req, As_top_mid_req, As_top_j_req)
        As_bot_cont_req = max(As_bot_i_req, As_bot_mid_req, As_bot_j_req)

        # §18.6.3.3: As_bot >= 0.5 As_top  (rebar is continuous)
        if self.seismic and As_bot_cont_req < 0.5 * As_top_cont_req:
            As_bot_cont_req = 0.5 * As_top_cont_req
            notes.append("§18.6.3.3 — As_bot raised to 0.5 As_top.")

        # Warn if the required As exceeds rho_max·bw·d (= 0.5·rho_b·bw·d)
        if As_top_cont_req > As_max:
            notes.append(
                f"As_top required {As_top_cont_req:.0f} mm² exceeds rho_max·bw·d = "
                f"{As_max:.0f} mm² (rho_max = 0.5·rho_b = {rho_max:.4f}). "
                f"Consider larger section or compression steel."
            )
        if As_bot_cont_req > As_max:
            notes.append(
                f"As_bot required {As_bot_cont_req:.0f} mm² exceeds rho_max·bw·d = "
                f"{As_max:.0f} mm² (rho_max = 0.5·rho_b = {rho_max:.4f})."
            )

        # ---- s_max in confined zone (§18.6.4.4) ----
        if self.seismic:
            s_conf = s_max_seismic_smf(
                d=d_pos, db_long_min=self.db_long_min,
                db_hoop=db_stirrup or self.db_stirrup,
                grade=s_steel.grade,
            )
        else:
            s_conf = d_pos / 2.0
        s_middle = s_max_seismic_outside(d=d_pos) if self.seismic else d_pos / 2.0

        # ---- shear demand per mode ----
        Vu_i_used = Vu_mid_used = Vu_j_used = None
        Vu_design = None       # the controlling Vu for av/s
        vc_zero = False

        if mode == "demand" and demands is not None:
            Vu_i_used   = abs(demands.Vu_i)
            Vu_mid_used = abs(demands.Vu_mid)
            Vu_j_used   = abs(demands.Vu_j)
            Vu_design = max(Vu_i_used, Vu_mid_used, Vu_j_used)
        elif mode == "capacity":
            Mpr_pos = self._mpr(direction="pos")
            Mpr_neg = self._mpr(direction="neg")
            V_seismic = v_seismic_from_mpr(
                Mpr_pos_i=Mpr_pos, Mpr_neg_i=Mpr_neg,
                Mpr_pos_j=Mpr_pos, Mpr_neg_j=Mpr_neg,
                ln=self._ln_m,
            )
            vg_i, vg_mid, vg_j = self._resolve_vg(demands)
            Vu_i_used   = V_seismic + vg_i
            Vu_mid_used = V_seismic + vg_mid
            Vu_j_used   = V_seismic + vg_j
            Vu_design = max(Vu_i_used, Vu_j_used)   # governs lo stirrup design

            # §18.6.5.2 — Vc = 0 in lo when BOTH:
            #   (a) earthquake-induced shear >= 1/2 total Vu within lo
            #   (b) Pu < Ag*fc/20  (always true for beams since Pu ≈ 0)
            seismic_share = V_seismic / Vu_design if Vu_design > 0 else 0.0
            cond_a = seismic_share >= 0.5
            cond_b = True   # Pu ≈ 0 for beams
            if cond_a and cond_b:
                vc_zero = True
                notes.append(
                    f"§18.6.5.2 — Vc = 0 in lo "
                    f"(seismic share = {seismic_share*100:.0f}% of Ve, Pu ≈ 0)."
                )
            else:
                notes.append(
                    f"§18.6.5.2 — Vc retained "
                    f"(seismic share = {seismic_share*100:.0f}% of Ve, < 50%)."
                )

        av_s_req = self._av_s_required_for(Vu_design, vc_zero=vc_zero)
        av_s_req_min = av_s_minimum(b=bw, concrete=c, fyt=self.fyt)
        av_s_req = max(av_s_req or 0.0, av_s_req_min)

        chosen_db_stirrup, chosen_n_legs, chosen_s = self._pick_stirrup(
            av_s_req, s_conf, db_stirrup, n_legs,
        )
        a_per_leg = pi * chosen_db_stirrup ** 2 / 4.0
        av_prov = chosen_n_legs * a_per_leg
        if chosen_s < s_conf - 1e-6:
            notes.append(
                f"Spacing tightened to {chosen_s:.0f} mm "
                f"(below §18.6.4.4 cap of {s_conf:.0f} mm) to satisfy Av/s demand."
            )

        # ---- pick longitudinal bars ----
        chosen_db_top, chosen_n_top = self._pick_longitudinal(
            As_top_cont_req, chosen_db_stirrup, db_top, n_top,
        )
        chosen_db_bot, chosen_n_bot = self._pick_longitudinal(
            As_bot_cont_req, chosen_db_stirrup, db_bot, n_bot,
        )
        As_top_provided = chosen_n_top * pi * chosen_db_top ** 2 / 4.0
        As_bot_provided = chosen_n_bot * pi * chosen_db_bot ** 2 / 4.0

        # §18.6.3.2: at least two bars continuous top/bottom
        two_top_ok = chosen_n_top >= 2
        two_bot_ok = chosen_n_bot >= 2

        # ---- phi_Vn provided by the proposed stirrups (use chosen_s) ----
        Vc = vc_simplified(b=bw, d=d_pos, concrete=c)
        Vs_prov = vs_capacity(Av=av_prov, fyt=self.fyt, d=d_pos, s=chosen_s)
        phi_Vn_prov = phi_shear() * (Vc + Vs_prov)

        # ---- Mpr / Ve_capacity (always reported when seismic) ----
        Mpr_pos = Mpr_neg = Ve_cap = ratio_VePhiVn = None
        if self.seismic:
            Mpr_pos = self._mpr(direction="pos")
            Mpr_neg = self._mpr(direction="neg")
            V_seis = v_seismic_from_mpr(
                Mpr_pos_i=Mpr_pos, Mpr_neg_i=Mpr_neg,
                Mpr_pos_j=Mpr_pos, Mpr_neg_j=Mpr_neg,
                ln=self._ln_m,
            )
            vg_i, vg_mid, vg_j = self._resolve_vg(demands)
            Ve_cap = V_seis + max(vg_i, vg_j)    # governs lo design
            if phi_Vn_prov > 0:
                ratio_VePhiVn = Ve_cap / phi_Vn_prov

        # §18.6.3 ratios from the chosen continuous reinforcement
        ratio_pos_neg_i = (As_bot_provided / As_top_provided
                           if As_top_provided > 0 else None)
        ratio_pos_neg_j = ratio_pos_neg_i
        face_max_prov = max(As_top_provided, As_bot_provided)
        min_any_prov  = min(As_top_provided, As_bot_provided)
        ratio_min_max = min_any_prov / face_max_prov if face_max_prov > 0 else None

        return BeamDesignProposal(
            mode=mode,
            As_top_i_required=As_top_i_req,
            As_top_mid_required=As_top_mid_req,
            As_top_j_required=As_top_j_req,
            As_bot_i_required=As_bot_i_req,
            As_bot_mid_required=As_bot_mid_req,
            As_bot_j_required=As_bot_j_req,
            As_top_continuous=As_top_provided,
            As_bot_continuous=As_bot_provided,
            db_top=chosen_db_top, n_top=chosen_n_top,
            db_bot=chosen_db_bot, n_bot=chosen_n_bot,
            db_stirrup=chosen_db_stirrup, n_legs=chosen_n_legs,
            av_provided=av_prov,
            spacing_confined=chosen_s, spacing_middle=s_middle,
            lo=2.0 * h,
            Vu_i_used=Vu_i_used, Vu_mid_used=Vu_mid_used, Vu_j_used=Vu_j_used,
            av_s_required=av_s_req, vc_zero_active=vc_zero,
            Mpr_pos=Mpr_pos, Mpr_neg=Mpr_neg,
            Ve_capacity=Ve_cap, phi_Vn=phi_Vn_prov,
            ratio_Ve_over_phiVn=ratio_VePhiVn,
            ratio_pos_over_neg_at_i=ratio_pos_neg_i,
            ratio_pos_over_neg_at_j=ratio_pos_neg_j,
            ratio_min_over_max_anywhere=ratio_min_max,
            two_bars_top_ok=two_top_ok,
            two_bars_bot_ok=two_bot_ok,
            notes=tuple(notes),
        )

    @staticmethod
    def _design_envelope(prop_min, prop_dem, prop_cap) -> BeamDesignProposal:
        modes = [prop_min] + ([prop_dem] if prop_dem else []) + [prop_cap]

        def steel_index(p: BeamDesignProposal) -> float:
            long_a = p.As_top_continuous + p.As_bot_continuous
            trans_a = (p.db_stirrup ** 2 * p.n_legs
                       / max(p.spacing_confined, 1e-9))
            return long_a + trans_a * 100.0

        worst = max(modes, key=steel_index)
        all_notes = tuple(f"[{p.mode}] {n}" for p in modes for n in p.notes)

        return BeamDesignProposal(
            mode="envelope",
            As_top_i_required=max(p.As_top_i_required for p in modes),
            As_top_mid_required=max(p.As_top_mid_required for p in modes),
            As_top_j_required=max(p.As_top_j_required for p in modes),
            As_bot_i_required=max(p.As_bot_i_required for p in modes),
            As_bot_mid_required=max(p.As_bot_mid_required for p in modes),
            As_bot_j_required=max(p.As_bot_j_required for p in modes),
            As_top_continuous=worst.As_top_continuous,
            As_bot_continuous=worst.As_bot_continuous,
            db_top=worst.db_top, n_top=worst.n_top,
            db_bot=worst.db_bot, n_bot=worst.n_bot,
            db_stirrup=worst.db_stirrup, n_legs=worst.n_legs,
            av_provided=worst.av_provided,
            spacing_confined=worst.spacing_confined,
            spacing_middle=worst.spacing_middle,
            lo=worst.lo,
            Vu_i_used=max((p.Vu_i_used or 0.0) for p in modes) or None,
            Vu_mid_used=max((p.Vu_mid_used or 0.0) for p in modes) or None,
            Vu_j_used=max((p.Vu_j_used or 0.0) for p in modes) or None,
            av_s_required=max((p.av_s_required or 0.0) for p in modes) or None,
            vc_zero_active=any(p.vc_zero_active for p in modes),
            Mpr_pos=prop_cap.Mpr_pos, Mpr_neg=prop_cap.Mpr_neg,
            Ve_capacity=prop_cap.Ve_capacity,
            phi_Vn=worst.phi_Vn,
            ratio_Ve_over_phiVn=worst.ratio_Ve_over_phiVn,
            ratio_pos_over_neg_at_i=worst.ratio_pos_over_neg_at_i,
            ratio_pos_over_neg_at_j=worst.ratio_pos_over_neg_at_j,
            ratio_min_over_max_anywhere=worst.ratio_min_over_max_anywhere,
            two_bars_top_ok=all(p.two_bars_top_ok for p in modes),
            two_bars_bot_ok=all(p.two_bars_bot_ok for p in modes),
            notes=all_notes,
        )

    # =============================================================== #
    # Plot proxy
    # =============================================================== #
    @property
    def plot(self) -> _BeamPlotter:
        if self._plotter is None:
            self._plotter = _BeamPlotter(self)
        return self._plotter

    # =============================================================== #
    # Reports
    # =============================================================== #
    # ---- formatting helpers (sensitive to the active unit system) ----
    def _fmt_len(self, mm_value: float) -> str:
        """Format a length value (internally mm) in the user's units."""
        u = self.units
        val = mm_value * u.length_factor
        if u.length == "m":   return f"{val:.3f}"
        if u.length == "cm":  return f"{val:.1f}"
        if u.length == "in":  return f"{val:.2f}"
        if u.length == "ft":  return f"{val:.3f}"
        return f"{val:.0f}"

    def _fmt_area(self, mm2_value: float) -> str:
        """Format an area value (internally mm²) in the user's units."""
        u = self.units
        val = mm2_value * u.area_factor
        if u.length == "m":  return f"{val*1e6:.0f} mm^2"   # mm² regardless
        if u.length == "cm": return f"{val:.1f} cm^2"
        if u.length == "in": return f"{val:.2f} in^2"
        if u.length == "ft": return f"{val*144:.0f} in^2"
        return f"{val:.0f} mm^2"

    def _fmt_force(self, kN_value: float) -> str:
        u = self.units
        return f"{kN_value * u.force_factor:.1f} {u.force}"

    def _fmt_moment(self, kNm_value: float) -> str:
        u = self.units
        return f"{kNm_value * u.moment_factor:.1f} {u.force}.{u.length}"

    def summary(self) -> None:
        u = self.units
        s = self.section
        rebar = s.rebar
        n_bars = sum(g.n for g in rebar.groups)
        Ast = rebar.total_area
        cap = self.capacity()

        print(f"=== {self.label} ===  (units: {u.name})")
        print(f"Section   : {self._fmt_len(s.width())}x{self._fmt_len(s.height())} {u.length}")
        print(f"Materials : fc = {s.concrete.fc:.0f} MPa, fy = {self.steel.fy:.0f} MPa")
        print(f"Rebar     : {n_bars} bars, As_tot = {self._fmt_area(Ast)}")
        print(f"            rho  = {cap.rho:.4f}  (bottom)")
        print(f"            rho' = {cap.rho_p:.4f}  (top)")
        print(f"            range [rho_min = {cap.rho_min:.4f} .. rho_max = {cap.rho_max:.4f}]")
        print(f"            rho_b = {cap.rho_balanced:.4f}  (rho_max = 0.5*rho_b)")
        print(f"phi*Mn+   : {self._fmt_moment(cap.phi_Mn_pos)}")
        print(f"phi*Mn-   : {self._fmt_moment(cap.phi_Mn_neg)}")
        print(f"phi*Vn    : {self._fmt_force(cap.phi_Vn)}")
        if cap.Mpr_pos:
            print(f"Mpr+      : {self._fmt_moment(cap.Mpr_pos)}")
            print(f"Mpr-      : {self._fmt_moment(cap.Mpr_neg)}")
        print(f"TC        : {cap.tension_controlled}")

    def print_checks(self, demands: BeamDemands | None = None) -> None:
        """Pretty-print the full set of ACI 318-25 checks."""
        cap = self.capacity()
        u = self.units
        chk = self.check(demands) if demands else None

        print(f"=== {self.label} -- ACI 318-25 Checks ===  (units: {u.name})")
        # Cuantia
        rho_ok_bot = cap.rho_min <= cap.rho <= cap.rho_max
        rho_ok_top = cap.rho_min <= cap.rho_p <= cap.rho_max
        print(f"§9.6.1.2 / 22.2  Cuantia (range 0.5*rho_b):")
        print(f"  rho  (bot) = {cap.rho:.4f}    [{cap.rho_min:.4f} .. {cap.rho_max:.4f}]   {'OK' if rho_ok_bot else 'FAIL'}")
        print(f"  rho' (top) = {cap.rho_p:.4f}    [{cap.rho_min:.4f} .. {cap.rho_max:.4f}]   {'OK' if rho_ok_top else 'FAIL'}")
        print(f"  rho_balanced = {cap.rho_balanced:.4f}    (ACI §18.6.3.1 SMF cap = {cap.rho_max_aci_smf:.4f})")
        # Tension-controlled
        print(f"§21.2.2          Tension-controlled                          : {'OK' if cap.tension_controlled else 'FAIL'}")
        # SMF continuity (only meaningful with seismic)
        if self.seismic:
            r_pos_neg = (cap.phi_Mn_pos / cap.phi_Mn_neg) if cap.phi_Mn_neg > 0 else float("inf")
            face_max = max(cap.phi_Mn_pos, cap.phi_Mn_neg)
            face_min = min(cap.phi_Mn_pos, cap.phi_Mn_neg)
            r_min_max = face_min / face_max if face_max > 0 else 0.0
            print(f"§18.6.3.2        >=2 bars continuous (top, bot)            : "
                  f"requires layout check at apply()")
            print(f"§18.6.3.3        phi_Mn+ >= 0.5 phi_Mn-                    : "
                  f"{r_pos_neg:.2f}   {'OK' if r_pos_neg >= 0.5 else 'FAIL'}")
            print(f"§18.6.3.4        phi_Mn anywhere >= 0.25 max               : "
                  f"{r_min_max:.2f}   {'OK' if r_min_max >= 0.25 else 'FAIL'}")
        # Demand-based ratios
        if chk is not None:
            print(f"Demand ratios (per station):")
            print(f"  M (i)  +/-: {chk.ratio_M_pos_i:.2f} / {chk.ratio_M_neg_i:.2f}")
            print(f"  M (mid)+/-: {chk.ratio_M_pos_mid:.2f} / {chk.ratio_M_neg_mid:.2f}")
            print(f"  M (j)  +/-: {chk.ratio_M_pos_j:.2f} / {chk.ratio_M_neg_j:.2f}")
            print(f"  V (i/mid/j): {chk.ratio_V_i:.2f} / {chk.ratio_V_mid:.2f} / {chk.ratio_V_j:.2f}")
            print(f"  Overall passed = {chk.passed}")

    def current_proposal(
        self, demands: BeamDemands | None = None,
    ) -> BeamDesignProposal:
        """Build a BeamDesignProposal reflecting the CURRENT beam detailing.

        Useful to (a) snapshot the as-input config before exploring other
        proposals, (b) revert via `beam.apply(snapshot)` after adopting a
        different mode:

            snapshot = beam.current_proposal(demands)
            beam.apply(results.envelope)
            ...
            beam.apply(snapshot)         # revert to the original
        """
        cap = self.capacity()
        db_top, n_top = self._infer_layer_bars("top")
        db_bot, n_bot = self._infer_layer_bars("bot")
        As_top = self._As_in_layer("top")
        As_bot = self._As_in_layer("bot")
        n_legs = self._current_n_legs()
        s_used = self.s if (self.s and self.s > 0) else float("nan")

        bw = self.section.width()
        h = self.section.height()
        d_pos, _, _, _ = self._depths()

        # Demand-driven required As (for the feasibility check that
        # downstream code may run).
        as_min = rho_min_beam(self.section.concrete.fc, self.steel.fy) * bw * d_pos
        c = self.section.concrete
        s_steel = self.steel
        if demands is not None:
            As_top_i = max(as_min, _as_req(demands.Mu_neg_i, bw, d_pos, c, s_steel))
            As_top_m = max(as_min, _as_req(demands.Mu_neg_mid, bw, d_pos, c, s_steel))
            As_top_j = max(as_min, _as_req(demands.Mu_neg_j, bw, d_pos, c, s_steel))
            As_bot_i = max(as_min, _as_req(demands.Mu_pos_i, bw, d_pos, c, s_steel))
            As_bot_m = max(as_min, _as_req(demands.Mu_pos_mid, bw, d_pos, c, s_steel))
            As_bot_j = max(as_min, _as_req(demands.Mu_pos_j, bw, d_pos, c, s_steel))
        else:
            As_top_i = As_top_m = As_top_j = as_min
            As_bot_i = As_bot_m = As_bot_j = as_min

        # Seismic / Ve at current state
        Mpr_pos = cap.Mn_pr_pos
        Mpr_neg = cap.Mn_pr_neg
        if self.seismic and Mpr_pos is not None:
            vg_i, vg_mid, vg_j = self._resolve_vg(demands)
            V_seis = v_seismic_from_mpr(
                Mpr_pos_i=Mpr_pos, Mpr_neg_i=Mpr_neg,
                Mpr_pos_j=Mpr_pos, Mpr_neg_j=Mpr_neg,
                ln=self._ln_m,
            )
            Ve_cap = V_seis + max(vg_i, vg_j)
            Vu_i_used = V_seis + vg_i
            Vu_mid_used = V_seis + vg_mid
            Vu_j_used = V_seis + vg_j
        else:
            Ve_cap = None
            Vu_i_used = Vu_mid_used = Vu_j_used = None

        # Av/s required at current spacing (for feasibility comparison)
        av_s_req = (self._av_s_required_for(Ve_cap, vc_zero=False)
                    if Ve_cap is not None else None)
        a_per_leg = pi * self.db_stirrup ** 2 / 4.0
        av_prov = n_legs * a_per_leg

        ratio_pos_neg = (As_bot / As_top) if As_top > 0 else None
        face_max = max(As_top, As_bot)
        ratio_min_max = (min(As_top, As_bot) / face_max) if face_max > 0 else None

        return BeamDesignProposal(
            mode="provided",
            As_top_i_required=As_top_i, As_top_mid_required=As_top_m,
            As_top_j_required=As_top_j,
            As_bot_i_required=As_bot_i, As_bot_mid_required=As_bot_m,
            As_bot_j_required=As_bot_j,
            As_top_continuous=As_top, As_bot_continuous=As_bot,
            db_top=db_top, n_top=n_top, db_bot=db_bot, n_bot=n_bot,
            db_stirrup=self.db_stirrup, n_legs=n_legs,
            av_provided=av_prov,
            spacing_confined=s_used, spacing_middle=s_used, lo=2.0 * h,
            Vu_i_used=Vu_i_used, Vu_mid_used=Vu_mid_used,
            Vu_j_used=Vu_j_used,
            av_s_required=av_s_req,
            vc_zero_active=False,
            Mpr_pos=Mpr_pos, Mpr_neg=Mpr_neg,
            Ve_capacity=Ve_cap, phi_Vn=cap.phi_Vn,
            ratio_Ve_over_phiVn=(Ve_cap / cap.phi_Vn
                                 if (Ve_cap and cap.phi_Vn > 0) else None),
            ratio_pos_over_neg_at_i=ratio_pos_neg,
            ratio_pos_over_neg_at_j=ratio_pos_neg,
            ratio_min_over_max_anywhere=ratio_min_max,
            two_bars_top_ok=n_top >= 2,
            two_bars_bot_ok=n_bot >= 2,
            notes=(),
        )

    def print_state(self, demands: BeamDemands | None = None) -> None:
        """Print the CURRENT beam state in the print_design format.

        Use this to see the capacity of the as-input design, optionally
        with demand ratios per station.
        """
        u = self.units
        cap = self.capacity()
        db_top, n_top = self._infer_layer_bars("top")
        db_bot, n_bot = self._infer_layer_bars("bot")
        As_top = self._As_in_layer("top")
        As_bot = self._As_in_layer("bot")
        n_legs = self._current_n_legs()

        print(f"--- PROVIDED ---")
        print(f"  Top         : {n_top}φ{db_top:.0f}    (As = {self._fmt_area(As_top)})")
        print(f"  Bottom      : {n_bot}φ{db_bot:.0f}    (As = {self._fmt_area(As_bot)})")
        if self.s and self.s > 0:
            print(f"  Stirrups    : φ{self.db_stirrup:.0f} x {n_legs} legs @ "
                  f"{self._fmt_len(self.s)} {u.length}")
        else:
            print(f"  Stirrups    : φ{self.db_stirrup:.0f} x {n_legs} legs  "
                  f"(spacing not set)")
        print(f"  phi*Mn (+/-): {self._fmt_moment(cap.phi_Mn_pos)} / "
              f"{self._fmt_moment(cap.phi_Mn_neg)}")
        print(f"  phi*Vn      : {self._fmt_force(cap.phi_Vn)}")
        if cap.Mpr_pos is not None:
            vg_i, _, vg_j = self._resolve_vg(demands)
            V_seis = (cap.Mpr_pos + cap.Mpr_neg) / self._ln_m
            Ve_cap = V_seis + max(vg_i, vg_j)
            print(f"  Mpr (+/-)   : {self._fmt_moment(cap.Mpr_pos)} / "
                  f"{self._fmt_moment(cap.Mpr_neg)}"
                  f"   -> Ve_capacity = {self._fmt_force(Ve_cap)}")
        # rho summary
        rho_ok_bot = cap.rho_min <= cap.rho   <= cap.rho_max
        rho_ok_top = cap.rho_min <= cap.rho_p <= cap.rho_max
        print(f"  rho / rho'  : {cap.rho:.4f} / {cap.rho_p:.4f}    "
              f"range [{cap.rho_min:.4f} .. {cap.rho_max:.4f}]    "
              f"{'OK' if (rho_ok_bot and rho_ok_top) else 'FAIL'}")
        # If demands given: ratios and verdict
        if demands is not None:
            chk = self.check(demands)
            print(f"  Demand check: ratio M = {chk.ratio_M_overall:.2f}   "
                  f"ratio V (i/m/j) = {chk.ratio_V_i:.2f} / "
                  f"{chk.ratio_V_mid:.2f} / {chk.ratio_V_j:.2f}   "
                  f"passed = {chk.passed}")
        print()

    def print_design(
        self,
        results: BeamDesignResults | None = None,
        demands: BeamDemands | None = None,
        *,
        include_provided: bool = True,
    ) -> None:
        """Print the three-mode design results.

        When `include_provided=True` (default), the current beam state is
        shown first as a PROVIDED section so the as-input design is
        directly comparable against the proposals.
        """
        if results is None:
            results = self.design_results
        if results is None:
            raise ValueError("No design results to print. Call beam.design(demands) first.")
        u = self.units

        def fmt(p: BeamDesignProposal, header: str) -> None:
            print(f"--- {header} ---")
            print(f"  Top         : {p.n_top}φ{p.db_top:.0f}    (As = {self._fmt_area(p.As_top_continuous)})")
            print(f"  Bottom      : {p.n_bot}φ{p.db_bot:.0f}    (As = {self._fmt_area(p.As_bot_continuous)})")
            print(f"  Stirrups    : φ{p.db_stirrup:.0f} x {p.n_legs} legs @ {self._fmt_len(p.spacing_confined)} {u.length}"
                  f"   (middle s = {self._fmt_len(p.spacing_middle)} {u.length}, lo = {self._fmt_len(p.lo)} {u.length})")
            if p.Mpr_pos is not None:
                print(f"  Mpr (+/-)   : {self._fmt_moment(p.Mpr_pos)} / {self._fmt_moment(p.Mpr_neg)}"
                      f"   -> Ve_capacity = {self._fmt_force(p.Ve_capacity)}")
            if p.Vu_i_used is not None:
                print(f"  Ve (i/m/j)  : {self._fmt_force(p.Vu_i_used)}"
                      f" / {self._fmt_force(p.Vu_mid_used or 0.0)}"
                      f" / {self._fmt_force(p.Vu_j_used or 0.0)}")
            if p.phi_Vn is not None:
                print(f"  phi*Vn      : {self._fmt_force(p.phi_Vn)}"
                      + (f"   Ve/phiVn = {p.ratio_Ve_over_phiVn:.2f}"
                         if p.ratio_Ve_over_phiVn is not None else ""))
            if p.vc_zero_active:
                print(f"  Vc = 0 in lo (§18.6.5.2)")
            if p.notes:
                for n in p.notes:
                    print(f"  note: {n}")
            print()

        print(f"=== {self.label} Design Results ===  (units: {u.name})\n")
        if include_provided:
            self.print_state(demands)
        fmt(results.minimum, "MINIMUM")
        if results.demand is not None:
            fmt(results.demand, "DEMAND")
        fmt(results.capacity, "CAPACITY")
        fmt(results.envelope, "ENVELOPE")

    def report(self) -> dict:
        u = self.units
        cap = self.capacity()
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
                "fy_MPa": self.steel.fy,
            },
            "rebar": {
                "n_bars": sum(g.n for g in s.rebar.groups),
                "As_total": s.rebar.total_area * u.area_factor,
                "rho": cap.rho, "rho_p": cap.rho_p,
            },
            "capacity": {
                "phi_Mn_pos": cap.phi_Mn_pos * u.moment_factor,
                "phi_Mn_neg": cap.phi_Mn_neg * u.moment_factor,
                "phi_Vn": cap.phi_Vn * u.force_factor,
                "Mn_pr_pos": (cap.Mn_pr_pos or 0.0) * u.moment_factor,
                "Mn_pr_neg": (cap.Mn_pr_neg or 0.0) * u.moment_factor,
            },
            "params": {
                "ln_m": self.ln, "cover_mm": self.cover,
                "seismic": self.seismic,
                "s_mm": self.s, "Vg_kN": self.Vg,
            },
        }

    def __repr__(self) -> str:
        s = self.section
        return (
            f"Beam(label={self.label!r}, section={s.width():.0f}×{s.height():.0f} mm, "
            f"fc={s.concrete.fc:.0f} MPa, units={self.units.name})"
        )


# ---------------------------------------------------------------- #
# Module-level helper
# ---------------------------------------------------------------- #
def _as_req(Mu: float, bw: float, d: float, c: Concrete, s: Steel) -> float:
    if Mu is None or Mu <= 0:
        return 0.0
    As = as_required_singly(Mu=Mu, b=bw, d=d, concrete=c, steel=s)
    if not np.isfinite(As):
        return 1e9
    return As


def _build_beam_rebar(
    *,
    bw: float, h: float, cover: float, db_stirrup: float,
    n_top: int, db_top: float,
    n_bot: int, db_bot: float,
    steel: Steel,
) -> RebarLayout:
    """Build a RebarLayout for a beam with two layers (top + bottom).

    `cover` is the CLEAR cover from the outer face to the OUTSIDE EDGE
    of the stirrup. Bar centroids sit at  cover + db_stirrup + db/2
    from the corresponding face. The two layers can have different
    diameters without distortion (each layer keeps its own offset).
    """
    y_top = h / 2.0 - (cover + db_stirrup + db_top / 2.0)
    y_bot = -h / 2.0 + (cover + db_stirrup + db_bot / 2.0)

    # Span available for centroids in each layer (face-to-face)
    half_span_top = bw / 2.0 - cover - db_stirrup - db_top / 2.0
    half_span_bot = bw / 2.0 - cover - db_stirrup - db_bot / 2.0

    xs_top = (np.linspace(-half_span_top, half_span_top, n_top)
              if n_top > 1 else np.array([0.0]))
    xs_bot = (np.linspace(-half_span_bot, half_span_bot, n_bot)
              if n_bot > 1 else np.array([0.0]))

    a_top = pi * db_top ** 2 / 4.0
    a_bot = pi * db_bot ** 2 / 4.0

    bars = [Rebar(x=float(x), y=y_top, area=a_top) for x in xs_top]
    bars += [Rebar(x=float(x), y=y_bot, area=a_bot) for x in xs_bot]
    return RebarLayout(groups=(RebarGroup(bars=tuple(bars), steel=steel),))
