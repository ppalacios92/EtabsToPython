"""Beam-column joint (nudo) design per ACI 318-25 §18.8 / §15.5.2.

A :class:`BeamColumnJoint` is assembled from real ``Column`` and
``Beam`` objects, exactly like ``Wall`` composes ``Column`` boundary
elements. You hand it the column above/below and up to four framing
beams; the joint derives its own demand and capacity.

Design philosophy (agreed with the user):
    * Member moments come from the INTERACTION DIAGRAM, never from a
      hand ``1.25 fy As`` calculation. For beams the default is the
      moment at ``P = 0`` (``moment_condition='P0'``); ``'balanced'``
      uses the peak (probable) moment. ``probable=True`` builds the
      curve with ``fy_pr = 1.25 fy`` and ``phi = 1.0``.
    * Joint shear is checked SEPARATELY in each principal direction
      (§15.4.2.1). Left/right beams load the X direction; up/down beams
      load the Y direction. Beams on the other axis act as transverse
      confinement.
    * Beam eccentricity is handled only through the reduced effective
      area Aj (§15.5.2.2). The induced torsion ``T = M * e`` is NOT a
      joint action — it is reported so it can be fed back to the column
      (§22.7). An ``e > b_c/8`` warning follows ACI PRC-352.
    * The angle on a framing beam is GEOMETRIC only (it does not rotate
      forces): demands are assumed already resolved into principal
      directions by the analysis model.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from math import pi
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from design.columns.interaction import interaction_diagram
from design.common.element_base import ElementBase
from design.common.probable import probable_interaction_diagram
from design.common.spacing import s_max_confined_column
from design.joints.provisions import (
    effective_joint_area,
    gamma_joint,
    joint_shear_capacity,
    ldh_hook,
    min_joint_depth,
    phi_joint_shear,
    transverse_beam_confines,
)

if TYPE_CHECKING:
    from design.beams.beam import Beam
    from design.columns.column import Column
    from design.sections.base import Section

MomentCondition = Literal["P0", "balanced", "demand"]

# Internal lever arm jd/d for converting a beam moment into a tension
# force at the joint (ACI PRC-352 joint model). T = M / jd.
_JOINT_LEVER_ARM = 0.875


# --------------------------------------------------------------------------
# Framing beam wrapper
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FramingBeam:
    """A beam framing into one face of the joint.

    ``angle_deg`` is GEOMETRIC (skew of the beam in plan); it does not
    project forces. ``eccentricity`` is the offset (mm) of the beam axis
    from the column centroid, perpendicular to the beam — it penalizes
    the effective joint area per §15.5.2.2.
    """
    beam: "Beam"
    angle_deg: float = 0.0
    eccentricity: float = 0.0


# --------------------------------------------------------------------------
# Result dataclasses
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class JointDemands:
    """Optional manual overrides of the per-direction joint shear (kN).

    Leave a field ``None`` to let the joint derive it from the members.
    """
    Vu_x: Optional[float] = None
    Vu_y: Optional[float] = None


@dataclass(frozen=True, slots=True)
class JointDirectionCapacity:
    axis: str
    Aj: float                 # mm2
    b_eff: float              # mm
    h_joint: float            # mm (column depth along the shear)
    gamma: float              # psi coefficient (Table 18.8.4.3)
    phi_Vn: float             # kN
    column_continuous: bool
    beam_continuous: bool
    transverse_confined: bool


@dataclass(frozen=True, slots=True)
class JointCapacity:
    x: JointDirectionCapacity
    y: JointDirectionCapacity
    joint_type: str
    phi_Vn: float             # governing (min) — satisfies Capacity protocol


@dataclass(frozen=True, slots=True)
class JointDirectionCheck:
    axis: str
    Vu: float                 # kN
    phi_Vn: float             # kN
    ratio: float
    bars_in_core: bool        # §18.8.5 — beam bars within column core
    torsion: float            # kN·m, M*e induced on the column (§22.7)
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JointCheck:
    capacity: JointCapacity
    x: JointDirectionCheck
    y: JointDirectionCheck
    ratio: float
    passed: bool
    notes: tuple[str, ...]

    def summary(self) -> None:
        """Print a structured one-line-per-axis report.

        Joints do NOT report steel quantities — confinement steel here
        comes from the column hoops, not from a joint-owned cage.
        """
        tag = lambda b: "PASS" if b else "FAIL"
        print(f"=== Joint check — {self.capacity.joint_type} ===")
        for axis, ck in (("X", self.x), ("Y", self.y)):
            note_extra = "  (bars in core)" if ck.bars_in_core else ""
            print(f"  [{axis}]  Vu = {ck.Vu:7.1f} kN    "
                  f"phi*Vn = {ck.phi_Vn:7.1f} kN    "
                  f"ratio = {ck.ratio:.3f}    [{tag(ck.ratio <= 1.0)}]{note_extra}")
            if abs(ck.torsion) > 1e-6:
                print(f"       torsion M*e = {ck.torsion:.1f} kN·m  (§22.7)")
        print(f"  ---")
        print(f"  passed         : {self.passed}")
        if self.notes:
            print(f"  Notes:")
            for n in self.notes:
                print(f"    - {n}")


@dataclass(frozen=True, slots=True)
class JointDesignProposal:
    mode: str
    s_max_confined: float     # §18.8.3 / §18.7.5.3 hoop spacing (mm)
    ash_reduction: bool       # §18.8.3.2 (4 beams, >=3/4 width) -> 1/2 Ash, s<=150
    ldh_required: float       # §18.8.5.1 for the largest beam bar (mm)
    min_depth_x: float        # §18.8.2.3
    min_depth_y: float
    depth_provided_x: float
    depth_provided_y: float
    depth_ok: bool
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JointDesignResults:
    minimum: JointDesignProposal
    capacity: JointDesignProposal
    envelope: JointDesignProposal

    def __iter__(self):
        yield self.minimum
        yield self.capacity
        yield self.envelope

    def as_dict(self) -> dict:
        return {"minimum": self.minimum,
                "capacity": self.capacity,
                "envelope": self.envelope}


# --------------------------------------------------------------------------
# Small section helpers (interaction-diagram driven)
# --------------------------------------------------------------------------
def _moment_at_condition(diagram, condition: MomentCondition,
                         Nu: float = 0.0) -> float:
    """Nominal/probable moment (kN·m) from a P-M diagram at a condition."""
    if condition == "balanced":
        return diagram.balanced_point().Mn
    order = np.argsort(diagram.Pn)
    return float(np.interp(Nu, diagram.Pn[order], diagram.Mn[order],
                           left=0.0, right=0.0))


def _beam_moment(section: "Section", sense: str, *,
                 condition: MomentCondition, probable: bool) -> float:
    """Beam moment for one bending sense from its interaction diagram.

    ``sense='pos'`` -> tension at bottom (angle 0); ``'neg'`` -> tension
    at top (angle 180). ``probable`` builds the 1.25 fy / phi=1.0 curve.
    """
    angle = 0.0 if sense == "pos" else 180.0
    if probable:
        diagram = probable_interaction_diagram(section, angle_deg=angle)
    else:
        diagram = interaction_diagram(section, angle_deg=angle)
    return _moment_at_condition(diagram, condition)


def _beam_effective_depth(section: "Section", sense: str) -> float:
    x0, y0, x1, y1 = section.bounding_box()
    ys = [r.y for r, _ in section.rebar.iter_bars()]
    if not ys:
        return 0.9 * (y1 - y0)
    if sense == "pos":          # tension at bottom -> d from top fiber
        return y1 - min(ys)
    return max(ys) - y0         # tension at top -> d from bottom fiber


def _largest_db_by_grade(section: "Section") -> tuple[float, float]:
    """Largest longitudinal bar diameter (mm) for Grade<=60 and Grade>=80."""
    g60 = g80 = 0.0
    for r, s in section.rebar.iter_bars():
        db = (4.0 * r.area / pi) ** 0.5
        if s.grade >= 80:
            g80 = max(g80, db)
        else:
            g60 = max(g60, db)
    return g60, g80


# --------------------------------------------------------------------------
# Joint
# --------------------------------------------------------------------------
class BeamColumnJoint(ElementBase):
    """Reinforced-concrete beam-column joint, ACI 318-25 §18.8 / §15.5.2.

    Internal units (per ``design/AGENTS.md``): MPa, mm, kN, kN·m, mm2.

    At least one column (``col_above`` or ``col_below``) is required.
    Beams default to ``None`` (an absent face), which lets the joint
    infer its type (interior / exterior / corner / roof).
    """

    def __init__(
        self,
        *,
        col_above: Optional["Column"] = None,
        col_below: Optional["Column"] = None,
        beam_left: Optional[FramingBeam] = None,
        beam_right: Optional[FramingBeam] = None,
        beam_up: Optional[FramingBeam] = None,
        beam_down: Optional[FramingBeam] = None,
        moment_condition: MomentCondition = "P0",
        probable: bool = True,
        label: str = "Joint",
        units: int | str | None = None,
    ) -> None:
        if col_above is None and col_below is None:
            raise ValueError("A joint needs at least one column "
                             "(col_above or col_below).")
        self.col_above = col_above
        self.col_below = col_below
        self.beam_left = beam_left
        self.beam_right = beam_right
        self.beam_up = beam_up
        self.beam_down = beam_down
        self.moment_condition: MomentCondition = moment_condition
        self.probable = probable
        self.label = label
        self.units = units

        self._capacity: Optional[JointCapacity] = None

    # ---- representative geometry --------------------------------------
    @property
    def _col(self) -> "Column":
        return self.col_below or self.col_above  # type: ignore[return-value]

    @property
    def _concrete(self):
        return self._col.section.concrete

    def _story_height(self) -> float:
        lus = [c.lu for c in (self.col_above, self.col_below) if c is not None]
        return sum(lus) / len(lus) if lus else 0.0

    def _beams(self) -> tuple[FramingBeam, ...]:
        return tuple(b for b in (self.beam_left, self.beam_right,
                                 self.beam_up, self.beam_down) if b is not None)

    # ---- joint type ---------------------------------------------------
    def joint_type(self) -> str:
        n = len(self._beams())
        roof = self.col_above is None or self.col_below is None
        base = {4: "interior", 3: "exterior", 2: "corner",
                1: "exterior", 0: "column-only"}.get(n, "interior")
        return f"roof {base}" if roof else base

    # ---- per-direction tension force ----------------------------------
    def _tension(self, fb: Optional[FramingBeam], sense: str
                 ) -> tuple[float, float]:
        """Return (T kN, M kN·m) for one framing beam and bending sense."""
        if fb is None:
            return 0.0, 0.0
        M = _beam_moment(fb.beam.section, sense,
                         condition=self.moment_condition,
                         probable=self.probable)
        d = _beam_effective_depth(fb.beam.section, sense)
        if d <= 0:
            return 0.0, M
        T = M * 1000.0 / (_JOINT_LEVER_ARM * d)   # kN  (M kN·m -> N·mm /jd mm)
        return T, M

    def _axis_demand(self, b1: Optional[FramingBeam], b2: Optional[FramingBeam],
                     lc: float) -> dict:
        """Worst-sway joint shear for one principal axis.

        b1 hinges in one sense while b2 hinges in the opposite sense; the
        column shear that equilibrates the two beam moments relieves the
        joint (§18.8.4.1 — column shear consistent with beam Mpr).
        """
        best = {"Vu": -float("inf")}
        for s1, s2 in (("neg", "pos"), ("pos", "neg")):
            T1, M1 = self._tension(b1, s1)
            T2, M2 = self._tension(b2, s2)
            Vcol = (M1 + M2) * 1000.0 / lc if lc > 0 else 0.0   # kN
            Vu = T1 + T2 - Vcol
            if Vu > best["Vu"]:
                best = {"Vu": Vu, "T1": T1, "T2": T2,
                        "M1": M1, "M2": M2, "Vcol": Vcol}
        return best

    # ---- per-direction capacity ---------------------------------------
    def _direction_capacity(self, axis: str) -> JointDirectionCapacity:
        col = self._col
        c = self._concrete
        if axis == "x":
            beams = (self.beam_left, self.beam_right)
            transverse = (self.beam_up, self.beam_down)
            h_joint = col.section.width()          # column dim along X
            b_column = col.section.height()        # perpendicular (Y)
            col_face = col.section.width()         # face the Y-beams frame into
        else:
            beams = (self.beam_up, self.beam_down)
            transverse = (self.beam_left, self.beam_right)
            h_joint = col.section.height()
            b_column = col.section.width()
            col_face = col.section.height()

        present = [b for b in beams if b is not None]
        if present:
            gov = max(present, key=lambda fb: fb.beam.section.width())
            b_beam = gov.beam.section.width()
            e = gov.eccentricity
        else:
            b_beam, e = 0.0, 0.0

        Aj, b_eff = effective_joint_area(
            h_joint=h_joint, b_column=b_column, b_beam=b_beam,
            eccentricity=e, gross_area=col.section.gross_area(),
        )

        column_continuous = (self.col_above is not None
                             and self.col_below is not None)
        beam_continuous = all(b is not None for b in beams)
        tw = tuple(t.beam.section.width() for t in transverse if t is not None)
        transverse_confined = transverse_beam_confines(
            transverse_widths=tw, column_face_width=col_face,
        )

        gamma = gamma_joint(
            column_continuous=column_continuous,
            beam_continuous=beam_continuous,
            transverse_confined=transverse_confined,
        )
        phi_Vn = joint_shear_capacity(
            gamma=gamma, lam=c.lam, sqrt_fc=c.sqrt_fc, Aj=Aj, reduced=True,
        )
        return JointDirectionCapacity(
            axis=axis, Aj=Aj, b_eff=b_eff, h_joint=h_joint, gamma=gamma,
            phi_Vn=phi_Vn, column_continuous=column_continuous,
            beam_continuous=beam_continuous,
            transverse_confined=transverse_confined,
        )

    # ---- contract: run / capacity / check / design --------------------
    def run(self) -> "BeamColumnJoint":
        self._capacity = JointCapacity(
            x=self._direction_capacity("x"),
            y=self._direction_capacity("y"),
            joint_type=self.joint_type(),
            phi_Vn=0.0,
        )
        gov = min(self._capacity.x.phi_Vn, self._capacity.y.phi_Vn)
        self._capacity = replace(self._capacity, phi_Vn=gov)
        return self

    def capacity(self) -> JointCapacity:
        if self._capacity is None:
            self.run()
        return self._capacity  # type: ignore[return-value]

    def _direction_check(self, axis: str, cap: JointDirectionCapacity,
                         override: Optional[float]) -> JointDirectionCheck:
        col = self._col
        if axis == "x":
            beams = (self.beam_left, self.beam_right)
            b_column = col.section.height()
        else:
            beams = (self.beam_up, self.beam_down)
            b_column = col.section.width()

        if override is not None:
            Vu = override
            demand = {"Vu": override, "M1": 0.0, "M2": 0.0, "Vcol": 0.0}
        else:
            demand = self._axis_demand(beams[0], beams[1], self._story_height())
            Vu = max(demand["Vu"], 0.0)

        ratio = Vu / cap.phi_Vn if cap.phi_Vn > 0 else float("inf")

        # §18.8.5 — beam bars must stay inside the column core.
        present = [b for b in beams if b is not None]
        bars_in_core = True
        for fb in present:
            b_beam = fb.beam.section.width()
            if abs(fb.eccentricity) + b_beam / 2.0 > b_column / 2.0 + 1e-6:
                bars_in_core = False

        # Torsion induced by eccentric beams (reported, fed to column §22.7).
        torsion = 0.0
        for fb in present:
            M = max(abs(demand.get("M1", 0.0)), abs(demand.get("M2", 0.0)))
            torsion = max(torsion, M * abs(fb.eccentricity) / 1000.0)

        notes: list[str] = []
        if not bars_in_core:
            notes.append(f"[{axis}] beam bars fall outside the column core "
                         f"(§18.8.5) — configuration not admissible.")
        for fb in present:
            if abs(fb.eccentricity) > b_column / 8.0:
                notes.append(f"[{axis}] eccentricity {fb.eccentricity:.0f} mm "
                             f"> b_c/8 ({b_column/8.0:.0f} mm) — ACI PRC-352 "
                             f"recommends centering beam and column.")
        if torsion > 0:
            notes.append(f"[{axis}] eccentric beam induces T = M*e = "
                         f"{torsion:.1f} kN·m on the column (check §22.7).")
        return JointDirectionCheck(
            axis=axis, Vu=Vu, phi_Vn=cap.phi_Vn, ratio=ratio,
            bars_in_core=bars_in_core, torsion=torsion, notes=tuple(notes),
        )

    def check(self, demands: Optional[JointDemands] = None) -> JointCheck:
        cap = self.capacity()
        d = demands or JointDemands()
        cx = self._direction_check("x", cap.x, d.Vu_x)
        cy = self._direction_check("y", cap.y, d.Vu_y)
        ratio = max(cx.ratio, cy.ratio)
        passed = (ratio <= 1.0) and cx.bars_in_core and cy.bars_in_core
        notes = cx.notes + cy.notes
        return JointCheck(capacity=cap, x=cx, y=cy, ratio=ratio,
                          passed=passed, notes=notes)

    def design(self, demands: Optional[JointDemands] = None
               ) -> JointDesignResults:
        """Joint detailing requirements (§18.8.2/3/5). Does not mutate self.

        The joint shear strength does not depend on the transverse
        reinforcement once the §18.7.5 minimum is provided (R18.8.4), so
        the three modes share the same detailing; they exist for contract
        symmetry with Beam/Column/Wall.
        """
        col = self._col
        c = self._concrete

        # §18.8.3.2 relief: beams on all four faces, each >= 3/4 col width.
        four_beams = all(b is not None for b in
                         (self.beam_left, self.beam_right,
                          self.beam_up, self.beam_down))
        wide_enough = True
        for fb, face in ((self.beam_left, col.section.height()),
                         (self.beam_right, col.section.height()),
                         (self.beam_up, col.section.width()),
                         (self.beam_down, col.section.width())):
            if fb is None or fb.beam.section.width() < 0.75 * face:
                wide_enough = False
        ash_reduction = four_beams and wide_enough

        # Confined hoop spacing (§18.7.5.3), relaxed to 150 mm by §18.8.3.2.
        col_db_g60, col_db_g80 = _largest_db_by_grade(col.section)
        col_db_min = min((d for d in (col_db_g60, col_db_g80) if d > 0),
                         default=20.0)
        col_grade = max((s.grade for _, s in col.section.rebar.iter_bars()),
                        default=60)
        s_col = s_max_confined_column(
            h_min=col.h_min, db_long_min=col_db_min, hx=col.hx, grade=col_grade,
        )
        s_max_confined = 150.0 if ash_reduction else s_col

        # §18.8.5.1 — ldh for the largest hooked beam bar.
        ldh_required = 0.0
        for fb in self._beams():
            g60, g80 = _largest_db_by_grade(fb.beam.section)
            for db, grade in ((g60, 60), (g80, 80)):
                if db <= 0:
                    continue
                fy = next((s.fy for _, s in fb.beam.section.rebar.iter_bars()
                           if (s.grade >= 80) == (grade >= 80)), 420.0)
                ldh_required = max(ldh_required,
                                   ldh_hook(fy=fy, db=db, lam=c.lam,
                                            sqrt_fc=c.sqrt_fc))

        # §18.8.2.3 — minimum joint depth in each direction.
        def _min_depth(beams) -> float:
            g60 = g80 = h_beam = 0.0
            for fb in beams:
                if fb is None:
                    continue
                a, b = _largest_db_by_grade(fb.beam.section)
                g60, g80 = max(g60, a), max(g80, b)
                h_beam = max(h_beam, fb.beam.section.height())
            return min_joint_depth(db_grade60=g60, db_grade80=g80,
                                   h_beam_max=h_beam, lam=c.lam)

        min_depth_x = _min_depth((self.beam_left, self.beam_right))
        min_depth_y = _min_depth((self.beam_up, self.beam_down))
        depth_provided_x = col.section.width()
        depth_provided_y = col.section.height()
        depth_ok = (depth_provided_x >= min_depth_x - 1e-6
                    and depth_provided_y >= min_depth_y - 1e-6)

        notes: list[str] = []
        if ash_reduction:
            notes.append("§18.8.3.2: 4 beams >= 3/4 column width — confining "
                         "Ash may be halved and spacing relaxed to 150 mm.")
        else:
            notes.append("Transverse reinforcement per §18.7.5.4 (use the "
                         "column's confinement design).")
        if not depth_ok:
            notes.append("§18.8.2.3: column depth is below the minimum for "
                         "beam-bar anchorage — increase the column or use "
                         "headed bars.")

        proposal = JointDesignProposal(
            mode="minimum", s_max_confined=s_max_confined,
            ash_reduction=ash_reduction, ldh_required=ldh_required,
            min_depth_x=min_depth_x, min_depth_y=min_depth_y,
            depth_provided_x=depth_provided_x, depth_provided_y=depth_provided_y,
            depth_ok=depth_ok, notes=tuple(notes),
        )
        return JointDesignResults(
            minimum=proposal,
            capacity=replace(proposal, mode="capacity"),
            envelope=replace(proposal, mode="envelope"),
        )

    # ---- immutability / units -----------------------------------------
    def _clone(self, **overrides) -> "BeamColumnJoint":
        kwargs = dict(
            col_above=self.col_above, col_below=self.col_below,
            beam_left=self.beam_left, beam_right=self.beam_right,
            beam_up=self.beam_up, beam_down=self.beam_down,
            moment_condition=self.moment_condition, probable=self.probable,
            label=self.label, units=self.units,
        )
        kwargs.update(overrides)
        return BeamColumnJoint(**kwargs)

    def evolve(self, proposal: JointDesignProposal) -> "BeamColumnJoint":
        """Return a NEW joint (immutable). Joint geometry is member-driven,
        so the proposal only carries detailing — the clone is unchanged
        except for cache invalidation."""
        return self._clone()

    @property
    def plot(self):
        """Lazy matplotlib helper bound to this joint (see ``joints.plotting``)."""
        from design.joints.plotting import _JointPlotter
        plotter = getattr(self, "_plotter", None)
        if plotter is None:
            plotter = _JointPlotter(self)
            self._plotter = plotter
        return plotter

    def clear_cache(self) -> None:
        """Invalidate the cached capacity (mirrors the other elements)."""
        self._capacity = None

    # ---- reporting ----------------------------------------------------
    def summary(self) -> None:
        cap = self.capacity()
        chk = self.check()
        print(f"Joint {self.label} — {cap.joint_type}")
        for axis, cd, ck in (("X", cap.x, chk.x), ("Y", cap.y, chk.y)):
            print(f"  [{axis}] Aj={cd.Aj:.0f} mm²  γ={cd.gamma:.0f}  "
                  f"φVn={cd.phi_Vn:.0f} kN  Vu={ck.Vu:.0f} kN  "
                  f"ratio={ck.ratio:.2f}")
        print(f"  passed = {chk.passed}")
        for n in chk.notes:
            print(f"    - {n}")

    def print_design(
        self,
        results: "JointDesignResults",
        *,
        modes: tuple[str, ...] = ("minimum", "capacity", "envelope"),
    ) -> None:
        """Print the joint design modes (English).

        Joints do NOT report steel quantities — the hoops live in the
        column. We show the prescriptive checks instead.
        """
        print()
        print("=" * 70)
        print(f"  JOINT DESIGN — {self.label}    ({self.joint_type()})")
        print("=" * 70)
        for m in modes:
            p = getattr(results, m, None)
            if p is None:
                continue
            print()
            print(f"-- [ {m.upper():^8} ] --".center(70, "-"))
            print(f"  s_max confined : {p.s_max_confined:.0f} mm   (§18.8.3 / §18.7.5.3)")
            print(f"  Ash reduction  : {p.ash_reduction}        (§18.8.3.2)")
            print(f"  ldh required   : {p.ldh_required:.0f} mm   (§18.8.5.1)")
            print(f"  min depth X    : {p.min_depth_x:.0f} mm    "
                  f"provided = {p.depth_provided_x:.0f} mm")
            print(f"  min depth Y    : {p.min_depth_y:.0f} mm    "
                  f"provided = {p.depth_provided_y:.0f} mm")
            print(f"  depth ok       : {p.depth_ok}")
            if p.notes:
                for n in p.notes:
                    print(f"  note: {n}")
        print("=" * 70)

    def print_final(
        self,
        proposal: "JointDesignProposal",
        demands: "JointDemands | None" = None,
    ) -> None:
        """Print the FINAL joint design in one consolidated block (English).

        Joints do NOT report steel quantities — confinement hoops belong
        to the column. The block shows: geometry, joint type, capacities,
        prescriptive limits, and (if demands given) the full check.
        """
        cap = self.capacity()
        bar_w = 72
        print()
        print("=" * bar_w)
        print(f"  JOINT — FINAL DESIGN   |   {self.label}   |   {cap.joint_type}")
        print("=" * bar_w)
        print(f"  Capacities (gross shear):")
        for axis, cd in (("X", cap.x), ("Y", cap.y)):
            print(f"    [{axis}]  Aj = {cd.Aj:7.0f} mm²    gamma = {cd.gamma:.0f}    "
                  f"phi*Vn = {cd.phi_Vn:6.0f} kN")
        print("-" * bar_w)
        print(f"  Detailing (proposal):")
        print(f"    s_max confined : {proposal.s_max_confined:.0f} mm   "
              f"(§18.8.3 / §18.7.5.3)")
        print(f"    Ash reduction  : {proposal.ash_reduction}   (§18.8.3.2)")
        print(f"    ldh required   : {proposal.ldh_required:.0f} mm   (§18.8.5.1)")
        print(f"    min depth X    : {proposal.min_depth_x:.0f} mm    "
              f"provided = {proposal.depth_provided_x:.0f}    "
              f"[{'PASS' if proposal.depth_ok else 'FAIL'}]")
        print(f"    min depth Y    : {proposal.min_depth_y:.0f} mm    "
              f"provided = {proposal.depth_provided_y:.0f}")
        # NOTE: No steel quantities here — confinement belongs to the column.

        if demands is not None:
            chk = self.check(demands)
            tag = lambda b: "PASS" if b else "FAIL"
            print("-" * bar_w)
            print(f"  Code check vs demands:")
            for axis, ck in (("X", chk.x), ("Y", chk.y)):
                print(f"    [{axis}]  Vu = {ck.Vu:7.1f} kN    "
                      f"phi*Vn = {ck.phi_Vn:7.1f} kN    "
                      f"ratio = {ck.ratio:.3f}    [{tag(ck.ratio <= 1.0)}]")
            print("  " + "-" * (bar_w - 4))
            print(f"  OVERALL          : {tag(chk.passed)}")
            if chk.notes:
                print(f"  Notes:")
                for n in chk.notes:
                    print(f"    - {n}")
        print("=" * bar_w)

    def report(self) -> dict:
        cap = self.capacity()
        chk = self.check()
        return {
            "label": self.label,
            "joint_type": cap.joint_type,
            "moment_condition": self.moment_condition,
            "probable": self.probable,
            "phi_Vn_x": cap.x.phi_Vn,
            "phi_Vn_y": cap.y.phi_Vn,
            "Aj_x": cap.x.Aj,
            "Aj_y": cap.y.Aj,
            "gamma_x": cap.x.gamma,
            "gamma_y": cap.y.gamma,
            "Vu_x": chk.x.Vu,
            "Vu_y": chk.y.Vu,
            "ratio": chk.ratio,
            "passed": chk.passed,
            "notes": list(chk.notes),
        }

    def __repr__(self) -> str:
        return (f"BeamColumnJoint(label={self.label!r}, "
                f"type={self.joint_type()!r})")
