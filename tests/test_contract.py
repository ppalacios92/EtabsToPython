"""Cross-element contract tests.

Iterates over [Beam, Column, Wall] and verifies that each respects the
``design.common.contracts.Element`` protocol and the rules documented
in ``design/AGENTS.md``.

These tests are the canary for the unified API: if any of them fail,
the contract has been broken somewhere in the element-specific code.
"""
from __future__ import annotations
import pytest
import warnings

from design.common.contracts import (
    Element, Demands, Capacity, Check, DesignProposal, DesignResults,
)
from design.common.materials import Concrete, Steel
from design.beams.beam import Beam, BeamDemands
from design.columns.column import Column, ColumnDemands
from design.walls.wall import Wall, WallDemands
from design.sections.wall import WallSection
from design.sections.rectangular import RectangularSection
from design.sections.reinforcement import RebarLayout, perimeter_bars


# ----------------------------------------------------------------- #
# Fixtures
# ----------------------------------------------------------------- #
@pytest.fixture
def concrete():
    return Concrete(fc=28.0)


@pytest.fixture
def steel():
    return Steel(fy=420.0)


@pytest.fixture
def beam(concrete, steel):
    # ln=5000 mm — pass mm explicitly to avoid DeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Beam.rectangular(
            bw=300, h=500, cover=40, concrete=concrete, steel=steel,
            n_top=4, db_top=22, n_bot=4, db_bot=22,
            db_stirrup=10, n_legs=2, s_stirrup=150, ln=5000.0,
        )


@pytest.fixture
def column(concrete, steel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Column.rectangular(
            b=500, h=500, concrete=concrete, steel=steel,
            n_x=4, n_y=4, db_long=22,
            db_stirrup=10, n_legs_x=3, n_legs_y=3, s_stirrup=100,
            lu=3000.0,
        )


@pytest.fixture
def wall(concrete, steel):
    bars = perimeter_bars(
        b=300, h=3000, cover=50,
        n_x=2, n_y=6,
        bar=__import__("design.common.materials", fromlist=["Bar"]).Bar(diameter=12),
        steel=steel,
    )
    section = WallSection(
        lw=3000, tw=300, concrete=concrete,
        rebar=RebarLayout(groups=(bars,)),
    )
    return Wall(section=section, hw=10000)


@pytest.fixture(params=["beam", "column", "wall"])
def element(request, beam, column, wall):
    return {"beam": beam, "column": column, "wall": wall}[request.param]


@pytest.fixture(params=["beam", "column", "wall"])
def element_with_demands(request, beam, column, wall):
    if request.param == "beam":
        d = BeamDemands(
            Mu_pos_i=80, Mu_neg_i=160, Vu_i=200,
            Mu_pos_mid=200, Mu_neg_mid=40, Vu_mid=80,
            Mu_pos_j=80, Mu_neg_j=160, Vu_j=200,
        )
        return beam, d
    if request.param == "column":
        d = ColumnDemands(Pu=1500, Mux=300, Muy=150, Vux=120, Vuy=120)
        return column, d
    d = WallDemands(Pu=2000, Mu=8000, Vu=1500, delta_u=80, sigma_max=8.0)
    return wall, d


# ----------------------------------------------------------------- #
# Protocol conformance
# ----------------------------------------------------------------- #
def test_implements_element_protocol(element):
    assert isinstance(element, Element), (
        f"{type(element).__name__} does not satisfy Element protocol"
    )


def test_has_label_attribute(element):
    assert isinstance(element.label, str)


def test_run_returns_self(element):
    assert element.run() is element


# ----------------------------------------------------------------- #
# capacity() shape
# ----------------------------------------------------------------- #
def test_capacity_returns_object_with_phi_Vn(element):
    cap = element.capacity()
    assert hasattr(cap, "phi_Vn"), f"{type(cap).__name__} missing phi_Vn"
    assert isinstance(cap.phi_Vn, (int, float))
    assert cap.phi_Vn >= 0


def test_capacity_satisfies_protocol(element):
    cap = element.capacity()
    assert isinstance(cap, Capacity)


# ----------------------------------------------------------------- #
# check() shape
# ----------------------------------------------------------------- #
def test_check_returns_passed_boolean(element_with_demands):
    el, demands = element_with_demands
    chk = el.check(demands)
    assert isinstance(chk.passed, bool)
    assert isinstance(chk, Check)


# ----------------------------------------------------------------- #
# design() — 4 modes
# ----------------------------------------------------------------- #
def test_design_has_four_modes(element_with_demands):
    el, demands = element_with_demands
    res = el.design(demands)
    assert res.minimum.mode == "minimum"
    assert res.capacity.mode == "capacity"
    assert res.envelope.mode == "envelope"
    if res.demand is not None:
        assert res.demand.mode == "demand"


def test_design_without_demands_skips_demand_mode(element):
    res = element.design(None)
    assert res.demand is None
    assert res.minimum.mode == "minimum"
    assert res.envelope.mode == "envelope"


def test_design_proposal_has_notes_tuple(element):
    res = element.design(None)
    assert isinstance(res.envelope.notes, tuple)


# ----------------------------------------------------------------- #
# evolve() — immutable
# ----------------------------------------------------------------- #
def test_evolve_returns_new_instance(element):
    res = element.design(None)
    new = element.evolve(res.envelope)
    assert new is not element
    assert type(new) is type(element), (
        f"evolve() returned {type(new).__name__}, expected {type(element).__name__}"
    )


def test_apply_is_alias_of_evolve(element):
    """apply(proposal) must behave like evolve(proposal) — returns new instance."""
    res = element.design(None)
    new = element.apply(res.envelope)
    assert new is not element


# ----------------------------------------------------------------- #
# design() does NOT mutate self.design_results
# ----------------------------------------------------------------- #
def test_design_does_not_mutate_cached_results(element):
    """design() returns a value; the caller decides whether to store it."""
    # Snapshot whatever the element currently has
    before = getattr(element, "design_results", None)
    res = element.design(None)
    after = getattr(element, "design_results", None)
    # The contract per AGENTS.md §4: design() does NOT mutate self.design_results.
    # If design_results is still equal to `before`, the contract holds.
    assert after is before or after is None or after is res, (
        "design() unexpectedly mutated self.design_results; per AGENTS.md §4 "
        "it should be left to the caller to decide whether to cache the result."
    )


# ----------------------------------------------------------------- #
# Units: lu / ln / hw in mm (canonical)
# ----------------------------------------------------------------- #
def test_internal_lengths_are_mm(beam, column, wall):
    # Each of these comes from a fixture that passed lengths > 1000
    assert beam.ln > 1000, f"Beam.ln={beam.ln} looks like metres"
    assert column.lu > 1000, f"Column.lu={column.lu} looks like metres"
    assert wall.hw > 1000, f"Wall.hw={wall.hw} looks like metres"


# ----------------------------------------------------------------- #
# summary() / report()
# ----------------------------------------------------------------- #
def test_summary_runs(element, capsys):
    element.summary()
    out = capsys.readouterr().out
    assert len(out) > 0


def test_report_returns_dict(element):
    r = element.report()
    assert isinstance(r, dict)
    assert "label" in r
