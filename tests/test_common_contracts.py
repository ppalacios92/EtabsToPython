"""Tests for design.common.contracts — Protocols and wall_demands_to_be."""
from __future__ import annotations
import pytest

from design.common.contracts import (
    Demands,
    Capacity,
    Check,
    DesignProposal,
    DesignResults,
    Element,
    wall_demands_to_be,
)


# ---------------------------------------------------------------------------
# Protocols are runtime_checkable
# ---------------------------------------------------------------------------
def test_protocols_are_runtime_checkable():
    """isinstance() against the Protocols must not raise."""
    # Empty object obviously is not an Element, but the runtime check
    # itself must succeed (i.e. the Protocols are runtime_checkable).
    obj = object()
    assert isinstance(obj, Demands)        # Demands has no required attrs
    assert not isinstance(obj, Capacity)   # missing phi_Vn
    assert not isinstance(obj, Check)      # missing passed
    assert not isinstance(obj, Element)    # missing methods


def test_capacity_protocol_recognizes_duck_type():
    """A simple object exposing phi_Vn satisfies Capacity."""
    class Fake:
        phi_Vn = 1.0

    assert isinstance(Fake(), Capacity)


def test_check_protocol_recognizes_duck_type():
    class Fake:
        passed = True

    assert isinstance(Fake(), Check)


def test_design_proposal_protocol_recognizes_duck_type():
    class Fake:
        mode = "demand"
        notes = ()

    assert isinstance(Fake(), DesignProposal)


def test_design_results_protocol_recognizes_duck_type():
    class FakeProp:
        mode = "demand"
        notes = ()

    class FakeResults:
        minimum = FakeProp()
        capacity = FakeProp()
        envelope = FakeProp()

    assert isinstance(FakeResults(), DesignResults)


def test_element_protocol_recognizes_duck_type():
    """An object with the right methods + label satisfies Element."""
    class Fake:
        label = "X-1"
        def run(self): return self
        def capacity(self): return None
        def check(self, demands): return None
        def design(self, demands=None): return None
        def evolve(self, proposal): return self
        def summary(self): return None
        def report(self): return {}

    assert isinstance(Fake(), Element)


# ---------------------------------------------------------------------------
# wall_demands_to_be — canonical mapping
# ---------------------------------------------------------------------------
def test_wall_demands_to_be_mapping():
    from design.walls.wall import WallDemands
    d = WallDemands(
        Pu=1000.0, Mu=5000.0, Mu_out=200.0,
        Vu=400.0, Vu_out=80.0,
    )
    cd = wall_demands_to_be(d)
    assert cd.Pu == 1000.0
    assert cd.Mux == 5000.0     # in-plane bending -> strong axis
    assert cd.Muy == 200.0      # out-of-plane bending -> weak axis
    assert cd.Vuy == 400.0      # in-plane shear -> Vuy
    assert cd.Vux == 80.0       # out-of-plane shear -> Vux


def test_wall_demands_to_be_defaults():
    """Mapping must still work when out-of-plane components are zero."""
    from design.walls.wall import WallDemands
    d = WallDemands(Pu=500.0, Mu=1000.0, Vu=100.0)
    cd = wall_demands_to_be(d)
    assert cd.Pu == 500.0
    assert cd.Mux == 1000.0
    assert cd.Muy == 0.0
    assert cd.Vuy == 100.0
    assert cd.Vux == 0.0
