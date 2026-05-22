"""Structural contracts for the ``design`` package.

This module turns the prose contract of ``design/AGENTS.md`` into
``Protocol`` objects with ``runtime_checkable=True``. The intent is:

    isinstance(beam, Element)  # True
    isinstance(col,  Element)  # True
    isinstance(wall, Element)  # True

Concrete classes (``Beam``, ``Column``, ``Wall``) do not need to
inherit from these Protocols — duck-typing is enough.

Also exposes :func:`wall_demands_to_be`, the canonical mapping
``WallDemands -> ColumnDemands`` used when designing wall boundary
elements.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class Demands(Protocol):
    """Marker protocol for ``XDemands`` dataclasses."""
    pass


@runtime_checkable
class Capacity(Protocol):
    """Marker protocol for ``XCapacity`` dataclasses."""

    phi_Vn: float


@runtime_checkable
class Check(Protocol):
    """Marker protocol for ``XCheck`` dataclasses."""

    passed: bool


@runtime_checkable
class DesignProposal(Protocol):
    """Marker protocol for ``XDesignProposal`` dataclasses."""

    mode: str
    notes: tuple


@runtime_checkable
class DesignResults(Protocol):
    """Marker protocol for the ``XDesignResults`` envelope object.

    The three modes (minimum / capacity / demand) are merged into the
    ``envelope`` proposal.
    """

    minimum: DesignProposal
    capacity: DesignProposal
    envelope: DesignProposal


@runtime_checkable
class Element(Protocol):
    """The Beam/Column/Wall contract.

    Concrete classes implement these methods; ``isinstance(elem, Element)``
    is the canonical test. See ``design/AGENTS.md`` for the full contract.
    """

    label: str

    def run(self): ...
    def capacity(self): ...
    def check(self, demands): ...
    def design(self, demands=None): ...
    def evolve(self, proposal): ...
    def summary(self): ...
    def report(self) -> dict: ...


# ---------------------------------------------------------------------------
# Wall -> Boundary Element column demands
# ---------------------------------------------------------------------------
def wall_demands_to_be(demands):
    """Map ``WallDemands`` to ``ColumnDemands`` for boundary-element design.

    A wall's boundary element acts as a column whose **strong axis is
    along the wall length** (``lw``) and whose **weak axis is the wall
    thickness** (``tw``). Therefore:

        in-plane Mu       -> Mux   (strong-axis bending)
        out-of-plane Mu   -> Muy   (weak-axis bending)
        in-plane Vu       -> Vuy   (perpendicular to strong axis)
        out-of-plane Vu   -> Vux   (perpendicular to weak axis)

    Pu is unchanged.

    Returns a ``design.columns.column.ColumnDemands`` instance.
    """
    # Lazy import to avoid a circular import at package load time.
    from design.columns.column import ColumnDemands
    return ColumnDemands(
        Pu=demands.Pu,
        Mux=demands.Mu,
        Muy=demands.Mu_out,
        Vux=demands.Vu_out,
        Vuy=demands.Vu,
    )


__all__ = [
    "Demands",
    "Capacity",
    "Check",
    "DesignProposal",
    "DesignResults",
    "Element",
    "wall_demands_to_be",
]
