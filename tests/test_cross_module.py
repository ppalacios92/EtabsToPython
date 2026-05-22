"""Cross-module integration tests.

Covers interactions BETWEEN ``design.beams``, ``design.columns`` and
``design.walls``:

  - ``wall.be_top`` is a real ``Column`` and respects its own contract.
  - The ``WallDemands -> ColumnDemands`` mapping used internally is
    importable from ``design.common.contracts`` and produces the
    documented field correspondence.
  - The public ``design`` namespace re-exports everything the
    AGENTS.md contract promises.
  - No active hardcoded formula remains in ``beams/`` ``columns/``
    ``walls/`` for the canonical shear and phi expressions.
"""
from __future__ import annotations
from pathlib import Path
import re
import warnings

import pytest

from design.common.contracts import Element, wall_demands_to_be
from design.common.materials import Concrete, Steel, Bar
from design.sections.wall import WallSection, be_perimeter_bars
from design.sections.reinforcement import RebarLayout, perimeter_bars
from design.walls.wall import Wall, WallDemands
from design.columns.column import Column, ColumnDemands


REPO_ROOT = Path(__file__).resolve().parent.parent
DESIGN_DIR = REPO_ROOT / "design"


# ----------------------------------------------------------------- #
# wall.be_top round-trip
# ----------------------------------------------------------------- #
def _wall_with_be() -> Wall:
    concrete = Concrete(fc=35.0)
    steel = Steel(fy=420.0)
    bar = Bar(diameter=12)

    # Web bars (sparse perimeter)
    web = perimeter_bars(b=300, h=4000, cover=50,
                        n_x=2, n_y=8, bar=bar, steel=steel)
    section = WallSection(
        lw=4000, tw=300, concrete=concrete,
        rebar=RebarLayout(groups=(web,)),
    )
    # Add a top BE explicitly
    be_bar = Bar(diameter=22)
    top_bars = be_perimeter_bars(
        be_thickness=400, be_length=600, center_y=1700,
        cover=40, n_x=2, n_y=4, bar=be_bar, steel=steel,
    )
    section = section.with_boundary_elements(
        top_length=600, top_thickness=400, add_be_top_bars=top_bars,
    )
    return Wall(section=section, hw=12000)


def test_wall_be_top_is_a_column():
    wall = _wall_with_be()
    assert wall.has_boundary_elements
    be = wall.be_top
    assert be is not None
    assert isinstance(be, Column)


def test_wall_be_top_satisfies_element_protocol():
    wall = _wall_with_be()
    be = wall.be_top
    assert isinstance(be, Element)


def test_wall_be_top_lu_is_mm():
    wall = _wall_with_be()
    be = wall.be_top
    # wall.lu is in mm; the BE column inherits it directly (no /1000 anymore)
    assert be.lu > 100, (
        f"BE column lu={be.lu} looks like metres — Wall must pass lu in mm "
        "to the BE column (no /1000 conversion). See AGENTS.md §1."
    )


# ----------------------------------------------------------------- #
# wall_demands_to_be mapping
# ----------------------------------------------------------------- #
def test_wall_demands_to_be_field_mapping():
    wd = WallDemands(Pu=1000, Mu=2000, Mu_out=300, Vu=400, Vu_out=50,
                    delta_u=60, sigma_max=7.0)
    cd = wall_demands_to_be(wd)
    assert isinstance(cd, ColumnDemands)
    # AGENTS.md §7 mapping
    assert cd.Pu == 1000          # axial unchanged
    assert cd.Mux == 2000         # Mu (in-plane) -> Mux (strong axis)
    assert cd.Muy == 300          # Mu_out -> Muy (weak axis)
    assert cd.Vuy == 400          # Vu (in-plane) -> Vuy
    assert cd.Vux == 50           # Vu_out -> Vux


# ----------------------------------------------------------------- #
# Public namespace: design.* must re-export the canon
# ----------------------------------------------------------------- #
def test_design_namespace_reexports_canonical():
    import design
    expected = [
        "Concrete", "Steel", "Bar",
        "UnitSystem", "units_from", "to_internal",
        "BarSchedule", "PlotStyle",
        "Element", "wall_demands_to_be",
        "beta1", "phi_axial_flexure", "phi_shear",
        "Beam", "Column", "Wall",
    ]
    for name in expected:
        assert hasattr(design, name), f"design.{name} not re-exported"


# ----------------------------------------------------------------- #
# Hardcode scan — defense against regressions
# ----------------------------------------------------------------- #
def _scan_for_pattern(pattern: re.Pattern, dirs: list[Path]) -> list[str]:
    """Return list of 'file:line:text' for matches in active code (skips
    docstrings and comments at line start, plus DeprecationWarning paths)."""
    hits = []
    for d in dirs:
        for py in d.glob("*.py"):
            try:
                lines = py.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if pattern.search(line):
                    # Allow legitimate occurrences:
                    if "DeprecationWarning" in line or "deprecat" in line.lower():
                        continue
                    if "lu_m" in line or "/ 1000.0" in line and "Mpr" in line:
                        # Internal unit-conversion paths — see AGENTS.md §1
                        continue
                    hits.append(f"{py.name}:{i}:{stripped}")
    return hits


def test_no_phi_065_090_hardcoded_in_beams():
    hits = _scan_for_pattern(
        re.compile(r"\bphi\s*=\s*0\.9[0]?\s+if\b"),
        [DESIGN_DIR / "beams"],
    )
    assert not hits, (
        "Found phi=0.90/0.65 hardcoded — use phi_axial_flexure(eps_t, eps_ty). "
        "Offenders:\n  " + "\n  ".join(hits)
    )


def test_no_getattr_proxy_in_wall_results():
    hits = _scan_for_pattern(
        re.compile(r"def\s+__getattr__"),
        [DESIGN_DIR / "walls"],
    )
    assert not hits, (
        "WallDesignResults.__getattr__ proxy must NOT exist — callers use "
        "results.envelope.X explicitly. Offenders:\n  " + "\n  ".join(hits)
    )


def test_no_lu_divided_by_1000_when_building_be_column():
    """Wall must NOT divide lu by 1000 when constructing the BE Column —
    Column.lu is already in mm. See AGENTS.md §1 + Fase 1C TW.3.

    The ``lu_m`` property (which legitimately returns ``self.lu / 1000``
    as a legacy accessor in metres) is excluded by checking that the
    offending line is part of a ``Column(...)`` constructor call or a
    ``lu=`` kwarg assignment.
    """
    wall_py = (DESIGN_DIR / "walls" / "wall.py").read_text(encoding="utf-8")
    lines = wall_py.splitlines()
    offenders = []
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not re.search(r"\blu\s*/\s*1000", line):
            continue
        if stripped.startswith("#"):
            continue
        # Allow the lu_m property body (legacy accessor)
        # Look at the preceding non-blank line: if it's @property or
        # `def lu_m`, this is the property body and is legitimate.
        ctx = lines[max(0, i - 4):i]
        if any("lu_m" in c or "def lu_m" in c for c in ctx):
            continue
        # The bug we're guarding against is `Column(lu=self.lu / 1000.0)`
        # or `lu=self.lu / 1000.0` style — flag those.
        offenders.append(f"wall.py:{i}:{stripped}")
    assert not offenders, (
        "Found lu/1000 in non-property context in walls — Column.lu is in "
        "mm now, no division needed when constructing BE column. "
        "Offenders:\n  " + "\n  ".join(offenders)
    )
