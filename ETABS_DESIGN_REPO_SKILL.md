---
name: etabs-design-repo
description: 'Understand and work on the EtabsToPython repository, especially the concrete design package under design/. Use when Codex must analyze, modify, test, explain, refactor, extend, debug, or document Beam, Column, Wall, common ACI helpers, section analysis, ETABS extraction, plotting, examples, notebooks, smoke tests, or repo architecture in C:\Dropbox\01. Brain\11. GitHub\EtabsToPython.'
---

# EtabsToPython Design Repo Skill

Use this skill as the onboarding map for this repository. It is written for a future Codex instance that must understand the repo quickly and then make changes safely.

## First Moves

1. Work from repo root:
   `C:\Dropbox\01. Brain\11. GitHub\EtabsToPython`
2. Read these files before touching `design/`:
   - `design/AGENTS.md`: binding contract for Beam / Column / Wall.
   - `design/REFACTOR_GUIDE.md`: refactor history, ownership matrix, and acceptance criteria.
   - `design/__init__.py`: public namespace and canonical re-exports.
   - `tests/test_contract.py` and `tests/test_cross_module.py`: executable contract.
3. Prefer `rg` for search:
   - `rg "pattern" design tests`
   - `rg "def (run|capacity|check|design|evolve|apply)" design`
4. Always keep internal units in mind:
   - Stress: MPa
   - Length: mm
   - Force: kN
   - Moment: kN*m
   - Area: mm2

## Repo Purpose

The repository has two main layers:

1. `etabstopython/`: ETABS COM extraction and plotting.
   - `model.py` connects to an active ETABS session through `comtypes`, pulls database tables into pandas DataFrames, builds geometry/connectivity, and stores story, frame, wall, floor, force, displacement and reaction results.
   - `plotter.py` and `tools.py` visualize and post-process extracted ETABS data.

2. `design/`: reinforced concrete design toolkit, currently the most developed part of the repo.
   - Implements ACI 318-25-style design helpers and element objects.
   - Core elements are `Beam`, `Column`, and `Wall`.
   - Shared code lives in `design/common/`.
   - Geometry and reinforcement sections live in `design/sections/`.
   - Analysis kernels live in `design/analysis/`.
   - Plotting wrappers live in `design/plotting/` and per-element `plotting.py` modules.

The `design/` package is not a loose collection of formulas. It has a strict bidirectional contract tested across beams, columns, and walls.

## Design Package Mental Model

The central abstraction is an element object:

```python
element.run()                  # lazy compute/cache, returns self
element.capacity()             # returns XCapacity
element.check(demands)         # returns XCheck
element.design(demands=None)   # returns XDesignResults
element.evolve(proposal)       # returns NEW element
element.apply(proposal)        # alias of evolve()
element.summary()
element.report()
element.set_units(code_or_name)
element.plot
```

`Beam`, `Column`, and `Wall` should satisfy `isinstance(element, design.common.contracts.Element)`.

Important design rule: `evolve()` is immutable and returns a new object. `run()` is allowed to mutate internal caches. `design()` returns a result object; do not rely on it mutating `self.design_results`.

## Public Namespace

`design/__init__.py` re-exports the canonical API:

- Materials: `Concrete`, `Steel`, `Bar`
- Units: `UnitSystem`, `units_from`, `to_internal`
- Shared helpers: `BarSchedule`, `PlotStyle`
- Protocols: `Element`, `Demands`, `Capacity`, `Check`, `DesignProposal`, `DesignResults`
- ACI factors: `beta1`, `phi_axial_flexure`, `phi_shear`, `lambda_concrete`, `omega_v_wall`
- Sections: `RebarGroup`, `RebarLayout`, `RectangularSection`, `WallSection`
- Elements: `Beam`, `Column`, `Wall`

Prefer importing canonical shared helpers from `design.common` or `design`, not from legacy shims unless maintaining compatibility.

## Shared Common Layer

`design/common/` is the contract and formula library. Treat it as the foundation.

- `materials.py`: frozen value types `Concrete`, `Steel`, `Bar`.
  - `Concrete.sqrt_fc` caps `sqrt(fc)` at `sqrt(70 MPa)` for shear equations.
  - `Concrete.lam` delegates lightweight factor to `lambda_concrete`.
  - `Steel.eps_ty` is the tension-control threshold input for phi.
- `units.py`: presentation unit system based on ETABS unit codes 1..16.
  - Internals remain MPa, mm, kN, kN*m.
  - `UnitSystem` gives display factors and conversions to internal values.
- `factors.py`: `beta1`, `phi_axial_flexure`, `phi_shear`, wall/column seismic amplification defaults.
- `shear.py`: canonical shear equations for beams, columns, and walls.
  - `vc_simplified`, `vc_detailed`, `vs_capacity`, `vs_max`
  - `av_s_required`, `av_s_minimum`
  - `vc_zero_seismic`
  - `alpha_c`, `vn_wall`, `vn_wall_max`, `ve_in_plane_capacity_wall`
- `spacing.py`: canonical confinement/spacing helpers for beams and columns.
- `probable.py`: probable strength helpers using `fy_factor`, including `mpr_envelope`.
- `contracts.py`: runtime-checkable protocols and `wall_demands_to_be`.
- `optimize.py`: common `OptimizeAlternative` and `format_optimize_table`.
- `plot_style.py`: single `PlotStyle` used through shims in element packages.
- `bar_schedule.py`: single reinforcement bar schedule abstraction.

Do not duplicate formulas inside `beams/`, `columns/`, or `walls/`. Use `common/`.

## Sections And Analysis

`design/sections/` defines geometry and rebar layouts.

- `base.py`: abstract `Section`.
- `rectangular.py`: `RectangularSection` with `b`, `h`, polygons, gross area, bounding box.
- `wall.py`: `WallSection` with web plus optional boundary elements.
  - Local axes: `x` across thickness, `y` along wall length.
  - In-plane bending is `angle_deg=0`; out-of-plane is `angle_deg=90`.
  - `with_boundary_elements(...)` returns a new section.
  - Boundary elements can be converted into standalone rectangular sections for `Column`.
- `reinforcement.py`: `Rebar`, `RebarGroup`, `RebarLayout`, `perimeter_bars`, `two_layer_bars`.

`design/analysis/section_analysis.py` performs strain compatibility:

- `pm_at_neutral_axis`
- `pm_curve`
- `pn_max`
- `to_capacity`

The beam inspection P-M curve, column interaction diagrams, and wall interaction diagrams build on these section-analysis routines.

## Beam Module

Main file: `design/beams/beam.py`.

Beam is primarily a flexure + shear element for design, but it now also exposes an inspection-grade bidirectional P-M curve. Use the closed-form flexure/shear values for normal beam design and the P-M curve when the user asks to inspect axial-load interaction, compare `Mn` at a specific `Nu`, or plot positive/negative bending envelopes.

Core dataclasses:

- `BeamDemands`
- `BeamCapacity`
- `BeamCheck`
- `BeamDesignProposal`
- `BeamDesignResults`

Important behavior:

- `Beam.rectangular(...)` is the convenient factory.
- `ln` is stored in mm. Factory may accept metres by heuristic if `ln < 30`, with `DeprecationWarning`.
- `ln_m` is the legacy metre view.
- `Beam.run()` computes lazy caches for `phi_Mn_pos`, `phi_Mn_neg`, `phi_Vn`, and seismic `Mpr` caches.
- `capacity()` returns `BeamCapacity` with canonical `Mpr_pos` and `Mpr_neg`; `Mn_pr_*` are legacy aliases.
- `design()` returns minimum, demand, capacity, and envelope proposals.
- `evolve()` rebuilds a new beam section and stirrup state from a proposal.
- `pm` lazily builds and caches a `BeamPMCurve` with two `InteractionDiagram` objects:
  - `pm.pos`: positive bending, bottom fiber in tension, column-kernel `angle_deg=0`.
  - `pm.neg`: negative bending, top fiber in tension, column-kernel `angle_deg=180`.
- `Mn`, `phi_Mn`, `Pn`, and `phi_Pn` are `_BeamFieldView` proxies over `beam.pm`, not plain dicts.
  - `beam.Mn(direction="pos")` returns the full curve as a NumPy array.
  - `beam.phi_Mn.at(Nu=0, direction="pos")` interpolates reduced moment at axial load.
  - `beam.Mn.peak()` returns the max over both bending senses.
- `clear_cache()` invalidates only the beam P-M curve cache.

Formula sources:

- Flexure helpers in `design/beams/flexure.py`
- Seismic beam helpers in `design/beams/seismic.py`
- Inspection P-M curve in `design/beams/interaction.py`
- Beam probable P-M / Mpr-by-axial helpers in `design/beams/probable.py`
- Beam result proxies in `design/beams/proxies.py`
- Shear uses `design.common.shear`
- Spacing uses `design.common.spacing`

When changing beams:

- Never reintroduce hardcoded `phi = 0.90 if ... else 0.65`; use `phi_axial_flexure`.
- Never write raw `Av * fyt * d / (s * 1000)`; use `vs_capacity`.
- Keep `BeamDesignResults.__iter__()` behavior consistent with tests.
- Do not assume `beam.Mn` or `beam.phi_Mn` is a dict; after the P-M update they are proxy objects. Use `beam.capacity().phi_Mn_pos` / `phi_Mn_neg` for scalar design strengths, and use `beam.phi_Mn.at(...)` or `beam.phi_Mn(direction=...)` for interaction-diagram queries.
- Keep the distinction clear: `capacity().phi_Mn_pos` is the closed-form design scalar; `beam.phi_Mn.at(Nu=0, direction="pos")` comes from strain compatibility and may differ slightly because it uses the interaction kernel.
- If editing `ln`, be careful: design shear uses moments in kN*m divided by span in m. The canonical stored length is intended to be mm per `design/AGENTS.md`; conversion helpers must keep capacity-design shear dimensionally consistent.

## Column Module

Main file: `design/columns/column.py`.

Column is a P-M-M plus shear/confinement element. It builds interaction diagrams and a biaxial surface.

Core dataclasses:

- `ColumnDemands`
- `ColumnCapacity`
- `ColumnCheck`
- `ColumnDesignProposal`
- `ColumnDesignResults`

Important behavior:

- `Column.rectangular(...)` is the convenient factory.
- `lu` is stored in mm. Factory may accept metres by heuristic if `lu < 30`, with `DeprecationWarning`.
- `lu_m` is the legacy metre view.
- `run()` builds Po, To, phi Pn max, diagrams/surfaces, and shear caches.
- `capacity(biaxial=True, n_angles=..., n_points_per_curve=...)` can be computationally heavier than beam capacity.
- `design()` has minimum, demand, capacity, and envelope modes.
- `evolve()` returns a new `Column` with proposal detailing applied.

Related files:

- `interaction.py`: uniaxial interaction diagram.
- `interaction_surface.py`: biaxial surface.
- `surface.py`: envelope/slice access.
- `proxies.py`: result-field views such as `Mn`, `phi_Mn`, `Pn`, `phi_Pn`.
- `confinement.py`, `slenderness.py`, `optimize.py`, `plotting.py`.

When changing columns:

- Use `vc_simplified`, `vs_capacity`, `av_s_required`, and `vc_zero_seismic` from `common.shear`.
- Use `OMEGA_0_COLUMN_DEFAULT` from `common.factors`.
- Keep direction conventions in `ColumnDesignProposal` intact:
  - `Mpr_x`: moment around local X.
  - `Mpr_y`: moment around local Y.
  - `Ve_y`: shear caused by `Mpr_x`.
  - `Ve_x`: shear caused by `Mpr_y`.
- Do not silently switch `lu` back to metres.

## Wall Module

Main file: `design/walls/wall.py`.

Wall composes a `WallSection` and optional boundary elements, which are real `Column` objects. The wall owns web design; boundary element confinement is delegated to the column module.

Core dataclasses:

- `WallDemands`
- `WallCapacity`
- `WallCheck`
- `WallDesignProposal`
- `WallDesignResults`

Important behavior:

- `Wall.rectangular(...)` is the convenient factory for web plus optional boundary elements.
- `hw` and `lu` are stored in mm.
- `capacity()` can run in-plane or biaxial.
- `check()` evaluates PM/shear, distributed reinforcement, double curtain, and boundary element triggers.
- `design()` can iterate geometry/detailing and returns original/final wall plus history.
- `WallDesignResults` intentionally has no `__getattr__` proxy. Use `results.envelope.X` explicitly.
- `be_top` and `be_bot` are `Column` instances when boundary elements exist.

Related files:

- `boundary.py`: special boundary element triggers and proposals.
- `distributed.py`: web reinforcement rho/spacing/layers.
- `shear.py`: wall-specific shim/helpers; canonical formulas are in `common.shear`.
- `probable.py`: probable in-plane/out-of-plane strength wrappers.
- `coupling_beam.py`: coupling beam classification and diagonal reinforcement.
- `wall_pier.py`: wall pier classification.
- `optimize.py`: wall web/detailing alternatives.
- `plotting.py`: wall-specific plotter.

Critical cross-module mapping:

Use `design.common.contracts.wall_demands_to_be(demands)` when converting wall demands into boundary-element column demands.

Canonical mapping:

```python
Pu     -> Pu
Mu     -> Mux
Mu_out -> Muy
Vu     -> Vuy
Vu_out -> Vux
```

When changing walls:

- Do not divide `lu` by 1000 when constructing BE columns.
- Do not resurrect `WallDesignResults.__getattr__`.
- Keep BE columns satisfying the shared `Element` protocol.
- Use `phi_shear()` rather than raw `0.75`.

## Optimization Pattern

Each element exposes `run_optimize(...)`:

- Beam: `design/beams/optimize.py`
- Column: `design/columns/optimize.py`
- Wall: `design/walls/optimize.py`

The common result shape is `design.common.optimize.OptimizeAlternative`:

```python
proposal: object
detailing: dict
rho_transverse: float
rho_longitudinal: float
rho_total: float
feasible: bool
is_baseline: bool = False
is_provided: bool = False
notes: tuple[str, ...] = ()
```

Use `format_optimize_table(...)` from common where possible.

## Plotting Pattern

There is one canonical style object: `design.common.plot_style.PlotStyle`.

Per-element style modules are compatibility shims:

- `design/beams/style.py`
- `design/columns/style.py`
- `design/walls/style.py`

Plotters should read `self.element.style` / `self.wall.style` rather than creating their own divergent style system.

Beam plotting now includes inspection P-M curves through `beam.plot.pm(...)`, which reads `beam.pm`. Older comments saying beams have no P-M curve are stale; the correct nuance is that beams do not use a PMM surface for ordinary ACI beam design, but they do expose a P-M diagram for inspection.

## Tests And Verification

Use these according to the risk of the change:

```powershell
python -m pytest tests\test_common_shear.py tests\test_common_probable.py tests\test_common_contracts.py -q
python -m pytest tests\test_contract.py tests\test_cross_module.py -q
python -m pytest -q
python -m design._smoke_test
python -m design._smoke_test --plot
```

Quick manual check for the beam P-M curve:

```powershell
python -c "from design import Concrete,Steel,Beam; c=Concrete(fc=28); s=Steel(fy=420); b=Beam.rectangular(bw=300,h=500,cover=40,concrete=c,steel=s,n_top=4,db_top=22,n_bot=4,db_bot=22,db_stirrup=10,n_legs=2,s_stirrup=150,ln=5000); print(type(b.pm).__name__, len(b.pm.pos.points), len(b.pm.neg.points)); print(type(b.Mn).__name__, b.Mn.peak(), b.phi_Mn.at(Nu=0,direction='pos')); print(b.capacity().phi_Mn_pos)"
```

Important tests:

- `tests/test_common_shear.py`: shared shear formulas and wall shear equations.
- `tests/test_common_probable.py`: probable strength envelopes.
- `tests/test_common_contracts.py`: runtime protocols and Wall -> BE mapping.
- `tests/test_contract.py`: Beam / Column / Wall unified API.
- `tests/test_cross_module.py`: BE column integration, namespace exports, hardcode regression scans.
- Add or update beam-specific tests when changing `design/beams/interaction.py`, `design/beams/probable.py`, or `design/beams/proxies.py`; the current cross-element contract tests do not fully exercise beam P-M proxy semantics.

If changing notebooks, regenerate or at least smoke-check the corresponding example:

- `design/example_beam*.ipynb`
- `design/example_column*.ipynb`
- `design/example_wall*.ipynb`
- `design/example_aci_check.ipynb`

## Regression Traps

Avoid these unless deliberately changing the contract and updating tests/docs:

- Lengths stored as metres inside elements.
- `lu / 1000` when building wall boundary-element columns.
- Hardcoded shear/flexure factors in element modules.
- Mutating `self.design_results` inside `design()`.
- `apply()` mutating the original object.
- Per-element duplicate `PlotStyle`.
- Direct inline Wall -> Column demand mapping.
- Reintroducing `WallDesignResults.__getattr__`.
- Treating `beam.Mn` / `beam.phi_Mn` as dicts after the beam P-M update.
- Letting stale docs claim beams have no P-M diagram at all. The correct wording is "no PMM surface for ordinary beam design; optional P-M curve for inspection."
- Touching unrelated element modules during a focused phase.

## Common Workflows

### Explain an element API

Read:

1. The element module docstring.
2. Its demand/capacity/check/proposal/results dataclasses.
3. `run`, `capacity`, `check`, `design`, `evolve`, `report`.
4. Related tests in `tests/test_contract.py`.

### Fix a formula bug

1. Search for existing canonical helper in `design/common/`.
2. If the formula is shared, add/change it in `common/` and test it in `tests/test_common_*.py`.
3. Replace per-element active formulas with calls to common helpers.
4. Run contract and cross-module tests.

### Add a new design output

1. Add the field to `XDesignProposal` or `XCapacity`.
2. Populate it in all relevant design modes.
3. Preserve existing aliases if users may depend on them.
4. Update print/report/plot code only where the new value is user-facing.
5. Add a focused test.

### Modify wall boundary elements

1. Read `WallSection` boundary element helpers.
2. Read `Wall.be_top` / `Wall.be_bot` construction.
3. Use `wall_demands_to_be`.
4. Verify `tests/test_cross_module.py`.
5. Confirm `Column.lu` remains in mm.

### Debug capacity mismatch

1. Check internal units first.
2. Check whether the value is nominal, reduced, or probable:
   - `Mn`, `Pn`: nominal.
   - `phi_Mn`, `phi_Pn`, `phi_Vn`: reduced.
   - `Mpr`: probable, usually `fy_pr = 1.25 fy` and `phi = 1.0`.
3. Check angle convention for columns/walls.
4. Reproduce with the smallest fixture-like example in tests.

For beams, decide which capacity source you are comparing:

- `beam.capacity().phi_Mn_pos` / `phi_Mn_neg`: closed-form design scalars used by standard beam design.
- `beam.phi_Mn.at(Nu=0, direction="pos")`: interpolation from the strain-compatibility P-M diagram.
- `beam.Mn.peak(direction="pos")`: peak nominal moment on the full interaction curve, not necessarily the same point as `Nu=0`.

## Current Architecture Summary

The design package is in a post-refactor state where common helpers, contracts, plot style, optimization result shape, probable strength, spacing, and shear are centralized. Beam, Column, and Wall are intended to be parallel public objects with their own engineering behavior but the same outer API.

The most important thing when working here is to preserve the contract, not just make a local calculation pass. If a local fix breaks `tests/test_contract.py` or `tests/test_cross_module.py`, it is probably fighting the architecture.
