# `design.walls` — Reinforced concrete wall module

Self-contained engineering of concrete walls per **ACI 318-25 §18.10**.
A `Wall` is a **composition** of a web (alma) and zero, one, or two
**boundary elements**. Each boundary element is a real `Column` object
from `design.columns` — same PMM machinery, same Ash / hoops / Mpr
pipeline.

The wall is **immutable**. Every call that would change geometry or
detailing returns a NEW `Wall` instance through `wall.evolve(proposal)`.
That lets you keep the original wall, the proposed wall, and every
intermediate iteration side-by-side for comparison.

`wall.design(demands)` runs **three modes** (`minimum`, `demand`,
`capacity`) plus an `envelope` aggregate. With `auto_update=True` (the
default when demands are passed), it **iterates** — proposing BE
geometry, rebuilding the wall, recomputing the PMM — until the BE
length stabilizes.

Internal numerics: **MPa, mm, kN, kN·m**. Presentation units
configurable through `wall.set_units(...)`.

---

## 1. Quick start

```python
from design import (
    Concrete, Steel, Bar, WallSection, Wall,
)
from design.walls import WallDemands
from design.sections.reinforcement import RebarLayout, perimeter_bars

# Materials
concrete = Concrete(fc=35)
steel    = Steel(fy=420, grade=60)

# Web steel — single perimeter group along the full lw
web = perimeter_bars(
    b=300, h=4000, cover=50, n_x=2, n_y=14,
    bar=Bar(diameter=12), steel=steel,
)

section = WallSection(
    lw=4000, tw=300,
    concrete=concrete, rebar=RebarLayout(groups=(web,)),
)
wall = Wall(section=section, hw=30_000, lu=3_000, label="W-1")
wall.set_units(11)   # Tonf_mm_C for display

# 1) Capacity of the bare-bones geometry
cap = wall.capacity()
print(cap.phi_Vn, max(cap.diagram.phi_Mn))

# 2) Design with demands — auto_update=True is the default
results = wall.design(WallDemands(
    Pu=2000.0, Mu=8000.0, Vu=1500.0,
    delta_u=180.0, sigma_max=8.0,
))

# 3) Inspect everything
print(results.envelope.be_required)             # True
print(results.envelope.be_length_proposed)      # mm
print(results.iterations, results.converged)

# 4) The original wall is untouched
assert results.original_wall is wall
assert not wall.has_boundary_elements
assert results.final_wall.has_boundary_elements

# 5) The BE on the new wall is a Column — full columns/ API works
final = results.final_wall
final.be_top.design()         # SMF column detailing
final.be_top.run_optimize()   # explore alternatives
final.be_top.plot.section()
```

---

## 2. File layout

```
design/walls/
├── README.md                 ← this document
├── __init__.py               ← public re-exports
├── wall.py                   ← Wall + dataclasses + iteration loop
├── plotting.py               ← _WallPlotter — bound plot proxy
├── shear.py                  ← alpha_c, Vn, omega_v, Av/s, Ve
├── boundary.py               ← BE checks + propose_be_geometry()
├── distributed.py            ← web rho_min, spacing, two-curtain
├── probable.py               ← Mpr wrappers (in-plane / out-of-plane)
├── coupling_beam.py          ← §18.10.7
└── wall_pier.py              ← §18.10.8
```

Shared collaborators (reused, not duplicated):

```
design/columns/        Column, Surface, _DiagramFieldView, BarSchedule,
                       PlotStyle, mpr_envelope, ash_required, hx_check,
                       s_max_confined, lo_length
design/sections/       WallSection (with BE partitioning)
design/common/         materials, factors, units, geometry
design/analysis/       section_analysis (pm_at_neutral_axis)
```

---

## 3. The `Wall` class — public surface

### 3.1 Construction

```python
Wall(
    *,
    section: WallSection,
    hw: float,                          # total wall height (mm)
    lu: float | None = None,            # unbraced height (mm), default = hw
    rho_t: float = 0.0025,              # provided distributed transverse
    rho_l: float = 0.0025,
    fyt: float = 420.0,
    seismic: bool = True,
    omega_v_factor: float = 1.5,        # §18.10.3.1.2 capacity-design shear amp
    hx: float = 200.0,                  # §18.7.5.2 inside the BE
    transverse_bar_diameter: float = 10.0,
    web_bar_diameter: float = 10.0,
    units: int | str | UnitSystem | None = None,
    bar_schedule: BarSchedule | None = None,
    label: str = "Wall",
    run_settings: dict | None = None,
)
```

### 3.2 Lazy compute

```python
wall.run(n_angles=19, n_points=60)    # build PMM surface, cache scalars
wall.clear_cache()
```

Anything below is lazy on first access:

```python
wall.surface                          # Surface (same facade as columns)
wall.surface[0]                       # in-plane diagram
wall.surface[90]                      # out-of-plane diagram
wall.Po, wall.To, wall.phi_Pn_max, wall.phi_Vn, wall.Vn_max

# Proxies — same API as columns
wall.Mn(angle=0)
wall.Mn.at(Pu=2000, angle=0)
wall.Mn.peak(angle=0)
wall.Mn[0, 2000]
```

### 3.3 Composition — BE as Column

```python
wall.has_boundary_elements            # bool
wall.be_top                           # Column | None
wall.be_bot                           # Column | None
```

When the section has a BE, `wall.be_top` is a full `Column` whose
`section` is the BE rectangle. All of:

```python
wall.be_top.design()
wall.be_top.run_optimize()
wall.be_top.Mn.peak()
wall.be_top.plot.section()
wall.be_top.set_units(...)
```

work as if you had built the column yourself.

### 3.4 capacity / check / design

```python
cap = wall.capacity(biaxial=True)                   # WallCapacity
chk = wall.check(WallDemands(Pu=..., Mu=..., Vu=...))  # WallCheck
results = wall.design(demands, *,
    auto_update=True,                # propose + rebuild + iterate
    max_iter=5, tolerance=25.0,      # convergence on BE length (mm)
    clear_cover=25.0,                # to outside edge of BE hoop
)
```

`results` carries:

```python
results.minimum     WallDesignProposal — ACI floor (rho_min, two-curtain)
results.demand      using demands for §18.10.6 and §22.5 Av/s
results.capacity    Mpr-based Ve and Ash inside the BE
results.envelope    worst-of across modes
results.original_wall   = self (untouched)
results.final_wall      the wall after iteration (may === original if no BE added)
results.history         tuple[Wall, ...] one per iteration
results.iterations
results.converged
```

### 3.5 `evolve()` / `apply()`

Both names refer to the same operation:

```python
new_wall = wall.evolve(proposal)      # returns a NEW Wall, does not mutate
```

Updates: rho_t, rho_l, section if `proposal.proposed_section` is set,
and propagates BE column proposals to the BE columns of the new wall.

### 3.6 Plotting

```python
wall.plot.section()
wall.plot.pm(plane='in-plane')          # or 'out-of-plane'
wall.plot.pm_compare()
wall.plot.pmm_volume(n_Pu=25)
wall.plot.pmm_slice(Pu, demand=(Mu, Mu_out))
wall.plot.demand(demands)
wall.plot.be_section(which='top')       # delegates to wall.be_top.plot.section()
wall.plot.dashboard(demands=demands)
wall.plot.iteration_history(results)    # BE length vs iteration
```

---

## 4. Three design modes

### 4.1 `minimum`

ACI floor without demands. Sets:

- `rho_l_required = rho_t_required = max(0.0020, 0.0025)` per §18.10.2.1
- `web_bar_spacing ≤ min(lw/5, 3·tw, 450 mm)` per §11.7.3
- `two_curtain_required` per §18.10.2.2 / §11.7.2.3
- BE **not** forced (no demand → no displacement / stress check)

### 4.2 `demand`

Uses demands explicitly:

- `Pu_used = demands.Pu` (drives Ash equation iii in the BE)
- `Vu_used = demands.Vu` for `Av/s` per §22.5
- `delta_u, sigma_max` drive §18.10.6.2 and §18.10.6.3

### 4.3 `capacity`

Capacity-design (§18.10.3):

- `Mpr_in_plane` from probable interaction diagram (fy_pr = 1.25·fy, φ = 1.0)
- `Pu_for_Mpr` = axial at which the plastic hinge actually forms
- `Ve_capacity = max(ω_v · Vu, (Mpr / Mu) · Vu)`
- `Av/s` derived from `Ve_capacity`

### 4.4 `envelope`

Worst of the three modes:

- `rho_t_required = max(...)` over modes
- `be_required = any(...)`
- `be_length_proposed = max(...)`
- `proposed_section` = the first non-None across modes (capacity is the
  usual driver)

---

## 5. Iteration loop

```
LOOP it = 1..max_iter:
    1. Build Surface for current wall
    2. Compute c at the demand point on the in-plane PM diagram
    3. Run the three design modes on the current geometry
    4. Read envelope.proposed_section
    5. If no proposal → DONE
    6. If proposal.be_length matches current within tolerance → DONE
    7. Otherwise current = current._with_section(new_section); repeat
```

Bars in the new BE are picked from `bar_schedule.longitudinal` and
arranged as a 2 × n_y perimeter, targeting roughly `ρ_BE ≈ 1%` per
§18.10.6.4(b). The web bars are NOT replaced — they keep running edge
to edge (continuous detailing).

---

## 6. `WallDesignProposal` — full field list

```python
mode                       'minimum' | 'demand' | 'capacity' | 'envelope'

# Web distributed reinforcement
rho_l_required, rho_t_required               (ratio)
rho_l_provided, rho_t_provided               (ratio)
double_curtain_required                       bool
web_bar_db, web_bar_spacing, web_bar_layers   (mm, mm, int)

# BE decision
be_required, be_required_disp, be_required_stress    bool
be_length_proposed, be_thickness_proposed             mm
be_extension_above, be_extension_below                mm  (§18.10.6.2)
c_used                                                mm

# BE detailing (delegated to columns.Column on each BE)
be_top_proposal, be_bot_proposal              ColumnDesignProposal | None

# Capacity-design shear
Mpr_in_plane, Pu_for_Mpr, omega_v
Ve_capacity, Ve_used, amplification           (kN, kN, -)

# Wall shear (§18.10.4 + §22.5)
alpha_c_factor, phi_Vn, phi_Vn_max            (kN)
av_s_required, ratio_Ve_over_phiVn

# Demand context
Pu_used, Pu_over_Ag_fc

# Geometry for evolve()
proposed_section: WallSection | None

# Diagnostics
notes: tuple[str, ...]
```

---

## 7. ACI 318-25 clauses consolidated

| Topic | Section | Where in code |
|---|---|---|
| Acv (in-plane shear) | §11.5.4 | `WallSection.Acv` |
| Distributed rho minimum | §11.6 / §18.10.2.1 | `distributed.rho_min_distributed` |
| Bar spacing / curtains | §11.7 / §18.10.2.2 | `distributed.web_bar_spacing_max`, `two_curtain_required` |
| Vn wall (alpha_c) | §18.10.4.1 (Table 18.10.4.1) | `shear.alpha_c`, `shear.vn_wall` |
| Vn cap (0.83 √fc Acv) | §18.10.4.4 | `shear.vn_wall_max` |
| Av/s for Vu | §22.5 / §11.5 | `shear.av_s_required_wall` |
| Ve capacity (omega_v) | §18.10.3.1 | `shear.ve_in_plane_capacity`, `shear.omega_v` |
| BE displacement check | §18.10.6.2 | `boundary.boundary_element_required_displacement` |
| BE stress check | §18.10.6.3 | `boundary.boundary_element_required_stress` |
| BE length | §18.10.6.4(a) | `boundary.boundary_extension_length` |
| BE thickness min | §18.10.6.4(c) | `boundary.be_thickness_minimum` |
| BE Ash | §18.10.6.4(g) | `boundary.ash_required_be` → delegated to `columns.confinement.ash_required` |
| Coupling beams | §18.10.7 | `coupling_beam.py` |
| Wall piers | §18.10.8 | `wall_pier.py` |

---

## 8. Conventions

### 8.1 Sign convention

- `Pu > 0` is compression, `Pu < 0` tension
- `Mu` is in-plane (about the section x-axis after WallSection's axis convention)
- `Mu_out` is out-of-plane (about the y-axis)
- `Vu` and `Vu_out` are magnitudes; sign irrelevant for detailing

### 8.2 Local axes

- `x` across thickness (`tw`)
- `y` along wall length (`lw`), top = +lw/2
- `angle_deg = 0` → in-plane bending; `angle_deg = 90` → out-of-plane

### 8.3 Internal units

| Quantity | Unit |
|---|---|
| Stress (fc, fy) | MPa |
| Length (lw, tw, BE) | mm |
| Force (Pn, Vn) | kN |
| Moment (Mn) | kN·m |

`wall.set_units(code)` only changes display.

---

## 9. Known simplifications

These are explicit choices, not bugs.

1. **`omega_v` is constant (default 1.5)** — the strict height-dependent
   form of §18.10.3.1.2(b) is not implemented. Pass a different value
   to `Wall(..., omega_v_factor=...)` if needed.
2. **BE convergence is on BE length only** (within `tolerance`, default
   25 mm). Thickness changes are accepted at any iteration.
3. **BE bars target ρ ≈ 1%** as a baseline (§18.10.6.4(b)). The picker
   uses the largest schedule diameter ≤ 25 mm and lays them out as
   2 × n_y perimeter bars. Override by overriding `bar_schedule`.
4. **Out-of-plane shear is not detailed** — `WallCheck.ratio_shear_out`
   is always `None`. The PMM surface still covers Mu_out for flexure.
5. **`c_at_demand` interpolation** picks the nearest PM point to
   `(Pu/0.9, Mu/0.9)`. With dense diagrams (n_points ≥ 60) the result
   is within a few mm of the true `c`.
6. **BE bars added in `evolve()` REPLACE web bars in the BE region.**
   The wall starts at its web — when there is no BE, the distributed
   reinforcement runs the full `lw`. When a BE is added, web bars
   inside the new BE rectangle are dropped and the BE bars take their
   place. No stacked bars.
7. **Two-curtain decision is binary**. Below the threshold we report
   1 curtain; above we report 2. The actual web design with two layers
   should account for spacing in both faces of the wall.

---

## 10. Smoke test

`python -m design._smoke_test` exercises beam / column / wall / coupling
beam / wall pier. The wall block sets up a 4000 × 300 mm wall with two
explicit BEs (600 × 400 mm each) and exercises in-plane PM, biaxial PM,
BE checks, distributed design proposal, and the iteration history.

---

## 11. Notebooks

| File | What it shows |
|---|---|
| `design/example_wall.ipynb` | Full tour: section, capacity, PM, PMM, demands, design with auto_update, BE as Column, evolve(), units, plots, history |
| `design/example_wall_simple.ipynb` | Quick path — section → capacity → design → BE inspection |

Both expose `INPUT_UNITS` and `DISPLAY_UNITS` at the top.

---

## 12. Design decisions log

- **Wall is immutable.** Every shape-changing operation returns a NEW
  `Wall`. This is the v3 of the plan — earlier drafts had `apply()`
  mutating in place; we changed it so the original and final designs
  stay side-by-side.
- **BE = Column.** Boundary elements are not a special-case path; they
  are real `Column` instances built from `WallSection.be_*_as_rectangular()`.
- **`design()` iterates by default when demands are passed.** That
  matches the ACI workflow: you provide demands, the code proposes BE
  geometry, the geometry changes the PMM, the PMM might change `c`,
  which might change the BE proposal — so we loop until BE length is
  stable.
- **Web bars stay continuous through the BE.** In the real wall the web
  steel runs edge-to-edge and the BE adds extra bars on top.
- **`Po` is descriptive, not a driver** — same principle as in columns.
- **`omega_v = 1.5` default**, knowing it is a simplification of the
  full §18.10.3.1.2 formula.
- **`section.Acv = tw · lw`**, not `tw · 0.8·lw`. ACI §11.5.4 defines
  Acv on the full gross area; the 0.8 factor lives in `d_eff` for
  Av/s only.
