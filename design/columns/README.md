# `design.columns` — Reinforced concrete column module

Self-contained engineering of concrete columns per **ACI 318-25**. Builds
the P-M-M interaction surface from a `RectangularSection` + `RebarLayout`,
lets you query Mn / Pn / φMn / φPn through chained proxies, checks demands
in 3-D, and proposes transverse-reinforcement detailing in three
independent modes (minimum / demand / capacity / envelope), with an
optimization pass that ranks alternatives by **kg of steel per m³ of
concrete**.

This module is **pure engineering**. It does not import anything from
ETABS, HDF5 readers, or COM. The same `Column` object works whether you
build it manually or hand it a `Section` adapted from an external model.

Internal numerics: **MPa, mm, kN, kN·m**. Presentation units are
configurable through `col.set_units(...)` (ETABS codes 1..16). To enter
values in another system, use `to_internal('force', value, units=9)`.

---

## 1. Quick start

```python
from design import (
    Concrete, Steel, Bar, RectangularSection, Column,
    to_internal, units_from,
)
from design.columns.column import ColumnDemands
from design.sections.reinforcement import RebarLayout, perimeter_bars

# Materials + geometry
concrete = Concrete(fc=28)
steel    = Steel(fy=420, grade=60, density=7850)   # kg/m³ for take-offs
bar      = Bar(diameter=22)

group   = perimeter_bars(b=500, h=500, cover=40,
                         n_x=4, n_y=4, bar=bar, steel=steel)
section = RectangularSection(b=500, h=500, concrete=concrete,
                             rebar=RebarLayout(groups=(group,)))

# Build the column
column = Column(section=section, lu=3.0, seismic=True, label='C-1A')
column.set_units(11)         # Tonf_mm_C for display
column.run()

# Capacity + proxies
print(column.Po, column.To, column.phi_Pn_max)
print(column.phi_Mn.peak(angle=0))

# Demand (inputs in N_mm, converted to internal)
demands = ColumnDemands(
    Pu  = to_internal('force',  2_500_000, units=9),   # 2500 kN
    Mux = to_internal('moment', 200_000_000, units=9), # 200 kN·m
    Muy = to_internal('moment', 120_000_000, units=9),
    Vux = to_internal('force',  150_000, units=9),
)
chk = column.check(demands)

# Three-mode design + envelope, with directional Mpr / Ve / φVn
results = column.design(demands, clear_cover=25)
print(results.envelope.db_hoop, results.envelope.n_legs_x, results.envelope.n_legs_y)
print(results.capacity.Mpr_x, results.capacity.Mpr_y)
print(results.capacity.Ve_x_capacity, results.capacity.Ve_y_capacity)

# Adopt the envelope into the column (in-place)
column.apply(results.envelope)

# Optimize around the baseline — rank by kg/m³
alts = column.run_optimize(demands)
column.apply(alts[0].proposal)
```

---

## 2. File layout

```
design/columns/
├── README.md                ← this document
├── __init__.py              ← public re-exports
├── column.py                ← Column + dataclasses + envelope picker
├── interaction.py           ← InteractionDiagram + interaction_diagram()
├── interaction_surface.py   ← InteractionSurface (internal, wrapped by Surface)
├── surface.py               ← Surface (the user-facing facade)
├── proxies.py               ← _DiagramFieldView (Mn / Pn / φMn / φPn)
├── plotting.py              ← _ColumnPlotter — bound plot proxy
├── style.py                 ← PlotStyle
├── bar_schedule.py          ← BarSchedule (preferred hoop / long diameters)
├── confinement.py           ← Ash_required, s_max_*, lo_length, hx_check
├── probable.py              ← probable_interaction_diagram + mpr_envelope
├── slenderness.py           ← k·lu/r check, Cm, moment magnifier
└── optimize.py              ← run_optimize + steel-quantity helpers
```

Shared collaborators:

```
design/common/
├── materials.py             ← Concrete, Steel (with density), Bar
├── factors.py               ← β1, phi(εt), λ
├── units.py                 ← UnitSystem + to_internal helper
└── geometry.py              ← polygon clipping (Sutherland-Hodgman)

design/sections/
├── base.py                  ← Section ABC
├── reinforcement.py         ← Rebar / RebarGroup / RebarLayout + builders
└── rectangular.py           ← RectangularSection

design/analysis/
└── section_analysis.py      ← strain compatibility kernel (pm_at_neutral_axis)
```

---

## 3. The `Column` class — public surface

### 3.1 Construction

```python
Column(
    *, section: Section,
       lu: float,                       # m
       k: float = 1.0,
       spiral: bool = False,
       Av: float = 0.0,                 # used only until apply()
       fyt: float = 420.0,
       s: float | None = None,
       seismic: bool = True,
       hx: float = 200.0,               # §18.7.5.2
       h_min: float | None = None,
       transverse_bar_diameter: float = 10.0,
       units: int | str | UnitSystem | None = None,
       bar_schedule: BarSchedule | None = None,
       label: str = "Column",
       run_settings: dict | None = None,
)
```

`omega_0` (default 3.0) lives on the instance — `column.omega_0 = ...` —
and is used by §18.7.6.1.1 to cap `Ve_capacity` by `Ωo·Vu` when demands
are passed.

### 3.2 `run(n_angles=37, n_points=60)`

Builds the PMM surface and caches `Po`, `To`, `phi_Pn_max`, `phi_Vn`.
Lazy on first access to any of those.

### 3.3 Surface, scalars, proxies

```python
column.surface                     # Surface
column.surface[0]                  # InteractionDiagram around X (Mxx)
column.surface[90]                 # InteractionDiagram around Y (Myy)
column.surface.at_Pu(Pu)           # (Mx, My) arrays of the biaxial envelope at Pu
column.surface.volume(n_Pu=25)     # (Pu_grid, Mx_grid, My_grid) for 3D plots
column.surface.check(Pu=, Mux=, Muy=)   # (passed, ratio) — 3D biaxial check

column.Po, column.To
column.phi_Pn_max, column.phi_Vn

# Proxies — all four share the same interface
column.Mn(angle=0)                 # full curve (np.ndarray, kN·m)
column.Mn.at(Pu=0, angle=0)        # scalar at Pu, theta
column.Mn.peak(angle=0)            # balanced moment at one angle
column.Mn[0, 2500]                 # == column.Mn.at(angle=0, Pu=2500)
```

### 3.4 Three-mode `design()`

```python
results = column.design(
    demands=None,                  # ColumnDemands or None
    *,
    db_hoop=None, n_legs_x=None, n_legs_y=None,
    db_long_min=20.0,
    clear_cover=25.0,              # to the OUTSIDE edge of the hoop (§18.7.5.4)
)

results.minimum    # eqs (i), (ii) — no axial input
results.demand     # adds eq (iii) with demands.Pu (None if no demands)
results.capacity   # capacity-design: Mpr-driven; eq (iii) with Pu_for_Mpr
results.envelope   # most demanding of the three by steel index
```

### 3.5 `apply(proposal)`

Adopts the proposed detailing **in-place**:
`column.transverse_bar_diameter`, `column.s`, `column.Av` are updated;
`column.phi_Vn` is recomputed. The PMM surface is **not** rebuilt — it
only depends on the longitudinal layout.

### 3.6 `run_optimize(demands)`

Explores `(db_hoop × n_legs_x × n_legs_y)` alternatives around the
design envelope, ranks by kg/m³, returns a list of
`OptimizeAlternative`. The baseline (the `design()` envelope) is always
included and marked with `is_baseline=True`.

### 3.7 Plotting

```python
column.plot.section()
column.plot.pm(angle=0)
column.plot.pm_compare(angles=[0, 30, 45, 60, 90])
column.plot.pmm_volume(n_Pu=25)
column.plot.pmm_slice(Pu, demand=(Mux, Muy))
column.plot.pmm_with_demand(Pu, Mux, Muy)
column.plot.demand(demands)
column.plot.dashboard(demands=demands)
```

Styles live in `column.plot.style` (a `PlotStyle` dataclass). All
methods read `column.units` and translate axis labels and tick values.

---

## 4. Units — input vs display

Internal numerics never change (MPa, mm, kN, kN·m). Two layers around them:

### 4.1 Input — `to_internal(kind, value, units=...)`

```python
from design import to_internal

Pu_kN = to_internal('force',  2_500_000, units=9)        # N      → 2500 kN
Mu    = to_internal('moment', 200_000_000, units=9)      # N·mm   → 200 kN·m
b_mm  = to_internal('length', 0.5,        units=12)      # m      → 500 mm
fc_MPa = to_internal('stress', 280,       units=14)      # kgf/cm² → 27.5 MPa
```

Four `kind`s: `'force'`, `'length'`, `'moment'`, `'stress'`. Units arg
accepts the ETABS code (1..16), a name like `'Tonf_m_C'`, or a
`UnitSystem` instance.

### 4.2 Display — `column.set_units(code_or_name)`

```python
column.set_units(11)            # by ETABS code
column.set_units('Tonf_m_C')    # by name

column.units.force_factor       # kN → target force
column.units.length_factor      # mm → target length
column.units.moment_factor      # kN·m → target moment
column.units.area_factor        # = length_factor²
column.units.stress_factor      # MPa → target stress

# Reverse direction — UnitSystem can convert back:
column.units.to_internal_force(value)     # value in display force → kN
column.units.to_internal_length(value)    # to mm
column.units.to_internal_moment(value)    # to kN·m
column.units.to_internal_stress(value)    # to MPa
```

Each plot title, axis label, summary, and report uses `column.units`.

---

## 5. Three design modes — details

### 5.1 `minimum`

Pure code minimum from §18.7.5.4. Eqs (i) and (ii) only.

- `Pu_used = None`, `eq_iii_active = False`.
- Mpr is still computed for diagnostics (it costs nothing extra), but
  not consumed by the detailing.

### 5.2 `demand`

ACI capacity-design **assuming a known set of factored demands**:

- `Pu_used = demands.Pu` → activates eq (iii) when `Pu/(Ag·fc') > 0.3`.
- `Vu_used = max(|Vux|, |Vuy|)` — the analysis shear.
- `av_s_required_x/y` derived from §22.5 for the **demand** shear.

### 5.3 `capacity`

Full Mpr-based capacity design (§18.7.6.1):

- `Mpr_x, Mpr_y` from `probable_interaction_diagram` (uses `fy_pr = 1.25·fy`,
  `φ = 1.0`) — `mpr_envelope()` sweeps the diagram and returns the peak
  moment **and** the Pu at which the peak occurs.
- `Ve_y_capacity = 2·Mpr_x / lu` (Vy is induced by Mx; resisted by legs ∥ X).
- `Ve_x_capacity = 2·Mpr_y / lu` (Vx is induced by My; resisted by legs ∥ Y).
- `Pu_used = max(Pu_for_Mpr_x, Pu_for_Mpr_y)` — the axial **at which the
  plastic hinge actually forms**, not `Po`. Po would be unrealistic (the
  column does not reach pure compression while plastifying in flexure).
- If `demands` is passed, `Ve_*_used = min(Ve_*_capacity, Ωo·Vu_*)`
  (§18.7.6.1.1).
- `av_s_required_x/y` derived from §22.5 for the **capacity** shear.

### 5.4 `envelope`

The most demanding proposal across all modes. Picked by total
transverse-steel intensity (`db² · (n_legs_x + n_legs_y) / spacing`),
**not** by Ash alone — that fixes the case where minimum and capacity
have the same Ash (eq iii dormant) but capacity needed extra legs to
cover shear.

---

## 6. The hoop / leg picker

Selects `(db_hoop, n_legs_x, n_legs_y)` from `bar_schedule.hoops` and the
physical bar-count limit. For each diameter, runs **two gates**:

**Gate A — required area (combines confinement + capacity shear):**

```python
n_x_conf  = ceil(Ash_x / a_per_leg)                  # §18.7.5.4
n_y_conf  = ceil(Ash_y / a_per_leg)
n_x_shear = ceil((Av/s_req_y · s) / a_per_leg)       # legs ∥ X resist Vy
n_y_shear = ceil((Av/s_req_x · s) / a_per_leg)       # legs ∥ Y resist Vx
n_x_req   = max(n_x_conf, n_x_shear, 2)
n_y_req   = max(n_y_conf, n_y_shear, 2)
```

**Gate B — `hx ≤ 350 mm` (§18.7.5.2):**

If `hx = b_core / (n_legs − 1) > 350 mm` with the area-required count,
the picker **increases n_legs** at the same diameter (`bump_to_hx`)
until `hx` passes or the physical limit `max_x` / `max_y` is reached.
Only if no n_legs satisfies both gates the next diameter is tried.

### 6.1 Physical bar-count limit

`_count_longitudinal_per_face` identifies the **extreme layer** of bars
(top / bottom / left / right faces) and counts how many share that
layer (within `layer_tol = 5 mm`). Result:

- `max_legs_x` = bars on left/right faces = bars distributed along Y in
  a perimeter layout.
- `max_legs_y` = bars on top/bottom faces = bars distributed along X.

---

## 7. ACI 318-25 clauses consolidated

| Topic | Section | Where in code |
|---|---|---|
| Whitney stress-block depth (β₁) | Table 22.2.2.4.3 | `common/factors.py::beta1` |
| Extreme concrete strain εcu = 0.003 | §22.2.2.1 | `materials.py::Concrete.eps_cu` |
| Strength reduction φ in flexure-axial | §21.2.2 (Table 21.2.2) | `common/factors.py::phi_axial_flexure` |
| Strength reduction φ in shear | §21.2.1 | `common/factors.py::phi_shear` |
| Po — pure compression | §22.4.2.1 | `analysis/section_analysis.py::pn_max` |
| Pn,max cap (0.80 ties / 0.85 spirals) | §22.4.2.1 | same |
| Vc simplified | §22.5.5.1 | `Column._av_s_required_for_direction` |
| Mpr — probable moment (1.25·fy, φ=1.0) | §18.6.5 / §18.7.6.1 | `columns/probable.py` |
| lo confined region | §18.7.5.1 | `confinement.py::lo_length` |
| hx ≤ 350 mm | §18.7.5.2 | `confinement.py::hx_check` + picker bump |
| Spacing in lo (s) | §18.7.5.3 | `confinement.py::s_max_confined` |
| Ash required (eqs i, ii, iii) | §18.7.5.4 — Table 18.7.5.4 | `confinement.py::ash_required` |
| Spacing outside lo | §18.7.5.5 | `confinement.py::s_max_middle_zone` |
| Ve from Mpr | §18.7.6.1 | `Column._design_one` (capacity branch) |
| Ωo·Vu cap on Ve | §18.7.6.1.1 | same |
| Slenderness | §6.6.4 | `slenderness.py` |

---

## 8. The full `ColumnDesignProposal` — every field

```python
mode                      str        'minimum' | 'demand' | 'capacity' | 'envelope'

# Confinement — §18.7.5.4 with directional bc
ash_required_x            float      mm²  (bc = b − 2·clear_cover)
ash_required_y            float      mm²  (bc = h − 2·clear_cover)

# Chosen detailing
db_hoop                   float      mm
n_legs_x                  int        legs ∥ X axis
n_legs_y                  int        legs ∥ Y axis
av_provided_x             float      mm²  = n_legs_x · π·db²/4
av_provided_y             float      mm²

# Spacings
spacing_confined          float      mm  in the lo region (§18.7.5.3)
spacing_middle            float      mm  outside lo      (§18.7.5.5)
lo                        float      mm  (§18.7.5.1)

# §18.7.5.2 geometry check
hx_x_face                 float      mm  = b_core / (n_legs_x − 1)
hx_y_face                 float      mm
hx_ok                     bool       both faces ≤ 350 mm

# Demand context (where applicable)
Pu_used                   float|None kN
Pu_over_Ag_fc             float|None
eq_iii_active             bool       True if Pu/(Ag·fc') > 0.3

# Mpr / Ve / Av/s — PER DIRECTION (computed for every mode where capacity matters)
Mpr_x                     float|None kN·m  bending around X (fy_pr=1.25·fy, φ=1.0)
Mpr_y                     float|None kN·m  bending around Y
Pu_for_Mpr_x              float|None kN    Pu at the peak Mpr_x
Pu_for_Mpr_y              float|None kN
Ve_x_capacity             float|None kN    = 2·Mpr_y / lu — resisted by legs ∥ Y
Ve_y_capacity             float|None kN    = 2·Mpr_x / lu — resisted by legs ∥ X
Ve_x_used                 float|None kN    capped by Ωo·Vux if demands provided
Ve_y_used                 float|None kN    capped by Ωo·Vuy
amplification_x           float|None       Ve_x_used / Vux_demand
amplification_y           float|None

# Shear capacity of the *proposed* hoops, per direction
phi_Vn_x                  float|None kN    φ·(Vc_x + Vs_x)  bw=h, d=0.9·b
phi_Vn_y                  float|None kN    φ·(Vc_y + Vs_y)  bw=b, d=0.9·h
av_s_required_x           float|None mm²/mm  Av/s needed for Vex (legs ∥ Y)
av_s_required_y           float|None mm²/mm  Av/s needed for Vey (legs ∥ X)
ratio_Ve_over_phiVn_x     float|None   Ve_x_capacity / phi_Vn_x
ratio_Ve_over_phiVn_y     float|None   Ve_y_capacity / phi_Vn_y

# Diagnostics
notes                     tuple[str]
```

**Legacy aliases (back-compat)** — these are `@property` shortcuts so
existing code keeps working:

```python
Mpr_top, Mpr_bot          → Mpr_x        (single-curvature assumption)
Pu_for_Mpr                → Pu_for_Mpr_x
Ve_capacity, Ve_used      → Ve_y_capacity, Ve_y_used
Vu_used                   → Ve_y_used
amplification_vs_demand   → amplification_y
av_s_required             → max(av_s_required_x, av_s_required_y)
av_s_provided             → max provided per direction
ratio_shear               → max(ratio_Ve_over_phiVn_x, ratio_Ve_over_phiVn_y)
```

---

## 9. Conventions

### 9.1 Sign convention

- `Pu > 0` is compression, `Pu < 0` is tension.
- `Mux` = moment about local X (bending around X, extreme fiber in ±y).
- `Muy` = moment about local Y (bending around Y, extreme fiber in ±x).
- **Vy is induced by Mx, and resisted by legs ∥ X (the "ramas cortas"
  in a tall section).**
- **Vx is induced by My, resisted by legs ∥ Y ("ramas largas").**

### 9.2 Local axes (rectangular section)

- `x` horizontal, positive to the right, width `b`.
- `y` vertical, positive upward, height `h`.
- Origin at the geometric centroid of the gross section.

### 9.3 `clear_cover` (NEW — replaces `cover_to_hoop_center`)

`clear_cover` is the cover to the **outside edge** of the transverse
reinforcement, which is the same definition ACI 318-25 Table 18.7.5.4
uses for `bc` and `Ach`:

```
bc = b − 2·clear_cover    (X dimension of the core)
hc = h − 2·clear_cover    (Y dimension of the core)
Ach = bc · hc             (scalar — not directional, ACI definition)
```

Typical values: 20–30 mm. Default in `column.design()` is **25 mm**.

### 9.4 Internal units

| Quantity | Unit |
|---|---|
| Stress (fc, fy, Es) | MPa |
| Length (b, h, cover, d, c) | mm |
| Area (Ag, Ach, As, Av) | mm² |
| Force (Pn, Vn) | kN |
| Moment (Mn) | kN·m |
| Steel density | kg/m³ (in `Steel.density`) |

`column.set_units(code)` only changes display; the formulas never see
anything but the internal system.

---

## 10. Po — where it enters (and where it doesn't)

`Po = 0.85·fc·(Ag − Ast) + fy·Ast / 1000` lives at
[section_analysis.py:174](design/analysis/section_analysis.py:174).

| What | Uses Po? | Why |
|---|---|---|
| PM curve anchor point at the top of the diagram | ✓ | `(Pn=Po, Mn=0)` — geometric closure |
| `Pn_max_phi` cap (§22.4.2.1) | ✓ | `phi · 0.80·Po` (ties) — the horizontal cap on the φ curve |
| `column.Po` exposed for inspection / summary / report / plot labels | ✓ | descriptive only |
| Confinement Ash (§18.7.5.4 eq iii) | **No** | uses `Pu_for_Mpr` in capacity mode, `demands.Pu` in demand mode |
| Capacity shear Ve | **No** | uses `Mpr_x`, `Mpr_y` |
| Hoop-leg picker | **No** | sees Ash_required and Av/s_required, neither of which involves Po |

**Po is a descriptive output, not a design driver in the current code.**
Earlier versions used `Pu_used = Po` in capacity mode (overly
conservative — Po assumes pure compression, which a plastifying column
never reaches). Replaced with `Pu_used = max(Pu_for_Mpr_x, Pu_for_Mpr_y)`,
i.e. the Pu at which the plastic hinge actually forms.

---

## 11. Optimization — `column.run_optimize(demands)`

Enumerates feasible `(db_hoop × n_legs_x × n_legs_y)` combinations from
`bar_schedule.hoops` and the physical bar-count limit, recomputes
`design()` for each, and ranks by **kg of steel per m³ of concrete**.

```python
@dataclass(frozen=True)
class OptimizeAlternative:
    proposal:        ColumnDesignProposal
    db_hoop:         float
    n_legs_x:        int
    n_legs_y:        int
    rho_transverse:  float          # kg/m³ — hoops only (lo region)
    rho_longitudinal:float          # kg/m³ — long bars (constant across alts)
    rho_total:       float
    feasible:        bool           # passes Ash + hx + bar-count
    is_baseline:     bool           # True if this is design() envelope
    notes:           tuple[str]
```

**The baseline (the design envelope) is always included** in the list
and marked with `is_baseline=True`. The algorithm explores *around* it;
nothing is generated from scratch.

Quantity formula (transverse only):

```
legs_per_layer  = n_legs_x · bc + n_legs_y · hc                [mm]
layers_per_m    = 1000 / spacing_confined                       [-]
length_per_m    = legs_per_layer · layers_per_m                 [mm]
volume_steel    = length_per_m · a_per_leg                       [mm³]
mass_steel      = volume_steel · steel.density · 1e-9            [kg]
volume_concrete = Ag · 1000 · 1e-9                                [m³]
rho_transverse  = mass_steel / volume_concrete                    [kg/m³]
```

---

## 12. Lazy evaluation and caching

| Object | When built | When invalidated |
|---|---|---|
| `column.surface` | first read or explicit `run()` | `clear_cache()`, section/rebar change |
| `column.Po / To / phi_Pn_max / phi_Vn` | first read (via `_ensure_run`) | same |
| `column.design_results` | when `design()` is called | next `design()` call |
| `column._plotter` | first read of `column.plot` | never (state lives in plotter) |
| Proxies (`Mn` etc.) | cheap per access; data from cached surface | — |

There is no eager evaluation. A `Column(...)` call is O(1); the first
access to a result triggers the surface build (~150 ms for 500×500 at
37 angles × 60 points).

---

## 13. Known simplifications

These are explicit choices — not bugs. Documented so you can replace
them with stricter forms if your project demands it.

1. **`Mpr_top = Mpr_bot`** — single-curvature assumption. Asymmetric
   top/bot moments require a separate analysis-pair input.
2. **`Pu_for_Mpr` is the Pu that maximizes `Mn_pr`** in the probable
   diagram. If you want a different Pu range, call `mpr_envelope` with
   `Pu_range=(lo, hi)` directly.
3. **Bresler-like 3D check.** `surface.check(Pu, Mux, Muy)` interpolates
   the angle-resolved envelope — not the strict ACI biaxial method.
4. **eq (iii) of §18.7.5.4 is the simplified form** `0.2·(Pu/(Ag·fc'))·…`
   — the 318-19/25 strict form uses `kf · kn · Pu / (fyt · Ach)` with
   `kf = fc'/175 + 0.6 ≥ 1` and `kn = nl / (nl − 2)`. Not implemented.
5. **`apply()` collapses `n_legs_x` and `n_legs_y` to a scalar `Av`**
   using `min(n_legs_x, n_legs_y) · a_per_leg`. The directional info
   stays in the proposal; `column.phi_Vn` after apply is therefore
   conservative.
6. **`_count_longitudinal_per_face` is layer-based** with
   `layer_tol = 5 mm`. Works for perimeter-bar layouts; non-standard
   layouts may need a wider tolerance.
7. **`Vc` uses the simplified §22.5.5.1 form** (`0.17·λ·√fc·bw·d`).
   The detailed Vc with `rho_w` and axial effects is available in
   `beams/shear.py::vc_detailed` but not consumed by `Column`.
8. **`Ve_capacity` is computed at θ=0 and θ=90 only.** Off-axis seismic
   shears (rotated frames) would require sweeping Mpr at intermediate
   angles. Not implemented.
9. **Slenderness** is exposed via `slenderness.py` but not auto-applied
   to demands inside `column.check()`. The caller must magnify Mu before
   passing it in (or call the helpers directly).

---

## 14. Integration with an ETABS-imported model (planned)

The `Column` class is ready to be consumed. What is still missing lives
**outside** `design/columns/`:

### 14.1 Section adapter

```python
class EtabsSectionAdapter:
    def section_from(self, etabs_section_name: str) -> Section: ...
```

Translates an ETABS `FrameSectionPropertyDefinition` into a
`RectangularSection` + `RebarLayout`.

### 14.2 Demands adapter

```python
class DemandsAdapter:
    def demands_from(self, frame_id, combo, location='end_i') -> ColumnDemands:
        # Reads FrameForces table, returns Demands in MPa/mm/kN
```

### 14.3 Manager glue

```python
class FrameManager:
    def as_column(self, frame_id: int) -> Column:
        section = self.section_adapter.section_from(frame.SectProp)
        return Column(section=section, lu=frame.length, ...)
```

### 14.4 Bulk design facade

```python
model.design.columns.apply(
    ids=[...] or selection_set,
    combos=['Combo1', 'Combo2'],
    location='critical',
)   # → DesignReport (DataFrame of one row per frame × combo)
```

None of these will require touching `design/columns/`. The module is a
clean dependency that the ETABS-side code consumes.

---

## 15. Public re-exports

```python
from design.columns import (
    Column,
    ColumnDemands, ColumnCapacity, ColumnCheck,
    ColumnDesignProposal, ColumnDesignResults,
    Surface, BarSchedule, PlotStyle,
    InteractionDiagram, interaction_diagram,
    ash_required, s_max_confined, lo_length,
    s_max_middle_zone, hx_check,
    is_slender, cm_factor, moment_magnifier,
    OptimizeAlternative, run_optimize,
    transverse_steel_quantity, longitudinal_steel_quantity,
)
```

Also at the top-level `design` package:

```python
from design import (
    Concrete, Steel, Bar, RectangularSection, Column,
    BarSchedule, PlotStyle,
    UnitSystem, units_from, to_internal,
)
```

---

## 16. Notebooks

| File | What it shows |
|---|---|
| `design/example_column.ipynb` | Full tour — every constructor arg, every plot, every design mode, unit hot-swap, manual override, sweep, three-mode comparison, apply, optimize |
| `design/example_column_simple.ipynb` | Quick path — section, P-M, P-M-M volume, design without + with demands, apply, optimize |

Both expose `INPUT_UNITS` and `DISPLAY_UNITS` at the top so you enter
in any ETABS code and the plots/reports come out in any other.

---

## 17. Smoke test

`design/_smoke_test.py` exercises every element class (beam, column,
wall, coupling beam, wall pier) and emits PMM plots when called with
`--plot`:

```bash
python -m design._smoke_test
python -m design._smoke_test --angles 73 --points 80
python -m design._smoke_test --plot
```

---

## 18. Design decisions log

A short, honest record of the bigger calls made during construction.
Each item is something we discussed and decided together.

- **Always run the three modes.** `column.design()` returns a container
  with `minimum / demand / capacity / envelope`.
- **`apply()` is in-place.** Mutates `column.Av`, `column.s`, etc. The
  PMM surface is not rebuilt — it does not depend on transverse layout.
- **Capacity-design Ve is computed in every mode** as a property of
  section + lu (same value across modes). Each mode then carries its own
  `φVn` and `ratio_Ve_over_phiVn` so the user sees, per mode, whether
  the proposed transverse steel covers Ve.
- **Mpr in BOTH directions** (`Mpr_x`, `Mpr_y`) — not single. For a
  rectangular section with `h > b` these differ by roughly `h/b`.
- **`phi_Vn_x` uses `d = 0.9·b`** (depth in X) and `bw = h` (width
  perpendicular to Vx). `phi_Vn_y` uses `d = 0.9·h` and `bw = b`. Earlier
  versions used `d = 0.9·h` for both, which silently doubled `Vs_x`.
- **Hoop picker combines confinement Ash + capacity shear Av/s.** When
  `n_legs` from the area gate violates `hx ≤ 350 mm`, the picker
  **increases n_legs** at the same diameter (`bump_to_hx`) before
  trying the next diameter.
- **Envelope picker ranks by transverse-steel intensity**
  `db² · (n_legs_x + n_legs_y) / spacing`, not by `Ash_required`. This
  ensures the envelope picks the mode that *also* needed extra legs for
  capacity shear, even when Ash is the same across modes.
- **`Pu_used` in capacity mode is `max(Pu_for_Mpr_x, Pu_for_Mpr_y)`** —
  the axial at the plastic hinge formation, not `Po`. `Po` is descriptive
  only and never drives the detailing.
- **`clear_cover` replaces `cover_to_hoop_center`.** ACI defines `bc` to
  the outside edge of the hoop, so the parameter is now the clear cover
  (typical 20–30 mm). Default 25 mm.
- **`Surface` wraps `InteractionSurface`** so the public class count
  stays small. New code should not import `InteractionSurface` directly.
- **`probable_interaction_diagram` reuses the same strain-compatibility
  kernel** with a scaled rebar layout (`fy_pr = 1.25·fy`) and `φ = 1.0`.
  No duplicate machinery.
- **Polygon clipping is Sutherland-Hodgman** (with Gauss-Green for
  area / centroid). In `common/geometry.py`.
- **The αc for in-plane wall shear is the SI value** (0.25 to 0.17),
  not the kg/cm² value (0.80 to 0.53). The legacy repo we drew
  inspiration from was in kg/cm² and the constants tripped us up once.
