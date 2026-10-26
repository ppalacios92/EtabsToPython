# Columns

[`design.columns.column.Column`](../api/columns.md) is a reinforced-concrete column
with a full **biaxial P-M-M interaction surface**, directional shear, and
confinement design per ACI 318-25 §18.7. It implements the
[element contract](design_overview.md).

Internal units: **mm · MPa · kN · kN·m · mm²**. `units=` is presentation only.

## Building a column

### `Column.rectangular()`

```python
from design import Column, Concrete, Steel

col = Column.rectangular(
    b=500, h=500, cover=40, lu=3000,
    concrete=Concrete(fc=28), steel=Steel(fy=420),
    n_x=4, n_y=4, db_long=25,
    db_stirrup=10, n_legs_x=4, n_legs_y=4, s_stirrup=100,
    k=1.0, spiral=False, seismic=True, hx=200,
)
```

| Parameter | Type | Default | Meaning (units) |
|---|---|---|---|
| `b` / `h` | float | — | Section width / height (mm) |
| `clear_cover` | float \| None | `None` | Cover to **outside edge of hoop** (mm); alias `cover` |
| `cover` | float \| None | `None` | Alias for `clear_cover` |
| `concrete` / `steel` | — | — | Materials (fc, fy in MPa) |
| `n_x` | int | — | Bars on each **X face** (left & right) |
| `n_y` | int | — | Bars on each **Y face** (top & bottom) |
| `db_long` | float | — | Longitudinal bar diameter, uniform (mm) |
| `db_stirrup` | float | `10.0` | Hoop/tie diameter (mm) |
| `n_legs_x` / `n_legs_y` | int \| None | `None` | Hoop legs ∥ X / ∥ Y (design later if `None`) |
| `s_stirrup` | float \| None | `None` | Confined-zone hoop spacing (mm) |
| `lu` | float | — | Unbraced length (mm) |
| `k` | float | `1.0` | Effective-length factor |
| `spiral` | bool | `False` | Spiral vs ties — affects φ and `phi_Pn_max` cap |
| `seismic` | bool | `True` | Apply §18.7 seismic rules |
| `hx` | float | `200.0` | Max spacing of supported bars for `hx` check (mm) |
| `h_min` | float \| None | `None` | Member dim for spacing checks (mm); defaults to `section.width()` |
| `steel_transverse` | `Steel` \| None | `None` | Transverse steel (defaults to longitudinal) |
| `units` / `bar_schedule` / `label` | — | — | Presentation, hoop schedule, identifier |
| `run_settings` | dict \| None | `None` | Merged with `{"n_angles": 37, "n_points": 60}` |

Longitudinal layout is a **perimeter pattern**; bar centroids at
`clear_cover + db_stirrup + db_long/2` from each face.

### `Column()` — bare constructor

```python
Column(*, section, lu, k=1.0, spiral=False, Av=0.0,
       Av_x=None, Av_y=None, fyt=420.0, s=None, seismic=True,
       hx=200.0, h_min=None, transverse_bar_diameter=10.0,
       units=None, bar_schedule=None, label="Column", run_settings=None)
```

| Parameter | Default | Meaning (units) |
|---|---|---|
| `section` | — | `Section` (geometry + materials + rebar) |
| `lu` | — | Unbraced length (mm) |
| `k` / `spiral` | `1.0` / `False` | Effective-length factor / spiral vs ties |
| `Av` | `0.0` | Legacy scalar; seeds `Av_x`/`Av_y` if both omitted (mm²) |
| `Av_x` / `Av_y` | `None` | **Directional** transverse area, legs ∥ X / ∥ Y (mm²) |
| `fyt` | `420.0` | Transverse steel yield (MPa) |
| `s` | `None` | Confined-zone hoop spacing (mm) |
| `seismic` | `True` | §18.7.6 seismic rules |
| `hx` | `200.0` | `hx` limit (mm) |
| `h_min` | `None` | Member dim for spacing; defaults to `section.width()` |
| `transverse_bar_diameter` | `10.0` | Hoop diameter (mm) |
| `run_settings` | `None` | Surface resolution overrides |

## Demands — biaxial

### `demands()`

```python
demands(*, Pu=0, Mux=0, Muy=0, Vux=0, Vuy=0, units=None) -> ColumnDemands
```

| Parameter | Default | Meaning (units) |
|---|---|---|
| `Pu` | `0.0` | Axial force, **+ compression** (kN) |
| `Mux` / `Muy` | `0.0` | Moment about local X / Y axis (kN·m) |
| `Vux` / `Vuy` | `0.0` | Shear in X / Y direction (kN) |
| `units` | `None` | Input units; defaults to `self.units` |

```python
d = col.demands(Pu=2200, Mux=180, Muy=120, Vux=160, Vuy=140)
```

`ColumnDemands` fields mirror these five arguments (stored internally in kN, kN·m).

## The contract

### `run()`

```python
run(*, n_angles=None, n_points=None) -> Column
```

Build the P-M-M surface by sweeping the neutral-axis angle 0°→180° and cache `Po`,
`To`, `phi_Pn_max`, `phi_Vn_x`, `phi_Vn_y`. `n_angles` (default 37) and `n_points`
(default 60) come from `run_settings`. Returns `self`.

### `capacity()`

```python
capacity(*, biaxial=True, n_angles=19, n_points_per_curve=60) -> ColumnCapacity
```

Back-compat capacity object. Re-runs if `n_angles`/`n_points_per_curve` differ from
the cache.

??? abstract "`ColumnCapacity` fields"
    | Field | Meaning (units) |
    |---|---|
    | `Po` / `To` | Pure compression / tension capacity (kN) |
    | `phi_Pn_max` | Reduced axial cap, 0.80·Po (ties) / 0.85·Po (spirals) (kN) |
    | `surface` | the PMM `Surface` |
    | `diagram_x` / `diagram_y` | uniaxial diagrams at 0° / 90° |
    | `phi_Vn` | shear capacity (kN) |

### `check()`

```python
check(demands, *, clear_cover=None) -> ColumnCheck
```

All gates: **PMM** biaxial interaction + directional **shear** + confinement
**Ash** (§18.7.5.4, evaluated for demand-Pu and capacity-Pu) + **hx** (§18.7.5.2,
≤350) + **s_max** (§18.7.5.3 confined / §18.7.5.5 middle). `ColumnCheck.summary()`
prints a per-direction report.

??? abstract "`ColumnCheck` fields (selected)"
    | Group | Fields |
    |---|---|
    | Summary | `ratio_pmm`, `ratio_shear`, `passed`, `notes` |
    | Shear X/Y | `phi_Vn_x/y`, `ratio_shear_x/y`, `ratio_Vu_x/y`, `ratio_Ve_x/y`, `Ve_x/y_capacity`, `Mpr_x/y` |
    | Confinement | `Av_x/y_provided`, `Ash_x/y_demand`, `Ash_x/y_capacity`, `ratio_ash_x/y`, `ash_ok` |
    | Geometry | `hx_x_face`, `hx_y_face`, `hx_ok` |
    | Spacing | `s_provided`, `s_max_confined`, `s_max_middle`, `s_confined_ok` |
    | Quantities | `rho_long`, `rho_trans`, `rho_total` (kg/m³) |

### `design()`

```python
design(demands=None, *, db_hoop=None, n_legs_x=None, n_legs_y=None,
       db_long_min=20.0, clear_cover=None, cover=None) -> ColumnDesignResults
```

Designs the **confined (`lo`)** and **middle** zones in four modes. The capacity
mode applies overstrength (`Ω₀ = 3.0`) and probable moments (§18.7.6.1), and
evaluates the §18.7.6.2.1 Vc=0 condition.

| Parameter | Default | Meaning |
|---|---|---|
| `demands` | `None` | If `None`, only the minimum mode runs |
| `db_hoop` | `None` | Force hoop diameter (mm); else auto-pick from `bar_schedule.hoops` |
| `n_legs_x` / `n_legs_y` | `None` | Force leg counts; else auto-pick |
| `db_long_min` | `20.0` | Min longitudinal bar diameter for spacing limits (mm) |
| `clear_cover` / `cover` | `None` | Cover to hoop outside edge (mm); defaults to `section.clear_cover` |

??? abstract "`ColumnDesignProposal` fields (one per mode)"
    | Group | Fields |
    |---|---|
    | Mode | `mode` |
    | Required / chosen | `ash_required_x/y`, `db_hoop`, `n_legs_x/y`, `av_provided_x/y` |
    | Spacing | `spacing_confined`, `spacing_middle`, `lo`, `hx_x/y_face`, `hx_ok` |
    | Drivers | `Pu_used`, `Pu_over_Ag_fc`, `eq_iii_active` |
    | Capacity design | `Mpr_x/y`, `Pu_for_Mpr_x/y`, `Ve_x/y_capacity`, `Ve_x/y_used`, `amplification_x/y` |
    | Shear of proposal | `phi_Vn_x/y`, `av_s_required_x/y`, `ratio_Ve_over_phiVn_x/y`, `ratio_Vu_over_phiVn_x/y` |
    | PMM / Vc=0 | `ratio_pmm`, `vc_zero_x/y` |
    | Middle zone | `middle_db_hoop`, `middle_n_legs_x/y`, `middle_spacing`, `middle_av_provided_x/y`, `middle_phi_Vn_x/y`, … |
    | Diagnostics | `picker_trace_confined`, `picker_trace_middle`, `notes` |

    `ColumnDesignResults` holds `minimum`, `demand` (or `None`), `capacity`,
    `envelope`; iterable; `as_dict()`.

### `evolve()`, `run_optimize()`, `clear_cache()`

```python
evolve(proposal) -> Column
run_optimize(demands=None, *, db_hoop_list=None, clear_cover=25.0,
             include_infeasible=False, sort_by="rho_total")
clear_cache() -> None
```

`evolve` returns a **new** column with the proposal's transverse detailing (same
longitudinal section). `run_optimize` sweeps `(db_hoop, n_legs_x, n_legs_y)` ranked
by steel weight (kg/m³); the `design()` envelope is included as `is_baseline`.
`clear_cache` invalidates the surface and cached scalars.

## The PMM surface

| Property | Meaning (units) |
|---|---|
| `surface` | `Surface` facade over the 3D P-M-M volume |
| `Po` / `To` | axial capacity at zero moment / pure tension (kN) |
| `phi_Pn_max` | reduced axial cap (kN) |
| `phi_Vn_x` / `phi_Vn_y` | reduced shear by direction — legs ∥ Y / ∥ X (kN) |
| `phi_Vn` | legacy scalar `min(phi_Vn_x, phi_Vn_y)` (kN) |
| `Mn` / `phi_Mn` / `Pn` / `phi_Pn` | field-view proxies (`.at(...)`, `.peak(...)`) |
| `plot` | bound `_ColumnPlotter` |

### `Surface` methods

```python
col.surface.check(Pu=2200, Mux=180, Muy=120)   # (passed, ratio) — Bresler §22.4.2.2
col.surface.at_Pu(2200, full_circle=True)       # (Mx, My) envelope at fixed Pu
col.surface.volume(n_Pu=25, Pu_min=None, Pu_max=None, full_circle=True, close_top=True)  # (Pu, Mx, My) grids
col.surface[45.0]                                # InteractionDiagram at 45°
col.surface.angles                               # all neutral-axis angles (deg)
col.phi_Mn.at(Pu=2200, angle=0.0)                # interpolate φMn at a Pu/angle
```

The underlying `InteractionSurface` adds `at_angle(deg)`, `envelope_at_Pu(Pu,
full_circle=True)`, `volume_slices(n_Pu=30, …)`, and `check_demand(Pu, Mux, Muy)`.

`close_top=True` (the default) closes the sampled P-M-M volume with the flat
`φPn,max` plateau: a final row has `Mx = My = 0` at `Pu_max`, rather than
leaving the top as an open ring.

### One-way shear

Column shear uses `design.common.shear.vc_one_way` and Table 22.5.5.1: the
lesser tabulated expression when `Av ≥ Av,min`, the `λs` size-effect expression
otherwise, and the SI cap `0.42·λ·√fc·bw·d`.

## Reporting

```python
col.summary()
col.print_design(results, modes=("demand","capacity","envelope"), show_trace=True)
col.print_final(results.envelope, demands=d)
col.report()                       # dict in display units
col.set_units("Tonf_m_C")
```

## Plotting

```python
col.plot.section(show_bar_labels=False, bar_scale=None)
col.plot.pm(angle=0.0)
col.plot.pm_compare(angles=(0, 45, 90))
col.plot.pmm_volume(n_Pu=25, wireframe=True)
col.plot.pmm_slice(Pu=2200, demand=None, fill=True)
col.plot.pmm_with_demand(Pu=2200, Mux=180, Muy=120)
col.plot.dashboard(demands=d)
```

## Reference

Full [`design.columns` API](../api/columns.md) — `Column`, the dataclasses,
`interaction`, `interaction_surface`, `surface`, `optimize`.
