# Walls

[`design.walls.wall.Wall`](../api/walls.md) is a structural wall: a web plus
optional **boundary elements (BEs)** at each end, per ACI 318-25 §18.10. Two things
make walls special: `design()` is **iterative** (it grows the BE until it
converges), and **each boundary element is a real `Column`**.

Internal units: **mm · MPa · kN · kN·m**. `units=` is presentation only.

## Building a wall

### `Wall.rectangular()`

```python
from design import Wall, Concrete, Steel

wall = Wall.rectangular(
    lw=4000, tw=300, hw=12000, lu=3000,
    concrete=Concrete(fc=28), steel=Steel(fy=420),
    n_y_web=10, db_web=12, cover=25,
    be_top_length=600, be_bot_length=600, be_align="center",
    n_x_be=3, n_y_be=5, db_be=22, db_stirrup=10,
    seismic=True, omega_v_factor=1.5, hx=200,
)
```

| Parameter | Type | Default | Meaning (units) |
|---|---|---|---|
| `lw` / `tw` / `hw` | float | — | Wall length / web thickness / clear height (mm) |
| `lu` | float \| None | `None` → `hw` | Unsupported length (mm) |
| `concrete` / `steel` | — | — | Materials (MPa) |
| `n_y_web` | int | `8` | Web vertical bars per long face |
| `db_web` | float | `10.0` | Web bar diameter (mm) |
| `cover` | float | `25.0` | Clear cover to bar centroid (mm) |
| `be_top_length` / `be_bot_length` | float | `0.0` | BE length along `lw`; 0 = no BE (mm) |
| `be_top_thickness` / `be_bot_thickness` | float \| None | `None` → `tw` | BE thickness across `tw` (mm) |
| `be_align` | str \| None | `None` → `'center'` | BE alignment: `'left'`·`'center'`·`'right'` |
| `be_top_align` / `be_bot_align` | str \| None | `None` | Per-end alignment override |
| `n_x_be` | int | `3` | Bars across BE thickness (must be ≥2; hoops and hx govern transverse detailing) |
| `n_walls_sharing` | int | `1` | Identical wall segments sharing the storey shear |
| `n_y_be` | int | `4` | Bars along BE length |
| `db_be` | float | `20.0` | BE bar diameter (mm) |
| `db_stirrup` | float | `10.0` | BE hoop diameter (mm) |
| `fyt` | float \| None | `None` → `steel.fy` | Transverse steel yield (MPa) |
| `seismic` | bool | `True` | §18.10 special-wall detailing |
| `omega_v_factor` | float | `1.5` | Capacity-design shear amplifier |
| `hx` | float | `200.0` | BE hoop spacing parameter (mm) |
| `rho_t` / `rho_l` | float | `0.0025` | As-built transverse / longitudinal web ratios |
| `units` / `bar_schedule` / `label` / `run_settings` | — | — | Presentation, schedule, identifier, surface resolution |

**Alignment** builds the shape: `center` → symmetric barbell; `left`/`right` →
BE flush to one web face (C, T, L, asymmetric shapes).

### `Wall()` — bare constructor

```python
Wall(*, section: WallSection, hw, lu=None, rho_t=0.0025, rho_l=0.0025,
     fyt=420.0, seismic=True, omega_v_factor=1.5, hx=200.0,
     transverse_bar_diameter=10.0, web_bar_diameter=10.0,
     n_x_be=3, n_walls_sharing=1, be_db_default=22.0, be_n_y_default=5,
     units=None, bar_schedule=None, label="Wall", run_settings=None)
```

| Parameter | Default | Meaning (units) |
|---|---|---|
| `section` | — | `WallSection` (web + optional BEs) |
| `hw` | — | Clear story height (mm) |
| `lu` | `None` → `hw` | Unsupported length (mm) |
| `rho_t` / `rho_l` | `0.0025` | As-built transverse / longitudinal ratios |
| `fyt` | `420.0` | Transverse yield (MPa) |
| `omega_v_factor` | `1.5` | Capacity-design shear amplifier |
| `hx` | `200.0` | BE hoop spacing parameter (mm) |
| `transverse_bar_diameter` / `web_bar_diameter` | `10.0` | Hoop / web bar diameters (mm) |
| `n_x_be` | `3` | Bars across BE thickness (≥2, else `ValueError`) |
| `n_walls_sharing` | `1` | Identical wall segments sharing the storey shear |
| `be_db_default` / `be_n_y_default` | `22.0` / `5` | As-built BE bar diameter / count |
| `run_settings` | `None` | Merged with `{"n_angles": 19, "n_points": 60}` |

## Demands — in-plane + out-of-plane

### `demands()`

```python
demands(*, Pu=0, Mu=0, Mu_out=0, Vu=0, Vu_out=0,
        delta_u=0, sigma_max=0, units=None) -> WallDemands
```

| Parameter | Default | Meaning (units) |
|---|---|---|
| `Pu` | `0.0` | Axial force, + compression (kN) |
| `Mu` / `Mu_out` | `0.0` | In-plane / out-of-plane moment (kN·m) |
| `Vu` / `Vu_out` | `0.0` | In-plane / out-of-plane shear (kN) |
| `delta_u` | `0.0` | Inelastic top displacement, §18.10.6.2 (mm) |
| `sigma_max` | `0.0` | Extreme compressive fiber stress, §18.10.6.3 (MPa) |
| `units` | `None` | Input units; defaults to `self.units` |

```python
d = wall.demands(Pu=3000, Mu=8000, Vu=1200, delta_u=60, sigma_max=18)
```

## The contract

### `run()`

```python
run(*, n_angles=None, n_points=None) -> Wall
```

Build the P-M-M surface and cache `Po`, `To`, `phi_Pn_max`, `phi_Vn`. Returns `self`.

### `capacity()`

```python
capacity(*, biaxial=False, n_angles=19, n_points=60) -> WallCapacity
```

Capacity snapshot. With `biaxial=True` also computes the out-of-plane diagram.

??? abstract "`WallCapacity` fields"
    | Field | Meaning (units) |
    |---|---|
    | `Po` / `To` / `phi_Pn_max` | axial caps (kN) |
    | `surface` / `diagram` / `diagram_out` | PMM surface + in/out-of-plane diagrams |
    | `phi_Vn` / `Vn_max` | reduced / governing max in-plane shear (kN) |
    | `Vn_max_individual` / `Vn_max_group_per_wall` | individual / common-lateral-force limit (kN) |
    | `rho_t_provided` / `rho_l_provided` | provided web ratios |
    | `rho_t_min` / `rho_l_min` | minimum web ratios |
    | `double_curtain_required` | two reinforcement layers needed |

### `check()`

```python
check(demands) -> WallCheck
```

Every gate: P-M (biaxial if `Mu_out≠0`), in-plane shear, distributed-rho minimums,
double-curtain feasibility, BE triggers (displacement §18.10.6.2 + stress
§18.10.6.3), tension-controlled (§21.2.2), and the neutral-axis depth `c` at the
demand. `WallCheck.summary(units=None)` prints one line per gate.

??? abstract "`WallCheck` fields"
    | Field | Meaning (units) |
    |---|---|
    | `ratio_pm` / `ratio_shear` / `ratio_shear_out` | demand/capacity ratios |
    | `be_required_disp` / `be_required_stress` | §18.10.6.2 / §18.10.6.3 triggers |
    | `be_length_required` / `be_thickness_min` | required BE geometry (mm) |
    | `distributed_rho_ok` / `double_curtain_ok` | web reinforcement gates |
    | `c_at_demand` / `a_at_demand` | neutral-axis depth / block depth (mm) |
    | `tension_controlled` / `eps_t_at_demand` / `phi_at_demand` | §21.2.2 state |
    | `passed` | all gates pass |
    | `rho_long` / `rho_trans` / `rho_total` | steel quantities (kg/m³) |

### `design()`

```python
design(demands=None, *, auto_update=True, max_iter=5,
       tolerance=25.0, clear_cover=25.0) -> WallDesignResults
```

Three-mode design with **iterative BE auto-update**: when a BE is required but
missing/too short, a new section is built and the design re-runs until the BE
length stabilizes within `tolerance` (mm), up to `max_iter` iterations.

| Parameter | Default | Meaning |
|---|---|---|
| `demands` | `None` | If `None`, only minimum mode runs |
| `auto_update` | `True` | Grow the BE automatically when required |
| `max_iter` | `5` | Max iterations before `converged=False` |
| `tolerance` | `25.0` | BE-length convergence tolerance (mm) |
| `clear_cover` | `25.0` | Cover for newly added BE bars (mm) |

??? abstract "`WallDesignProposal` fields (one per mode)"
    | Group | Fields |
    |---|---|
    | Mode | `mode` |
    | Web reinforcement | `rho_l_required`, `rho_t_required`, `rho_l_provided`, `rho_t_provided`, `double_curtain_required`, `web_bar_db`, `web_bar_spacing`, `web_bar_layers` |
    | BE decision | `be_required`, `be_required_disp`, `be_required_stress`, `be_length_proposed`, `be_thickness_proposed`, `be_extension_above`, `be_extension_below`, `c_used`, `a_used` |
    | BE confinement | `be_top_proposal`, `be_bot_proposal` (each a `ColumnDesignProposal`) |
    | Capacity shear | `Mpr_in_plane`, `Pu_for_Mpr`, `omega_v`, `Ve_capacity`, `Ve_used`, `amplification` |
    | Wall shear | `alpha_c_factor`, `phi_Vn`, `phi_Vn_max`, `av_s_required`, `ratio_Ve_over_phiVn` |
    | Context | `Pu_used`, `Pu_over_Ag_fc`, `proposed_section`, `notes` |

    `WallDesignResults` holds `minimum`, `demand` (or `None`), `capacity`,
    `envelope`, plus `original_wall`, `final_wall`, `history`, `iterations`,
    `converged`.

### `evolve()`, `run_optimize()`, `clear_cache()`

```python
evolve(proposal) -> Wall
run_optimize(demands=None, *, db_web_list=None, spacing_list=None,
             layers_options=(1, 2), include_infeasible=False, sort_by="rho_total")
clear_cache() -> None
```

`evolve` returns a **new** wall (updated section, web ratios, and BE column
detailing). `run_optimize` sweeps web `(db_web, spacing, layers)` ranked by steel
weight. `clear_cache` resets the surface, scalars, and BE columns.

## Capacity properties

| Property | Meaning (units) |
|---|---|
| `surface` | PMM `Surface` (lazy) |
| `Po` / `To` / `phi_Pn_max` | axial caps (kN) |
| `phi_Vn` / `Vn_max` | reduced / governing max in-plane shear (kN) |
| `Vn_max_individual` / `Vn_max_group_per_wall` | individual / common-lateral-force limit (kN) |
| `Mn` / `phi_Mn` / `Pn` / `phi_Pn` | field-view proxies |
| `has_boundary_elements` | whether BEs are present |
| `be_top` / `be_bot` | `Optional[Column]` — the BEs as real columns (lazy) |
| `plot` | bound `_WallPlotter` |

## Boundary elements *are* columns

```python
wall.be_top      # Column | None — top BE, with its own capacity/check/design/plot
wall.be_bot      # Column | None
```

### `wall_demands_to_be()`

```python
wall_demands_to_be(WallDemands) -> ColumnDemands
```

Canonical mapping (`design.common.contracts`), used internally to build BE column
demands:

```text
Pu      -> Pu
Mu      -> Mux   (in-plane bending = BE strong axis)
Mu_out  -> Muy   (out-of-plane = BE weak axis)
Vu      -> Vuy   (in-plane shear)
Vu_out  -> Vux   (out-of-plane shear)
```

## Capacity-design shear

In-plane shear is amplified by `omega_v` (§18.10.3.1.2): default `1.5`, dropping to
`1.0` for short walls. The wall reports `Ve_capacity`, `Ve_used`, `amplification`,
and the `alpha_c` factor (§18.10.4.1, interpolated from `hw/lw`).

### §18.10.4.4 limits and distributed reinforcement

`alpha_sh(*, acf, bcf, bw)` is 1.0 for a rectangular wall and otherwise
`0.7(1 + sqrt(Acf/(bcf·bw)))`, bounded to 1.0…1.2. The individual cap is
`αsh·0.83·√fc·Acv`; a line of walls sharing storey shear is also checked by
`vn_wall_group_max`, using `0.66·√fc·Acv` per vertical segment. `Vn_max` is
the governing individual/group per-wall limit.

For special walls, `rho_l = rho_t = 0.0025` unless the §18.10.2.1
low-shear exception applies. Otherwise `rho_min_distributed` uses Table
11.6.1 with `Vu`, `db`, and `fy`.

## Boundary-element & coupling helpers

The §18.10 provisions are exposed as pure functions:

??? abstract "`walls.boundary` (§18.10.6)"
    | Function | Returns |
    |---|---|
    | `boundary_element_required_displacement(*, c, lw, delta_u, hw)` | bool (§18.10.6.2) |
    | `boundary_element_required_stress(*, sigma_max_compressive, fc)` | bool, σ≥0.20fc (§18.10.6.3) |
    | `boundary_extension_length(*, c, lw)` | `max(c−0.1lw, c/2)` (mm) |
    | `be_thickness_minimum(*, lu, hw_over_lw, c, lw)` | min BE thickness (mm) |
    | `propose_be_geometry(*, c, lw, tw, hw, lu, delta_u, sigma_max_compressive, fc, min_length=300.0, min_thickness=None, extension_factor=1.0)` | `BEGeometryProposal` |
    | `c_at_demand(diagram, *, Pu, Mu)` | neutral-axis depth (mm) |
    | `a_at_demand(diagram, *, Pu, Mu, beta1)` | block depth a=β1·c (mm) |
    | `tension_controlled_at_demand(diagram, *, Pu, eps_ty=0.002)` | `(tc, eps_t, phi)` |
    | `ash_required_be(*, s, bc, Ag_be, Ach_be, fc, fyt, Pu_over_Ag_fc=0.0)` | Ash (mm²/mm) (§18.10.6.4(g)) |

??? abstract "`walls.coupling_beam` (§18.10.7) and `walls.wall_pier` (§18.10.8)"
    | Function / class | Returns |
    |---|---|
    | `coupling_beam_classification(*, ln, h, Vu, Acw, fc, lam=1.0)` | `CouplingBeamClassification` (conventional / diagonal / either) |
    | `vn_diagonal_coupling(*, Avd, fy, alpha_deg, fc, Acw, lam=1.0)` | `(Vn, Vn_cap)` (kN) |
    | `CouplingBeam(*, ln, h, b, concrete, steel)` | `.classify(Vu)`, `.diagonal_capacity(*, Avd, alpha_deg)` |
    | `classify_wall_pier(*, hw, lw, bw)` | `WallPierClassification` (column-like if `lw/bw ≤ 2.5`) |
    | `WallPier(*, hw, lw, bw)` | `.classify()` |

## Reporting

```python
wall.summary()
wall.print_design(results, modes=("demand","capacity","envelope"))
wall.print_final(results.envelope, demands=d)
wall.report()                 # dict in display units
wall.set_units("Tonf_m_C")
```

## Plotting

```python
wall.plot.section(show_bar_labels=False, bar_scale=None)
wall.plot.be_section(which="top")        # "top" | "bot"
wall.plot.pm(plane="in-plane")           # "in-plane" | "out-of-plane"
wall.plot.pm_compare()                   # in-plane vs out-of-plane
wall.plot.pmm_volume(n_Pu=25, wireframe=True)
wall.plot.pmm_slice(Pu=3000, demand=None)
wall.plot.dashboard(demands=d)
wall.plot.iteration_history(results)     # BE-length convergence
```

## Reference

Full [`design.walls` API](../api/walls.md).
