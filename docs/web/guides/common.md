# The common layer

`design.common` is the shared foundation under every element: the **contract**, the
**unit system**, the **ACI factors**, and the **centralized code formulas**.
Elements *consume* this layer — per `design/AGENTS.md` they never re-implement a
formula that lives here.

## Contracts

[`design.common.contracts`](../api/common.md) defines the runtime-checkable
`Element` protocol and the marker protocols for the exchanged data objects:

| Protocol | Required surface |
|---|---|
| `Element` | `label`; methods `run`, `capacity`, `check`, `design`, `evolve`, `apply`, `set_units`, `summary`, `report` |
| `Demands` | marker (no required attrs) |
| `Capacity` | `phi_Vn: float` |
| `Check` | `passed: bool` |
| `DesignProposal` | `mode: str`, `notes: tuple` |
| `DesignResults` | `minimum`, `capacity`, `envelope` |

### `wall_demands_to_be()`

```python
wall_demands_to_be(demands: WallDemands) -> ColumnDemands
```

The canonical wall→boundary-element demand mapping (Pu→Pu, Mu→Mux, Mu_out→Muy,
Vu→Vuy, Vu_out→Vux).

## Units

[`design.common.units`](../api/common.md) implements the **presentation** unit
system. Golden rule: geometry is always mm; only loads convert.

### `UnitSystem`

```python
UnitSystem(code, force, length, temperature)
```

| Field | Meaning |
|---|---|
| `code` | ETABS unit code (1..16) |
| `force` | N · kN · kgf · Tonf · lb · kip |
| `length` | mm · cm · m · in · ft |
| `temperature` | F · C (stored, unused in math) |

**Factors** (internal → display): `force_factor`, `length_factor`, `moment_factor`,
`area_factor`, `stress_factor`. **Labels**: `name`, `length_label`, `area_label`
(always `mm^2`), `force_label`, `moment_label` (e.g. `kN.m`), `stress_label`
(e.g. `N/mm^2`). **Formatters** (return ASCII): `fmt_length`, `fmt_area`,
`fmt_force`, `fmt_moment`, `fmt_stress` — each `(value, *, unit=True)`.
**To-internal**: `to_internal_force`, `to_internal_length`, `to_internal_moment`,
`to_internal_stress`.

### `units_from()` and `to_internal()`

```python
units_from(code_or_name: int | str) -> UnitSystem          # 12 or "Tonf_m_C"
to_internal(kind, value, *, units) -> float                # kind ∈ force/length/moment/stress
```

`to_internal`'s `units` keyword is **required** — there is no global state.
`DEFAULT_UNITS = units_from(9)` (`N_mm_C`), where the conversion is the identity.

??? abstract "ETABS code map"
    | code | system | code | system |
    |---|---|---|---|
    | 1 | lb_in_F | 9 | N_mm_C |
    | 2 | lb_ft_F | 10 | N_m_C |
    | 3 | kip_in_F | 11 | Tonf_mm_C |
    | 4 | kip_ft_F | 12 | Tonf_m_C |
    | 5 | kN_mm_C | 13 | kN_cm_C |
    | 6 | kN_m_C | 14 | kgf_cm_C |
    | 7 | kgf_mm_C | 15 | N_cm_C |
    | 8 | kgf_m_C | 16 | Tonf_cm_C |

## ACI factors

[`design.common.factors`](../api/common.md), with citations:

| Symbol | Function / constant | §ACI |
|---|---|---|
| β₁ | `beta1(fc)` → 0.85…0.65 | Table 22.2.2.4.3 |
| λ | `lambda_concrete(lightweight, sand_lightweight)` → 1.0/0.85/0.75 | §19.2.4 |
| φ (axial-flexure) | `phi_axial_flexure(eps_t, eps_ty, *, spiral=False)` | §21.2.2 |
| φ (shear) | `phi_shear()` → 0.75 | §21.2.1 |
| φ (bearing) | `phi_bearing()` → 0.65 | — |
| φ (strut-tie) | `phi_compression_strut_tie()` → 0.75 | — |
| Ωᵥ (wall) | `omega_v_wall(*, n_stories=None, hw_over_lw=None)` → 1.5 (1.0 short) | §18.10.3.1.2 |
| — | `OMEGA_0_COLUMN_DEFAULT = 3.0` | §18.7.6.1.1 |
| — | `OMEGA_V_WALL_DEFAULT = 1.5` | §18.10.3.1.2 |

`phi_axial_flexure` linearly interpolates: φ_compression (0.65 ties / 0.75 spiral)
for `eps_t ≤ eps_ty`, 0.90 for `eps_t ≥ eps_ty + 0.003`, interpolated between.

## Centralized formulas

Every shear / spacing / strength formula lives in one place; element files are
forbidden from inlining them (`design/AGENTS.md` §5). All return **kN** unless
noted; lengths in **mm**, stress **MPa**.

### Shear — [`design.common.shear`](../api/common.md)

| Function | §ACI | Notes |
|---|---|---|
| `vc_simplified(*, b, d, concrete)` | §22.5.5.1 | `0.17·λ·√fc·b·d` |
| `lambda_s(d)` | §22.5.5.1.3 / §22.6.5.3 | `min(sqrt(2/(1+d/254)), 1.0)` |
| `vc_one_way(*, b, d, rho_w, concrete, av_ge_av_min, Nu=0, Ag=0)` | Table 22.5.5.1 | lesser tabulated expression with Av≥Av,min; size-effect expression otherwise; cap `0.42λ√fc·b·d` |
| `vc_detailed(*, b, d, Nu, Ag, rho_w, concrete)` | §22.5.5.1 | with axial `Nu` (kN) + `rho_w` |
| `vs_capacity(*, Av, fyt, d, s)` | §22.5.10.5.3 | `Av·fyt·d/s` |
| `vs_max(*, b, d, concrete)` | §22.5.1.2 | web-crushing cap `0.66√fc·b·d` |
| `av_s_required(*, Vu, Vc, fyt, d, phi=None)` | §22.5 | Av/s (mm²/mm); `None` if Vu≤0; `phi` defaults `phi_shear()` |
| `av_s_minimum(*, b, concrete, fyt)` | §9.6.3.4 | minimum Av/s (mm²/mm) |
| `vc_zero_seismic(*, V_seismic, V_total, Pu, Ag, fc)` | §18.6.5.2 / §18.7.6.2.1 | bool: `V_seismic≥0.5V_total` **and** `Pu<Ag·fc/20` |
| `alpha_c(hw_over_lw)` | Table 18.10.4.1 | 0.25 (≤1.5) → 0.17 (≥2.0) |
| `vn_wall(*, Acv, fc, rho_t, fyt, hw_over_lw, lam=1.0)` | §18.10.4.1 | `(α_c·λ·√fc + rho_t·fyt)·Acv` |
| `alpha_sh(*, acf=0, bcf=None, bw)` | §18.10.4.4 | `0.7(1+sqrt(Acf/(bcf·bw)))`, bounded 1.0…1.2 |
| `vn_wall_max(*, Acv, fc, alpha_sh=1.0)` | §18.10.4.4 | individual cap `αsh·0.83·√fc·Acv` |
| `vn_wall_group_max(*, acv_list, fc, alpha_sh_list=None)` | §18.10.4.4 | common-lateral-force group cap `Σ(αsh·0.66·√fc·Acv)` |
| `ve_in_plane_capacity_wall(*, Mpr, Mu=None, Vu=None, omega_v_factor=1.5)` | §18.10.3.1 | capacity-design Ve |

### Two-way shear

`bo_rectangular(*, c1, c2, d, position='interior')` returns the §22.6.4
critical perimeter at `d/2` from the supported-area faces. For edge and corner
supports, `c1` is perpendicular to the free edge. `vc_two_way(...)` returns the
least §22.6.5.3 stress expression in MPa:

```text
min(0.33, 0.17(1 + 2/β), 0.083(2 + αs·d/bo)) · λs·λ·√fc
```

where `αs` is 40 / 30 / 20 for interior / edge / corner supports.
`vc_two_way_force(...)` returns `vc·bo·d` in kN. With stirrups,
`vc_two_way_max_with_stirrups(*, d, concrete)` returns the Table 22.6.6.1
concrete term `0.17λsλ√fc`; evaluate the steel term separately as
`vs = Av·fyt/(bo·s)`.

### Confinement & spacing — [`design.common.spacing`](../api/common.md)

| Function | §ACI | Returns (mm) |
|---|---|---|
| `lo_length_beam(*, h_member)` | §18.6.4.1 | `2·h_member` |
| `lo_length_column(*, h_member, lu_clear_mm)` | §18.7.5.1 | `max(h_member, lu/6, 450)` |
| `s_max_seismic_smf_beam(*, d, db_long_min, db_hoop, grade=60)` | §18.6.4.4 | `min(d/4, (6 or 5)·db, 150)` |
| `s_max_seismic_outside_beam(*, d)` | §18.6.4.6 | `d/2` |
| `s_max_confined_column(*, h_min, db_long_min, hx, grade=60)` | §18.7.5.3 | `min(h_min/4, (6 or 5)·db, so)` |
| `s_max_middle_zone_column(*, db_long_min, grade=60)` | §18.7.5.5 | `min((6 or 5)·db, 150)` |
| `hx_check(*, b_core, n_legs, limit=350.0)` | §18.7.5.2 | `(hx, ok)` |
| `ash_required(*, s, bc, Ag, Ach, fc, fyt, Pu_over_Ag_fc=0.0)` | §18.7.5.4 | Ash (mm²) — max of the table eqs |

### Geometry kernel — [`design.common.geometry`](../api/common.md)

`polygon_area(poly)`, `polygon_centroid(poly)`, `clip_above_y(poly, y_cut)`
(Sutherland-Hodgman), `rectangle(b, h, *, center=(0,0))` — the polygon math behind
section analysis.

## Shared infrastructure

| Module / type | Provides |
|---|---|
| `element_base.ElementBase` | the contract parts shared by all elements (`apply` = alias of `evolve`; `set_units`) |
| `bar_schedule.BarSchedule` | `longitudinal=[12,16,20,22,25,28,32]`, `hoops=[10,12,16]` — diameter sets for `design()` |
| `plot_style.PlotStyle` | one shared plot-style object (colours, line widths, `figsize_*`); reached via `element.style` |
| `probable` | `probable_interaction_diagram(...)`, `mpr_envelope(...)` (Mpr, §18.6.5 / §18.7.6) |
| `optimize.OptimizeAlternative` | base record for `run_optimize` (proposal, detailing, rho_*, feasible, flags, notes) + `format_optimize_table(...)` |
| `codes.ACI` | a small annotation type: `ACI(section, equation=None, description="")` |

## Reference

Full [`design.common` API](../api/common.md).
