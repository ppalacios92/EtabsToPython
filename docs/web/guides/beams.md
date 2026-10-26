# Beams

[`design.beams.beam.Beam`](../api/beams.md) is a self-contained reinforced-concrete
beam: closed-form flexure, shear, and seismic capacity design per ACI 318-25 §18.6
(SMF). It implements the [element contract](design_overview.md) in full.

All geometry is **mm**, stress **MPa**, force **kN**, moment **kN·m**. The `units=`
argument is presentation only — it never reinterprets geometry.

## Building a beam

### `Beam.rectangular()`

The one-shot factory: builds the `RectangularSection`, lays out two bar layers, and
returns a ready-to-run beam.

```python
from design import Beam, Concrete, Steel

beam = Beam.rectangular(
    bw=300, h=600, cover=40, ln=6000,
    concrete=Concrete(fc=28), steel=Steel(fy=420),
    n_top=3, db_top=20, n_bot=2, db_bot=20,
    db_stirrup=10, n_legs=2, s_stirrup=100, seismic=True,
)
```

| Parameter | Type | Default | Meaning (units) |
|---|---|---|---|
| `bw` | float | — | Section width (mm) |
| `h` | float | — | Section height (mm) |
| `cover` | float | — | Clear cover to **outside edge of stirrup** (mm) |
| `concrete` | `Concrete` | — | Concrete material (fc in MPa) |
| `steel` | `Steel` | — | Primary longitudinal (and default transverse) steel |
| `n_top` / `db_top` | int / float | — | Count and diameter (mm) of the top layer |
| `n_bot` / `db_bot` | int / float | — | Count and diameter (mm) of the bottom layer |
| `db_stirrup` | float | `10.0` | Stirrup diameter (mm) |
| `n_legs` | int | `2` | Vertical stirrup legs ∥ Y (resist V_y); `Av = n_legs·π·db_stirrup²/4` |
| `s_stirrup` | float \| None | `None` | Stirrup spacing (mm); alias `s` |
| `ln` | float | — | Clear span (mm) |
| `Vg` | float | `0.0` | Gravity shear at face of support (kN) |
| `seismic` | bool | `True` | Enable §18.6 (SMF) rules |
| `steel_transverse` | `Steel` \| None | `None` | Transverse steel grade (defaults to primary) |
| `units` | int·str·UnitSystem \| None | `None` | Presentation units (ETABS code 1..16 or name) |
| `bar_schedule` | `BarSchedule` \| None | `None` | Diameter set for `design()` auto-pick |
| `label` | str | `"Beam"` | Identifier for reports |

Bar centroids sit at `cover + db_stirrup + db/2` from the corresponding face.

### `Beam()` — bare constructor

From a pre-built `Section` (internal units assumed):

```python
Beam(*, section, ln, cover=40.0, Av=0.0, s=None, seismic=True,
     db_long_min=16.0, db_stirrup=10.0, Vg=0.0,
     steel_transverse=None, units=None, bar_schedule=None, label="Beam")
```

| Parameter | Type | Default | Meaning (units) |
|---|---|---|---|
| `section` | `Section` | — | Geometry + materials + rebar layout |
| `ln` | float | — | Clear span (mm) |
| `cover` | float | `40.0` | Clear cover to stirrup outside edge (mm) |
| `Av` | float | `0.0` | Total stirrup area per leg group (mm²) |
| `s` | float \| None | `None` | Initial stirrup spacing (mm) |
| `seismic` | bool | `True` | Enable §18.6 rules |
| `db_long_min` | float | `16.0` | Min longitudinal bar diameter for `s_max` (mm) |
| `db_stirrup` | float | `10.0` | Stirrup diameter (mm) |
| `Vg` | float | `0.0` | Gravity shear at support face (kN) |
| `steel_transverse` | `Steel` \| None | `None` | Transverse steel (defaults to primary) |
| `units` / `bar_schedule` / `label` | — | — | As in the factory |

## Demands — three stations

Beams are the only element with **three stations**: `i` (left face), `mid`
(mid-span), `j` (right face), each with positive (bottom-tension) and negative
(top-tension) moments plus shear.

### `demands()`

```python
demands(*, Mu_pos_i=0, Mu_neg_i=0, Vu_i=0,
        Mu_pos_mid=0, Mu_neg_mid=0, Vu_mid=0,
        Mu_pos_j=0, Mu_neg_j=0, Vu_j=0,
        Vg_i=None, Vg_mid=None, Vg_j=None, units=None) -> BeamDemands
```

| Parameter | Default | Meaning (units) |
|---|---|---|
| `Mu_pos_i` / `Mu_neg_i` | `0.0` | Moment at left face, bottom / top tension (kN·m) |
| `Vu_i` | `0.0` | Shear at left face (kN) |
| `Mu_pos_mid` / `Mu_neg_mid` / `Vu_mid` | `0.0` | Same at mid-span (kN·m, kN) |
| `Mu_pos_j` / `Mu_neg_j` / `Vu_j` | `0.0` | Same at right face (kN·m, kN) |
| `Vg_i` / `Vg_mid` / `Vg_j` | `None` | Per-station gravity shear (kN); fall back to `beam.Vg` (ends) and 0 (mid) |
| `units` | `None` | Input units; defaults to `self.units` |

```python
d = beam.demands(Mu_neg_i=180, Mu_pos_mid=120, Mu_neg_j=170,
                 Vu_i=150, Vu_j=150, Vg_i=60, Vg_j=60)
```

??? abstract "`BeamDemands` fields"
    Property `Vu_max` returns the maximum absolute shear across stations.

    | Field | Default | Meaning (units) |
    |---|---|---|
    | `Mu_pos_i`, `Mu_neg_i`, `Vu_i` | `0.0` | Left face: +M, −M (kN·m), V (kN) |
    | `Mu_pos_mid`, `Mu_neg_mid`, `Vu_mid` | `0.0` | Mid-span |
    | `Mu_pos_j`, `Mu_neg_j`, `Vu_j` | `0.0` | Right face |
    | `Vg_i`, `Vg_mid`, `Vg_j` | `None` | Per-station gravity shear (kN) |

## The contract

```python
beam.run()                               # cache φMn±, φVn, Mpr±
cap     = beam.capacity()                # BeamCapacity
chk     = beam.check(d)                  # BeamCheck → PASS/FAIL
results = beam.design(d)                 # BeamDesignResults (4 modes)
beam    = beam.evolve(results.envelope)  # NEW immutable beam
```

### `run()`

```python
run() -> Beam
```

Lazily compute and cache `φMn_pos`, `φMn_neg`, `φVn`, `Mpr_pos`, `Mpr_neg`.
Idempotent; returns `self` for chaining.

### `capacity()`

```python
capacity() -> BeamCapacity
```

Capacity of the as-built reinforcement from closed-form flexure and shear.

??? abstract "`BeamCapacity` fields"
    | Field | Default | Meaning (units) |
    |---|---|---|
    | `phi_Mn_pos` / `phi_Mn_neg` | — | Design moment, bottom / top tension (kN·m) |
    | `phi_Vn` | — | Design shear strength (kN) |
    | `Mpr_pos` / `Mpr_neg` | `None` | Probable moments §18.6.5.1 (kN·m) |
    | `rho` / `rho_p` | `0.0` | Tension / compression ratio under +M |
    | `rho_min` | `0.0` | Minimum ratio §9.6.1.2 |
    | `rho_max` | `0.0` | `0.5·rho_balanced` (ductility cap) |
    | `rho_balanced` | `0.0` | Balanced ratio §22.2 |
    | `rho_max_aci_smf` | `0.0` | Legacy §18.6.3.1 cap (reporting only) |
    | `tension_controlled` | `True` | `eps_t ≥ eps_ty + 0.003` (§21.2.2) |

### `check()`

```python
check(demands) -> BeamCheck
```

Demand ratios at every station and direction, plus rho limits,
tension-controlled, and SMF continuity (§18.6.3) flags.

??? abstract "`BeamCheck` fields"
    | Field | Meaning |
    |---|---|
    | `ratio_M_pos_i` … `ratio_M_neg_j` | Per-station moment ratios (6 total) |
    | `ratio_M_overall` | Max moment ratio across stations/directions |
    | `ratio_V_i` / `ratio_V_mid` / `ratio_V_j` | Per-station shear ratios |
    | `ratio_V_overall` | Max shear ratio |
    | `rho_ok` | `rho_min ≤ max(rho, rho_p) ≤ rho_max` |
    | `tc_ok` | Tension-controlled |
    | `smf_continuity_ok` | §18.6.3 continuity (seismic) |
    | `passed` | All checks pass |
    | `rho_long` / `rho_trans` / `rho_total` | Steel quantities (kg/m³) |

### `design()`

```python
design(demands=None, *, db_top=None, n_top=None,
       db_bot=None, n_bot=None,
       db_stirrup=None, n_legs=None) -> BeamDesignResults
```

Runs **minimum → demand → capacity → envelope**. Optional overrides fix bar
sizing; if all `None`, the auto-picker chooses from `bar_schedule`.

| Parameter | Default | Meaning |
|---|---|---|
| `demands` | `None` | If `None`, only the minimum mode runs |
| `db_top` / `n_top` / `db_bot` / `n_bot` | `None` | Force longitudinal sizing |
| `db_stirrup` / `n_legs` | `None` | Force stirrup sizing |

??? abstract "`BeamDesignProposal` fields (one per mode)"
    | Field | Meaning (units) |
    |---|---|
    | `mode` | `'minimum'`·`'demand'`·`'capacity'`·`'envelope'`·`'provided'` |
    | `As_top_{i,mid,j}_required` / `As_bot_{i,mid,j}_required` | Required areas per station (mm²) |
    | `As_top_continuous` / `As_bot_continuous` | Continuous areas provided (mm²) |
    | `db_top` / `n_top` / `db_bot` / `n_bot` | Longitudinal detailing |
    | `db_stirrup` / `n_legs` / `av_provided` | Stirrup detailing (mm, –, mm²) |
    | `spacing_confined` / `spacing_middle` / `lo` | §18.6.4 spacings & confined length (mm) |
    | `Vu_{i,mid,j}_used` | Design shear per station (kN) |
    | `av_s_required` | Required Av/s (mm²/mm) |
    | `vc_zero_active` | §18.6.5.2 Vc=0 active in `lo` |
    | `Mpr_pos` / `Mpr_neg` | Probable moments (kN·m) |
    | `Ve_capacity` / `phi_Vn` / `ratio_Ve_over_phiVn` | Capacity-design shear gate |
    | `ratio_pos_over_neg_at_{i,j}` | §18.6.3.3 bottom/top ratio at faces |
    | `ratio_min_over_max_anywhere` | §18.6.3.4 (≥0.25) |
    | `two_bars_top_ok` / `two_bars_bot_ok` | §18.6.3.2 |
    | `notes` | Design notes |

    `BeamDesignResults` holds `minimum`, `demand` (or `None`), `capacity`,
    `envelope`; it is iterable and has `as_dict()`.

### `evolve()` and `run_optimize()`

```python
evolve(proposal) -> Beam
run_optimize(demands=None, *, db_long_list=None, db_stirrup_list=None,
             n_legs_list=None, include_provided=True,
             include_infeasible=False, sort_by="rho_total")
```

`evolve` returns a **new** beam with the proposal applied (immutable; `self`
unchanged). `run_optimize` enumerates `(db_top, db_bot, db_stirrup, n_legs)`,
runs `design()` for each, and ranks the alternatives by steel weight (kg/m³).

## Capacity properties

| Property | Meaning (units) |
|---|---|
| `phi_Mn_pos` / `phi_Mn_neg` | Design moment, bottom / top tension (kN·m) |
| `phi_Mn` | `{'pos': …, 'neg': …}` (kN·m) |
| `Mn` | nominal moments `{'pos': …, 'neg': …}` (kN·m) |
| `phi_Vn` | Design shear φ(Vc + Vs) (kN) |
| `steel` / `fyt` | Primary longitudinal steel / transverse fy (MPa) |
| `ln_m` | Clear span in metres (view of canonical mm `ln`) |
| `pm` | `BeamPMCurve` — bidirectional P-M curve (inspection only) |
| `plot` | bound `_BeamPlotter` |

## Reporting

```python
beam.summary()                  # section, materials, rebar, capacities, quantities
beam.print_state(d)             # current (as-built) config + demand ratios
beam.print_checks(d)            # all ACI checks
beam.print_design(results, d)   # minimum / demand / capacity / envelope side-by-side
beam.print_final(results.envelope, d)   # one consolidated final design block
beam.report()                   # dict (in display units)
beam.set_units("Tonf_m_C")      # change presentation units
```

## Seismic specifics

- **Probable moments** `Mpr±` use `fy_pr = 1.25·fy`, `φ = 1.0` (§18.6.5.1).
- **Capacity-design shear** `Ve` from the end-moment couple over both sway senses:
  `Ve = max[(Mpr_neg_i + Mpr_pos_j)/ln, (Mpr_pos_i + Mpr_neg_j)/ln] + Vg`
  (`v_seismic_from_mpr`, `ve_capacity_design`).
- **Vc = 0** in `lo` when `V_seismic ≥ 0.5·V_total` **and** `Pu < Ag·fc/20`
  (§18.6.5.2), via `design.common.shear.vc_zero_seismic`.

### Flexure & seismic helper functions

These pure functions back the beam and are importable directly:

| Function | Returns (units) |
|---|---|
| `flexure.rho_min_beam(fc, fy)` | `max(0.25√fc/fy, 1.4/fy)` (§9.6.1.2) |
| `flexure.rho_balanced(fc, fy, *, beta1_val=None, Es=200000, eps_cu=0.003)` | balanced ratio (§22.2) |
| `flexure.rho_max_beam(fc, fy)` | `0.5·rho_balanced` |
| `flexure.rho_max_seismic(fy)` | legacy §18.6.3.1 cap (0.025/0.020) |
| `flexure.mn_singly(*, b, d, As, concrete, steel)` | Mn singly reinforced (kN·m) |
| `flexure.mn_doubly(*, b, d, d_prime, As, As_prime, concrete, steel)` | Mn doubly reinforced (kN·m) |
| `flexure.as_required_singly(*, Mu, b, d, concrete, steel, phi=0.9)` | required As (mm²) |
| `flexure.is_tension_controlled(*, b, d, As, concrete, steel)` | bool (§21.2.2) |
| `seismic.mpr(*, b, d, As, concrete, steel)` | probable moment, one sense (kN·m) |
| `seismic.v_seismic_from_mpr(*, Mpr_pos_i, Mpr_neg_i, Mpr_pos_j, Mpr_neg_j, ln)` | seismic shear Ve (kN) |
| `seismic.ve_capacity_design(*, Mpr_pos_i, Mpr_neg_i, Mpr_pos_j, Mpr_neg_j, ln, Vg=0.0)` | design shear Ve + Vg (kN) |

## Plotting

```python
beam.plot.section(show_bar_labels=False, bar_scale=None)
beam.plot.moment_envelope(d)     # Mu± per station vs φMn±
beam.plot.shear_envelope(d)      # Vu per station vs φVn / Ve
beam.plot.pm(direction="both")   # inspection-grade P-M curve ('pos'|'neg'|'both')
beam.plot.dashboard(demands=d)   # 2×2: section, capacity, moment, shear
```

The inspection P-M curve (`beam.pm`) interpolates `phi_Mn_at(Nu=…, direction=…)`,
`Mn_at(...)`, `peak(...)`, and `check(Nu=…, Mu_pos=…, Mu_neg=…)`.

## Reference

Full [`design.beams` API](../api/beams.md) — `Beam`, the dataclasses, `optimize`,
`flexure`, `seismic`, `interaction`, `probable`.
