# `design.beams` — Reinforced concrete beam module

Self-contained engineering of concrete beams per **ACI 318-25**. Beams
are flexural-shear elements: this module computes `phi*Mn+`, `phi*Mn-`,
`phi*Vn`, `Mpr`, applies the SMF detailing rules of §18.6, and proposes
longitudinal + transverse reinforcement in three independent modes
(minimum / demand / capacity) plus a pointwise envelope.

This module is **pure engineering**. It does not import anything from
ETABS, HDF5 readers, or COM. The same `Beam` object works whether you
build it manually or hand it a `Section` adapted from an external model.

**Beams have no P-M interaction diagram.** ACI §9 and §18.6 treat them
as flexural elements; M-N coupling lives in Cap. 10 (columns) and §22.4.
`Mn` is computed in closed form by `mn_doubly` and `Mpr` by `mpr` per
§18.6.5.1 (both in MPa, mm, kN·m).

---

## 1. Quick start

```python
from design import Concrete, Steel
from design.beams import Beam, BeamDemands

concrete = Concrete(fc=28)
steel    = Steel(fy=420, grade=60)

# One-shot construction — all detailing in a single call
beam = Beam.rectangular(
    bw=400, h=500, cover=40,
    concrete=concrete, steel=steel,
    n_top=5, db_top=22,
    n_bot=5, db_bot=22,
    db_stirrup=10, n_legs=3, s_stirrup=100,
    ln=5.0, seismic=True, Vg=80.0,
    units=6,                          # kN_m_C
    label='B-1A',
)

# Demands at three stations: i (left face), mid, j (right face).
# Each carries Mu_pos, Mu_neg, Vu. Optional Vg_i/Vg_mid/Vg_j.
demands = BeamDemands(
    Mu_pos_i=80,    Mu_neg_i=200,   Vu_i=150,
    Mu_pos_mid=300, Mu_neg_mid=80,  Vu_mid=80,
    Mu_pos_j=80,    Mu_neg_j=200,   Vu_j=150,
)

# 1) Does the provided design meet code?
beam.print_checks(demands)

# 2) Run the three design modes
results = beam.design(demands)
beam.print_design(results, demands=demands)   # shows PROVIDED + 4 modes

# 3) Adopt any mode (minimum / demand / capacity / envelope)
beam.apply(results.envelope)
beam.print_checks(demands)                    # verify the adopted design
```

---

## 2. File layout

```
design/beams/
├── README.md                ← this document
├── __init__.py              ← public re-exports
├── beam.py                  ← Beam class + dataclasses
├── flexure.py               ← mn_singly / mn_doubly / rho_min / rho_balanced / rho_max
├── shear.py                 ← Vc / Vs / Vn / av_s / s_max_seismic_*
├── seismic.py               ← mpr / ve_capacity_design / v_seismic_from_mpr
├── plotting.py              ← _BeamPlotter — bound to one beam
├── style.py                 ← BeamPlotStyle
└── optimize.py              ← run_optimize + format_optimize_table + quantities
```

Plus shared collaborators:

```
design/common/
├── materials.py             ← Concrete, Steel, Bar (frozen dataclasses)
├── bar_schedule.py          ← BarSchedule (preferred long / hoop diameters)
├── factors.py               ← β1, phi(εt), λ
└── units.py                 ← UnitSystem with 16 ETABS codes

design/sections/
├── base.py                  ← Section ABC
├── reinforcement.py         ← Rebar / RebarGroup / RebarLayout + builders
└── rectangular.py           ← RectangularSection
```

---

## 3. The `Beam` class — public surface

### 3.1 Construction — two paths

#### 3.1a `Beam.rectangular(...)` — one-shot factory (recommended)

```python
Beam.rectangular(
    *, bw: float, h: float, cover: float,
       concrete: Concrete, steel: Steel,
       # Longitudinal (two layers, top + bottom)
       n_top: int, db_top: float,
       n_bot: int, db_bot: float,
       # Stirrups
       db_stirrup: float = 10.0,
       n_legs: int = 2,                  # vertical legs (parallel to Y) → resist V_y
       s_stirrup: float | None = None,
       # Member
       ln: float,
       Vg: float = 0.0,                  # gravity shear at supports (kN)
       seismic: bool = True,
       steel_transverse: Steel | None = None,
       units: int | str | UnitSystem | None = None,
       bar_schedule: BarSchedule | None = None,
       label: str = "Beam",
)
```

`cover` is the **clear cover** to the outside edge of the stirrup,
applied symmetrically to top and bottom. Bar centroids sit at
`cover + db_stirrup + db/2` from the corresponding face — top and bottom
layers can have different diameters without distortion.

`Av` is computed internally as `n_legs · π·db_stirrup²/4` — you don't pass
it directly. There is **no `fyt` scalar** — the transverse steel grade
comes from `steel_transverse` (defaults to the primary steel).

#### 3.1b `Beam(section=..., ...)` — low-level path

Use this when you already have a custom `Section` + `RebarLayout` (e.g.
adapted from ETABS). Accepts the same operational parameters as
`rectangular()` but takes the rebar layout directly through the section.

### 3.2 Capacity / Check / Design

```python
beam.capacity()
        # → BeamCapacity (phi_Mn_pos, phi_Mn_neg, phi_Vn,
        #                 Mn_pr_pos, Mn_pr_neg,
        #                 rho, rho_p, rho_min, rho_max, rho_balanced,
        #                 rho_max_aci_smf, tension_controlled)

beam.check(demands)
        # → BeamCheck (ratios per station, V per station, SMF continuity,
        #              rho_ok, tc_ok, smf_continuity_ok, passed)

beam.design(demands=None, *, db_top=None, n_top=None,
            db_bot=None, n_bot=None,
            db_stirrup=None, n_legs=None)
        # → BeamDesignResults (minimum / demand / capacity / envelope)
```

The `db_*` and `n_*` overrides in `design()` are the way to force a
specific detailing instead of letting the picker choose.

### 3.3 `apply(proposal) -> Beam`

Adopts the detailing described by a `BeamDesignProposal` into the beam
**inplace**. Accepts any of the four modes returned by `design()`
(`minimum`, `demand`, `capacity`, `envelope`) or a snapshot from
`current_proposal()`.

- `beam.section.rebar` ← new two-layer layout from `db_top/n_top/db_bot/n_bot`
- `beam.db_stirrup` ← `proposal.db_stirrup`
- `beam.s` ← `proposal.spacing_confined`
- `beam.Av` ← `n_legs · π·db_stirrup²/4`

### 3.4 `current_proposal(demands=None) -> BeamDesignProposal`

Snapshot of the beam's **current** detailing as a proposal. Two use cases:

1. **Snapshot/revert** — save the as-input state before exploring:
   ```python
   snapshot = beam.current_proposal(demands)
   beam.apply(results.envelope)        # adopt envelope
   # ... evaluate, decide it's not what we want ...
   beam.apply(snapshot)                 # revert to the original
   ```

2. **Audit** — compare the as-built design against ACI; this is what
   `print_state()` shows as the `--- PROVIDED ---` block.

### 3.5 Forcing diameters at design time

`beam.design()` accepts overrides for `db_top`, `db_bot`, `db_stirrup`,
`n_legs`. The picker fixes those and computes everything else:

```python
# Fix only diameters; picker computes the bar counts and n_legs at s_max
results = beam.design(demands,
                      db_top=20, db_bot=20, db_stirrup=10)

# Fix db AND n_legs; the spacing adapts to the demand:
#   s = min(s_max_§18.6.4.4, av_provided / av_s_required)
results = beam.design(demands,
                      db_top=20, db_bot=20,
                      db_stirrup=10, n_legs=2)
```

When the adaptive spacing falls below the §18.6.4.4 cap, the proposal
adds a note explaining why.

### 3.6 Units, reports, plotting

```python
beam.set_units(6)               # by ETABS code  (kN_m_C)
beam.set_units('Tonf_m_C')      # by name
beam.units.force_factor         # multiply kN to get target force
beam.units.moment_factor        # multiply kN·m to get target moment

beam.summary()                  # geometry + materials + capacities
beam.print_checks(demands)      # full ACI check list
beam.print_state(demands)       # the PROVIDED block (in print_design format)
beam.print_design(results, demands=demands)   # PROVIDED + 4 modes
beam.report()                   # → dict
print(beam)                     # one-liner

beam.plot.section()             # section + bars
beam.plot.moment_envelope(demands)   # Mu+/Mu- per station vs phi*Mn
beam.plot.shear_envelope(demands)    # Vu_i / Vu_mid / Vu_j vs phi*Vn / Ve
beam.plot.demand(demands)            # alias for moment_envelope
beam.plot.dashboard(demands=demands) # 2x2 grid
```

The 16 ETABS unit codes are all supported. There is **no PM curve plot**
— beams don't have one.

---

## 4. Demands — three stations × (M+, M-, V) + per-station Vg

```python
@dataclass(frozen=True, slots=True)
class BeamDemands:
    Mu_pos_i:   float = 0.0
    Mu_neg_i:   float = 0.0
    Vu_i:       float = 0.0
    Mu_pos_mid: float = 0.0
    Mu_neg_mid: float = 0.0
    Vu_mid:     float = 0.0
    Mu_pos_j:   float = 0.0
    Mu_neg_j:   float = 0.0
    Vu_j:       float = 0.0
    Vg_i:       Optional[float] = None    # gravity shear at left support
    Vg_mid:     Optional[float] = None    # gravity shear at midspan
    Vg_j:       Optional[float] = None    # gravity shear at right support

    @property
    def Vu_max(self) -> float: ...   # max(|Vu_i|, |Vu_mid|, |Vu_j|)
```

ETABS naturally emits this: `FrameForces` has a `Station` column (0%,
50%, 100% of the frame length); each row also carries M2/M3 and V2/V3
per combo. The adapter collapses combos to (Mu+, Mu-, Vu) per station.

**Per-station `Vg`** lets you express asymmetric gravity loading
(point loads, non-uniform distributed loads, cantilever). If left
`None`, the beam's scalar `Vg` is used at both supports and `0` at
midspan — the uniformly-loaded-beam assumption.

There is **no `Nu`** field. Beams do not see significant axial.

---

## 5. Three design modes

`beam.design(demands=None, ...)` always runs three modes and returns a
`BeamDesignResults` container with four `BeamDesignProposal` objects:

```python
results = beam.design(demands)
results.minimum    # ACI minimum (§9.6.1.2 + §9.6.3.4 + §18.6.4). No demands used.
results.demand     # As required from Mu_*_X per station; av_s from max(|Vu|).
results.capacity   # Mpr-based Ve per station with V_seismic + Vg_X.
                   # Vc=0 in lo if §18.6.5.2 (a)+(b) hold.
results.envelope   # Pointwise worst of the three.
```

### 5.1 Minimum
- `rho_min` per §9.6.1.2 (or §18.6.3.1 if seismic)
- `av_s_minimum` per §9.6.3.4
- `s_max` per §18.6.4.4 (confined) and §18.6.4.6 (middle) when seismic
- No Mu / Vu used.

### 5.2 Demand
- `As_top_X_required = as_required_singly(Mu_neg_X)` for X ∈ {i, mid, j}.
- `As_bot_X_required = as_required_singly(Mu_pos_X)` for X ∈ {i, mid, j}.
- `Vu_design = max(|Vu_i|, |Vu_mid|, |Vu_j|)`.
- `av_s_required = (Vu_design/φ − Vc) / (fyt · d)`.

### 5.3 Capacity (§18.6.5)

Mpr is computed in both senses (positive and negative bending):
```
Mpr_pos = As_bot · 1.25·fy · (d_pos − a_pos/2)
Mpr_neg = As_top · 1.25·fy · (d_neg − a_neg/2)
```

The seismic-induced shear from the end-moment couple is **constant**
along the beam:
```
V_seismic = max( (Mpr_neg_i + Mpr_pos_j) / ln,    ← sway right
                 (Mpr_pos_i + Mpr_neg_j) / ln )   ← sway left
```

(For continuous reinforcement, both directions give the same magnitude.)

The design shear at each station adds the local gravity shear:
```
Ve_i   = V_seismic + |Vg_i|
Ve_mid = V_seismic + |Vg_mid|
Ve_j   = V_seismic + |Vg_j|
Ve_capacity = max(Ve_i, Ve_j)         ← governs lo stirrup design
```

**§18.6.5.2 — `Vc = 0` in lo** when BOTH:
- **(a)** earthquake-induced shear ≥ ½ total Ve in lo
  → equivalent to `V_seismic ≥ 0.5 · Ve_capacity`
- **(b)** Pu < Ag·fc/20 — always true for beams (Pu ≈ 0)

The proposal note reports the seismic-share percentage and whether
Vc=0 is applied.

**No Ωo·Vu cap** — that's a §18.7.6.1.1 column rule.

### 5.4 §18.6.3 — SMF continuity

The proposal enforces three continuity rules on the continuous
reinforcement (constant along the beam):

- **§18.6.3.2**: `n_top ≥ 2`, `n_bot ≥ 2`.
- **§18.6.3.3**: `As_bot_continuous ≥ 0.5 · As_top_continuous`.
- **§18.6.3.4**: `As anywhere ≥ ¼ · max(As at face of either joint)` —
  raises the As required at mid-span when the joint face is the
  controlling station.

The proposal reports the realized ratios:

```python
proposal.ratio_pos_over_neg_at_i        # ≥ 0.5 expected
proposal.ratio_pos_over_neg_at_j        # ≥ 0.5
proposal.ratio_min_over_max_anywhere    # ≥ 0.25
proposal.two_bars_top_ok                # True
proposal.two_bars_bot_ok                # True
```

### 5.5 `BeamDesignProposal` — every field

```python
mode                          str   'minimum' | 'demand' | 'capacity' | 'envelope' | 'provided'

# As required per station (3 stations × 2 senses)
As_top_i_required             float  mm²
As_top_mid_required           float  mm²
As_top_j_required             float  mm²
As_bot_i_required             float  mm²
As_bot_mid_required           float  mm²
As_bot_j_required             float  mm²

# Chosen continuous reinforcement
As_top_continuous             float  mm²
As_bot_continuous             float  mm²
db_top                        float  mm
n_top                         int
db_bot                        float  mm
n_bot                         int

# Transverse
db_stirrup                    float  mm
n_legs                        int           ← vertical legs (parallel to Y)
av_provided                   float  mm²
spacing_confined              float  mm     ← may be < §18.6.4.4 cap if adaptive
spacing_middle                float  mm     §18.6.4.6
lo                            float  mm     = 2 h  (§18.6.4.1)

# Shear demand per station + §18.6.5.2 flag
Vu_i_used                     float|None  kN
Vu_mid_used                   float|None  kN
Vu_j_used                     float|None  kN
av_s_required                 float|None  mm²/mm
vc_zero_active                bool

# Capacity-design quantities
Mpr_pos                       float|None  kN·m
Mpr_neg                       float|None  kN·m
Ve_capacity                   float|None  kN
phi_Vn                        float|None  kN  (proposed stirrups)
ratio_Ve_over_phiVn           float|None

# §18.6.3 SMF continuity ratios
ratio_pos_over_neg_at_i       float|None  ≥ 0.5
ratio_pos_over_neg_at_j       float|None  ≥ 0.5
ratio_min_over_max_anywhere   float|None  ≥ 0.25
two_bars_top_ok               bool
two_bars_bot_ok               bool

notes                         tuple[str]
```

---

## 6. The picker

### 6.1 Longitudinal — `_pick_longitudinal`

For each diameter `db` in `beam.bar_schedule.longitudinal` (ascending):

1. `a = π·db²/4`.
2. `n_required = max(2, ceil(As_required / a))`.
3. Cap `n_required` by `_max_bars_in_layer(db, db_stirrup)`:
   ```
   span  = bw − 2·(cover + db_stirrup) − db
   clear = max(db, 25 mm, 1.33·ag_max)   (ACI §25.2)
   pitch = db + clear
   n_max = floor(span / pitch) + 1
   ```
4. If `n_required ≤ n_max`, use this `(db, n_required)`.

User overrides: `db_top`, `n_top`, `db_bot`, `n_bot` (any of them) force
the picker to use those values exactly.

### 6.2 Transverse — `_pick_stirrup`

Three cases, depending on which user overrides are given:

- **User fixes BOTH `db_stirrup` and `n_legs`**: spacing adapts to the
  demand: `s = min(s_max_§18.6.4.4, av_provided / av_s_required)`.
- **User fixes only `db_stirrup`**: picker computes `n_legs` at
  `s = s_max_§18.6.4.4`, accepts the smallest `n_legs ≤ 6`.
- **No overrides**: iterate from smallest diameter upward; first one
  that gives `n_legs ≤ 6` at `s_max_§18.6.4.4` wins.

---

## 7. ACI 318-25 clauses used (consolidated)

| Topic | Section | Where |
|---|---|---|
| Whitney stress-block depth (β₁) | Table 22.2.2.4.3 | `common/factors.py::beta1` |
| Strength reduction φ in flexure | §21.2.2 | hard-coded 0.9 (tension-controlled) / 0.65 |
| Strength reduction φ in shear | §21.2.1 | `common/factors.py::phi_shear` |
| Mn (singly reinforced) | §22.2 | `flexure.py::mn_singly` |
| Mn (doubly reinforced) | §22.2 | `flexure.py::mn_doubly` |
| h minimum (predimensioning) | Table 9.3.1.1 | manual — note in summary() |
| ρ_min (flexural tension) | §9.6.1.2 | `flexure.py::rho_min_beam` |
| ρ_balanced | §22.2 | `flexure.py::rho_balanced` |
| ρ_max (this module) | §22.2 limit | `flexure.py::rho_max_beam` = 0.5 · ρ_b |
| ρ_max (informative, SMF) | §18.6.3.1 | `flexure.py::rho_max_seismic` = 0.025 |
| Two bars continuous top/bot | §18.6.3.2 | `_pick_longitudinal` |
| φMn⁺ at face ≥ ½ φMn⁻ at face | §18.6.3.3 | `_design_one` |
| φMn anywhere ≥ ¼ max φMn at face | §18.6.3.4 | `_design_one` |
| lo length (confined zone) | §18.6.4.1 | `lo = 2·h` in `_design_one` |
| s_max in confined zone | §18.6.4.4 | `shear.py::s_max_seismic_smf` |
| s_max outside confined zone | §18.6.4.6 | `shear.py::s_max_seismic_outside` |
| Mpr (1.25·fy, φ=1) | §18.6.5.1 | `seismic.py::mpr` |
| Ve from Mpr (both sway directions) | §18.6.5.1 + R18.6.5 | `seismic.py::v_seismic_from_mpr` |
| Ve = V_seismic + Vg per station | §18.6.5.1 | `_design_one` |
| Vc = 0 in lo (cond. a + b) | §18.6.5.2 | `_design_one(capacity)` |
| Vc simplified | §22.5.5.1 | `shear.py::vc_simplified` |
| Av/s minimum | §9.6.3.4 | `shear.py::av_s_minimum` |
| Bar spacing limits | §25.2 | `_max_bars_in_layer` |

---

## 8. Conventions

### 8.1 Sign convention

- **`Mu_pos > 0`** produces tension on the bottom fiber (gravity-like).
- **`Mu_neg > 0`** is a magnitude (tension on the top fiber).
- **`Vu_*`** is the factored shear at one station (the envelope of combos).
- **`Vg_*`** is positive magnitude — sign decided internally by the
  worst-case combination.

### 8.2 Reinforcement ratios

- **`ρ = As_bot / (bw·d)`** — the steel that carries tension under
  positive bending (the conventional "ρ").
- **`ρ' = As_top / (bw·d)`** — the steel that lies on the compression
  side under positive bending.
- The d in both expressions is the positive-bending effective depth
  (from top fiber to bottom-layer centroid).
- The module-wide `ρ_max = 0.5 · ρ_balanced` is more restrictive than
  the §18.6.3.1 SMF cap of 0.025; we report both.

### 8.3 Stations

Three demand stations along the longitudinal axis:
- `i` — face of the left joint (where −M usually peaks)
- `mid` — middle of clear span (where +M usually peaks)
- `j` — face of the right joint (where −M usually peaks)

`ln` is the clear span (face to face of the joints) in metres.

### 8.4 Stirrup leg counting

For a beam, only **vertical legs** (parallel to Y) resist the vertical
shear V_y. `n_legs` counts these. A basic closed perimeter stirrup has
`n_legs = 2`; each interior crosstie parallel to Y adds one.

The two horizontal legs (parallel to X) of the closed stirrup don't
contribute to V_y — they're implicit in the geometry and not counted.

### 8.5 Internal units

| Quantity | Unit |
|---|---|
| Stress (fc, fy, Es) | MPa |
| Length (bw, h, cover, d, ...) | mm |
| Area (Ag, As, Av) | mm² |
| Force (Vn) | kN |
| Moment (Mn, Mpr) | kN·m |
| Clear span (ln) | m |

`beam.set_units(code)` only changes how numbers are presented.

---

## 9. Optimization — `run_optimize`

```python
from design.beams import run_optimize, format_optimize_table

alts = run_optimize(
    beam, demands,
    db_long_list=None,            # default = bar_schedule.longitudinal
    db_stirrup_list=None,         # default = bar_schedule.hoops
    n_legs_list=[2, 3, 4],        # vertical-leg counts to try
    include_provided=True,        # PROVIDED entry always in the ranking
    include_infeasible=False,     # show only feasible alternatives
    sort_by='rho_total',          # 'rho_total' | 'rho_transverse'
)

print(format_optimize_table(alts, top_n=10))
```

Sweeps `(db_top, db_bot, db_stirrup, n_legs)` from the schedules and
ranks by total steel mass in kg/m³ of concrete. Deduplicates by realized
result (different inputs that map to the same detailing only appear once).

The output table **always** shows the PROVIDED entry (the as-input beam,
tagged `provided`) and the baseline (the envelope from `design()`),
even when their rank is outside the displayed `top_n`. They get
appended below a `....` separator.

Tags column:
- `baseline` — the proposal from `design()` envelope
- `provided` — the beam as input (or after `apply()`)
- `infeasible` — does not satisfy As required or Av/s required

---

## 10. Known simplifications

1. **No PM diagram**. Beams in ACI §9 / §18.6 are flexural-shear elements.
   M-N coupling needs Cap. 10 / Cap. 11 — use a `Column` or `Wall` instead.

2. **Mpr ignores compression steel**. `mpr(As=...)` uses only the tension
   area (standard §18.6.5.1 closed-form interpretation).

3. **Two-layer geometry**. `Beam.rectangular()` produces a two-layer
   (top + bottom) rebar layout. For bundled bars, multiple layers, or
   skin reinforcement, use `Beam(section=...)` with a custom
   `RebarLayout`.

4. **Stirrups are perimeter hoops with interior crossties parallel to Y**.
   `n_legs ≥ 2` always. Crossties for intermediate longitudinal bars
   need an extra check this module does not perform.

5. **`_max_bars_in_layer` is geometric only**. It does not account for
   the corner radius of the stirrup or for the spacer between bundled
   bars. Conservative for typical detailing.

6. **§18.6.5.2 `Vc = 0`**: condition (b) is treated as always-true
   because beams have Pu ≈ 0. Condition (a) (seismic share ≥ 50%) is
   verified explicitly — the proposal note reports the share.

7. **Single `steel_transverse`**. If you need bundled stirrups of
   different grades, override `beam._steel_transverse` directly.

---

## 11. Integration with an ETABS-imported model (planned)

```python
class EtabsBeamAdapter:
    def section_from(self, etabs_section_name: str) -> Section: ...
    def demands_from(self, frame_id, combos) -> BeamDemands:
        """Reads FrameForces; for each station (i / mid / j) takes
        the max |Mu+| and max |Mu-| over combos, plus the envelope |Vu|
        at that station. Adds Vg_i/Vg_mid/Vg_j from a gravity-only combo."""
        ...

model.design.beams.apply(
    ids=[...], combos=['Combo1', 'Combo2'],
)
```

None of these changes will require touching `design/beams/`.

---

## 12. Public re-exports

```python
from design.beams import (
    Beam,
    BeamDemands, BeamCapacity, BeamCheck,
    BeamDesignProposal, BeamDesignResults,
    BeamPlotStyle,
    OptimizeAlternative, run_optimize, format_optimize_table,
    transverse_steel_quantity, longitudinal_steel_quantity,
    # closed-form utilities
    as_required_singly, mn_singly, mn_doubly,
    rho_min_beam, rho_balanced, rho_max_beam, rho_max_seismic,
    is_tension_controlled,
    vc_simplified, vc_detailed, vs_capacity, vs_max,
    vn_beam, av_s_required, av_s_minimum,
    s_max_seismic_smf, s_max_seismic_outside,
    mpr, ve_capacity_design, v_seismic_from_mpr,
)
```

Also at the top-level `design` package:

```python
from design import Concrete, Steel, Bar, RectangularSection, Beam
```

---

## 13. Two notebooks ship with the module

| File | What it shows |
|---|---|
| `design/example_beam.ipynb` | Full tour — every constructor arg, every plot, every design mode, ACI checks, unit hot-swap, manual override, snapshot/revert, optimize sweep |
| `design/example_beam_simple.ipynb` | Quick path — `Beam.rectangular`, demands, checks, design, apply with snapshot/revert |

Both are runnable from the repo root.

---

## 14. Smoke test

`design/_smoke_test.py` exercises every element class. From the repo root:

```bash
python -m design._smoke_test                       # numeric only
python -m design._smoke_test --plot                # writes PNGs to _smoke_plots/
```

---

## 15. Design decisions log

- **`Beam.rectangular(...)` factory**. One-shot constructor that builds
  geometry, rebar layout, and Beam in a single call. `Beam(section=...)`
  remains the low-level path for custom sections.
- **No PM diagram for beams**. ACI §9 / §18.6 treat beams as flexural-shear
  elements. Removing the diagram simplified the public surface, the
  README, and the plotter; nothing was lost on the engineering side.
- **`Vu` is per station, `Vg` is per station**. ETABS naturally emits
  `V` at every frame station; `BeamDemands` keeps the three values.
  Per-station `Vg` (optional) handles non-uniform gravity loading.
- **No `fyt`**. The transverse steel grade is `steel_transverse`
  (defaults to the primary `Steel` of the section). The property
  `beam.fyt` reads from it.
- **No `Av` in the public API**. `n_legs` is the input; `Av` is computed
  internally. Stirrup config is fully specified by `(db_stirrup, n_legs, s_stirrup)`.
- **`ρ_max = 0.5 · ρ_balanced`** as the module-wide cap (more restrictive
  than the SMF 0.025 cap). The 0.025 is still reported.
- **Adaptive stirrup spacing**. When the user fixes both `db_stirrup`
  and `n_legs`, the spacing is computed as
  `min(s_max_§18.6.4.4, av_provided / av_s_required)`.
- **Ve takes both sway directions**. `max((Mpr_neg_i+Mpr_pos_j)/ln,
  (Mpr_pos_i+Mpr_neg_j)/ln)`, then adds Vg per station.
- **`Vc = 0` in lo only when §18.6.5.2 (a)+(b) hold**. Condition (a) is
  verified — the seismic share must be ≥ 50% of the total Ve. The
  proposal note reports the share.
- **`Ve` without Ωo·Vu cap**. ACI §18.6.5.1 confirms — that cap is in
  §18.7.6.1.1 for columns only.
- **`apply` is inplace and rewrites the two-layer rebar**. Different
  from columns, because the longitudinal IS what `design()` chooses.
- **`current_proposal(demands)`** lets you snapshot the as-input state
  and revert via `beam.apply(snapshot)` after exploring other modes.
- **`run_optimize` shows PROVIDED and baseline always**. Even when
  ranked outside the displayed `top_n`, they're appended after a
  separator. Includes per-leg-count sweep (`n_legs_list`).
- **Two-layer geometry only via factory**. Multi-layer / skin
  reinforcement requires building a `RebarLayout` manually (kernel
  supports any layout; the factory and `apply()` don't).
