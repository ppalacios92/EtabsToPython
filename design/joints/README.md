# `design.joints` — Beam-column joints (ACI 318-25 §18.8 / §15.5.2)

A `BeamColumnJoint` is **assembled from real members**, the same way a
`Wall` composes its boundary `Column`s. You hand it the column above and
below and up to four framing beams; the joint derives its own demand and
capacity. No hand `1.25·fy·As` — every member force comes from the
interaction diagram.

```python
from design import BeamColumnJoint, FramingBeam, Column, Beam

joint = BeamColumnJoint(
    col_above=col_top, col_below=col_bot,
    beam_left=FramingBeam(b_x),  beam_right=FramingBeam(b_x),
    beam_up=FramingBeam(b_y),    beam_down=FramingBeam(b_y),
    moment_condition="P0", probable=True, label="J-A2",
)
joint.run()           # build capacity (lazy + cached)
joint.summary()       # per-direction Aj, gamma, phi*Vn, Vu, ratio
chk  = joint.check()  # JointCheck — ratio, passed, notes
prop = joint.design() # JointDesignResults — detailing (depth, ldh, hoops)
```

## Element contract

Implements the package-wide protocol (`run / capacity / check / design /
evolve / apply / set_units / summary / report / plot`). Internal units are
**MPa, mm, kN, kN·m, mm²** (see `design/AGENTS.md`). All result objects are
frozen dataclasses; `evolve()` returns a NEW joint.

## How it works

### Assembly and joint type
- At least one column (`col_above` / `col_below`) is required.
- A missing column → roof joint; missing beam faces → exterior / corner.
- `joint_type()` infers `interior / exterior / corner / roof …` from the
  count of present beams and columns.

### Demand — from the interaction diagram, per direction (§15.4.2.1)
Joint shear is checked **separately in each principal direction**:
- **X** is loaded by `beam_left` / `beam_right`; **Y** by `beam_up` /
  `beam_down`. Beams on the *other* axis act as transverse confinement.
- For each beam, the bending moment is read from its P-M diagram at the
  chosen `moment_condition` — `"P0"` (moment at axial = 0, the default),
  `"balanced"` (peak/probable moment), or `"demand"`. With `probable=True`
  the curve is built with `fy_pr = 1.25·fy` and `phi = 1.0`.
- That moment is converted to a flange tension `T = M / jd` with
  `jd = 0.875·d` (ACI PRC-352 joint model).
- The worst sway is taken over the two hinge patterns (left-neg/right-pos
  and left-pos/right-neg). The column shear that equilibrates the beam
  moments relieves the joint: `Vu = T1 + T2 − Vcol`, with
  `Vcol = (M1 + M2) / lc` (§18.8.4.1 — column shear *consistent with beam
  Mpr*). You can override either direction with `JointDemands(Vu_x, Vu_y)`.

### Capacity (Table 18.8.4.3, §15.5.2.2, §21.2.4.4)
`phi*Vn = phi · gamma · 0.083 · lambda · sqrt(fc') · Aj`, with `phi = 0.85`.
- `gamma ∈ {20, 15, 12, 8}` (psi coefficient) from three binary axes:
  column continuity, beam continuity *in the direction of Vu*, and
  transverse-beam confinement (`provisions.gamma_joint`).
- `Aj` is the **effective** joint area (`provisions.effective_joint_area`,
  §15.5.2.2): `b_eff = b_column` if the beam is wider than the column,
  otherwise `min(b_beam + h, b_column − 2|e|)`; `Aj ≤ column gross area`.
- The governing direction (smaller `phi*Vn`) is exposed as
  `JointCapacity.phi_Vn` to satisfy the Capacity protocol.

### Eccentric joints
Beam eccentricity (`FramingBeam(..., eccentricity=mm)`) is handled by two
distinct, physically-correct paths — **not** by inventing a joint torsion
action:
1. **Reduced `Aj`** via the `b_column − 2|e|` term in §15.5.2.2 (in the
   capacity).
2. **Torsion `T = M·e`** is reported on the check as a **column** demand
   (§22.7), since the joint itself has no torsional limit state — feed it
   back to the column design.
3. `|e| > b_c/8` raises an ACI PRC-352 advisory note.

### Anchorage (§18.8.5) — hard check
Beam bars must terminate **inside the confined column core**. If
`|e| + b_beam/2 > b_column/2` the bars fall outside the core, `bars_in_core`
is `False`, and the joint **fails** regardless of shear ratio.

### Detailing (`design()`)
- **§18.8.2.3** minimum column depth along each shear direction: greatest
  of `(20/λ)·db` (Gr 60), `26·db` (Gr 80), `h_beam/2`. Compared against the
  provided column dimension (`depth_ok`).
- **§18.8.5.1** hooked-bar development length
  `ldh = fy·db / (65·λ·sqrt(fc'))`, floored at `8db / 150 mm`
  (normalweight), for the largest framing-beam bar.
- **§18.8.3 / §18.7.5.3** confined hoop spacing (from the column's
  confinement design). **§18.8.3.2** relief: when beams frame all four
  faces and each covers ≥ ¾ of the column width, confining `Ash` may be
  halved and spacing relaxed to 150 mm (`ash_reduction`).

Joint shear strength does not depend on transverse reinforcement once the
§18.7.5 minimum is met (R18.8.4), so `design()`'s three modes
(minimum / capacity / envelope) carry the same detailing — they exist only
for contract symmetry with Beam / Column / Wall.

### `angle_deg` is geometric only
`FramingBeam(..., angle_deg=...)` describes the beam's skew in plan. It does
**not** project forces: demands are assumed already resolved into the
principal directions by the analysis model.

## Files
- `provisions.py` — pure ACI functions (no element state): `gamma_joint`,
  `effective_joint_area`, `joint_shear_capacity`, `min_joint_depth`,
  `ldh_hook`, `transverse_beam_confines`, `phi_joint_shear`.
- `joint.py` — `BeamColumnJoint`, `FramingBeam`, and the result dataclasses.

## Clause map
| Clause | Topic | Where |
|---|---|---|
| §21.2.4.4 | `phi = 0.85` joint shear | `provisions.phi_joint_shear` |
| §18.8.2.3 | minimum joint depth | `provisions.min_joint_depth` |
| Table 18.8.4.3 | `gamma` coefficient | `provisions.gamma_joint` |
| §18.8.4.1 | `Vu` from beam Mpr / column shear | `joint._axis_demand` |
| §18.8.4.3 | `Vn = gamma·0.083·λ·√fc'·Aj` | `provisions.joint_shear_capacity` |
| §18.8.3.2 | 4-beam Ash relief | `joint.design` |
| §18.8.5.1 | hooked-bar `ldh` | `provisions.ldh_hook` |
| §18.8.5 | bars in confined core | `joint._direction_check` |
| §15.5.2.2 | effective area `Aj` | `provisions.effective_joint_area` |
| §15.5.2.5 | transverse-beam confinement | `provisions.transverse_beam_confines` |
| §15.4.2.1 | shear checked per direction | `joint._direction_capacity` |
| §22.7 | torsion `M·e` → column | `joint._direction_check` |
