# Beam-column joints

[`design.joints.joint.BeamColumnJoint`](../api/joints.md) checks and designs a
reinforced-concrete beam-column joint per ACI 318-25 §18.8 / §15.5.2. Unlike the
other elements, a joint is **assembled from its members** — the columns above and
below, and the beams framing in from up to four sides.

Internal units: **mm · MPa · kN · kN·m · mm²**.

## Building a joint

A joint has **no `rectangular()` factory** — it is composed from existing `Column`
and `Beam` objects, the beams wrapped as `FramingBeam`s.

```python
from design import BeamColumnJoint, FramingBeam, Column, Beam, Concrete, Steel

# build col_above, col_below (Column) and beam_x, beam_y (Beam) first ...

joint = BeamColumnJoint(
    col_above=col_above,
    col_below=col_below,
    beam_left=FramingBeam(beam=beam_x, angle_deg=0.0),
    beam_right=FramingBeam(beam=beam_x, angle_deg=0.0),
    beam_up=FramingBeam(beam=beam_y, angle_deg=90.0),
    beam_down=FramingBeam(beam=beam_y, angle_deg=90.0),
    moment_condition="P0",
    probable=True,
)
```

### `BeamColumnJoint()`

```python
BeamColumnJoint(*, col_above=None, col_below=None,
                beam_left=None, beam_right=None,
                beam_up=None, beam_down=None,
                moment_condition="P0", probable=True,
                label="Joint", units=None)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `col_above` / `col_below` | `Column` \| None | `None` | Columns above / below. **At least one required** (else `ValueError`). |
| `beam_left` / `beam_right` | `FramingBeam` \| None | `None` | Beams on the −X / +X faces (X-direction shear) |
| `beam_up` / `beam_down` | `FramingBeam` \| None | `None` | Beams on the +Y / −Y faces (Y-direction shear) |
| `moment_condition` | `"P0"`·`"balanced"`·`"demand"` | `"P0"` | Which beam moment drives joint shear: at zero axial / peak probable / hand-specified |
| `probable` | bool | `True` | Use probable beam moments (`fy_pr = 1.25 fy`, `φ = 1.0`) |
| `label` | str | `"Joint"` | Identifier |
| `units` | int·str \| None | `None` | Presentation units |

### `FramingBeam()`

```python
FramingBeam(beam, angle_deg=0.0, eccentricity=0.0)
```

| Field | Default | Meaning (units) |
|---|---|---|
| `beam` | — | The `Beam` framing in (carries its section + rebar) |
| `angle_deg` | `0.0` | Geometric plan skew (does NOT rotate forces) (deg) |
| `eccentricity` | `0.0` | Offset of beam axis from column centroid; penalizes `Aj` per §15.5.2.2 (mm) |

## The contract

```python
joint.run()                  # compute joint shear capacity, both directions
cap     = joint.capacity()   # JointCapacity (x / y)
chk     = joint.check()      # JointCheck → PASS/FAIL
results = joint.design()     # JointDesignResults: minimum / capacity / envelope
```

### `run()`

```python
run() -> BeamColumnJoint
```

Compute and cache the joint shear capacity in both principal directions.

### `capacity()`

```python
capacity() -> JointCapacity
```

Per-direction capacity (`x`, `y`) plus the overall `joint_type` and governing
`phi_Vn`.

??? abstract "`JointCapacity` / `JointDirectionCapacity` fields"
    `JointCapacity`: `x`, `y` (`JointDirectionCapacity`), `joint_type`, `phi_Vn` (kN).

    `JointDirectionCapacity`:

    | Field | Meaning (units) |
    |---|---|
    | `axis` | `"x"` or `"y"` |
    | `Aj` / `b_eff` / `h_joint` | effective joint area / width / depth (mm², mm, mm) |
    | `gamma` | shear coefficient §18.8.4.3 (20/15/12/8) |
    | `phi_Vn` | reduced joint shear, φ=0.85 (kN) |
    | `column_continuous` / `beam_continuous` / `transverse_confined` | continuity & confinement flags |

### `check()`

```python
check(demands=None) -> JointCheck
```

Check shear demand vs `phi_Vn` in each direction. With `demands=None`, the joint
**derives** shear from the members; pass `JointDemands(Vu_x=…, Vu_y=…)` to override.
`JointCheck.summary()` prints a per-axis report.

??? abstract "`JointDemands` / `JointDirectionCheck` / `JointCheck` fields"
    `JointDemands`: `Vu_x`, `Vu_y` (both `Optional[float]`, kN; `None` ⇒ derive from members).

    `JointDirectionCheck`: `axis`, `Vu` (kN), `phi_Vn` (kN), `ratio`, `bars_in_core`
    (bool, §18.8.5), `torsion` (kN·m, from eccentric beams), `notes`.

    `JointCheck`: `capacity`, `x`, `y`, `ratio` (max), `passed` (ratio≤1 **and**
    bars-in-core both axes), `notes`.

### `design()`

```python
design(demands=None) -> JointDesignResults
```

Prescriptive design in **minimum / capacity / envelope** modes — there is **no
demand mode** (joints are capacity-designed to protect the beam/column mechanism).
The three modes share detailing, since joint shear strength is independent of
transverse steel once the §18.7.5 minimum is met (R18.8.4). `demands` is accepted
for API symmetry but unused.

??? abstract "`JointDesignProposal` / `JointDesignResults` fields"
    `JointDesignProposal`:

    | Field | Meaning (units) |
    |---|---|
    | `mode` | `"minimum"`·`"capacity"`·`"envelope"` |
    | `s_max_confined` | max hoop spacing in the joint (mm) |
    | `ash_reduction` | 4-beam ½·Ash allowance (§18.8.3.2) |
    | `ldh_required` | hooked-bar development length (mm) |
    | `min_depth_x` / `min_depth_y` | min column depth for anchorage §18.8.2.3 (mm) |
    | `depth_provided_x` / `depth_provided_y` / `depth_ok` | provided depths & verdict (mm) |
    | `notes` | prescriptive notes |

    `JointDesignResults`: `minimum`, `capacity`, `envelope`; iterable; `as_dict()`.

### `evolve()` and `clear_cache()`

```python
evolve(proposal) -> BeamColumnJoint
clear_cache() -> None
```

`evolve` returns a new joint (geometry is member-driven, so the clone is functionally
identical with a fresh cache). `clear_cache` invalidates the cached capacity.

!!! note "By-design differences"
    A joint has **no `rectangular()`**, **no `demand` mode**, and **no
    `run_optimize`** (transverse steel lives in the column). Everything else —
    `run`, `capacity`, `check`, `design`, `evolve`, `apply`, `summary`, `report`,
    `set_units`, `plot` — matches the other elements.

## Underlying provisions

The pure ACI functions live in [`design.joints.provisions`](../api/joints.md)
(constant `PHI_JOINT_SHEAR = 0.85`):

| Function | §ACI | Returns (units) |
|---|---|---|
| `phi_joint_shear()` | §21.2.4.4 | 0.85 |
| `gamma_joint(*, column_continuous, beam_continuous, transverse_confined)` | §18.8.4.3 | γ ∈ {20,15,12,8} |
| `effective_joint_area(*, h_joint, b_column, b_beam, eccentricity=0.0, gross_area=None)` | §15.5.2.2 | `(Aj, b_eff)` (mm², mm) |
| `joint_shear_capacity(*, gamma, lam, sqrt_fc, Aj, reduced=True)` | §18.8.4.3 | `Vn = γ·0.083·λ·√fc'·Aj` (kN) |
| `min_joint_depth(*, db_grade60=0.0, db_grade80=0.0, h_beam_max=0.0, lam=1.0)` | §18.8.2.3 | min column depth (mm) |
| `ldh_hook(*, fy, db, lam, sqrt_fc)` | §18.8.5.1 | `fy·db/(65·λ·√fc')` ≥ floor (mm) |
| `transverse_beam_confines(*, transverse_widths, column_face_width, n_faces_required=2)` | §15.5.2.5 | bool (¾-width rule) |

## Reporting

```python
joint.summary()
joint.print_design(results, modes=("minimum","capacity","envelope"))
joint.print_final(results.envelope, demands=None)
joint.report()              # dict
```

## Plotting

```python
joint.plot.section()     # plan: column core + framing-beam footprints, with Aj(X), Aj(Y)
joint.plot.demand()      # Vu vs φVn for X and Y, annotated with ratios
joint.plot.dashboard(figsize=None)   # section + demand
```

## Reference

Full [`design.joints` API](../api/joints.md).
