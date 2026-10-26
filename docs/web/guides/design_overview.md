# Design toolkit — overview

The `design` package designs and checks reinforced-concrete elements per
**ACI 318-25**. Its defining feature is that **every element obeys the same
contract**: learn it once for a beam and you already know how to drive a column,
a wall, or a joint.

## The element contract

`Beam`, `Column`, `Wall`, and `BeamColumnJoint` all implement the protocol in
[`design.common.contracts.Element`](../api/common.md):

```python
element.run()                  # lazy-compute + cache capacity → self
element.capacity()             # XCapacity of the as-built reinforcement
element.check(demands)         # XCheck → PASS / FAIL with demand ratios
element.design(demands=None)   # XDesignResults: minimum / demand / capacity / envelope
element.evolve(proposal)       # NEW immutable element with the proposal applied
element.apply(proposal)        # alias of evolve()
element.summary() / .report()  # human-readable / dict reporting
element.set_units(code_or_name)# change presentation units → self
element.plot.*                 # lazy plotter (sections, diagrams, dashboards)
```

`isinstance(element, design.common.contracts.Element)` is `True` for all four,
and `tests/test_contract.py` enforces the contract for every element.

### Immutability via `evolve()`

`evolve(proposal)` returns a **new** element; it never mutates `self`. `design()`
likewise does not store its result on the element — *you* keep the returned value.
Only `run()` mutates (internal capacity caches) and returns `self` for chaining.

```python
results = beam.design(demands)
beam = beam.evolve(results.envelope)   # adopt the design → fresh object
```

## The 12-step flow

The example notebooks drive every element through one identical sequence. This is
the canonical way to use any element:

| Step | Call | What it does |
|------|------|--------------|
| 1  | `Concrete(...)`, `Steel(...)` | materials (MPa) |
| 2  | `X.rectangular(...)` | create the element from scalar specs |
| 3  | `el.run()` | lazy-compute + cache capacity (returns `self`) |
| 4  | `el.capacity()` | capacity of the as-built reinforcement (no demand) |
| 5  | `el.demands(...)` | build the load set (display units → internal) |
| 6  | `el.check(demands)` | review the current rebar → PASS/FAIL |
| 7  | `el.design(demands)` | automatic proposal in 4 modes |
| 8  | `results.envelope.<field>` | read the result **explicitly** |
| 9  | `el.evolve(p)` / `el.apply(p)` | adopt the design → **new immutable** element |
| 10 | `el.report()` / `el.summary()` / `el.set_units()` | reporting |
| 11 | `el.plot.*` | every plotter method |
| 12 | `el.run_optimize(demands)` | sweep detailing alternatives, ranked by steel |

### The four design modes

`design()` returns an `XDesignResults` envelope holding up to four proposals:

- **minimum** — code-minimum detailing, ignoring demand.
- **demand** — the lightest detailing that satisfies the given demands.
- **capacity** — capacity-design detailing (probable strengths, overstrength).
- **envelope** — the governing combination of the above; usually what you adopt.

## Internal units (the golden rule)

All internal math is **MPa · mm · kN · kN·m · mm²**, with no magnitude
heuristics. The rules, from `design/AGENTS.md`:

- **Geometry is always millimetres** — bare constructors (`RectangularSection`,
  `Column.__init__`, `perimeter_bars`) are internal-only and assume mm/kN/MPa.
- The `units=` argument on factories like `.rectangular(...)` is **presentation
  only**: it sets `self.units` but never converts geometry.
- **Loads are unit-aware** and only enter through `element.demands(..., units=)`,
  which converts to internal units (falling back to `self.units`).
- There is **no global mutable unit state**; units live on each object. The default
  is `DEFAULT_UNITS = units_from(9)` (`N_mm_C`), where the conversion is identity.

See [The common layer](common.md) for `UnitSystem`, conversions, and the
centralized code formulas.

## Element family at a glance

| Element | Physics | By-design differences |
|---|---|---|
| [`Beam`](beams.md) | flexure + shear, SMF §18.6 | 3 demand stations (i/mid/j); P-M curve is inspection-only; `Vc=0` in `lo`. |
| [`Column`](columns.md) | biaxial P-M-M + confinement §18.7 | full PMM surface; two confinement zones; `hx` check; slenderness. |
| [`Wall`](walls.md) | web + boundary elements §18.10 | `design()` is **iterative**; each boundary element **is a real `Column`**. |
| [`BeamColumnJoint`](joints.md) | joint shear §18.8 | no `rectangular()` (assembled from members); no `demand` mode; no `run_optimize`. |

Everything else — `run`, `capacity`, `check`, `design`, `evolve`, `apply`,
`summary`, `report`, `set_units`, `plot` — is identical, provided by the shared
`design.common.element_base.ElementBase`.

## Reference

- [`design.common` API](../api/common.md) — contracts, materials, units, factors.
- [Sections & materials](sections.md) — building the geometry and rebar.
