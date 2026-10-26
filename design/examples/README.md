# `design.examples` — one notebook walkthrough per element

Four Jupyter notebooks demonstrate the **whole public API** of every design
element (`Beam`, `Column`, `Wall`, `BeamColumnJoint`) using the **same 12-step
flow**. The point is to make the *architecture* obvious: the skeleton is
identical across elements and only the physics (and a couple of by-design
exceptions) changes.

### Per-element walkthroughs (the 12-step contract)

| Notebook | Element |
|----------|---------|
| [`example_beam.ipynb`](example_beam.ipynb)   | flexure + shear (SMF §18.6) |
| [`example_column.ipynb`](example_column.ipynb) | biaxial P-M-M + confinement (§18.7) |
| [`example_wall.ipynb`](example_wall.ipynb)   | web + boundary elements, iterative design (§18.10) |
| [`example_joint.ipynb`](example_joint.ipynb)  | beam-column joint shear (§18.8) |

### Combination exercises — what beams and walls can do together

| Notebook | Explores |
|----------|----------|
| [`combo_coupled_walls.ipynb`](combo_coupled_walls.ipynb) | coupling beams tying two walls; aspect-ratio regimes + diagonal shear capacity (§18.10.7) |
| [`combo_wall_boundary_columns.ipynb`](combo_wall_boundary_columns.ipynb) | a wall's boundary element **is** a `Column`; `wall_demands_to_be`; §18.10.6 triggers |
| [`combo_parametric_studies.ipynb`](combo_parametric_studies.ipynb) | parametric sweeps (beam span, wall length) + `run_optimize` ranking |
| [`combo_coupled_wall_system.ipynb`](combo_coupled_wall_system.ipynb) | capstone: wall-pier classification + coupling beam + capacity-design shear `omega_v` |
| [`combo_wall_shapes.ipynb`](combo_wall_shapes.ipynb) | **wall shapes**: rectangular, barbell/cabezales, C, T, L, asymmetric — built from the BE thickness + alignment knobs (§18.10.6) |

## Run them

Open any notebook in **Jupyter Lab / Notebook / VS Code** and *Run All*. Each
notebook is self-contained: the first cell puts the repo on `sys.path`, so they
work whether or not `design` is pip-installed, and figures render inline. The
notebooks are committed **already executed**, so you can also just read them.

```bash
jupyter lab design/examples/example_beam.ipynb
```

To re-execute headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace design/examples/example_beam.ipynb
```

## The canonical 12-step flow

| Step | Method | What it does |
|------|--------|--------------|
| 1  | `Concrete(...)`, `Steel(...)` | materials (MPa) |
| 2  | `X.rectangular(...)` | create the element — **all input methods** shown |
| 3  | `el.run()` | lazy-compute + cache capacity (returns `self`) |
| 4  | `el.capacity()` | capacity of the as-built reinforcement (no demand) |
| 5  | `el.demands(...)` | build the load set (display units → internal) |
| 6  | `el.check(demands)` | review the current rebar → PASS/FAIL |
| 7  | `el.design(demands)` | automatic proposal in 4 modes |
| 8  | `results.envelope.<field>` | read the result **explicitly** (no magic proxy) |
| 9  | `el.evolve(p)` / `el.apply(p)` | adopt the design → **new immutable** element |
| 10 | `el.report()` / `el.summary()` / `el.set_units()` / `el.print_final()` | reporting |
| 11 | `el.plot.*` | every plotter method (rendered inline) |
| 12 | `el.run_optimize(demands)` | sweep detailing alternatives |

## Element-specific differences (by design, not inconsistency)

| Element | What is different |
|---------|-------------------|
| **Beam**   | 3 demand stations (i/mid/j); P-M curve is **inspection-only**; `Vc=0` in `lo` (§18.6.5.2). |
| **Column** | full **biaxial PMM** surface; two confinement zones (`lo` + middle); `hx` check; slenderness. |
| **Wall**   | `design()` is **iterative** (`auto_update` rebuilds the BE geometry until it converges); each boundary element **is a real `Column`**. |
| **Joint**  | **no `rectangular()`** factory (assembled from members); **no `demand` mode** in `design()` (capacity-designed to protect the beam/column mechanism); **no `run_optimize`** (steel lives in the column). |

Everything else — `run`, `capacity`, `check`, `design`, `evolve`, `apply`,
`summary`, `report`, `set_units`, `plot` — is **identical** across the four,
provided by the shared `design.common.element_base.ElementBase` and verified
by `tests/test_contract.py`.
