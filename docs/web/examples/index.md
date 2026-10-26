# Examples

The repository ships executable Jupyter notebooks for both layers. They are
committed **already executed**, so you can read them without ETABS or a kernel.

## Design notebooks — one walkthrough per element

Under [`design/examples/`](https://github.com/ppalacios92/EtabsToPython/tree/main/design/examples),
four notebooks drive every element through the **same 12-step flow**. The point is
that the skeleton is identical across elements — only the physics changes.

| Notebook | Element | Code |
|----------|---------|------|
| `example_beam.ipynb`   | Beam — flexure + shear (SMF §18.6) | [view](https://github.com/ppalacios92/EtabsToPython/blob/main/design/examples/example_beam.ipynb) |
| `example_column.ipynb` | Column — biaxial P-M-M + confinement (§18.7) | [view](https://github.com/ppalacios92/EtabsToPython/blob/main/design/examples/example_column.ipynb) |
| `example_wall.ipynb`   | Wall — web + boundary elements, iterative design (§18.10) | [view](https://github.com/ppalacios92/EtabsToPython/blob/main/design/examples/example_wall.ipynb) |
| `example_joint.ipynb`  | Joint — beam-column joint shear (§18.8) | [view](https://github.com/ppalacios92/EtabsToPython/blob/main/design/examples/example_joint.ipynb) |

### The canonical 12-step flow

| Step | Method | What it does |
|------|--------|--------------|
| 1  | `Concrete(...)`, `Steel(...)` | materials (MPa) |
| 2  | `X.rectangular(...)` | create the element — all input methods shown |
| 3  | `el.run()` | lazy-compute + cache capacity (returns `self`) |
| 4  | `el.capacity()` | capacity of the as-built reinforcement (no demand) |
| 5  | `el.demands(...)` | build the load set (display units → internal) |
| 6  | `el.check(demands)` | review the current rebar → PASS/FAIL |
| 7  | `el.design(demands)` | automatic proposal in 4 modes |
| 8  | `results.envelope.<field>` | read the result explicitly (no magic proxy) |
| 9  | `el.evolve(p)` / `el.apply(p)` | adopt the design → new immutable element |
| 10 | `el.report()` / `el.summary()` / `el.set_units()` / `el.print_final()` | reporting |
| 11 | `el.plot.*` | every plotter method (rendered inline) |
| 12 | `el.run_optimize(demands)` | sweep detailing alternatives |

## Combination studies — beams and walls together

| Notebook | Explores |
|----------|----------|
| `combo_coupled_walls.ipynb` | coupling beams tying two walls; aspect-ratio regimes + diagonal shear (§18.10.7) |
| `combo_wall_boundary_columns.ipynb` | a wall's boundary element **is** a `Column`; `wall_demands_to_be`; §18.10.6 triggers |
| `combo_parametric_studies.ipynb` | parametric sweeps (beam span, wall length) + `run_optimize` ranking |
| `combo_coupled_wall_system.ipynb` | capstone: wall-pier classification + coupling beam + capacity-design shear `omega_v` |
| `combo_wall_shapes.ipynb` | wall shapes: rectangular, barbell, C, T, L, asymmetric — from the BE thickness + alignment knobs |

## Element-specific differences (by design, not inconsistency)

| Element | What is different |
|---------|-------------------|
| **Beam**   | 3 demand stations (i/mid/j); P-M curve is inspection-only; `Vc=0` in `lo` (§18.6.5.2). |
| **Column** | full biaxial PMM surface; two confinement zones (`lo` + middle); `hx` check; slenderness. |
| **Wall**   | `design()` is iterative (`auto_update` rebuilds BE geometry until convergence); each BE **is a real `Column`**. |
| **Joint**  | no `rectangular()` (assembled from members); no `demand` mode; no `run_optimize` (steel lives in the column). |

## Running them

```bash
jupyter lab design/examples/example_beam.ipynb
```

Each notebook is self-contained: the first cell puts the repo on `sys.path`, so
they work whether or not `design` is pip-installed. To re-execute headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace design/examples/example_beam.ipynb
```

## ETABS notebook

[`EtabsToPython.ipynb`](https://github.com/ppalacios92/EtabsToPython/blob/main/EtabsToPython.ipynb)
demonstrates the ETABS extraction layer end-to-end — connecting to a live model,
loading the tables, and producing the 3D and story-envelope plots. It requires
ETABS running on Windows.
