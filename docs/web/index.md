---
hide:
  - navigation
---

<div class="ep-hero" markdown>
<img class="ep-hero__mark" src="assets/logo.svg" alt="EtabsToPython mark" />
<div>
  <h1 class="ep-hero__word">EtabsToPython</h1>
  <div class="ep-hero__sub">ETABS · Python · ACI 318-25</div>
</div>
</div>

**EtabsToPython** is two complementary layers in one package. The **ETABS layer**
pulls geometry, section properties, and analysis results straight out of an open
ETABS model through the COM API and hands them back as tidy `pandas` DataFrames,
ready to plot and post-process. The **design layer** is a self-contained
reinforced-concrete design toolkit that checks and designs beams, columns,
structural walls, and beam-column joints per **ACI 318-25** — every element
behind the *same* small contract.

!!! info "One package, two jobs"
    Use the **ETABS layer** when you want to get a real model *out of ETABS and
    into Python*. Use the **design layer** to *check or design* concrete sections
    against the code — with or without ETABS in the loop. They share a repo but
    have no hard dependency on each other, so the design toolkit runs anywhere
    (no Windows, no ETABS required).

## Key features

**ETABS extraction (`etabstopython`)**

- Connects directly to an open ETABS session via the COM API.
- Extracts story definitions, point/beam/column/brace/wall/floor connectivity,
  frame section definitions & assignments, modal mass ratios, element & story
  forces, and joint displacements — all as `pandas` DataFrames.
- 3D structural reconstruction with extruded frames, walls, floors, and supports.
- Story-level envelopes: displacement, drift, shear, and max/avg drift ratios.

**Concrete design (`design`)**

- Beams (flexure + shear, SMF §18.6), columns (biaxial P-M-M + confinement §18.7),
  walls (web + boundary elements, iterative design §18.10), and beam-column
  joints (§18.8) — all ACI 318-25.
- One **bidirectional contract** on every element:
  `run · capacity · check · design · evolve · apply`.
- Immutable value objects, explicit unit system, centralized code formulas with
  ACI citations, and a `run_optimize` detailing sweep ranked by steel weight.
- A rich plotting layer: sections, P-M / P-M-M diagrams, demand envelopes, and
  per-element dashboards.

## How it works

| Layer | Entry point | You give it | You get back |
|---|---|---|---|
| ETABS extraction | `EtabsModel(units, name)` | an open ETABS model | DataFrames + plots |
| Concrete design  | `Beam` · `Column` · `Wall` · `BeamColumnJoint` | section + demands | `check` / `design` results + plots |

The design layer works in a fixed **internal unit system** — MPa, mm, kN, kN·m —
while the `units=` knob controls only *presentation*. Every element implements
the identical contract, so once you know one element you know them all.

## A minimal example

```python
from design import Beam, Concrete, Steel

beam = Beam.rectangular(
    bw=300, h=600, cover=40, ln=6000,
    concrete=Concrete(fc=28), steel=Steel(fy=420),
    n_top=3, db_top=20, n_bot=2, db_bot=20,
)

demands = beam.demands(Mu_neg_i=180, Mu_pos_mid=120, Vu_i=150)  # kN·m, kN
check = beam.check(demands)
print(check.passed)

results = beam.design(demands)        # minimum / demand / capacity / envelope
beam = beam.evolve(results.envelope)  # adopt → new immutable beam
beam.plot.dashboard(demands=demands)
```

---

<!-- The "Where do you want to start?" cards go AT THE END — a friendly
     launchpad after the reader knows what the project is. -->

## Where do you want to start?

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } &nbsp; __[Getting started](guides/first_steps.md)__

    ---

    *I'm new — orient me.*

    Install the package and learn the two layers in five minutes.

-   :material-database-export:{ .lg .middle } &nbsp; __[ETABS extraction](guides/etabs_extraction.md)__

    ---

    *Get my model out of ETABS.*

    `EtabsModel`, the DataFrames it builds, and the plotting tools.

-   :material-floor-plan:{ .lg .middle } &nbsp; __[Design toolkit](guides/design_overview.md)__

    ---

    *Check or design a section.*

    The element contract, units, and the 12-step flow shared by every element.

-   :material-rocket-launch:{ .lg .middle } &nbsp; __[Examples](examples/index.md)__

    ---

    *Show me a working model.*

    Notebook walkthroughs per element plus coupled-wall combination studies.

-   :material-book-open-variant:{ .lg .middle } &nbsp; __[API reference](api/index.md)__

    ---

    *Look up a class or method.*

    Auto-generated docs for every public class, method, and function.

</div>

---

**Author:** Patricio Palacios B. — Structural Engineer · Python Developer · Seismic Modeler ·
[@ppalacios92](https://github.com/ppalacios92). Licensed under MIT.
