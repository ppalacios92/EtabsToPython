# Getting started

EtabsToPython ships **two layers** in one repository. This page installs the
package and shows the smallest useful program in each layer.

## What's in the box

| Layer | Package | Purpose | Needs ETABS / Windows? |
|---|---|---|---|
| ETABS extraction | `etabstopython` | Pull a live ETABS model into `pandas` | Yes (COM API) |
| Concrete design  | `design` | Check / design RC sections per ACI 318-25 | No — pure Python |

The two layers share a repo but are decoupled: the `design` toolkit imports
nothing from `etabstopython`, so it runs on any OS without ETABS installed.

## Installation

```bash
git clone https://github.com/ppalacios92/EtabsToPython.git
cd EtabsToPython
pip install -e .
```

This installs `comtypes`, `numpy`, `pandas`, `matplotlib`, and `ipython`.
Python 3.8+ is declared, though the `design` layer uses modern typing and is
exercised on 3.10+.

!!! warning "ETABS layer is Windows-only"
    `etabstopython` talks to ETABS through the Windows COM API (`comtypes`).
    ETABS must be installed **and running with a model open**. The `design`
    layer has no such requirement.

## Hello, ETABS

With a model open in ETABS:

```python
from etabstopython import EtabsModel

model = EtabsModel(units=12, name="PRY00")  # connects + loads every table
model.summary()                              # story / element / combo counts

model.frames_df          # beams + columns + braces, with coordinates & sections
model.story_definitions  # story heights
model.combos             # unique load combinations
```

`EtabsModel.__init__` connects to the active session and eagerly loads every
table into DataFrame attributes. See [ETABS extraction](etabs_extraction.md).

## Hello, design

No ETABS needed — design a beam from scalar specs:

```python
from design import Beam, Concrete, Steel

beam = Beam.rectangular(
    bw=300, h=600, cover=40, ln=6000,
    concrete=Concrete(fc=28), steel=Steel(fy=420),
    n_top=3, db_top=20, n_bot=2, db_bot=20,
)
demands = beam.demands(Mu_neg_i=180, Mu_pos_mid=120, Vu_i=150)
print(beam.check(demands).passed)
```

Every element — `Beam`, `Column`, `Wall`, `BeamColumnJoint` — follows the same
contract. Learn it once in [Design toolkit — overview](design_overview.md).

## Units in 30 seconds

The design layer computes **internally in MPa · mm · kN · kN·m**, always.

- **Geometry is canonical millimetres**, everywhere (`bw`, `h`, `ln`, `cover`, bar
  diameters). The `units=` argument is *presentation only* — it never reinterprets
  geometry.
- **Loads are unit-aware**: they pass through `element.demands(..., units=)`, which
  converts to internal units.
- `element.set_units(code)` changes what `summary()`, `report()`, and plots display
  (and how later `demands()` reads forces).

```python
beam.set_units("Tonf_m_C")     # presentation switches to tonf · m
beam.summary()                  # now prints in tonf, m
```

## Where to go next

- [ETABS extraction](etabs_extraction.md) — `EtabsModel` and the plotting tools.
- [Design toolkit — overview](design_overview.md) — the element contract & 12-step flow.
- [Examples](../examples/index.md) — runnable notebooks per element.
- [API reference](../api/index.md) — every public symbol.
