# Sections & materials

Before you build an element you build its **section**: concrete + a rebar layout.
Materials live in [`design.common.materials`](../api/common.md); sections in
[`design.sections`](../api/sections.md). **All geometry is millimetres.**

## Materials

Frozen value objects with derived ACI properties.

```python
from design import Concrete, Steel, Bar

c   = Concrete(fc=28)            # MPa
s   = Steel(fy=420)             # MPa
bar = Bar.from_imperial(8)      # #8 → 25.4 mm
```

### `Concrete()`

```python
Concrete(fc, unit_weight=24.0, lightweight=False, sand_lightweight=False)
```

| Field | Default | Meaning (units) |
|---|---|---|
| `fc` | — | Compressive strength (MPa) |
| `unit_weight` | `24.0` | Density (kN/m³) |
| `lightweight` | `False` | Lightweight concrete flag → affects `lam` |
| `sand_lightweight` | `False` | Sand-lightweight flag → affects `lam` |

Derived properties: `beta1` (Table 22.2.2.4.3), `Ec = 4700√fc` (§19.2.2),
`lam` (§19.2.4: 1.0 / 0.85 / 0.75), `sqrt_fc = √min(fc, 70)` (§22.5.3.1 cap),
`eps_cu = 0.003` (§22.2.2.1).

### `Steel()`

```python
Steel(fy, Es=200_000.0, fu=None, grade=60, density=7850.0)
```

| Field | Default | Meaning (units) |
|---|---|---|
| `fy` | — | Yield strength (MPa) |
| `Es` | `200_000.0` | Elastic modulus (MPa) |
| `fu` | `None` | Ultimate strength (MPa, reporting) |
| `grade` | `60` | Grade label (ksi; reporting only) |
| `density` | `7850.0` | Mass density (kg/m³) for take-offs |

Derived: `eps_y = fy/Es`, `eps_ty = fy/Es` (§21.2.2.1).

### `Bar()`

```python
Bar(diameter)                   # mm
Bar.from_imperial(number)       # #N (1/8-inch increments) → mm
```

`area = π·diameter²/4` (mm²); `str(Bar(16)) == "db16"`.

## Reinforcement layout

A section's steel is a `RebarLayout` of `RebarGroup`s (bars sharing one steel
grade). Builder functions produce the common patterns.

```python
from design import RebarGroup, RebarLayout
from design.sections.reinforcement import perimeter_bars, two_layer_bars

group  = perimeter_bars(b=500, h=500, cover=40, n_x=4, n_y=4, bar=Bar(25), steel=s)
layout = RebarLayout(groups=(group,))
layout.total_area               # mm²
```

### Data types

| Type | Fields / API |
|---|---|
| `Rebar` | `x`, `y`, `area` (mm) — one bar |
| `RebarGroup` | `bars: tuple[Rebar,…]`, `steel`; props `total_area`, `n` |
| `RebarLayout` | `groups: tuple[RebarGroup,…]`; prop `total_area`; `iter_bars()` → `(Rebar, Steel)`; `as_arrays()` → `RebarArrays` |
| `RebarArrays` | vectorized NumPy view: `x`, `y`, `area`, `fy`, `Es` (one row per bar) |

### `perimeter_bars()`

```python
perimeter_bars(*, b, h, cover, n_x, n_y, bar, steel, center=(0.0, 0.0)) -> RebarGroup
```

Bars evenly along the four edges of a rectangle (corners shared, deduplicated) —
the natural layout for **columns**. `n_x` = bars on bottom/top, `n_y` = bars on
left/right; `center` offsets the rectangle (mm).

### `two_layer_bars()`

```python
two_layer_bars(*, b, h, cover, n_top, n_bottom, top_bar, bottom_bar,
               steel, center=(0.0, 0.0)) -> RebarGroup
```

One top layer + one bottom layer — the natural layout for **beams**. A count of 1
places that layer's bar at the centerline.

### `be_perimeter_bars()`

```python
be_perimeter_bars(*, be_thickness, be_length, center_y, cover,
                  n_x, n_y, bar, steel, center_x=0.0) -> RebarGroup
```

Perimeter bars for a wall **boundary element**, returned in wall-local coordinates
(`center_y` positive = top BE, negative = bottom; `center_x` shifts asymmetric BEs).

## Sections

All sections subclass the abstract `Section` (`concrete` + `rebar`) and implement
`polygons()` (list of CCW `(N,2)` arrays), `gross_area()` (mm²), and
`bounding_box()` → `(x_min, y_min, x_max, y_max)`, plus helpers `y_top()`,
`y_bottom()`, `height()`, `width()`. Local axes: x horizontal (right +), y vertical
(up +), origin at the gross centroid.

### `RectangularSection()`

```python
RectangularSection(*, b, h, concrete, rebar, clear_cover=25.0)
```

| Parameter | Default | Meaning (units) |
|---|---|---|
| `b` / `h` | — | Width / height (mm) |
| `concrete` / `rebar` | — | Material / `RebarLayout` |
| `clear_cover` | `25.0` | Outer face to outside edge of hoop (mm) |

The basis for beams and columns; centered at the origin.

### `WallSection()`

```python
WallSection(*, lw, tw, concrete, rebar,
            be_length_bot=0.0, be_length_top=0.0,
            be_thickness_bot=None, be_thickness_top=None,
            be_top_x_offset=0.0, be_bot_x_offset=0.0)
```

| Parameter | Default | Meaning (units) |
|---|---|---|
| `lw` / `tw` | — | Wall length / web thickness (mm) |
| `concrete` / `rebar` | — | Material / full-section `RebarLayout` |
| `be_length_bot` / `be_length_top` | `0.0` | BE length along `lw`; 0 = no BE (mm) |
| `be_thickness_bot` / `be_thickness_top` | `None` → `tw` | BE thickness across `tw` (mm) |
| `be_top_x_offset` / `be_bot_x_offset` | `0.0` | BE centroid offset from web centerline (mm) — asymmetric shapes |

Axes: x across `tw` (web centerline at x=0), y along `lw` (top at +lw/2). Helpers:

| Method / property | Returns |
|---|---|
| `Acv` | gross in-plane shear area `tw·lw` (mm²) |
| `has_boundary_elements` | bool |
| `be_top_rect()` / `be_bot_rect()` | BE bounding rectangle or `None` |
| `web_rebar()` / `be_top_rebar()` / `be_bot_rebar()` | rebar partitioned web vs BE |
| `be_top_as_rectangular()` / `be_bot_as_rectangular()` | a BE as a `BoundaryElementSection` (→ becomes a `Column`) |
| `with_boundary_elements(*, top_length=None, bot_length=None, top_thickness=None, bot_thickness=None, top_x_offset=None, bot_x_offset=None, add_be_top_bars=None, add_be_bot_bars=None)` | a **new** `WallSection` (immutable) |

### `BoundaryElementSection()`

```python
BoundaryElementSection(*, b, h, concrete, rebar, wall_center_y=0.0)
```

A stand-alone rectangular view of one boundary element (produced by
`WallSection.be_*_as_rectangular()`), so the column machinery can consume it
directly. `wall_center_y` keeps the original wall-frame y-offset for plotting.

## Reference

[`design.sections` API](../api/sections.md) and
[`design.common` materials](../api/common.md).
