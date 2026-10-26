# ETABS extraction

The `etabstopython` layer connects to a running ETABS session through the COM
API and loads the whole model into `pandas` DataFrames. Everything is driven by
one class — [`EtabsModel`](../api/etabstopython.md) — plus a set of plotting and
helper functions.

!!! warning "Requirements"
    ETABS must be **installed, running, and have a model open** before you
    construct `EtabsModel`. This layer is Windows-only (`comtypes`).

## Connect and load

```python
from etabstopython import EtabsModel

model = EtabsModel(units=12, name="PRY00")
```

Constructing the model does everything in one shot: it attaches to the active
ETABS instance, sets the unit system, and runs every loader. After that the data
lives on the instance as DataFrame / array attributes.

### `EtabsModel(units=12, name="PRY00")`

Connect to ETABS, set units, and eagerly load all tables and properties. `units`
is the ETABS unit-system code; `name` is a model identifier you choose.

### `summary()`

Print a consolidated overview: story counts, element counts, load combinations,
and which tables were loaded. Run it right after construction as a sanity check.

## What gets loaded

`EtabsModel` exposes its data as attributes. The most useful ones:

| Attribute | Type | Contents |
|---|---|---|
| `story_definitions` | DataFrame | story names + heights |
| `floor_heights` | ndarray | accumulated height per floor |
| `point_object_connectivity` | DataFrame | node coordinates + floor labels |
| `frames_df` | DataFrame | **beams + columns + braces** unified, with end coordinates & section props |
| `wall_object_connectivity` | DataFrame | wall (shell) connectivity |
| `floor_object_connectivity` | DataFrame | floor (slab) connectivity |
| `floor_points_by_story` | ndarray | node indices grouped by story |
| `frame_section_property_definitions_concrete_rectangular` | DataFrame | concrete section dimensions |
| `frame_assignments_section_properties` | DataFrame | frame → section assignments |
| `section_color_dict` / `section_dim_dict` | dict | colours & `(t2, t3)` dims per section |
| `element_forces_columns` | DataFrame | column element forces |
| `story_forces` | DataFrame | story forces |
| `combos` | ndarray | unique load combination names |
| `modal_participating_mass_ratios` | DataFrame | modal mass participation |
| `joint_displacements` | DataFrame | joint displacements & rotations |
| `joint_reactions` | DataFrame | joint reactions |

## The loaders

`__init__` calls these for you; call individual ones to refresh a single table.

| Method | Builds |
|---|---|
| `load_story_definitions()` | `story_definitions`, `floor_heights` |
| `load_point_object_connectivity()` | `point_object_connectivity` (+ floor labels by Z) |
| `load_frame_section_property_definitions_summary()` | section definition summary |
| `load_frame_section_property_definitions_concrete_rectangular()` | concrete rectangular section props |
| `load_frame_assignments_section_properties()` | frame → section assignments |
| `load_beam_object_connectivity()` | beams |
| `load_column_object_connectivity()` | columns |
| `load_brace_object_connectivity()` | braces |
| `build_linear_elements_dataframe()` | `frames_df` (beams + columns + braces) |
| `load_wall_object_connectivity()` | walls |
| `load_floor_object_connectivity()` | floors + `floor_points_by_story` |
| `build_section_color_dict()` | `section_color_dict`, `section_dim_dict` |
| `load_element_forces_columns()` | `element_forces_columns` |
| `load_modal_participating_mass_ratios()` | `modal_participating_mass_ratios` |
| `load_story_forces()` | `story_forces`, `combos` |
| `load_joint_displacements()` | `joint_displacements` |
| `joint_reactions()` | `joint_reactions` |

## Visualizing the model

The `etabstopython.plotter` module renders the model and story-level envelopes.
Pass the loaded `model` straight in.

```python
from etabstopython import plotter

plotter.plot_structure_3d(model)
plotter.plot_story_drift_bounds(model, combos_comp=model.combos, highlight_combo="EQX")
```

### `plot_structure_3d(model)`

Render the 3D structure with extruded frames (beams/columns/braces), walls,
floors, and supports.

### `plot_story_displacement_bounds(model, combos_comp, ...)`

Per-story min/max joint displacements (`Ux`, `Uy`) across combinations, with an
optional `highlight_combo` and amplification `factor`.

### `plot_story_drift_bounds(model, combos_comp, ...)`

Story drift (%) vs height across combinations, with optional highlight.

### `plot_story_shear_bounds(model, combos_comp, ...)`

Story shear (`VX`, `VY`) envelope by height.

### `plot_story_max_over_avg_ratios(model, combos_comp, show=True, save_path=None)`

Bar chart of max-drift / avg-drift ratio per story (X right, Y left) — the
torsional irregularity indicator.

## Helper functions (`etabstopython.tools`)

Lower-level utilities the plotters build on; useful for custom figures.

| Function | Returns |
|---|---|
| `extrude_frame(ax, p1, p2, width, height, color)` | draws a rectangular prism between two 3D points |
| `compute_story_displacement_bounds(model, combos_comp, factor=1.0)` | DataFrame of `Ux/Uy` min/max per story |
| `compute_story_force_bounds(model, combos_comp)` | DataFrame of max `P/VX/VY/MX/MY` per story |
| `story_max_over_avg_ratios(model, combos_comp)` | `(stories, heights, data_dict)` of drift ratios |

## Reference

See the full [`etabstopython` API](../api/etabstopython.md).
