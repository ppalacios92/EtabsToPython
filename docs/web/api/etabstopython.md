# etabstopython (ETABS layer)

The ETABS extraction layer. Because it depends on the Windows COM API
(`comtypes`) and a running ETABS session, it cannot be imported on the Linux docs
runner — so this page is maintained by hand rather than auto-generated. For usage,
see [ETABS extraction](../guides/etabs_extraction.md).

---

## `etabstopython.model.EtabsModel`

Main interface to the ETABS COM API: connects to the active session, sets units,
and eagerly loads every model table into `pandas` DataFrames.

### Constructor

```python
EtabsModel(units: int = 12, name: str = "PRY00") -> None
```

Connect to ETABS, set the unit system (`units`, an ETABS code), tag the model
(`name`), and run every loader.

### Loader methods

| Method | Builds |
|---|---|
| `load_story_definitions()` | `story_definitions`, `floor_heights` |
| `load_point_object_connectivity()` | `point_object_connectivity` (+ floor labels) |
| `load_frame_section_property_definitions_summary()` | section definition summary |
| `load_frame_section_property_definitions_concrete_rectangular()` | concrete rectangular props |
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
| `summary()` | prints a consolidated overview of everything loaded |

### Data attributes

After construction the loaded data is available as instance attributes:
`story_definitions`, `floor_heights`, `point_object_connectivity`, `frames_df`,
`beam_object_connectivity`, `column_object_connectivity`,
`brace_object_connectivity`, `wall_object_connectivity`,
`floor_object_connectivity`, `floor_points_by_story`,
`frame_section_property_definitions_summary`,
`frame_section_property_definitions_concrete_rectangular`,
`frame_assignments_section_properties`, `section_color_dict`, `section_dim_dict`,
`element_forces_columns`, `modal_participating_mass_ratios`, `story_forces`,
`combos`, `joint_displacements`, `joint_reactions`.

---

## `etabstopython.plotter`

Matplotlib renderers. Pass the loaded `EtabsModel` as `model`.

| Function | Purpose |
|---|---|
| `plot_structure_3d(model)` | 3D structure: extruded frames, walls, floors, supports |
| `plot_story_displacement_bounds(model, combos_comp, color=[.7,.7,.7], lw=1.0, highlight_combo=None, factor=1.0)` | per-story min/max `Ux`, `Uy` |
| `plot_story_drift_bounds(model, combos_comp, color=[.7,.7,.7], lw=1.0, highlight_combo=None, factor=1.0)` | story drift (%) vs height |
| `plot_story_shear_bounds(model, combos_comp, color=[.7,.7,.7], lw=1.0, highlight_combo=None)` | story shear (`VX`, `VY`) envelope |
| `plot_story_max_over_avg_ratios(model, combos_comp, show=True, save_path=None)` | max/avg drift ratio per story (torsion) |

---

## `etabstopython.tools`

Lower-level helpers the plotters build on.

| Function | Returns |
|---|---|
| `extrude_frame(ax, p1, p2, width, height, color)` | draws a rectangular prism between two 3D points |
| `compute_story_displacement_bounds(model, combos_comp, factor=1.0)` | DataFrame of `Ux/Uy` min/max per story (+ `Height`) |
| `compute_story_force_bounds(model, combos_comp)` | DataFrame of max `P/VX/VY/MX/MY` per story |
| `story_max_over_avg_ratios(model, combos_comp)` | `(stories, heights, data_dict)` of drift ratios |
