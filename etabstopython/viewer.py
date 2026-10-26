"""PyVista viewer for ETABS geometry snapshots."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def view_model(
    model,
    *,
    show_frames: bool = True,
    show_floors: bool = True,
    show_walls: bool = True,
    show_supports: bool = True,
    show_points: bool = False,
    qt: bool = True,
    notebook: bool = False,
    point_size: float = 6.0,
    floor_opacity: float = 0.4,
    wall_opacity: float = 0.5,
    window_size: tuple[int, int] = (1200, 800),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    slicer: bool = True,
):
    """Open a PyVista/Qt view with numeric X/Y/Z limit controls."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "model.view() requires pyvista. Install it with `pip install pyvista pyvistaqt`."
        ) from exc

    plotter = _make_plotter(pv, qt=qt, notebook=notebook, window_size=window_size)
    building_bounds = _data_bounds(getattr(model, "point_object_connectivity", None))
    bounds = _clip_bounds_from_args(
        getattr(model, "point_object_connectivity", None),
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
    )

    entries = []
    if show_floors:
        mesh = _floor_mesh_like_plotter(pv, model)
        if mesh is not None:
            entries.append(_entry(mesh, "cells", color="gray", opacity=floor_opacity,
                                  show_edges=True, edge_color="black", label="Floors"))

    if show_walls:
        mesh = _shell_mesh(
            pv,
            getattr(model, "wall_object_connectivity", None),
            getattr(model, "point_object_connectivity", None),
        )
        if mesh is not None:
            entries.append(_entry(mesh, "cells", color="orange", opacity=wall_opacity,
                                  show_edges=True, edge_color="black", label="Walls"))

    if show_frames:
        entries.extend(
            _frame_mesh_entries(
                pv,
                getattr(model, "frames_df", None),
                getattr(model, "section_color_dict", None),
                getattr(model, "section_dim_dict", None),
            )
        )

    if show_supports:
        mesh = _support_points_mesh(pv, getattr(model, "frames_df", None))
        if mesh is not None:
            entries.append(_entry(mesh, "points", color="purple", point_size=12.0,
                                  render_points_as_spheres=False, label="Supports"))

    if show_points:
        mesh = _points_mesh(pv, getattr(model, "point_object_connectivity", None))
        if mesh is not None:
            entries.append(_entry(mesh, "points", color="#111827", point_size=point_size,
                                  render_points_as_spheres=True, label="Points"))

    _draw_entries(plotter, entries, bounds)

    if slicer:
        _add_limit_controls(plotter, entries, current_bounds=bounds, full_bounds=building_bounds)

    model_name = getattr(model, "name", "ETABS model")
    _add_title(plotter, f"3D View Structure - PRY: {model_name}")
    plotter.add_axes()
    plotter.show_grid()
    _set_equal_bounds(plotter, bounds)
    plotter.view_isometric()
    plotter.camera.SetViewUp(0.0, 0.0, 1.0)
    plotter.reset_camera()

    if not qt:
        plotter.show()
    return plotter


def _entry(mesh, kind: str, **kwargs):
    return {"mesh": mesh, "kind": kind, "kwargs": kwargs}


def _make_plotter(pv, *, qt: bool, notebook: bool, window_size: tuple[int, int]):
    if qt:
        try:
            from pyvistaqt import BackgroundPlotter
        except ImportError as exc:
            raise ImportError(
                "model.view(qt=True) requires pyvistaqt. Install it with "
                "`pip install pyvistaqt pyqt5` or call `model.view(qt=False)`."
            ) from exc
        return BackgroundPlotter(window_size=window_size)
    return pv.Plotter(notebook=notebook, window_size=window_size)


def _add_title(plotter, text: str) -> None:
    if hasattr(plotter, "add_title"):
        plotter.add_title(text, font_size=10)
        return
    plotter.add_text(text, position="upper_edge", font_size=10)


def _frame_mesh_entries(pv, frames_df, section_color_dict, section_dim_dict):
    entries = []
    if frames_df is None or frames_df.empty:
        return entries

    cols = ["PointIX", "PointIY", "PointIZ", "PointJX", "PointJY", "PointJZ"]
    if not set(cols).issubset(frames_df.columns):
        return entries

    if "SectProp" in frames_df.columns and section_color_dict:
        for section, group in frames_df.groupby("SectProp", dropna=False):
            mesh = _frame_prism_mesh(pv, group, section_dim_dict)
            if mesh is not None:
                entries.append(_entry(mesh, "cells", color=section_color_dict.get(section, "#4b5563"),
                                      opacity=0.9, show_edges=True, edge_color="black"))
        return entries

    mesh = _frame_prism_mesh(pv, frames_df, section_dim_dict)
    if mesh is not None:
        entries.append(_entry(mesh, "cells", color=(0.6, 0.6, 0.6),
                              opacity=0.9, show_edges=True, edge_color="black"))
    return entries


def _draw_entries(plotter, entries, bounds):
    actors = []
    for entry in entries:
        mesh = _clip_entry_mesh(entry, bounds)
        if mesh is None:
            continue
        actors.append(plotter.add_mesh(mesh, **entry["kwargs"]))
    plotter._e2p_slice_actors = actors


def _redraw_entries(plotter, entries, bounds):
    for actor in getattr(plotter, "_e2p_slice_actors", []):
        plotter.remove_actor(actor, reset_camera=False, render=False)
    _draw_entries(plotter, entries, tuple(bounds))
    plotter.render()


def _clip_entry_mesh(entry, bounds):
    if entry["kind"] == "points":
        mesh = _clip_points_by_bounds(entry["mesh"], bounds)
        return None if mesh is None or mesh.n_points == 0 else mesh
    mesh = _clip_mesh_by_bounds(entry["mesh"], bounds)
    return None if mesh is None or mesh.n_cells == 0 else mesh


def _add_limit_controls(plotter, entries, *, current_bounds, full_bounds):
    if current_bounds is None or full_bounds is None:
        return
    if not hasattr(plotter, "app_window"):
        return

    from PyQt5 import QtCore, QtWidgets

    full = [float(v) for v in full_bounds]
    current = [float(v) for v in current_bounds]
    labels = [
        ("X min", 0),
        ("X max", 1),
        ("Y min", 2),
        ("Y max", 3),
        ("Z min", 4),
        ("Z max", 5),
    ]

    dock = QtWidgets.QDockWidget("View limits", plotter.app_window)
    dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

    panel = QtWidgets.QWidget()
    layout = QtWidgets.QFormLayout(panel)
    layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

    boxes = {}

    def apply_limits():
        for index, box in boxes.items():
            current[index] = float(box.value())
        for lo, hi in ((0, 1), (2, 3), (4, 5)):
            if current[lo] > current[hi]:
                current[lo], current[hi] = current[hi], current[lo]
                boxes[lo].blockSignals(True)
                boxes[hi].blockSignals(True)
                boxes[lo].setValue(current[lo])
                boxes[hi].setValue(current[hi])
                boxes[lo].blockSignals(False)
                boxes[hi].blockSignals(False)
        _redraw_entries(plotter, entries, current)

    for title, index in labels:
        axis_base = index - index % 2
        box = QtWidgets.QDoubleSpinBox()
        box.setDecimals(3)
        box.setRange(full[axis_base], full[axis_base + 1])
        box.setValue(current[index])
        box.setSingleStep(max((full[axis_base + 1] - full[axis_base]) / 100.0, 0.001))
        box.editingFinished.connect(apply_limits)
        layout.addRow(title, box)
        boxes[index] = box

    reset = QtWidgets.QPushButton("RESET")

    def reset_limits():
        for index, value in enumerate(full):
            boxes[index].blockSignals(True)
            boxes[index].setValue(value)
            boxes[index].blockSignals(False)
            current[index] = value
        _redraw_entries(plotter, entries, current)
        _set_equal_bounds(plotter, tuple(full))
        plotter.reset_camera()

    reset.clicked.connect(reset_limits)
    layout.addRow(reset)

    dock.setWidget(panel)
    plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    plotter._e2p_limit_dock = dock
    plotter._e2p_limit_boxes = boxes


def _frame_prism_mesh(pv, frames_df, section_dim_dict):
    all_points = []
    all_faces = []
    for _, row in frames_df.iterrows():
        prism = _frame_prism_points(row, section_dim_dict)
        if prism is None:
            continue
        start = len(all_points)
        all_points.extend(prism)
        all_faces.extend(
            [
                4, start + 0, start + 1, start + 3, start + 2,
                4, start + 4, start + 5, start + 7, start + 6,
                4, start + 0, start + 1, start + 5, start + 4,
                4, start + 2, start + 3, start + 7, start + 6,
                4, start + 1, start + 3, start + 7, start + 5,
                4, start + 0, start + 2, start + 6, start + 4,
            ]
        )

    if not all_points:
        return None
    return pv.PolyData(np.asarray(all_points, dtype=float), faces=np.asarray(all_faces, dtype=np.int64))


def _frame_prism_points(row, section_dim_dict):
    try:
        p1 = np.asarray([row["PointIX"], row["PointIY"], row["PointIZ"]], dtype=float)
        p2 = np.asarray([row["PointJX"], row["PointJY"], row["PointJZ"]], dtype=float)
    except (TypeError, ValueError):
        return None

    axis = p2 - p1
    length = np.linalg.norm(axis)
    if length == 0:
        return None
    v = axis / length

    if np.allclose(v, [0.0, 0.0, 1.0]) or np.allclose(v, [0.0, 0.0, -1.0]):
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.cross(v, [0.0, 0.0, 1.0])
        u = u / np.linalg.norm(u)
    w = np.cross(v, u)

    dims = {}
    if section_dim_dict:
        dims = section_dim_dict.get(row.get("SectProp"), {}) or {}
    u *= _float_or_default(dims.get("t2"), 0.3) / 2.0
    w *= _float_or_default(dims.get("t3"), 0.5) / 2.0

    return [
        point + sign_u * u + sign_w * w
        for point in (p1, p2)
        for sign_u in (-1, 1)
        for sign_w in (-1, 1)
    ]


def _points_mesh(pv, points_df):
    if points_df is None or points_df.empty:
        return None
    if not {"X", "Y", "Z"}.issubset(points_df.columns):
        return None
    points = points_df[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").dropna()
    if points.empty:
        return None
    return pv.PolyData(points.to_numpy(dtype=float))


def _support_points_mesh(pv, frames_df):
    if frames_df is None or frames_df.empty:
        return None
    required = {"PointIX", "PointIY", "PointIZ", "PointJX", "PointJY", "PointJZ"}
    if not required.issubset(frames_df.columns):
        return None

    coords = []
    for prefix in ("I", "J"):
        xyz = frames_df[[f"Point{prefix}X", f"Point{prefix}Y", f"Point{prefix}Z"]].copy()
        xyz.columns = ["X", "Y", "Z"]
        xyz = xyz.apply(pd.to_numeric, errors="coerce").dropna()
        coords.append(xyz[xyz["Z"] == 0.0])

    supports = pd.concat(coords, ignore_index=True).drop_duplicates()
    if supports.empty:
        return None
    return pv.PolyData(supports[["X", "Y", "Z"]].to_numpy(dtype=float))


def _floor_mesh_like_plotter(pv, model):
    point_df = getattr(model, "point_object_connectivity", None)
    return _shell_mesh(pv, getattr(model, "floor_object_connectivity", None), point_df)


def _shell_mesh(pv, shells_df, points_df):
    if shells_df is None or points_df is None or shells_df.empty or points_df.empty:
        return None
    point_lookup = _point_lookup(points_df)
    if not point_lookup:
        return None

    shell_cols = [c for c in ("UniquePt1", "UniquePt2", "UniquePt3", "UniquePt4") if c in shells_df.columns]
    if len(shell_cols) < 3:
        return None

    points = []
    faces = []
    groups = shells_df.groupby("UniqueName", sort=False) if "UniqueName" in shells_df.columns else [(None, shells_df)]
    for _, group in groups:
        coords = []
        for _, row in group.iterrows():
            coords.extend(
                point_lookup[k]
                for k in _point_keys(row[c] for c in shell_cols)
                if k in point_lookup
            )
        coords = _dedupe_coords(coords)
        if len(coords) < 3:
            continue
        start = len(points)
        points.extend(coords)
        faces.extend([len(coords), *range(start, start + len(coords))])

    if not points:
        return None
    return pv.PolyData(np.asarray(points, dtype=float), faces=np.asarray(faces, dtype=np.int64))


def _clip_bounds_from_args(points_df, *, xlim=None, ylim=None, zlim=None):
    bounds = _data_bounds(points_df)
    if bounds is None:
        return None
    if xlim is not None:
        bounds[0], bounds[1] = sorted((float(xlim[0]), float(xlim[1])))
    if ylim is not None:
        bounds[2], bounds[3] = sorted((float(ylim[0]), float(ylim[1])))
    if zlim is not None:
        bounds[4], bounds[5] = sorted((float(zlim[0]), float(zlim[1])))
    return tuple(bounds)


def _data_bounds(points_df):
    if points_df is None or points_df.empty or not {"X", "Y", "Z"}.issubset(points_df.columns):
        return None
    coords = points_df[["X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce").dropna()
    if coords.empty:
        return None
    xyz = coords.to_numpy(dtype=float)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    return [float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2])]


def _clip_mesh_by_bounds(mesh, bounds):
    if bounds is None or mesh is None or mesh.n_cells == 0:
        return mesh
    centers = mesh.cell_centers().points
    mask = (
        (centers[:, 0] >= bounds[0]) & (centers[:, 0] <= bounds[1]) &
        (centers[:, 1] >= bounds[2]) & (centers[:, 1] <= bounds[3]) &
        (centers[:, 2] >= bounds[4]) & (centers[:, 2] <= bounds[5])
    )
    return mesh.extract_cells(mask)


def _clip_points_by_bounds(mesh, bounds):
    if bounds is None or mesh is None or mesh.n_points == 0:
        return mesh
    pts = mesh.points
    mask = (
        (pts[:, 0] >= bounds[0]) & (pts[:, 0] <= bounds[1]) &
        (pts[:, 1] >= bounds[2]) & (pts[:, 1] <= bounds[3]) &
        (pts[:, 2] >= bounds[4]) & (pts[:, 2] <= bounds[5])
    )
    return mesh.extract_points(mask, adjacent_cells=False)


def _set_equal_bounds(plotter, bounds):
    if bounds is None:
        return
    mins = np.array([bounds[0], bounds[2], bounds[4]], dtype=float)
    maxs = np.array([bounds[1], bounds[3], bounds[5]], dtype=float)
    mid = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    if radius == 0:
        radius = 1.0
    plotter.set_focus(tuple(mid))
    plotter.set_scale(1.0, 1.0, 1.0)
    plotter.show_bounds(
        bounds=(
            mid[0] - radius, mid[0] + radius,
            mid[1] - radius, mid[1] + radius,
            mid[2] - radius, mid[2] + radius,
        ),
        xtitle="Eje X (m)",
        ytitle="Eje Y (m)",
        ztitle="Eje Z (m)",
    )


def _point_lookup(points_df) -> dict[str, tuple[float, float, float]]:
    if not {"UniqueName", "X", "Y", "Z"}.issubset(points_df.columns):
        return {}

    lookup = {}
    for _, row in points_df.iterrows():
        key = _point_key(row["UniqueName"])
        xyz = pd.to_numeric(row[["X", "Y", "Z"]], errors="coerce")
        if not key or xyz.isna().any():
            continue
        lookup[key] = tuple(float(v) for v in xyz)
    return lookup


def _point_keys(values: Iterable) -> list[str]:
    return [key for key in (_point_key(v) for v in values) if key]


def _point_key(value) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        fvalue = float(value)
        if fvalue.is_integer():
            return str(int(fvalue))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _dedupe_coords(coords):
    unique = []
    seen = set()
    for coord in coords:
        key = tuple(round(v, 9) for v in coord)
        if key in seen:
            continue
        seen.add(key)
        unique.append(coord)
    return unique


def _float_or_default(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
