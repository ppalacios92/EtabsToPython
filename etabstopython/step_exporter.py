"""STEP export for ETABS line and shell connectivity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_step(model, filename: str, *, include="all", progress: bool = True) -> Path:
    """Export model geometry to STEP as frame lines and shell surfaces."""
    try:
        import gmsh
    except ImportError as exc:
        raise ImportError(
            "export_STEP requires gmsh. Install it with `pip install gmsh`."
        ) from exc

    path = Path(filename)
    if path.suffix.lower() not in {".step", ".stp"}:
        path = path.with_suffix(".step")
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)

    initialized_here = not gmsh.isInitialized()
    if initialized_here:
        gmsh.initialize()

    include_set = _normalize_include(include)
    counts = {"points": 0, "beams": 0, "columns": 0, "braces": 0, "floors": 0, "walls": 0}
    try:
        gmsh.model.add(path.stem)
        point_tags = _add_points(gmsh, _points_dataframe(model))
        counts["points"] = len(point_tags)

        if "beams" in include_set:
            counts["beams"] = _add_frame_lines(
                gmsh, _connectivity_table(model, "beam_object_connectivity", "Beam Object Connectivity"), point_tags
            )
        if "columns" in include_set:
            counts["columns"] = _add_frame_lines(
                gmsh, _connectivity_table(model, "column_object_connectivity", "Column Object Connectivity"), point_tags
            )
        if "braces" in include_set:
            counts["braces"] = _add_frame_lines(
                gmsh, _connectivity_table(model, "brace_object_connectivity", "Brace Object Connectivity"), point_tags
            )
        if "floors" in include_set:
            counts["floors"] = _add_shell_surfaces(
                gmsh,
                getattr(model, "floor_object_connectivity", None),
                point_tags,
                prefix="Floor",
            )
        if "walls" in include_set:
            counts["walls"] = _add_shell_surfaces(
                gmsh,
                getattr(model, "wall_object_connectivity", None),
                point_tags,
                prefix="Wall",
            )

        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        if initialized_here:
            gmsh.finalize()

    if progress:
        print(f"STEP exported: {path}")
        print(
            f"points={counts['points']} | beams={counts['beams']} | "
            f"columns={counts['columns']} | braces={counts['braces']} | "
            f"floors={counts['floors']} | walls={counts['walls']}"
        )
    return path


def _normalize_include(include) -> set[str]:
    allowed = {"beams", "columns", "braces", "floors", "walls"}
    if include == "all":
        return set(allowed)
    if isinstance(include, str):
        include = [include]
    include_set = set(include)
    invalid = sorted(include_set - allowed)
    if invalid:
        raise ValueError(
            f"Invalid STEP include item(s): {invalid}. "
            "Allowed: beams, columns, braces, floors, walls, all"
        )
    return include_set


def _points_dataframe(model) -> pd.DataFrame:
    if hasattr(model, "point_object_connectivity"):
        return model.point_object_connectivity
    return model._get_table_as_dataframe("Point Object Connectivity")


def _add_points(gmsh, points_df: pd.DataFrame) -> dict[str, int]:
    if points_df is None or points_df.empty:
        return {}
    required = {"UniqueName", "X", "Y", "Z"}
    if not required.issubset(points_df.columns):
        return {}

    point_tags = {}
    for _, row in points_df.iterrows():
        key = _point_key(row["UniqueName"])
        if not key:
            continue
        xyz = pd.to_numeric(row[["X", "Y", "Z"]], errors="coerce")
        if xyz.isna().any():
            continue
        point_tags[key] = gmsh.model.occ.addPoint(float(xyz["X"]), float(xyz["Y"]), float(xyz["Z"]))
    return point_tags


def _add_frame_lines(gmsh, frames: pd.DataFrame, point_tags: dict[str, int]) -> int:
    count = 0
    for _, row in frames.iterrows():
        pi = point_tags.get(_point_key(row.get("UniquePtI")))
        pj = point_tags.get(_point_key(row.get("UniquePtJ")))
        if pi is None or pj is None or pi == pj:
            continue
        line = gmsh.model.occ.addLine(pi, pj)
        name = row.get("UniqueName")
        if name is not None:
            gmsh.model.setEntityName(1, line, str(name))
        count += 1
    return count


def _connectivity_table(model, attr: str, table_name: str) -> pd.DataFrame:
    table = getattr(model, attr, None)
    if table is None and hasattr(model, "_get_table_as_dataframe"):
        try:
            table = model._get_table_as_dataframe(table_name)
        except Exception:
            table = None
    required = ["UniqueName", "UniquePtI", "UniquePtJ"]
    if table is None or table.empty or not all(col in table.columns for col in required):
        return pd.DataFrame(columns=required)
    return table[required].copy()


def _add_shell_surfaces(gmsh, shells_df, point_tags: dict[str, int], *, prefix: str) -> int:
    if shells_df is None or shells_df.empty:
        return 0

    point_cols = [col for col in ("UniquePt1", "UniquePt2", "UniquePt3", "UniquePt4") if col in shells_df.columns]
    if len(point_cols) < 3:
        return 0

    groups = shells_df.groupby("UniqueName", sort=False) if "UniqueName" in shells_df.columns else [(None, shells_df)]

    count = 0
    for name, group in groups:
        pts = []
        for _, row in group.iterrows():
            pts.extend(_point_key(row[col]) for col in point_cols)
        pts = [pt for pt in pts if pt and pt in point_tags]
        pts = _dedupe(pts)
        if len(pts) < 3:
            continue

        try:
            lines = []
            for i, pt in enumerate(pts):
                p1 = point_tags[pt]
                p2 = point_tags[pts[(i + 1) % len(pts)]]
                if p1 == p2:
                    raise ValueError("repeated point")
                lines.append(gmsh.model.occ.addLine(p1, p2))
            loop = gmsh.model.occ.addCurveLoop(lines)
            surface = gmsh.model.occ.addPlaneSurface([loop])
            gmsh.model.setEntityName(2, surface, f"{prefix}_{name}" if name is not None else prefix)
            count += 1
        except Exception:
            continue
    return count


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


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
