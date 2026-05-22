"""Polygon helpers for section analysis.

Polygons are stored as ordered numpy arrays of shape (N, 2). Vertices
counter-clockwise yield positive area (Gauss-Green).
"""
from __future__ import annotations
import numpy as np


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(poly: np.ndarray) -> tuple[float, float]:
    x = poly[:, 0]
    y = poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = 0.5 * cross.sum()
    if abs(a) < 1e-12:
        return float(x.mean()), float(y.mean())
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * a)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * a)
    return float(cx), float(cy)


def clip_above_y(poly: np.ndarray, y_cut: float) -> np.ndarray:
    """Return the portion of `poly` whose y >= y_cut.

    Sutherland-Hodgman polygon clipping against the half-plane y >= y_cut.
    Used to integrate the compression zone of a section above the
    Whitney stress-block boundary.
    """
    if len(poly) < 3:
        return np.empty((0, 2))

    output: list[np.ndarray] = []
    n = len(poly)
    for i in range(n):
        curr = poly[i]
        prev = poly[i - 1]
        curr_in = curr[1] >= y_cut
        prev_in = prev[1] >= y_cut
        if curr_in:
            if not prev_in:
                output.append(_intersect_y(prev, curr, y_cut))
            output.append(curr)
        elif prev_in:
            output.append(_intersect_y(prev, curr, y_cut))
    if not output:
        return np.empty((0, 2))
    return np.asarray(output)


def _intersect_y(p1: np.ndarray, p2: np.ndarray, y_cut: float) -> np.ndarray:
    dy = p2[1] - p1[1]
    if abs(dy) < 1e-12:
        return p1.copy()
    t = (y_cut - p1[1]) / dy
    return np.array([p1[0] + t * (p2[0] - p1[0]), y_cut])


def rectangle(b: float, h: float, *, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """CCW rectangle centered on `center` with width b and height h."""
    cx, cy = center
    return np.array([
        [cx - b / 2, cy - h / 2],
        [cx + b / 2, cy - h / 2],
        [cx + b / 2, cy + h / 2],
        [cx - b / 2, cy + h / 2],
    ])
