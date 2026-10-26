"""Section analysis by strain compatibility (Whitney stress block).

Reference: ACI 318-25 §22.2 (assumptions), §22.2.2.4 (rectangular block).

Sign convention: P > 0 means compression. Positive moment about the
section's local x-axis produces compression on the +y face. The
neutral axis is a horizontal line y = y_top - c, with c measured from
the top compression fiber downward.

This module never imports anything element-specific (Beam, Column,
Wall). It returns raw PM points; element classes interpret them and
add code-specific checks.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from design.common.factors import phi_axial_flexure
from design.common.geometry import clip_above_y, polygon_area, polygon_centroid
from design.common.materials import Steel
from design.sections.base import Section


@dataclass(frozen=True, slots=True)
class PMPoint:
    c: float            # neutral-axis depth from compression fiber (mm)
    a: float            # equivalent stress block depth (mm)
    Pn: float           # nominal axial (kN, + compression)
    Mn: float           # nominal moment perpendicular to the NA (kN·m)
    phi: float          # strength-reduction factor (-)
    phi_Pn: float
    phi_Mn: float
    eps_t: float        # strain in extreme tension steel (-)
    # Resultant moment components about the section's GLOBAL axes (kN·m).
    # For doubly-symmetric sections at 0°/90° these collapse to (Mn, 0) /
    # (0, Mn); for asymmetric sections (C/L/T) the resultant is NOT aligned
    # with the NA angle, so Mx/My carry the true biaxial direction.
    Mx: float = 0.0     # moment about global X axis (= sum F·y)
    My: float = 0.0     # moment about global Y axis (= sum F·x)


@dataclass(frozen=True, slots=True)
class PMResult:
    points: tuple[PMPoint, ...]
    spiral: bool

    @property
    def Po(self) -> float:
        return max(p.Pn for p in self.points)

    @property
    def To(self) -> float:
        return min(p.Pn for p in self.points)


# ---------- rotation helpers ----------

def _rotation_matrix(angle_deg: float) -> np.ndarray:
    """Rotation that maps section-local coords into the frame where the
    neutral axis is horizontal (sweeping +y). Shared by the PM routines so
    the depth used to bound `c` is consistent with how bars/polygons rotate.
    """
    c_a = np.cos(np.deg2rad(angle_deg))
    s_a = np.sin(np.deg2rad(angle_deg))
    return np.array([[c_a, s_a], [-s_a, c_a]])


def rotated_depth(section: Section, angle_deg: float = 0.0) -> float:
    """Section depth along the (rotated) y-axis: compression fiber to
    tension fiber, measured in the direction the neutral axis sweeps.

    This is the depth that bounds the neutral-axis position `c` — NOT the
    un-rotated bounding-box height. For a wall bent out-of-plane
    (angle_deg = 90) this returns the thickness ``tw``, not the length
    ``lw``; using ``section.height()`` there over-scales ``c_max`` by
    ``lw/tw`` and starves the curved part of the diagram (ACI 318-25 §22.2).
    """
    poly_list = section.polygons()
    if angle_deg != 0.0:
        rot = _rotation_matrix(angle_deg)
        poly_list = [p @ rot.T for p in poly_list]
    y_max = max(p[:, 1].max() for p in poly_list)
    y_min = min(p[:, 1].min() for p in poly_list)
    return y_max - y_min


# ---------- core routine ----------

def pm_at_neutral_axis(
    section: Section,
    *,
    c: float,
    angle_deg: float = 0.0,
    spiral: bool = False,
) -> PMPoint:
    """Evaluate Pn, Mn for a given neutral-axis depth c at orientation `angle_deg`.

    `angle_deg` rotates the neutral axis about the centroid; 0 means a
    horizontal NA (bending about local x). For PMM volume sweeps the
    column module rotates the section through 0..180°.
    """
    fc = section.concrete.fc
    beta = section.concrete.beta1
    eps_cu = section.concrete.eps_cu

    poly_list_g = section.polygons()             # global coords
    bars = section.rebar.as_arrays()
    x_bar_g, y_bar_g = bars.x, bars.y             # global bar coords
    a_bar, fy, Es = bars.area, bars.fy, bars.Es

    if angle_deg != 0.0:
        rot = _rotation_matrix(angle_deg)
        poly_list_r = [p @ rot.T for p in poly_list_g]   # rotated, for clipping
        coords_r = np.stack([x_bar_g, y_bar_g], axis=1) @ rot.T
        x_bar_r, y_bar_r = coords_r[:, 0], coords_r[:, 1]
        c_a, s_a = float(rot[0, 0]), float(rot[0, 1])    # cos, sin
    else:
        poly_list_r = poly_list_g
        x_bar_r, y_bar_r = x_bar_g, y_bar_g
        c_a, s_a = 1.0, 0.0

    y_top = max(p[:, 1].max() for p in poly_list_r)
    a = beta * c
    y_cut = y_top - a

    Cc = 0.0
    Mc_perp = 0.0          # about rotated x' (perpendicular to NA) -> legacy Mn
    Mc_x = 0.0             # about GLOBAL X (= sum f·y_global)
    Mc_y = 0.0             # about GLOBAL Y (= sum f·x_global)
    for poly_r in poly_list_r:
        clipped = clip_above_y(poly_r, y_cut)
        if len(clipped) < 3:
            continue
        Ac = polygon_area(clipped)
        if Ac <= 0:
            continue
        xcc_r, ycc_r = polygon_centroid(clipped)         # rotated centroid
        # rotate the centroid back to global coords: p_g = p_r @ rot
        xcc_g = c_a * xcc_r - s_a * ycc_r
        ycc_g = s_a * xcc_r + c_a * ycc_r
        f = 0.85 * fc * Ac
        Cc += f
        Mc_perp += f * ycc_r
        Mc_x += f * ycc_g
        Mc_y += f * xcc_g

    # Strain uses the rotated y (depth from the rotated top fiber).
    eps_s = eps_cu * ((y_top - c) - y_bar_r) / c  # +tension, -compression
    sigma = np.clip(Es * eps_s, -fy, fy)          # per-bar Es and fy
    Fs = sigma * a_bar
    Ms_perp = (Fs * y_bar_r).sum()
    Ms_x = (Fs * y_bar_g).sum()
    Ms_y = (Fs * x_bar_g).sum()

    Cc_kN = Cc / 1000.0
    Fs_kN = Fs / 1000.0

    # Net axial: compression of concrete minus tensile pull of bars
    Pn_kN = Cc_kN - Fs_kN.sum()
    Mn_kNm = (Mc_perp - Ms_perp) / 1e6            # legacy: perpendicular to NA
    Mx_kNm = (Mc_x - Ms_x) / 1e6                  # about global X (= sum F·y)
    # Right-hand rule: moment about global Y is M_Y = -sum F·x, so a
    # symmetric square at 45° gives (Mx, My) both positive.
    My_kNm = -(Mc_y - Ms_y) / 1e6                 # about global Y

    # phi depends on the extreme-tension bar's strain and yield strain
    if len(eps_s):
        i_t = int(np.argmax(eps_s))
        eps_t = float(eps_s[i_t])
        eps_ty = float(fy[i_t] / Es[i_t])
    else:
        eps_t, eps_ty = 0.0, 0.00207
    phi = phi_axial_flexure(eps_t, eps_ty, spiral=spiral)

    return PMPoint(
        c=c, a=a,
        Pn=Pn_kN, Mn=Mn_kNm,
        phi=phi,
        phi_Pn=phi * Pn_kN, phi_Mn=phi * Mn_kNm,
        eps_t=eps_t,
        Mx=Mx_kNm, My=My_kNm,
    )


def pm_curve(
    section: Section,
    *,
    n_points: int = 60,
    c_min: float | None = None,
    c_max: float | None = None,
    angle_deg: float = 0.0,
    spiral: bool = False,
) -> PMResult:
    """Sweep `c` to build the nominal PM curve at one orientation.

    Defaults span from near-pure-tension (c -> 0) to well past full
    compression (c >> h) so the curve reaches both anchors of the
    diagram. Use geometric spacing — points cluster near the small-c
    side where the curvature of the envelope is highest.

    ``c`` is bounded by the depth measured along the *rotated* y-axis
    (``rotated_depth``), so out-of-plane / rotated sweeps (e.g. a wall at
    angle_deg = 90) sample the curved region correctly instead of
    over-scaling ``c_max`` by the un-rotated height.
    """
    h = rotated_depth(section, angle_deg)
    if c_min is None:
        c_min = max(0.001 * h, 1.0)
    if c_max is None:
        c_max = 5.0 * h
    # geometric spacing so we sample the curved tension/balanced region
    # densely and the flat compression tail sparsely
    cs = np.geomspace(c_min, c_max, n_points)
    pts: list[PMPoint] = []
    for c in cs:
        pts.append(pm_at_neutral_axis(section, c=float(c), angle_deg=angle_deg, spiral=spiral))
    return PMResult(points=tuple(pts), spiral=spiral)


# ---------- helper queries ----------

def pn_max(section: Section, *, spiral: bool = False) -> tuple[float, float]:
    """Po and phi*Pn,max per §22.4.2.1.

    Po = 0.85 fc (Ag - Ast) + fy Ast
    Pn,max = 0.80 Po (ties) or 0.85 Po (spirals)
    phi = 0.65 (ties) or 0.75 (spirals)
    """
    Ast = section.rebar.total_area
    Ag = section.gross_area()
    fy_avg = _weighted_fy(section)
    fc = section.concrete.fc
    Po = (0.85 * fc * (Ag - Ast) + fy_avg * Ast) / 1000.0  # kN
    factor = 0.85 if spiral else 0.80
    phi = 0.75 if spiral else 0.65
    Pn_max = factor * Po
    return Po, phi * Pn_max


def to_capacity(section: Section) -> float:
    """Pure tension capacity = sum(fy * As) per group, in kN."""
    total = 0.0
    for g in section.rebar.groups:
        total += g.steel.fy * g.total_area
    return total / 1000.0


def _weighted_fy(section: Section) -> float:
    total_a = section.rebar.total_area
    if total_a == 0:
        return 0.0
    return sum(g.steel.fy * g.total_area for g in section.rebar.groups) / total_a
