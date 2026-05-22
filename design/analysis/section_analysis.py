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
    Mn: float           # nominal moment about section x-axis (kN·m)
    phi: float          # strength-reduction factor (-)
    phi_Pn: float
    phi_Mn: float
    eps_t: float        # strain in extreme tension steel (-)


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

    poly_list = section.polygons()
    bars = section.rebar.as_arrays()
    x_bar, y_bar = bars.x, bars.y
    a_bar, fy, Es = bars.area, bars.fy, bars.Es

    if angle_deg != 0.0:
        c_a = np.cos(np.deg2rad(angle_deg))
        s_a = np.sin(np.deg2rad(angle_deg))
        rot = np.array([[c_a, s_a], [-s_a, c_a]])
        poly_list = [p @ rot.T for p in poly_list]
        coords = np.stack([x_bar, y_bar], axis=1) @ rot.T
        x_bar, y_bar = coords[:, 0], coords[:, 1]

    y_top = max(p[:, 1].max() for p in poly_list)
    a = beta * c
    y_cut = y_top - a

    Cc = 0.0
    Mc = 0.0
    for poly in poly_list:
        clipped = clip_above_y(poly, y_cut)
        if len(clipped) < 3:
            continue
        Ac = polygon_area(clipped)
        if Ac <= 0:
            continue
        _, ycc = polygon_centroid(clipped)
        Cc += 0.85 * fc * Ac
        Mc += 0.85 * fc * Ac * ycc

    eps_s = eps_cu * ((y_top - c) - y_bar) / c  # +tension, -compression
    sigma = np.clip(Es * eps_s, -fy, fy)         # per-bar Es and fy
    Fs = sigma * a_bar
    Ms = Fs * y_bar

    Cc_kN = Cc / 1000.0
    Fs_kN = Fs / 1000.0

    # Net axial: compression of concrete minus tensile pull of bars
    Pn_kN = Cc_kN - Fs_kN.sum()
    Mn_kNm = (Mc - Ms.sum()) / 1e6

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
    """
    h = section.height()
    if c_min is None:
        c_min = 0.001 * h
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
