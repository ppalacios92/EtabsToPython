"""Beam-column joint provisions per ACI 318-25 §18.8 and §15.5.2.

Pure functions — no element state. The joint class in ``joint.py``
composes these with the assembled Column / Beam objects.

Clause map:
    §21.2.4.4    phi = 0.85 for beam-column joint shear
    §18.8.2.3    minimum joint depth (column dimension along joint shear)
    Table 18.8.4.3   nominal joint shear strength Vn = gamma*lambda*sqrt(fc')*Aj
    §15.5.2.2    effective joint area Aj (with beam eccentricity)
    §15.5.2.5    confinement by transverse beams (3/4-width rule)
    §18.8.5.1    development length of hooked bars, ldh
"""
from __future__ import annotations

# ACI 318-25 §21.2.4.4 — strength reduction factor for joint shear.
PHI_JOINT_SHEAR = 0.85

# SI conversion of the psi-based table coefficients: the table gives
# Vn = gamma*lambda*sqrt(fc')*Aj with sqrt(fc') in psi. In MPa the same
# expression carries the 1/12.04 ~ 0.083 factor.
_PSI_TO_MPA = 0.083


def phi_joint_shear() -> float:
    """ACI 318-25 §21.2.4.4 — phi for beam-column joint shear."""
    return PHI_JOINT_SHEAR


def gamma_joint(
    *,
    column_continuous: bool,
    beam_continuous: bool,
    transverse_confined: bool,
) -> float:
    """Joint shear coefficient gamma (psi units). ACI 318-25 Table 18.8.4.3.

    Three binary axes, evaluated for ONE principal direction of joint
    shear:
        column_continuous   — column continuous through the joint or
                               meets §15.5.2.3 (extension >= h).
        beam_continuous      — beams continuous across the joint in the
                               direction of Vu, or meeting §15.5.2.4.
        transverse_confined  — joint confined by transverse beams per
                               §15.5.2.5.

    Returns gamma in {20, 15, 12, 8} (psi coefficient). The caller
    applies the SI factor and lambda.
    """
    table = {
        (True,  True,  True):  20.0,
        (True,  True,  False): 15.0,
        (True,  False, True):  15.0,
        (True,  False, False): 12.0,
        (False, True,  True):  15.0,
        (False, True,  False): 12.0,
        (False, False, True):  12.0,
        (False, False, False): 8.0,
    }
    return table[(column_continuous, beam_continuous, transverse_confined)]


def effective_joint_area(
    *,
    h_joint: float,
    b_column: float,
    b_beam: float,
    eccentricity: float = 0.0,
    gross_area: float | None = None,
) -> tuple[float, float]:
    """Effective joint area Aj. ACI 318-25 §15.5.2.2.

    All lengths in mm; returns ``(Aj, b_eff)`` in (mm2, mm).

    Geometry, for the direction of joint shear considered:
        h_joint   — column depth ALONG the joint-shear direction
                    (= joint depth h).
        b_column  — column width PERPENDICULAR to the beam.
        b_beam    — beam width.
        eccentricity — offset of the beam axis from the column centroid,
                    measured perpendicular to the beam axis (mm).

    Rule:
        - beam wider than column  ->  b_eff = b_column (full width).
        - column wider than beam  ->  b_eff = min(b_beam + h_joint, 2*x),
          where x = b_column/2 - |e| is the distance from the beam axis
          to the NEAREST column face. With e: 2*x = b_column - 2|e|.

    Aj never exceeds the column gross area (R15.5.2.2).
    """
    e = abs(eccentricity)
    if b_beam >= b_column:
        b_eff = b_column
    else:
        b_eff = min(b_beam + h_joint, b_column - 2.0 * e)
    b_eff = max(b_eff, 0.0)
    Aj = h_joint * b_eff
    if gross_area is not None:
        Aj = min(Aj, gross_area)
    return Aj, b_eff


def joint_shear_capacity(
    *,
    gamma: float,
    lam: float,
    sqrt_fc: float,
    Aj: float,
    reduced: bool = True,
) -> float:
    """Joint shear strength in kN. ACI 318-25 §18.8.4.3 + Table.

        Vn = gamma * 0.083 * lambda * sqrt(fc') * Aj         [N]
        phi*Vn with phi = 0.85 (§21.2.4.4) when ``reduced``.

    ``sqrt_fc`` is sqrt(fc') in MPa, ``Aj`` in mm2.
    """
    Vn = gamma * _PSI_TO_MPA * lam * sqrt_fc * Aj   # N
    if reduced:
        Vn *= PHI_JOINT_SHEAR
    return Vn / 1000.0   # kN


def min_joint_depth(
    *,
    db_grade60: float = 0.0,
    db_grade80: float = 0.0,
    h_beam_max: float = 0.0,
    lam: float = 1.0,
) -> float:
    """Minimum column depth along the joint shear. ACI 318-25 §18.8.2.3.

    Greatest of:
        (a) (20/lambda) * db   (largest Grade 60 longitudinal bar)
        (b) 26 * db            (largest Grade 80 longitudinal bar)
        (c) h_beam / 2         (any beam generating joint shear)
    All in mm.
    """
    a = (20.0 / lam) * db_grade60 if db_grade60 > 0 else 0.0
    b = 26.0 * db_grade80 if db_grade80 > 0 else 0.0
    c = 0.5 * h_beam_max
    return max(a, b, c)


def ldh_hook(*, fy: float, db: float, lam: float, sqrt_fc: float) -> float:
    """Hooked-bar development length ldh. ACI 318-25 §18.8.5.1.

        ldh = fy * db / (65 * lambda * sqrt(fc'))   [mm]
        ldh >= max(8 db, 150 mm)  (normalweight)
             >= max(10 db, 190 mm) (lightweight, lam < 1)

    fy, sqrt_fc in MPa; db in mm.
    """
    ldh = fy * db / (65.0 * lam * sqrt_fc)
    if lam >= 1.0:
        floor = max(8.0 * db, 150.0)
    else:
        floor = max(10.0 * db, 190.0)
    return max(ldh, floor)


def transverse_beam_confines(
    *,
    transverse_widths: tuple[float, ...],
    column_face_width: float,
    n_faces_required: int = 2,
) -> bool:
    """Confinement by transverse beams. ACI 318-25 §15.5.2.5 (width rule).

    Returns True when transverse beams cover BOTH perpendicular faces and
    each beam covers at least 3/4 of the column face width it frames into.

    ``transverse_widths`` are the widths of the present transverse beams
    (mm); a missing face contributes nothing. This checks the §15.5.2.5(a)
    3/4-width condition; the depth/area conditions (b)-(d) are the
    designer's responsibility and flagged in the joint notes.
    """
    if len(transverse_widths) < n_faces_required or column_face_width <= 0:
        return False
    return all(w >= 0.75 * column_face_width for w in transverse_widths)
