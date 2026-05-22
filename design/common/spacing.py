"""Confinement / hoop-spacing limits shared by beams and columns.

ACI 318-25 references:
    §18.6.4.4   — SMF beam hoop spacing in confined region
    §18.6.4.6   — SMF beam hoop spacing outside confined region
    §18.7.5.1   — Column confined length lo
    §18.7.5.2   — hx limit
    §18.7.5.3   — Column hoop spacing in confined region
    §18.7.5.4   — Required Ash
    §18.7.5.5   — Column hoop spacing in middle (non-confined) zone

Existing element modules (``beams/shear.py``, ``columns/confinement.py``)
may continue to re-export these symbols for backwards compatibility, but
the canonical home is here.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Confined-length lo
# ---------------------------------------------------------------------------
def lo_length_beam(*, h_member: float) -> float:
    """Confined region length above and below joints for SMF beams.

    ACI 318-25 §18.6.4.1: lo >= 2 * h_member.
    """
    return 2.0 * h_member


def lo_length_column(*, h_member: float, lu_clear_mm: float) -> float:
    """Confined region length above and below joints for SMF columns.

    ACI 318-25 §18.7.5.1: lo >= max(h_member, lu_clear/6, 450 mm).

    ``lu_clear_mm`` is the clear column height in **mm**.
    """
    return max(h_member, lu_clear_mm / 6.0, 450.0)


# ---------------------------------------------------------------------------
# Hoop spacing: beams (§18.6.4)
# ---------------------------------------------------------------------------
def s_max_seismic_smf_beam(
    *,
    d: float,
    db_long_min: float,
    db_hoop: float,
    grade: int = 60,
) -> float:
    """Max hoop spacing in the confined region of SMF beams. §18.6.4.4.

    Grade 60: s <= min(d/4, 6 db_long, 150 mm).
    Grade 80+: factor 5 instead of 6.
    """
    factor = 6.0 if grade <= 60 else 5.0
    return min(d / 4.0, factor * db_long_min, 150.0)


def s_max_seismic_outside_beam(*, d: float) -> float:
    """Max hoop spacing outside the confined zone of SMF beams. §18.6.4.6."""
    return d / 2.0


# ---------------------------------------------------------------------------
# Hoop spacing: columns (§18.7.5)
# ---------------------------------------------------------------------------
def s_max_confined_column(
    *,
    h_min: float,
    db_long_min: float,
    hx: float,
    grade: int = 60,
) -> float:
    """Maximum hoop spacing in the confined zone. ACI 318-25 §18.7.5.3.

        s <= min(h_min/4, 6 db_long (Gr60) or 5 db_long (Gr80), so)
        so = 100 + (350 - hx)/3  [mm], clamped to [100, 150].
    """
    factor = 6.0 if grade <= 60 else 5.0
    so = 100.0 + (350.0 - hx) / 3.0
    so = max(100.0, min(so, 150.0))
    return min(h_min / 4.0, factor * db_long_min, so)


def s_max_middle_zone_column(*, db_long_min: float, grade: int = 60) -> float:
    """Hoop spacing outside the confined zone of SMF columns. §18.7.5.5.

        s <= min(6 db_long (Gr60) or 5 db_long (Gr80), 150 mm)
    """
    factor = 6.0 if grade <= 60 else 5.0
    return min(factor * db_long_min, 150.0)


# ---------------------------------------------------------------------------
# hx limit and Ash
# ---------------------------------------------------------------------------
def hx_check(
    *,
    b_core: float,
    n_legs: int,
    limit: float = 350.0,
) -> tuple[float, bool]:
    """Horizontal spacing between longitudinal bars supported by hoop legs.

    Returns ``(hx, ok)``. ``hx <= 350 mm`` per ACI 318-25 §18.7.5.2
    (450 mm is allowed outside plastic-hinge zones, but we use the
    strict value here).
    """
    if n_legs <= 1:
        return b_core, b_core <= limit
    hx = b_core / (n_legs - 1)
    return hx, hx <= limit


def ash_required(
    *,
    s: float,
    bc: float,
    Ag: float,
    Ach: float,
    fc: float,
    fyt: float,
    Pu_over_Ag_fc: float = 0.0,
) -> float:
    """Required area of confining reinforcement Ash in one direction.

    ACI 318-25 §18.7.5.4 (Table) — the larger of (i) and (ii):

        (i)  Ash = 0.3 * (Ag/Ach - 1) * s * bc * fc / fyt
        (ii) Ash = 0.09 * s * bc * fc / fyt

    A third equation applies for high axial load (Pu / (Ag*fc') > 0.3)
    in some configurations; passed in via ``Pu_over_Ag_fc``.
    """
    ash_i = 0.3 * (Ag / Ach - 1.0) * s * bc * fc / fyt
    ash_ii = 0.09 * s * bc * fc / fyt
    ash = max(ash_i, ash_ii)
    if Pu_over_Ag_fc > 0.3:
        ash_iii = 0.2 * (Pu_over_Ag_fc) * s * bc * fc / fyt
        ash = max(ash, ash_iii)
    return ash
