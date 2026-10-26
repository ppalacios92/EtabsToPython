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
    nl: int | None = None,
) -> float:
    """Required area of confining reinforcement Ash in one direction.

    ACI 318-25 §18.7.5.4, Table 18.7.5.4 (rectilinear hoops) — the
    greatest of the applicable expressions:

        (a) Ash = 0.3 * (Ag/Ach - 1) * s * bc * fc' / fyt
        (b) Ash = 0.09 * s * bc * fc' / fyt
        (c) Ash = 0.2 * kf * kn * Pu * s / (fyt * Ach)

    Expression (c) is triggered ONLY for high axial load
    (Pu/(Ag·fc') > 0.3) or high-strength concrete (fc' > 69 MPa), with

        kf = fc'/172.4 + 0.6 >= 1.0      (Eq. 18.7.5.4a)
        kn = nl/(nl - 2)                  (Eq. 18.7.5.4b)

    Expressions (a) and (b) are NOT multiplied by kf. ``Pu_over_Ag_fc``
    is the axial-load ratio Pu/(Ag·fc'); ``nl`` is the number of
    longitudinal bars laterally supported around the core perimeter
    (kn = 1 if not provided — conservative).
    """
    ash_a = 0.3 * (Ag / Ach - 1.0) * s * bc * fc / fyt
    ash_b = 0.09 * s * bc * fc / fyt
    ash = max(ash_a, ash_b)
    if Pu_over_Ag_fc > 0.3 or fc > 68.95:
        kf = max(fc / 172.4 + 0.6, 1.0)
        kn = (nl / (nl - 2.0)) if (nl is not None and nl > 2) else 1.0
        # Table 18.7.5.4 gives Ash/(s·bc); so Ash = expr · s · bc.
        # Pu [N] = Pu/(Ag·fc') · Ag · fc' ; (c) = 0.2·kf·kn·Pu·s·bc/(fyt·Ach)
        Pu_N = Pu_over_Ag_fc * Ag * fc
        ash_c = 0.2 * kf * kn * Pu_N * s * bc / (fyt * Ach)
        ash = max(ash, ash_c)
    return ash


def rho_s_spiral_required(
    *,
    Ag: float,
    Ach: float,
    fc: float,
    fyt: float,
    Pu_over_Ag_fc: float = 0.0,
) -> float:
    """Required volumetric ratio of spiral / circular-hoop reinforcement.

    ACI 318-25 §25.7.3.3 and Table 18.7.5.4 (spiral rows) — the greatest
    of the applicable expressions:

        (d) rho_s = 0.45 * (Ag/Ach - 1) * fc' / fyt
        (e) rho_s = 0.12 * fc' / fyt
        (f) rho_s = 0.35 * kf * Pu / (fyt * Ach)

    Expression (f) is triggered ONLY for high axial load
    (Pu/(Ag·fc') > 0.3) or high-strength concrete (fc' > 69 MPa), with
    kf = fc'/172.4 + 0.6 >= 1.0. ``fyt`` is capped at 689.5 MPa (100 ksi)
    per §25.7.3.3. ``Pu_over_Ag_fc`` is the axial-load ratio Pu/(Ag·fc').
    """
    fyt = min(fyt, 689.5)  # §25.7.3.3 fyt <= 100 ksi
    rho_d = 0.45 * (Ag / Ach - 1.0) * fc / fyt
    rho_e = 0.12 * fc / fyt
    rho = max(rho_d, rho_e)
    if Pu_over_Ag_fc > 0.3 or fc > 68.95:
        kf = max(fc / 172.4 + 0.6, 1.0)
        Pu_N = Pu_over_Ag_fc * Ag * fc
        rho_f = 0.35 * kf * Pu_N / (fyt * Ach)
        rho = max(rho, rho_f)
    return rho
