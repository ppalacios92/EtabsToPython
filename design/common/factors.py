"""ACI 318-25 strength reduction and material factors."""
from __future__ import annotations


def beta1(fc: float) -> float:
    """Whitney stress-block depth ratio. ACI 318-25 Table 22.2.2.4.3 (MPa)."""
    if fc <= 28.0:
        return 0.85
    if fc >= 56.0:
        return 0.65
    return 0.85 - 0.05 * (fc - 28.0) / 7.0


def lambda_concrete(lightweight: bool, sand_lightweight: bool) -> float:
    """Modification factor for lightweight concrete. ACI 318-25 §19.2.4."""
    if not lightweight:
        return 1.0
    if sand_lightweight:
        return 0.85
    return 0.75


def phi_axial_flexure(
    eps_t: float,
    eps_ty: float,
    *,
    spiral: bool = False,
) -> float:
    """Strength reduction factor for combined axial-flexural sections.

    ACI 318-25 §21.2.2 (Table 21.2.2). Linear interpolation between
    compression-controlled (eps_t <= eps_ty) and tension-controlled
    (eps_t >= eps_ty + 0.003).
    """
    phi_c = 0.75 if spiral else 0.65
    phi_t = 0.90
    if eps_t <= eps_ty:
        return phi_c
    if eps_t >= eps_ty + 0.003:
        return phi_t
    return phi_c + (phi_t - phi_c) * (eps_t - eps_ty) / 0.003


def phi_shear() -> float:
    """ACI 318-25 §21.2.1 — shear and torsion."""
    return 0.75


def phi_bearing() -> float:
    return 0.65


def phi_compression_strut_tie() -> float:
    return 0.75


# ---------------------------------------------------------------------------
# Capacity-design shear amplification constants
# ---------------------------------------------------------------------------
OMEGA_0_COLUMN_DEFAULT = 3.0   # §18.7.6.1.1 — column overstrength factor
OMEGA_V_WALL_DEFAULT = 1.5     # §18.10.3.1.2 — wall shear amplifier (simplified)


def omega_v_wall(
    *,
    n_stories: int | None = None,
    hw_over_lw: float | None = None,
) -> float:
    """Capacity-design shear amplifier for walls. ACI 318-25 §18.10.3.1.2.

    Default: 1.5. Drops to 1.0 for very short walls (n_stories <= 2 AND
    hw/lw <= 2). The strict height-dependent form is left to the caller
    via manual override.
    """
    if n_stories is not None and hw_over_lw is not None:
        if n_stories <= 2 and hw_over_lw <= 2.0:
            return 1.0
    return OMEGA_V_WALL_DEFAULT
