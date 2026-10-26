"""Distributed web reinforcement of structural walls. ACI 318-25 §18.10.2 / §11.6 / §11.7."""
from __future__ import annotations
from math import sqrt, pi

from design.common.factors import phi_shear


def rho_min_distributed(
    *,
    Vu: float,
    Acv: float,
    fc: float,
    lam: float = 1.0,
    special: bool = False,
    db: float = 16.0,
    fy: float = 420.0,
) -> tuple[float, float]:
    """Return (rho_l_min, rho_t_min).

    At high shear (``Vu > 0.083 lambda sqrt(fc) Acv``), §11.6.2 requires
    0.0025 in both directions.  At lower shear, Table 11.6.1 supplies the
    low-shear ratios for cast-in-place walls with deformed bars.  Special
    structural walls use those Table 11.6.1 values only through the
    §18.10.2.1 low-shear exception; otherwise their minimum is 0.0025.
    """
    threshold_high = 0.083 * lam * sqrt(fc) * Acv / 1000.0
    if Vu > threshold_high:
        return 0.0025, 0.0025
    threshold_low = 0.04 * phi_shear() * lam * sqrt(fc) * Acv / 1000.0
    if Vu <= threshold_low:
        if db <= 16.0 and fy >= 420.0:
            return 0.0012, 0.0020
        return 0.0015, 0.0025
    # This is the remaining low-shear Table 11.6.1 range.  For special
    # walls it is permitted by the §18.10.2.1 exception; for other walls it
    # is the table value as well.
    return 0.0015, 0.0025


def web_bar_spacing_max(*, lw: float, tw: float) -> float:
    """Maximum spacing of distributed bars. §11.7.2.1 / §11.7.3.1.

    s_max = min(lw/5, 3*tw, 450 mm).
    """
    return min(lw / 5.0, 3.0 * tw, 450.0)


def two_curtain_required(
    *,
    Vu: float,
    Acv: float,
    fc: float,
    lam: float,
    hw_over_lw: float,
    tw: float,
) -> bool:
    """Two curtains of reinforcement required if any of:
        - Vu > 0.17 lambda sqrt(fc) Acv     (§18.10.2.2)
        - hw/lw >= 2.0                       (§18.10.2.2)
        - tw >= 250 mm                       (§11.7.2.3)
    """
    threshold = 0.17 * lam * sqrt(fc) * Acv / 1000.0
    return (Vu > threshold) or (hw_over_lw >= 2.0) or (tw >= 250.0)


def web_bars_for_rho(
    *,
    rho: float,
    tw: float,
    db: float,
    layers: int = 2,
) -> float:
    """Spacing required to achieve a target rho with bars of diameter `db`.

    rho = (n_layers * A_bar) / (tw * s)  =>  s = n_layers * A_bar / (rho * tw)
    """
    if rho <= 0:
        return float("inf")
    a_bar = pi * db ** 2 / 4.0
    return layers * a_bar / (rho * tw)


def web_rho_provided(*, tw: float, db: float, spacing: float, layers: int = 2) -> float:
    """Return rho for a given (db, spacing, layers) detailing."""
    if spacing <= 0:
        return 0.0
    a_bar = pi * db ** 2 / 4.0
    return layers * a_bar / (tw * spacing)
