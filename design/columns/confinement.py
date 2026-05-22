"""Re-export shim. Canonical implementations live in ``design.common.spacing``.

Kept here for backward compatibility. ACI 318-25 §18.7.5 (confined
column detailing) is now centralized in :mod:`design.common.spacing`.
"""
from __future__ import annotations

from design.common.spacing import (
    ash_required,
    s_max_confined_column as s_max_confined,
    lo_length_column as lo_length,
    s_max_middle_zone_column as s_max_middle_zone,
    hx_check,
)

__all__ = [
    "ash_required",
    "s_max_confined",
    "lo_length",
    "s_max_middle_zone",
    "hx_check",
]
