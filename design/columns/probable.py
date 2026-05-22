"""Re-export — canonical implementation lives in :mod:`design.common.probable`.

This shim exists so existing imports keep working::

    from design.columns.probable import probable_interaction_diagram, mpr_envelope

New code should import from ``design.common.probable`` directly.
"""
from design.common.probable import (
    probable_interaction_diagram,
    mpr_envelope,
)

__all__ = ["probable_interaction_diagram", "mpr_envelope"]
