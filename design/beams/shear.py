"""Re-export shim — canonical shear/spacing live in :mod:`design.common`.

Beams now consume :mod:`design.common.shear` (Vc/Vs/Av-s) and
:mod:`design.common.spacing` (hoop spacing limits). This module is kept
as a thin re-export layer for backward compatibility — existing imports
such as ``from design.beams.shear import vc_simplified`` continue to
work, but the canonical home is :mod:`design.common`.

Local helper :func:`vn_beam` (= Vc + Vs capped) is kept here as it is a
simple composition with no element-specific logic.

ACI 318-25 references:
    §22.5         — one-way shear, Vc/Vs/Av_s
    §9.6.3.4      — minimum transverse reinforcement
    §18.6.4.4     — SMF beam hoop spacing in confined region
    §18.6.4.6     — SMF beam hoop spacing outside confined region
"""
from __future__ import annotations

from design.common.materials import Concrete
from design.common.shear import (
    vc_simplified,
    vc_detailed,
    vs_capacity,
    vs_max,
    av_s_required,
    av_s_minimum,
    vc_zero_seismic,
)
from design.common.spacing import (
    s_max_seismic_smf_beam as s_max_seismic_smf,
    s_max_seismic_outside_beam as s_max_seismic_outside,
)


def vn_beam(*, b: float, d: float, Av: float, fyt: float, s: float,
            concrete: Concrete) -> float:
    """Nominal shear strength Vn = Vc + Vs (capped by Vs,max). Returns kN.

    Composition kept here — wraps the canonical helpers in
    :mod:`design.common.shear`.
    """
    Vc = vc_simplified(b=b, d=d, concrete=concrete)
    Vs = min(
        vs_capacity(Av=Av, fyt=fyt, d=d, s=s),
        vs_max(b=b, d=d, concrete=concrete),
    )
    return Vc + Vs


__all__ = [
    "vc_simplified",
    "vc_detailed",
    "vs_capacity",
    "vs_max",
    "vn_beam",
    "av_s_required",
    "av_s_minimum",
    "vc_zero_seismic",
    "s_max_seismic_smf",
    "s_max_seismic_outside",
]
