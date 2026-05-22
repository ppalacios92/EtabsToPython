"""Coupling beams. ACI 318-25 §18.10.7."""
from __future__ import annotations
from dataclasses import dataclass
from math import sin, radians, sqrt
from typing import Literal


CouplingType = Literal["conventional", "diagonal", "either"]


@dataclass(frozen=True, slots=True)
class CouplingBeamClassification:
    ln_over_h: float
    type_required: CouplingType
    diagonal_mandatory: bool
    notes: tuple[str, ...]


def coupling_beam_classification(
    *,
    ln: float,
    h: float,
    Vu: float,
    Acw: float,
    fc: float,
    lam: float = 1.0,
) -> CouplingBeamClassification:
    """Classify a coupling beam per §18.10.7.1-3.

    - ln/h >= 4: design as SMF beam (§18.6). type = "conventional".
    - ln/h < 2 and Vu > 0.33 lambda sqrt(fc') Acw: diagonal reinforcement
      mandatory (§18.10.7.2).
    - intermediate or low-shear cases: designer's choice.
    """
    ln_over_h = ln / h if h > 0 else float("inf")
    notes: list[str] = []
    Vu_threshold = 0.33 * lam * sqrt(fc) * Acw / 1000.0  # kN

    if ln_over_h >= 4.0:
        notes.append("Design as SMF beam per §18.6.")
        return CouplingBeamClassification(
            ln_over_h=ln_over_h,
            type_required="conventional",
            diagonal_mandatory=False,
            notes=tuple(notes),
        )
    if ln_over_h < 2.0 and Vu > Vu_threshold:
        notes.append("Diagonal reinforcement mandatory: ln/h < 2 and Vu > 0.33 λ√f'c Acw.")
        return CouplingBeamClassification(
            ln_over_h=ln_over_h,
            type_required="diagonal",
            diagonal_mandatory=True,
            notes=tuple(notes),
        )
    notes.append("Either conventional or diagonal allowed; check serviceability.")
    return CouplingBeamClassification(
        ln_over_h=ln_over_h,
        type_required="either",
        diagonal_mandatory=False,
        notes=tuple(notes),
    )


def vn_diagonal_coupling(
    *,
    Avd: float,
    fy: float,
    alpha_deg: float,
    fc: float,
    Acw: float,
    lam: float = 1.0,
) -> tuple[float, float]:
    """Nominal shear of diagonally reinforced coupling beam. §18.10.7.4.

    Vn = 2 * Avd * fy * sin(alpha)  <=  0.83 * sqrt(fc) * Acw

    Returns (Vn, Vn_cap) in kN. Avd is total diagonal area in ONE group.
    """
    a_rad = radians(alpha_deg)
    Vn = 2.0 * Avd * fy * sin(a_rad) / 1000.0
    Vn_cap = 0.83 * lam * sqrt(fc) * Acw / 1000.0
    return min(Vn, Vn_cap), Vn_cap


class CouplingBeam:
    """High-level coupling beam wrapper for design/check workflows."""
    __slots__ = ("ln", "h", "b", "Acw", "concrete", "steel")

    def __init__(self, *, ln: float, h: float, b: float,
                 concrete, steel) -> None:
        self.ln = ln
        self.h = h
        self.b = b
        self.Acw = h * b
        self.concrete = concrete
        self.steel = steel

    def classify(self, Vu: float) -> CouplingBeamClassification:
        return coupling_beam_classification(
            ln=self.ln, h=self.h, Vu=Vu, Acw=self.Acw,
            fc=self.concrete.fc, lam=self.concrete.lam,
        )

    def diagonal_capacity(self, *, Avd: float, alpha_deg: float) -> float:
        Vn, _ = vn_diagonal_coupling(
            Avd=Avd, fy=self.steel.fy, alpha_deg=alpha_deg,
            fc=self.concrete.fc, Acw=self.Acw, lam=self.concrete.lam,
        )
        return Vn
