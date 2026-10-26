"""Material objects. Frozen value types — once built, they don't change."""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt, pi

from design.common.factors import beta1, lambda_concrete


@dataclass(frozen=True, slots=True)
class Concrete:
    """Normal- or lightweight-concrete properties. Units: MPa, kN/m3."""
    fc: float
    unit_weight: float = 24.0
    lightweight: bool = False
    sand_lightweight: bool = False

    def __post_init__(self) -> None:
        if self.fc <= 0:
            raise ValueError(f"fc must be positive, got {self.fc}")

    @property
    def beta1(self) -> float:
        return beta1(self.fc)

    @property
    def Ec(self) -> float:
        # ACI 318-25 §19.2.2: Ec = 4700 sqrt(fc) [MPa] for normalweight
        return 4700.0 * sqrt(self.fc)

    @property
    def lam(self) -> float:
        return lambda_concrete(self.lightweight, self.sand_lightweight)

    @property
    def sqrt_fc(self) -> float:
        # ACI 318-25 §22.5.3.1: sqrt(fc') shall not exceed 8.3 MPa
        # (= 100 psi) in shear, torsion and development equations.
        return min(sqrt(self.fc), 8.3)

    @property
    def fr(self) -> float:
        # ACI 318-25 §19.2.3.1: modulus of rupture for deflection /
        # cracking, fr = 0.62 lambda sqrt(fc') [MPa]. Uses the uncapped
        # sqrt(fc') (the 8.3 MPa cap is a shear-equation rule only).
        return 0.62 * self.lam * sqrt(self.fc)

    @property
    def eps_cu(self) -> float:
        return 0.003  # §22.2.2.1


@dataclass(frozen=True, slots=True)
class Steel:
    """Reinforcement steel. Units: MPa, kg/m³."""
    fy: float
    Es: float = 200_000.0
    fu: float | None = None
    grade: int = 60          # 60, 80, 100 (ksi labels — only for reporting)
    density: float = 7850.0  # kg/m³  — mass density for weight take-offs

    def __post_init__(self) -> None:
        if self.fy <= 0:
            raise ValueError(f"fy must be positive, got {self.fy}")

    @property
    def eps_y(self) -> float:
        return self.fy / self.Es

    @property
    def eps_ty(self) -> float:
        # §21.2.2.1 — yield strain threshold for tension-controlled limit
        return self.fy / self.Es


@dataclass(frozen=True, slots=True)
class Bar:
    """Reinforcing bar. Units: mm."""
    diameter: float

    @property
    def area(self) -> float:
        return pi * self.diameter ** 2 / 4.0

    @classmethod
    def from_imperial(cls, number: int) -> "Bar":
        # #N → diameter in mm (1/8 inch increments * 25.4)
        d_inch = number / 8.0
        return cls(diameter=d_inch * 25.4)

    def __str__(self) -> str:
        return f"db{self.diameter:.0f}"
