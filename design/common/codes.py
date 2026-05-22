"""Code reference helpers.

Used to annotate where a formula or check comes from. Renders nicely on
DesignReport tables and helps the reader verify against the standard.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ACI:
    section: str            # e.g. "22.5.5.1"
    equation: str | None = None   # e.g. "22.5.5.1"
    description: str = ""

    def __str__(self) -> str:
        if self.equation:
            return f"ACI 318-25 §{self.section} (Eq. {self.equation})"
        return f"ACI 318-25 §{self.section}"
