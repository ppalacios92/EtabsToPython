"""Shared base for ``run_optimize`` across Beam / Column / Wall.

The element-specific ``OptimizeAlternative`` dataclasses (in
``columns/optimize.py``, ``beams/optimize.py`` and ``walls/optimize.py``)
should either subclass this one or wrap its `detailing` dict pattern so
the three modules share a single ranking/printing layer.

Steel-quantity convention (kg/m³ of concrete):

    transverse:    mass of stirrups/hoops in the ``lo`` confined region,
                   per m³ of concrete (1 m slice of the member).
    longitudinal:  mass of longitudinal bars per m³ of concrete (the
                   bars run the full length).
    total:         transverse + longitudinal.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True, slots=True)
class OptimizeAlternative:
    """One detailing alternative explored by ``run_optimize``.

    Parameters
    ----------
    proposal :
        The element-specific ``XDesignProposal`` produced by ``element.design()``.
    detailing :
        Element-specific keys describing the variation (e.g.
        ``{"db_hoop": 10, "n_legs_x": 3, "n_legs_y": 3}`` for columns,
        ``{"db_top": 20, "n_top": 3, ...}`` for beams).
    rho_transverse :
        kg/m³ of concrete (lo region).
    rho_longitudinal :
        kg/m³ of concrete (full length).
    rho_total :
        Sum of the two.
    feasible :
        True if the alternative passes all code gates.
    is_baseline :
        True if this is the alternative returned by ``element.design()``.
    is_provided :
        True if this matches the as-input element (pre-``design()``).
    notes :
        Free-form notes from the proposal/design step.
    """

    proposal: object
    detailing: dict
    rho_transverse: float
    rho_longitudinal: float
    rho_total: float
    feasible: bool
    is_baseline: bool = False
    is_provided: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


def _default_detail_fmt(alt: OptimizeAlternative) -> str:
    """Best-effort generic formatter for the ``detailing`` dict."""
    if not alt.detailing:
        return ""
    parts = []
    for k, v in alt.detailing.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def format_optimize_table(
    alternatives: list,
    *,
    top_n: int = 12,
    detail_fmt: Callable | None = None,
) -> str:
    """Pretty-print a ranked list of optimize alternatives.

    Works with both :class:`OptimizeAlternative` and element-specific
    dataclasses that share its shape (``rho_*``, ``feasible``,
    ``is_baseline``, ``is_provided``, ``notes``).

    Parameters
    ----------
    alternatives :
        Already-sorted list (ascending by `rho_total` typically).
    top_n :
        How many rows to show in the main block. Baseline/provided
        entries ranked outside `top_n` are appended after a separator.
    detail_fmt :
        ``f(alt) -> str`` used for the "detail" column. If None,
        ``alt.detailing`` (if present) is rendered generically; otherwise
        a blank column is shown.
    """
    if detail_fmt is None:
        def _fmt(alt):
            if hasattr(alt, "detailing") and isinstance(alt.detailing, dict):
                return _default_detail_fmt(alt)
            return ""
        detail_fmt = _fmt

    header = (
        f"{'rank':>4}  {'detail':<30}  "
        f"{'rho_t':>8}  {'rho_l':>8}  {'rho_tot':>8}  {'tag':>22}"
    )
    units_line = (
        f"{'':>4}  {'':<30}  "
        f"{'(kg/m3)':>8}  {'(kg/m3)':>8}  {'(kg/m3)':>8}  {'':>22}"
    )
    lines = [header, units_line, "-" * len(header)]

    def _row(rank: int, a) -> str:
        det = detail_fmt(a)
        tags = []
        if getattr(a, "is_baseline", False):
            tags.append("baseline")
        if getattr(a, "is_provided", False):
            tags.append("provided")
        if not getattr(a, "feasible", True):
            tags.append("infeasible")
        tag = ", ".join(tags) if tags else ""
        return (
            f"{rank:>4}  {det:<30}  "
            f"{a.rho_transverse:>8.1f}  {a.rho_longitudinal:>8.1f}  "
            f"{a.rho_total:>8.1f}  {tag:>22}"
        )

    n = min(top_n, len(alternatives))
    for i in range(n):
        lines.append(_row(i + 1, alternatives[i]))

    # Always show provided/baseline ranked outside top_n
    extras = [
        (i, a) for i, a in enumerate(alternatives)
        if i >= n and (getattr(a, "is_provided", False)
                       or getattr(a, "is_baseline", False))
    ]
    if extras:
        lines.append("." * len(header))
        for orig_rank, a in extras:
            lines.append(_row(orig_rank + 1, a))
    return "\n".join(lines)
