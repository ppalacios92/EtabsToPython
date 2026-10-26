"""Shared base class for the design elements.

``Beam``, ``Column``, ``Wall`` and ``BeamColumnJoint`` all implement the
same external contract (see ``design/AGENTS.md`` §3). Most of that surface
is genuinely element-specific — ``run``/``capacity``/``check``/``design``/
``evolve``/``report``/``summary``/``plot`` each encode different physics, so
they live on the concrete classes. Only two pieces of the contract are
*identical* everywhere, and those are factored here so there is a single
source of truth:

    * :meth:`apply`     — a back-compat alias of ``evolve`` (immutable).
    * :meth:`set_units` — swap the presentation :class:`UnitSystem`.

Inheriting from :class:`ElementBase` therefore removes four copies of each
method and, in particular, fixes the historical ``BeamColumnJoint.set_units``
which stored the raw code instead of the resolved :class:`UnitSystem`.

The runtime-checkable ``Element`` protocol (``design.common.contracts``)
still drives ``isinstance`` checks via duck-typing; subclasses do not need
to be registered.
"""
from __future__ import annotations

from design.common.units import units_from


class ElementBase:
    """Mixin with the parts of the element contract shared verbatim.

    Subclasses (``Beam``/``Column``/``Wall``/``BeamColumnJoint``) MUST
    implement the element-specific surface themselves:

        run() capacity() check(demands) design(demands=None)
        evolve(proposal) summary() report() and the ``plot`` property.

    They inherit ``apply`` and ``set_units`` from here.
    """

    # Subclasses set ``self.units`` in their ``__init__``.
    units: object
    label: str

    def apply(self, proposal):
        """Alias of :meth:`evolve` — returns a NEW element (immutable).

        Kept for back-compat and because it reads well at call sites
        (``element.apply(results.envelope)``). It never mutates ``self``.
        """
        return self.evolve(proposal)

    def set_units(self, code_or_name) -> "ElementBase":
        """Change the presentation units; returns ``self`` for chaining.

        Accepts an ETABS unit code (1..16), a unit-system name, or a
        :class:`UnitSystem`. The value is resolved to a canonical
        :class:`UnitSystem` so every element stores the same type.
        """
        self.units = units_from(code_or_name)
        return self
