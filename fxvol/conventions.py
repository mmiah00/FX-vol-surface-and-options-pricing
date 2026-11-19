"""FX delta conventions.

Two independent switches define an FX delta quote:

* ``delta_type``: SPOT delta is the sensitivity to spot, ``exp(-rf*T) * N(d1)``
  for a call. FORWARD delta strips the foreign discount factor, ``N(d1)``.
* ``premium_included``: when True the delta is premium-adjusted, i.e. net of
  the premium paid in the foreign (underlying) currency, ``dV/dS - V/S``.

EUR/USD default
---------------
``EURUSD_DEFAULT`` is **spot delta, premium excluded**.

Rationale: for EUR/USD the premium is paid in USD, which is the domestic
(quote) currency. The delta hedge is a position in EUR, so the premium does
not eat into the hedge amount and no premium adjustment is needed. Spot delta
is the market standard for tenors up to 1Y. Beyond 1Y, forward delta is often
used instead; this module applies a single convention across all tenors, and
a different one can be passed explicitly (``DeltaConvention(DeltaType.FORWARD,
False)``) wherever a convention is accepted.

Premium-included deltas are used for pairs where the premium currency is the
foreign currency of the pair (e.g. USD/CAD-style quoting from the USD side).
They are supported here because the solver for them is the non-trivial part.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeltaType(Enum):
    SPOT = "spot"
    FORWARD = "forward"


@dataclass(frozen=True)
class DeltaConvention:
    delta_type: DeltaType = DeltaType.SPOT
    premium_included: bool = False

    def __str__(self) -> str:
        pa = "premium-included" if self.premium_included else "premium-excluded"
        return f"{self.delta_type.value} delta, {pa}"


EURUSD_DEFAULT = DeltaConvention(DeltaType.SPOT, premium_included=False)
