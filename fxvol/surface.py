"""Vol surface across tenors: per-tenor smiles joined by linear interpolation in total variance.

Total variance ``w = sigma^2 * T`` is what is interpolated, not vol. Flat forward variance
between pillars is the natural "no information" assumption, and total variance is the
quantity whose monotonicity in T rules out calendar arbitrage.

Interpolation coordinate
------------------------
Between pillars ``T_j < T < T_{j+1}`` the surface is

    w(k, T) = w_j(k) + (T - T_j)/(T_{j+1} - T_j) * (w_{j+1}(k) - w_j(k)),    k = ln(K / F_T)

i.e. at fixed **log forward-moneyness** ``k``, not fixed strike. The no-calendar-arbitrage
condition (Gatheral-Jacquier) is monotonicity of ``w`` in ``T`` at fixed ``k``; with a
carry (``rd != rf``) the forward moves with T, so a fixed *strike* is a different moneyness at
each tenor. :mod:`fxvol.arbitrage` checks both coordinates.

Extrapolation in T: before the first pillar and after the last, the vol at fixed ``k`` is held
flat (so ``w`` scales linearly with T there).

Rates: each pillar carries its own ``rd`` and ``rf`` (continuously compounded zero rates).
Between pillars the *integrated* rates ``r*T`` are interpolated linearly in T; beyond the last
pillar the last rate is held constant, and before the first, the first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .conventions import EURUSD_DEFAULT, DeltaConvention
from .smile import AtmType, Smile, SmileQuote, build_smile


@dataclass(frozen=True)
class TenorQuote:
    """Market quotes and rates for one tenor. Vols are decimals (RR = -0.005 is -0.5 vol)."""

    T: float
    rd: float
    rf: float
    atm: float
    rr: float
    bf: float


class VolSurface:
    def __init__(
        self,
        S: float,
        tenors: Sequence[TenorQuote],
        model: str = "malz",
        strangle: str = "smile",
        wing_delta: float = 0.25,
        convention: DeltaConvention = EURUSD_DEFAULT,
        atm_type: AtmType = AtmType.DELTA_NEUTRAL,
    ):
        tenors = sorted(tenors, key=lambda t: t.T)
        Ts = np.array([t.T for t in tenors])
        if len(tenors) < 1 or np.any(np.diff(Ts) <= 0) or Ts[0] <= 0:
            raise ValueError("tenors must be non-empty with strictly increasing positive T")
        self.S, self.model, self.strangle = S, model, strangle
        self.convention, self.atm_type, self.wing_delta = convention, atm_type, wing_delta
        self.tenors = tenors
        self.T = Ts
        self._int_rd = np.array([t.rd * t.T for t in tenors])
        self._int_rf = np.array([t.rf * t.T for t in tenors])
        self.smiles: list[Smile] = [
            build_smile(
                SmileQuote(S, t.T, t.rd, t.rf, t.atm, t.rr, t.bf, wing_delta, convention, atm_type),
                model,
                strangle,
            )
            for t in tenors
        ]

    # ------------------------------------------------------------------ rates / forwards
    def rates(self, T: float) -> tuple[float, float]:
        """(rd, rf) at maturity T from linearly interpolated integrated rates."""
        if T >= self.T[-1] or T <= self.T[0]:
            j = -1 if T >= self.T[-1] else 0
            return self.tenors[j].rd, self.tenors[j].rf
        return (
            float(np.interp(T, self.T, self._int_rd)) / T,
            float(np.interp(T, self.T, self._int_rf)) / T,
        )

    def forward(self, T: float) -> float:
        rd, rf = self.rates(T)
        return self.S * float(np.exp((rd - rf) * T))

    # ------------------------------------------------------------------ variance / vol
    def pillar_variance(self, j: int, k):
        """Total variance of pillar ``j`` at log forward-moneyness ``k`` (NaN where undefined)."""
        sm = self.smiles[j]
        K = sm.quote.F * np.exp(np.asarray(k, dtype=float))
        return np.asarray(sm.vol(K)) ** 2 * self.T[j]

    def total_variance(self, k, T: float):
        """``w(k, T)``: linear in T between pillars at fixed ``k``; flat vol outside."""
        k = np.asarray(k, dtype=float)
        if T <= 0:
            raise ValueError("T must be positive")
        j = int(np.searchsorted(self.T, T, side="right")) - 1
        if j < 0:  # shorter than the first pillar
            return self.pillar_variance(0, k) * (T / self.T[0])
        if j >= len(self.T) - 1:  # at or beyond the last pillar
            return self.pillar_variance(len(self.T) - 1, k) * (T / self.T[-1])
        lam = (T - self.T[j]) / (self.T[j + 1] - self.T[j])
        return (1.0 - lam) * self.pillar_variance(j, k) + lam * self.pillar_variance(j + 1, k)

    def vol_k(self, k, T: float):
        """Implied vol at log forward-moneyness ``k`` and maturity ``T``."""
        out = np.sqrt(self.total_variance(k, T) / T)
        return out if out.ndim else float(out)

    def vol(self, K, T: float):
        """Implied vol at strike ``K`` and maturity ``T``."""
        k = np.log(np.asarray(K, dtype=float) / self.forward(T))
        return self.vol_k(k, T)

    def atm_vol(self, T: float) -> float:
        """Vol at the forward (``k = 0``); a convenient scale and initial guess."""
        return float(self.vol_k(0.0, T))

    # ------------------------------------------------------------------ slices / deltas
    def slice(self, T: float) -> "SurfaceSlice":
        """The surface at one maturity, as a :class:`~fxvol.smile.Smile` (delta lookups etc.)."""
        return SurfaceSlice(self, T)

    def vol_at_delta(self, delta, T: float):
        return self.slice(T).vol_at_delta(delta)


class SurfaceSlice(Smile):
    """A fixed-T view of a surface that reuses the Smile delta machinery."""

    def __init__(self, surface: VolSurface, T: float):
        rd, rf = surface.rates(T)
        quote = SmileQuote(
            S=surface.S,
            T=T,
            rd=rd,
            rf=rf,
            atm=surface.atm_vol(T),  # initial guess for the delta fixed point
            rr=float("nan"),
            bf=float("nan"),
            wing_delta=surface.wing_delta,
            convention=surface.convention,
            atm_type=surface.atm_type,
        )
        super().__init__(quote, nodes=None)
        self.surface = surface

    def vol(self, K):
        return self.surface.vol(K, self.quote.T)
