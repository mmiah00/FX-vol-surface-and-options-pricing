"""Single-tenor FX smile from ATM / risk-reversal / butterfly quotes.

All vols are decimals (0.075 = 7.5 vol; RR = -0.005 is "-0.5 vol").

Quote definitions (``wing_delta`` = 25 by default)::

    RR = sigma(25D call) - sigma(25D put)
    BF = (sigma(25D call) + sigma(25D put)) / 2 - sigma(ATM)

Two butterfly conventions
-------------------------
``strangle="smile"``  (default, simple)
    Treat the quoted BF as the *smile* butterfly above, so
    ``sigma_c = ATM + BF + RR/2`` and ``sigma_p = ATM + BF - RR/2`` directly.

``strangle="market"``  (broker / market strangle)
    The quoted BF is the vol of a *flat-vol* strangle minus ATM:
    ``sigma_ST = ATM + BF``. The strangle's strikes are the 25D strikes computed
    with ``sigma_ST``, and its price at ``sigma_ST`` must equal the price of the same
    two options at the *smile* vols. That fixes the (unobserved) smile butterfly
    ``b`` by a one-dimensional root-find, because the smile vols at the strangle
    strikes come from the interpolator, which itself is built from ``b``. This is
    the Reiswich-Wystup style iteration. The result depends on the interpolation
    model, and ``b != BF`` in general. The gap scales with BF and vol level and
    shrinks with tenor (it can change sign at long tenors). Measured on synthetic
    inputs: under 0.01 vol point for EUR/USD-like ATM 7.5 / BF 0.25, and a few
    hundredths of a vol point for ATM 12 / BF 1.0.

Interpolators (built through three nodes: 25D put, ATM, 25D call)
-----------------------------------------------------------------
``MalzSmile``
    Quadratic in call delta ``x = N(d1)``, the forward call delta (premium
    excluded) of the strike *at its own smile vol*, so it is implicit in
    ``K``. With forward-delta nodes this is exactly Malz's
    ``sigma(x) = ATM - 2 RR (x - 1/2) + 16 BF (x - 1/2)^2``.
    Wings are bounded: as K -> 0 or inf, x -> 1 or 0, and the vol tends to a constant.

``VannaVolgaSmile``
    Castagna-Mercurio vanna-volga: price at flat ATM vol plus a correction that
    replicates vega, vanna and volga of the target option with the three pivot
    options, then back out the implied vol of the OTM option. Prices (not vols) are
    interpolated, so wings keep growing and the surface can break (no implied vol)
    far out; those strikes return NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from . import gk
from .conventions import EURUSD_DEFAULT, DeltaConvention
from .delta import delta_to_strike
from .gk import d1_d2, forward


class AtmType(Enum):
    DELTA_NEUTRAL = "delta_neutral"  # strike where call delta + put delta = 0 (straddle)
    FORWARD = "forward"  # K = F


def atm_strike(
    S,
    T,
    rd,
    rf,
    sigma,
    atm_type: AtmType = AtmType.DELTA_NEUTRAL,
    convention: DeltaConvention = EURUSD_DEFAULT,
):
    """ATM strike.

    Delta-neutral straddle: premium-excluded gives ``d1 = 0`` so ``K = F exp(+s^2/2)``;
    premium-included gives ``d2 = 0`` so ``K = F exp(-s^2/2)``, with ``s = sigma sqrt(T)``.
    Spot vs forward delta does not matter, as the factor is common to the call and put.
    """
    F = forward(S, T, rd, rf)
    if atm_type is AtmType.FORWARD:
        return F
    sign = -1.0 if convention.premium_included else 1.0
    return F * np.exp(sign * 0.5 * np.asarray(sigma) ** 2 * T)


@dataclass(frozen=True)
class SmileQuote:
    """One tenor of market quotes plus the conventions needed to interpret them.

    ``atm_type`` defaults to delta-neutral straddle and ``convention`` to
    :data:`EURUSD_DEFAULT` (spot delta, premium excluded).
    """

    S: float
    T: float
    rd: float
    rf: float
    atm: float
    rr: float
    bf: float
    wing_delta: float = 0.25
    convention: DeltaConvention = EURUSD_DEFAULT
    atm_type: AtmType = AtmType.DELTA_NEUTRAL

    @property
    def F(self) -> float:
        return float(forward(self.S, self.T, self.rd, self.rf))


def smile_strangle_vols(atm: float, rr: float, bf: float) -> tuple[float, float]:
    """(sigma_put, sigma_call) at the wing deltas, treating ``bf`` as the smile butterfly."""
    return atm + bf - 0.5 * rr, atm + bf + 0.5 * rr


@dataclass(frozen=True)
class SmileNodes:
    """The three calibration points, ordered (25D put, ATM, 25D call)."""

    strikes: np.ndarray
    vols: np.ndarray


def make_nodes(q: SmileQuote, vol_put: float, vol_call: float) -> SmileNodes:
    """Nodes at the wing-delta strikes (each solved at its own vol) and the ATM strike."""
    args = (q.S, q.T, q.rd, q.rf)
    k_put = delta_to_strike(-q.wing_delta, *args, vol_put, q.convention)
    k_call = delta_to_strike(+q.wing_delta, *args, vol_call, q.convention)
    k_atm = atm_strike(*args, q.atm, q.atm_type, q.convention)
    return SmileNodes(
        strikes=np.array([k_put, float(k_atm), k_call]),
        vols=np.array([vol_put, q.atm, vol_call]),
    )


# ------------------------------------------------------------------ interpolators


class Smile:
    """Base class: ``vol(K)`` is model-specific; everything else is shared."""

    #: filled in by :func:`build_smile`
    strangle: str = "smile"
    bf_smile: float = float("nan")

    def __init__(self, quote: SmileQuote, nodes: SmileNodes):
        self.quote = quote
        self.nodes = nodes

    def vol(self, K):  # pragma: no cover - abstract
        raise NotImplementedError

    def call_delta(self, K):
        """Spot call delta (premium excluded) of strike ``K`` at its smile vol."""
        q = self.quote
        sigma = self.vol(K)
        d1, _ = d1_d2(q.S, K, q.T, q.rd, q.rf, sigma)
        return np.exp(-q.rf * q.T) * ndtr(d1)

    def vol_at_delta(self, delta):
        """Smile vol at a signed delta in the quote's convention (fixed point in sigma).

        The strike for a delta depends on the vol at that strike, so iterate
        ``sigma <- vol(K(delta, sigma))``. This converges quickly because the smile slope
        is small relative to the sensitivity of K to sigma.
        """
        q = self.quote
        args = (q.S, q.T, q.rd, q.rf)

        def one(delta):
            sigma = q.atm
            for _ in range(200):
                K = delta_to_strike(delta, *args, sigma, q.convention)
                new = float(self.vol(K))
                if not np.isfinite(new):
                    return np.nan
                if abs(new - sigma) < 1e-13:
                    return new
                sigma = new
            raise RuntimeError(f"vol_at_delta({delta}) did not converge")

        out = np.vectorize(one, otypes=[float])(delta)
        return out if out.ndim else float(out)

    def strike_at_delta(self, delta):
        """Strike at a signed delta on this smile (consistent vol and strike)."""
        q = self.quote
        sigma = self.vol_at_delta(delta)
        return delta_to_strike(delta, q.S, q.T, q.rd, q.rf, sigma, q.convention)


def _scalarize(out):
    return out if out.ndim else float(out)


class MalzSmile(Smile):
    """Quadratic in call delta through the three nodes (exact fit)."""

    def __init__(self, quote: SmileQuote, nodes: SmileNodes):
        super().__init__(quote, nodes)
        x = self._x(nodes.strikes, nodes.vols)
        self.coeffs = np.polyfit(x, nodes.vols, 2)  # exact: 3 points, degree 2

    def _x(self, K, sigma):
        q = self.quote
        d1, _ = d1_d2(q.S, K, q.T, q.rd, q.rf, sigma)
        return ndtr(d1)

    def vol(self, K):
        """Solve ``sigma = q(N(d1(K, sigma)))`` for every strike at once (NaN if no root in [1e-4, 5])."""
        K = np.asarray(K, dtype=float)
        f = lambda s: s - np.polyval(self.coeffs, self._x(K, s))
        out, valid = gk._bisect(f, 1e-4, 5.0, K.shape)
        return _scalarize(np.where(valid, out, np.nan))

    def vol_at_delta(self, delta):
        # Premium-excluded conventions give x explicitly, so skip the fixed point.
        q = self.quote
        if q.convention.premium_included:
            return super().vol_at_delta(delta)
        from .conventions import DeltaType

        delta = np.asarray(delta, dtype=float)
        df = np.exp(-q.rf * q.T) if q.convention.delta_type is DeltaType.SPOT else 1.0
        fwd = delta / df  # forward delta
        x = np.where(delta > 0, fwd, 1.0 + fwd)  # call-equivalent forward delta N(d1)
        return _scalarize(np.polyval(self.coeffs, x))


class VannaVolgaSmile(Smile):
    """Three-pivot vanna-volga: ATM-vol Black price plus a vega/vanna/volga-matching correction."""

    def __init__(self, quote: SmileQuote, nodes: SmileNodes):
        super().__init__(quote, nodes)
        self._base = float(nodes.vols[1])  # ATM vol is the Black base
        q, K, s = quote, nodes.strikes, nodes.vols
        # 3x3: rows = (vega, vanna, volga), columns = pivots
        self._A = np.array([g(q.S, K, q.T, q.rd, q.rf, self._base) for g in (gk.vega, gk.vanna, gk.volga)])
        self._adj = gk.price(q.S, K, q.T, q.rd, q.rf, s, True) - gk.price(q.S, K, q.T, q.rd, q.rf, self._base, True)

    def weights(self, K):
        """Pivot weights x_i(K): sum_i x_i * greek_i(pivot) = greek(K) for vega, vanna, volga."""
        q = self.quote
        K = np.atleast_1d(np.asarray(K, dtype=float))
        rhs = np.array([g(q.S, K, q.T, q.rd, q.rf, self._base) for g in (gk.vega, gk.vanna, gk.volga)])
        return np.linalg.solve(self._A, rhs)  # shape (3, n)

    def price(self, K, is_call=True):
        """Vanna-volga price (call or put). Put-call parity is preserved by construction."""
        q = self.quote
        K = np.asarray(K, dtype=float)
        flat = gk.price(q.S, K, q.T, q.rd, q.rf, self._base, is_call)
        corr = (self.weights(K).T @ self._adj).reshape(K.shape)
        return flat + corr

    def vol(self, K):
        """Implied vol of the OTM option priced by vanna-volga (NaN where none exists)."""
        q = self.quote
        K = np.asarray(K, dtype=float)
        is_call = K >= q.F
        value = self.price(K, is_call)
        return gk.implied_vol(value, q.S, K, q.T, q.rd, q.rf, is_call)


MODELS = {"malz": MalzSmile, "vv": VannaVolgaSmile}


# --------------------------------------------------------------------- builders


def _market_strangle_bf(q: SmileQuote, model_cls) -> float:
    """Solve for the smile butterfly ``b`` that makes the flat-vol market strangle reprice."""
    args = (q.S, q.T, q.rd, q.rf)
    sigma_st = q.atm + q.bf
    k_c = delta_to_strike(+q.wing_delta, *args, sigma_st, q.convention)
    k_p = delta_to_strike(-q.wing_delta, *args, sigma_st, q.convention)
    target = float(gk.price(q.S, k_c, q.T, q.rd, q.rf, sigma_st, True)) + float(
        gk.price(q.S, k_p, q.T, q.rd, q.rf, sigma_st, False)
    )

    def residual(b):
        sp, sc = smile_strangle_vols(q.atm, q.rr, b)
        smile = model_cls(q, make_nodes(q, sp, sc))
        vc, vp = float(smile.vol(k_c)), float(smile.vol(k_p))
        return (
            float(gk.price(q.S, k_c, q.T, q.rd, q.rf, vc, True))
            + float(gk.price(q.S, k_p, q.T, q.rd, q.rf, vp, False))
            - target
        )

    # price is increasing in b; widen a bracket around the quoted BF
    width = 0.002
    for _ in range(8):
        lo, hi = q.bf - width, q.bf + width
        try:
            r_lo, r_hi = residual(lo), residual(hi)
        except ValueError:
            r_lo = r_hi = np.nan
        if np.isfinite(r_lo) and np.isfinite(r_hi) and r_lo < 0.0 < r_hi:
            return brentq(residual, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps)
        width *= 2.0
    raise RuntimeError("could not bracket the market-strangle butterfly")


def build_smile(q: SmileQuote, model: str = "malz", strangle: str = "smile") -> Smile:
    """Build a smile from ``q``.

    ``model``: ``"malz"`` or ``"vv"``.  ``strangle``: ``"smile"`` or ``"market"``.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}")
    if strangle not in ("smile", "market"):
        raise ValueError("strangle must be 'smile' or 'market'")
    cls = MODELS[model]
    b = q.bf if strangle == "smile" else _market_strangle_bf(q, cls)
    sp, sc = smile_strangle_vols(q.atm, q.rr, b)
    smile = cls(q, make_nodes(q, sp, sc))
    smile.strangle, smile.bf_smile = strangle, b
    return smile
