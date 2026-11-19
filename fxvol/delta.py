"""FX delta <-> strike under the four market delta conventions.

Notation: ``F`` forward, ``s = sigma*sqrt(T)``, ``phi = +1`` call / ``-1`` put,
``d1 = (ln(F/K) + s^2/2)/s``, ``d2 = d1 - s``. In terms of the *forward* delta
``D_f`` (spot delta = ``exp(-rf*T) * D_f`` in both cases):

    premium excluded:  D_f = phi * N(phi*d1)
    premium included:  D_f = phi * (K/F) * N(phi*d2)

The premium-included form is ``dV/dS - V/S``: the spot delta less the premium
(paid in foreign currency) expressed in units of foreign.

Delta -> strike
---------------
* Premium excluded has a closed form:
  ``K = F * exp(-phi*s*N^-1(phi*D_f) + s^2/2)``.
* Premium included needs a root-finder. For puts ``D_f`` is monotone in K, so
  the root is unique. For calls ``(K/F)*N(d2)`` rises to a maximum and then
  falls back to 0, so a given delta has **two** strikes (or none, if delta
  exceeds the maximum). We return the larger strike, i.e. the root on the
  branch continuous with the premium-excluded delta, which is the market
  convention for building smiles.

``delta`` is signed (calls positive, puts negative); the sign selects the
option type. The vol passed in is the vol *at that strike*; when building a
smile from delta quotes the caller iterates strike and vol to a fixed point.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from .conventions import EURUSD_DEFAULT, DeltaConvention, DeltaType
from .gk import _check, _pdf, _phi, d1_d2, forward

_TINY = 4.0 * np.finfo(float).eps


def _spot_to_forward_factor(T, rf, convention: DeltaConvention):
    """Multiplier turning a quoted delta into a forward delta."""
    if convention.delta_type is DeltaType.SPOT:
        return np.exp(np.asarray(rf) * np.asarray(T))
    return 1.0


def fx_delta(
    S, K, T, rd, rf, sigma, is_call=True, convention: DeltaConvention = EURUSD_DEFAULT
):
    """Delta of a vanilla under ``convention``. Puts are negative."""
    d1, d2 = d1_d2(S, K, T, rd, rf, sigma)
    phi = _phi(is_call)
    if convention.premium_included:
        F = forward(S, T, rd, rf)
        fwd = phi * (np.asarray(K) / F) * ndtr(phi * d2)
    else:
        fwd = phi * ndtr(phi * d1)
    if convention.delta_type is DeltaType.SPOT:
        fwd = fwd * np.exp(-np.asarray(rf) * T)
    return fwd


# ---------------------------------------------------------------- strike solvers


def _peak_log_moneyness(s: float) -> tuple[float, float]:
    """Location x* = ln(K/F) and height of the max of g(x) = e^x N(d2(x)).

    Stationarity gives N(d2) * s = n(d2), which has a unique root d2* > -s.
    """

    def h(d2):
        return ndtr(d2) * s - _pdf(d2)

    d2_star = brentq(h, -s, max(10.0, s + 10.0), xtol=1e-14, rtol=_TINY)
    x_star = -d2_star * s - 0.5 * s * s
    return x_star, np.exp(x_star) * ndtr(d2_star)


def _premium_included_log_moneyness(delta_f: float, s: float) -> float:
    """Solve phi*e^x*N(phi*d2(x)) = delta_f for x = ln(K/F)."""
    is_call = delta_f > 0
    phi = 1.0 if is_call else -1.0

    def g(x):
        d2 = (-x - 0.5 * s * s) / s
        return phi * np.exp(x) * ndtr(phi * d2) - delta_f

    if is_call:
        x_lo, g_max = _peak_log_moneyness(s)
        if delta_f >= g_max:
            raise ValueError(
                f"premium-included call delta {delta_f:.6g} (forward) exceeds the "
                f"maximum attainable {g_max:.6g}; no strike exists"
            )
        # g is decreasing on [x_peak, inf) from (g_max - delta_f) > 0 to -delta_f < 0
        step = 1.0
        x_hi = x_lo + step
        while g(x_hi) > 0:
            step *= 2.0
            x_hi = x_lo + step
    else:
        # g is decreasing on all of R from -delta_f > 0 to -inf
        x_lo, x_hi, step = -1.0, 1.0, 1.0
        while g(x_lo) < 0:
            step *= 2.0
            x_lo = -step
        step = 1.0
        while g(x_hi) > 0:
            step *= 2.0
            x_hi = step
    return brentq(g, x_lo, x_hi, xtol=1e-15, rtol=_TINY, maxiter=200)


_solve_pi = np.vectorize(_premium_included_log_moneyness, otypes=[float])


def delta_to_strike(
    delta, S, T, rd, rf, sigma, convention: DeltaConvention = EURUSD_DEFAULT
):
    """Strike whose delta equals ``delta`` (signed) under ``convention``.

    ``sigma`` is the vol at the sought strike. Broadcasts over numpy arrays.
    """
    delta = np.asarray(delta, dtype=float)
    if np.any(delta == 0):
        raise ValueError("delta must be non-zero (sign selects call/put)")
    _check(S, 1.0, T, sigma)
    F = forward(S, T, rd, rf)
    s = np.asarray(sigma) * np.sqrt(T)
    delta_f = delta * _spot_to_forward_factor(T, rf, convention)

    if convention.premium_included:
        x = _solve_pi(delta_f, s)
    else:
        phi = np.sign(delta_f)
        if np.any(np.abs(delta_f) >= 1.0):
            raise ValueError("|forward delta| must be below 1 (spot: below exp(-rf*T))")
        x = -phi * s * ndtri(phi * delta_f) + 0.5 * s * s
    K = F * np.exp(x)
    return K if K.ndim else float(K)
