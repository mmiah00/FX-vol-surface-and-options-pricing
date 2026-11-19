"""Garman-Kohlhagen pricing and analytic Greeks.

Conventions
-----------
* ``S`` is quoted as domestic per 1 unit of foreign (EUR/USD: USD per EUR).
* ``rd`` / ``rf`` are continuously-compounded domestic / foreign rates.
* ``T`` is in years, ``sigma`` is an annualised vol as a decimal (0.08 = 8%).
* Values are in domestic currency per 1 unit of foreign notional.
* All Greeks are raw partial derivatives (no per-vol-point or per-day
  scaling), so they can be compared directly with finite differences:

  - delta  = dV/dS               (spot delta, premium excluded)
  - gamma  = d2V/dS2
  - vega   = dV/dsigma           (per 1.00 of vol; divide by 100 per vol point)
  - theta  = dV/dt = -dV/dT      (calendar time, per year)
  - vanna  = d2V/(dS dsigma)
  - volga  = d2V/dsigma2

Premium-adjusted and forward deltas live in :mod:`fxvol.delta`.

All functions broadcast over numpy arrays. ``is_call`` may be a bool or a
boolean array.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _pdf(x):
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _check(S, K, T, sigma):
    if np.any(np.asarray(S) <= 0) or np.any(np.asarray(K) <= 0):
        raise ValueError("S and K must be positive")
    if np.any(np.asarray(T) <= 0):
        raise ValueError("T must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")


def _phi(is_call):
    return np.where(is_call, 1.0, -1.0)


def forward(S, T, rd, rf):
    """FX forward outright, ``S * exp((rd - rf) * T)``."""
    return np.asarray(S) * np.exp((np.asarray(rd) - np.asarray(rf)) * np.asarray(T))


def d1_d2(S, K, T, rd, rf, sigma):
    _check(S, K, T, sigma)
    S, K, T, rd, rf, sigma = map(np.asarray, (S, K, T, rd, rf, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(S, K, T, rd, rf, sigma, is_call=True):
    """Garman-Kohlhagen option value in domestic currency per unit foreign."""
    d1, d2 = d1_d2(S, K, T, rd, rf, sigma)
    phi = _phi(is_call)
    return phi * (
        np.asarray(S) * np.exp(-np.asarray(rf) * T) * ndtr(phi * d1)
        - np.asarray(K) * np.exp(-np.asarray(rd) * T) * ndtr(phi * d2)
    )


def delta(S, K, T, rd, rf, sigma, is_call=True):
    """Spot delta, dV/dS (premium excluded). Puts are negative."""
    d1, _ = d1_d2(S, K, T, rd, rf, sigma)
    phi = _phi(is_call)
    return phi * np.exp(-np.asarray(rf) * T) * ndtr(phi * d1)


def gamma(S, K, T, rd, rf, sigma):
    d1, _ = d1_d2(S, K, T, rd, rf, sigma)
    return (
        np.exp(-np.asarray(rf) * T) * _pdf(d1) / (np.asarray(S) * sigma * np.sqrt(T))
    )


def vega(S, K, T, rd, rf, sigma):
    d1, _ = d1_d2(S, K, T, rd, rf, sigma)
    return np.asarray(S) * np.exp(-np.asarray(rf) * T) * _pdf(d1) * np.sqrt(T)


def theta(S, K, T, rd, rf, sigma, is_call=True):
    """dV/dt per year (negative of the derivative w.r.t. time to expiry)."""
    d1, d2 = d1_d2(S, K, T, rd, rf, sigma)
    phi = _phi(is_call)
    S, K, T, rd, rf, sigma = map(np.asarray, (S, K, T, rd, rf, sigma))
    df_f, df_d = np.exp(-rf * T), np.exp(-rd * T)
    decay = -S * df_f * _pdf(d1) * sigma / (2.0 * np.sqrt(T))
    carry = phi * (rf * S * df_f * ndtr(phi * d1) - rd * K * df_d * ndtr(phi * d2))
    return decay + carry


def vanna(S, K, T, rd, rf, sigma):
    """d(delta)/d(sigma) = d(vega)/dS."""
    d1, d2 = d1_d2(S, K, T, rd, rf, sigma)
    return -np.exp(-np.asarray(rf) * T) * _pdf(d1) * d2 / sigma


def volga(S, K, T, rd, rf, sigma):
    """Vomma, d2V/dsigma2 = vega * d1 * d2 / sigma."""
    d1, d2 = d1_d2(S, K, T, rd, rf, sigma)
    return vega(S, K, T, rd, rf, sigma) * d1 * d2 / sigma


def greeks(S, K, T, rd, rf, sigma, is_call=True):
    """All of the above in one dict."""
    args = (S, K, T, rd, rf, sigma)
    return {
        "price": price(*args, is_call),
        "delta": delta(*args, is_call),
        "gamma": gamma(*args),
        "vega": vega(*args),
        "theta": theta(*args, is_call),
        "vanna": vanna(*args),
        "volga": volga(*args),
    }
