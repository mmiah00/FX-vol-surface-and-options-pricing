"""SYNTHETIC quote sets. None of these are market data.

``eurusd_like`` is an illustrative EUR/USD-style surface (ATM, RR and BF levels in the range
usually quoted for the pair, rates roughly 5.3% USD / 3.5% EUR). Every scenario function builds on it
by overriding one thing, so a break can be traced to a single cause.
"""

from __future__ import annotations

import numpy as np

from .surface import TenorQuote, VolSurface

PILLARS = {"1W": 7 / 365, "1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0}
T_PILLARS = np.array(list(PILLARS.values()))

_BASE = dict(
    atm=[0.068, 0.070, 0.073, 0.076, 0.079],
    rr=[-0.002, -0.0035, -0.005, -0.006, -0.007],
    bf=[0.0015, 0.0020, 0.0025, 0.0030, 0.0035],
    rd=[0.0530, 0.0525, 0.0515, 0.0500, 0.0470],
    rf=[0.0355, 0.0350, 0.0340, 0.0325, 0.0300],
)


def make_surface(S: float = 1.10, model: str = "malz", strangle: str = "smile", **per_tenor) -> VolSurface:
    """Surface on the five standard pillars. ``per_tenor`` keys: atm, rr, bf, rd, rf (5 values each)."""
    unknown = set(per_tenor) - set(_BASE)
    if unknown:
        raise TypeError(f"unknown overrides {sorted(unknown)}")
    p = {**_BASE, **per_tenor}
    quotes = [
        TenorQuote(T, rd, rf, atm, rr, bf)
        for T, rd, rf, atm, rr, bf in zip(T_PILLARS, p["rd"], p["rf"], p["atm"], p["rr"], p["bf"])
    ]
    return VolSurface(S, quotes, model=model, strangle=strangle)


def eurusd_like(model: str = "malz", strangle: str = "smile") -> VolSurface:
    """Clean baseline: upward ATM term structure, mild negative RR, modest BF."""
    return make_surface(1.10, model, strangle)


def vv_short_end_butterfly(bf_1w: float = 0.01) -> VolSurface:
    """Vanna-Volga with a 1.0-vol-point 1W butterfly: the density goes negative in the put wing."""
    bf = list(_BASE["bf"])
    bf[0] = bf_1w
    return make_surface(1.10, "vv", bf=bf)


def stale_6m_atm(atm_6m: float = 0.12) -> VolSurface:
    """A bad 6M ATM print (12% against neighbours at 7.3% and 7.9%): total variance falls from 6M to 1Y."""
    atm = list(_BASE["atm"])
    atm[3] = atm_6m
    return make_surface(1.10, "malz", atm=atm)


def high_carry_steep_front(carry: float = 0.10, rr: float = 0.01, a: float = 0.4) -> VolSurface:
    """ENGINEERED to separate the two calendar coordinates; not meant to look like a real pair.

    ATM vol falls like T^-a (slow total-variance growth) and rd - rf = ``carry``. Total variance is
    non-decreasing at fixed log forward-moneyness, so the surface is calendar-arbitrage-free, yet
    total variance falls with T at fixed *strike* because the forward drifts through the skew.
    """
    atm = np.clip(0.25 * (T_PILLARS / T_PILLARS[0]) ** (-a), 0.05, None)
    return make_surface(
        1.0, "malz", atm=list(atm), rr=[rr] * 5, bf=[0.003] * 5, rd=[0.02 + carry] * 5, rf=[0.02] * 5
    )
