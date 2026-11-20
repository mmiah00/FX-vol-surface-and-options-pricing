"""The ASSUMED layer: turn one day of REAL market state into a vol surface.

What is real, assumed and synthetic (also tabulated in the README)::

    REAL      spot, rd (SOFR), rf (EUR STR), EVZ                      -> fxvol.data
    ASSUMED   EVZ -> 1M ATM conversion, ATM term structure shape,
              RR and BF levels, flat rate curve across tenors          -> this module
    SYNTHETIC stress scenarios built on a real day                     -> ``stress_scenarios``

Assumptions, each a field of :class:`AssumedParams`
---------------------------------------------------
* **1M ATM from EVZ.** EVZ is a VIX-style 30-day index, closer to a variance-swap level than to ATM vol. Given the
  assumed smile below, ``atm_over_evz`` is the factor that makes the model's 30-day variance-swap vol equal EVZ.
  See :func:`calibrate_atm_over_evz`.
* **ATM term structure.** Forward variance mean-reverts to a long-run level ``theta``:
  ``v(t) = theta + (v0 - theta) exp(-kappa t)``, with ``v0`` fixed so the 30-day ATM vol matches the day's value.
  Total variance ``w(T) = int_0^T v`` is increasing by construction (``v > 0``), so the ATM term structure never
  breaks calendar arbitrage. ``theta`` should be set from data (the sample mean of EVZ); ``kappa`` is a guess.
* **RR and BF as fixed fractions of ATM** at each pillar. This scales the smile with the vol regime (high EVZ ->
  larger skew and wings). The fractions are plausible EUR/USD magnitudes (1M ATM 7: RR about -0.35, BF about 0.2;
  1Y ATM 8: RR about -0.7, BF about 0.35), NOT calibrated to market quotes.
* **Flat rate curve:** the same overnight ``rd`` and ``rf`` at every tenor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .surface import TenorQuote, VolSurface, variance_swap_vol
from .synthetic import T_PILLARS

T_EVZ = 30.0 / 365.0  # EVZ measures 30-day implied vol


@dataclass(frozen=True)
class AssumedParams:
    rr_over_atm: tuple[float, ...] = (-0.029, -0.050, -0.068, -0.079, -0.089)
    bf_over_atm: tuple[float, ...] = (0.022, 0.029, 0.034, 0.039, 0.044)
    long_run_vol: float = 0.077  # replace with the sample mean of EVZ
    kappa: float = 2.0  # mean-reversion speed of forward variance, per year (half-life about 4 months)
    atm_over_evz: float = 1.0  # see calibrate_atm_over_evz


def atm_term_structure(atm_30d: float, T=T_PILLARS, long_run_vol: float = 0.077, kappa: float = 2.0) -> np.ndarray:
    """ATM vol at maturities ``T`` from forward variance reverting to ``long_run_vol``^2, pinned at 30 days."""
    T = np.asarray(T, dtype=float)
    theta = long_run_vol**2
    w30 = atm_30d**2 * T_EVZ
    v0 = theta + (w30 - theta * T_EVZ) * kappa / (1.0 - np.exp(-kappa * T_EVZ))
    if v0 <= 0:
        raise ValueError(f"30-day vol {atm_30d:.4f} is too low for long-run {long_run_vol:.4f}: forward variance < 0")
    w = theta * T + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / kappa
    return np.sqrt(w / T)


def surface_from_state(
    S: float,
    rd: float,
    rf: float,
    evz: float,
    params: AssumedParams = AssumedParams(),
    model: str = "malz",
    atm: np.ndarray | None = None,
) -> VolSurface:
    """Five-pillar surface for one day. ``evz`` is a decimal vol; ``atm`` (5 values) overrides the term structure."""
    if atm is None:
        atm = atm_term_structure(evz * params.atm_over_evz, T_PILLARS, params.long_run_vol, params.kappa)
    atm = np.asarray(atm, dtype=float)
    rr = np.asarray(params.rr_over_atm) * atm
    bf = np.asarray(params.bf_over_atm) * atm
    quotes = [TenorQuote(float(T), rd, rf, float(a), float(r), float(b)) for T, a, r, b in zip(T_PILLARS, atm, rr, bf)]
    return VolSurface(S, quotes, model=model)


def calibrate_atm_over_evz(
    params: AssumedParams, S: float = 1.10, rd: float = 0.025, rf: float = 0.011, ref_evz: float | None = None
) -> float:
    """Factor ``c`` with ``atm_30d = c * EVZ`` such that the model's 30-day variance-swap vol equals EVZ.

    Because the smile scales with ATM, the variance-swap-to-ATM ratio is nearly independent of the vol level, so one
    ``c`` serves every day; :func:`atm_over_evz_sensitivity` shows how nearly.
    """
    ref = params.long_run_vol if ref_evz is None else ref_evz
    p = replace(params, atm_over_evz=1.0)
    s = surface_from_state(S, rd, rf, ref, p)
    return float(ref / variance_swap_vol(s, T_EVZ))


def atm_over_evz_sensitivity(params: AssumedParams, evz_levels=(0.04, 0.06, 0.08, 0.12, 0.20, 0.30), **kw) -> dict:
    """``variance-swap vol / ATM vol`` at 30 days for a range of vol levels (should be nearly constant)."""
    p = replace(params, atm_over_evz=1.0)
    out = {}
    for lvl in evz_levels:
        s = surface_from_state(kw.get("S", 1.10), kw.get("rd", 0.025), kw.get("rf", 0.011), lvl, p)
        out[lvl] = variance_swap_vol(s, T_EVZ) / s.vol_k(0.0, T_EVZ)
    return out


# ------------------------------------------------------------------ synthetic stress tests


def stress_scenarios(S: float, rd: float, rf: float, evz: float, params: AssumedParams) -> dict[str, dict]:
    """SYNTHETIC edge cases applied to a real day's state: keyword args for :func:`surface_from_state`.

    Each changes one thing relative to the base day, so a failure can be attributed.
    """
    scale = lambda xs, f: tuple(f * x for x in xs)
    base_atm = atm_term_structure(evz * params.atm_over_evz, T_PILLARS, params.long_run_vol, params.kappa)
    inverted = base_atm.copy()
    inverted[0] = 2.5 * inverted[1]  # event vol in the front week
    common = dict(S=S, rd=rd, rf=rf, evz=evz)
    return {
        "base day": dict(common, params=params),
        "extreme skew (RR x6)": dict(common, params=replace(params, rr_over_atm=scale(params.rr_over_atm, 6.0))),
        "extreme butterfly (BF x6)": dict(common, params=replace(params, bf_over_atm=scale(params.bf_over_atm, 6.0))),
        "vol spike (EVZ 30)": dict(common, params=params) | {"evz": 0.30},
        "vol crush (EVZ 4)": dict(common, params=params) | {"evz": 0.04},
        "inverted front end (1W = 2.5x 1M)": dict(common, params=params, atm=inverted),
    }
