"""Static-arbitrage diagnostics for a :class:`~fxvol.surface.VolSurface`.

Two conditions make a surface of European option prices free of static arbitrage:

1. **No butterfly arbitrage**: call price is convex in strike, i.e. the implied risk-neutral
   density ``q(K) = exp(rd*T) * d2C/dK2`` is non-negative.
2. **No calendar arbitrage**: total variance does not decrease with maturity.

Butterfly: two independent detectors
------------------------------------
* ``g(k)`` (Gatheral): with ``w(k)`` the total variance and ``k = ln(K/F)``,

      g = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2,

  and the density of ``k`` is ``g * exp(-d^2/2) / sqrt(2 pi w)`` with ``d = -k/sqrt(w) - sqrt(w)/2``.
  So ``g < 0`` is exactly "density < 0". ``w'`` and ``w''`` are central finite differences.
* A **model-free price butterfly** ``C(K-h) - 2C(K) + C(K+h)`` on the surface's own prices.
  A negative value is a literal arbitrage: buy the butterfly, collect a credit, and it can never pay
  negative.

The two should flag the same strikes; the tests check that they do.

Calendar: two coordinates
-------------------------
The surface is interpolated at fixed log forward-moneyness ``k``, and the Gatheral-Jacquier
condition is monotonicity in T at fixed ``k``. The prompt for this stage says "fixed strike";
the two coincide only with zero carry. With ``rd != rf`` the forward drifts, so a fixed strike is a
different ``k`` at each tenor. Both are reported, so the difference is visible.

All checks skip points where the model has no vol (NaN, e.g. Vanna-Volga far wings) and count
them separately as ``n_undefined``; a surface that is undefined is not the same as one that is
arbitrage-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

from . import gk
from .surface import VolSurface

G_TOL = -1e-6  # g(k) is O(1); below this it is a real violation, not finite-difference noise
FLY_TOL = -1e-9  # price-butterfly density, relative to its own peak
CAL_TOL = -1e-9  # calendar dw, relative to the shorter tenor's total variance


# --------------------------------------------------------------------------- helpers


def maturity_grid(surface: VolSurface, per_interval: int = 3) -> np.ndarray:
    """Pillars plus ``per_interval`` interior points between each pair (to catch interpolation-induced breaks)."""
    Ts = surface.T
    pts = [Ts[0]]
    for a, b in zip(Ts[:-1], Ts[1:]):
        pts.extend(a + (b - a) * np.arange(1, per_interval + 1) / (per_interval + 1))
        pts.append(b)
    return np.array(pts)


def signed_delta(surface: VolSurface, T: float, k) -> np.ndarray:
    """Spot, premium-excluded delta at ``k``: call delta for ``k >= 0``, put delta below."""
    k = np.asarray(k, dtype=float)
    rd, rf = surface.rates(T)
    K = surface.forward(T) * np.exp(k)
    sigma = np.asarray(surface.vol_k(k, T))
    d1, _ = gk.d1_d2(surface.S, K, T, rd, rf, sigma)
    df = np.exp(-rf * T)
    call = df * ndtr(d1)
    return np.where(k >= 0, call, call - df)


def _w_and_derivs(surface: VolSurface, T: float, k: np.ndarray, h: float):
    w = surface.total_variance(k, T)
    wp = (surface.total_variance(k + h, T) - surface.total_variance(k - h, T)) / (2 * h)
    wm = (surface.total_variance(k + h, T) - 2 * w + surface.total_variance(k - h, T)) / h**2
    return w, wp, wm


def density_g(surface: VolSurface, T: float, k, h: float | None = None) -> np.ndarray:
    """Gatheral's ``g(k)`` at maturity ``T``. Negative => negative implied density."""
    k = np.asarray(k, dtype=float)
    if h is None:
        h = 0.02 * np.sqrt(float(surface.total_variance(0.0, T)))
    w, wp, wm = _w_and_derivs(surface, T, k, h)
    return (1 - k * wp / (2 * w)) ** 2 - wp**2 / 4 * (1 / w + 0.25) + wm / 2


def density_from_g(surface: VolSurface, T: float, k, g) -> np.ndarray:
    """Strike-space density ``q(K) = p(k)/K`` implied by ``g``."""
    k = np.asarray(k, dtype=float)
    w = surface.total_variance(k, T)
    d = -k / np.sqrt(w) - np.sqrt(w) / 2
    K = surface.forward(T) * np.exp(k)
    return g * np.exp(-0.5 * d**2) / np.sqrt(2 * np.pi * w) / K


def price_butterfly_density(surface: VolSurface, T: float, k, rel_step: float = 0.01) -> np.ndarray:
    """``exp(rd*T) * [C(K-h) - 2C(K) + C(K+h)] / h^2`` on the surface's own call prices.

    ``h = rel_step * sigma_atm * sqrt(T) * F`` (equal-width wings, as a butterfly spread requires).
    """
    k = np.asarray(k, dtype=float)
    rd, rf = surface.rates(T)
    F = surface.forward(T)
    K = F * np.exp(k)
    h = rel_step * surface.atm_vol(T) * np.sqrt(T) * F

    def call(Kx):
        return gk.price(surface.S, Kx, T, rd, rf, surface.vol(Kx, T), True)

    return np.exp(rd * T) * (call(K - h) - 2 * call(K) + call(K + h)) / h**2


# --------------------------------------------------------------------------- results


@dataclass
class ButterflyResult:
    T: np.ndarray  # maturities checked
    k: list[np.ndarray]  # per-maturity k grid
    g: list[np.ndarray]
    density_price: list[np.ndarray]
    bad_g: list[np.ndarray]  # bool masks, per maturity
    bad_price: list[np.ndarray]
    n_undefined: int

    @property
    def n_violations(self) -> int:
        return int(sum(m.sum() for m in self.bad_g))

    @property
    def ok(self) -> bool:
        return self.n_violations == 0

    def worst(self) -> tuple[float, float, float] | None:
        """(T, k, g) at the most negative g, or None."""
        best = None
        for T, k, g, bad in zip(self.T, self.k, self.g, self.bad_g):
            if bad.any():
                i = np.nanargmin(np.where(bad, g, np.nan))
                if best is None or g[i] < best[2]:
                    best = (float(T), float(k[i]), float(g[i]))
        return best


@dataclass
class CalendarResult:
    """Calendar check over consecutive maturity-grid intervals.

    Row ``i`` compares maturities ``T_short[i] < T_long[i]`` on its own band of ``n`` points: ``x[i]``
    is ``k`` or the strike, and the band is +-``z_max`` standard deviations of the *shorter* tenor,
    which is where the shorter option has non-negligible value.
    """

    coordinate: str  # "forward_moneyness" or "strike"
    T_short: np.ndarray  # (m,)
    T_long: np.ndarray  # (m,)
    x: np.ndarray  # (m, n)
    w_short: np.ndarray  # (m, n)
    w_long: np.ndarray  # (m, n)
    bad: np.ndarray  # (m, n): w_long < w_short
    n_undefined: int

    @property
    def dw(self) -> np.ndarray:
        return self.w_long - self.w_short

    @property
    def n_violations(self) -> int:
        return int(self.bad.sum())

    @property
    def ok(self) -> bool:
        return self.n_violations == 0

    def worst(self) -> tuple[float, float, float, float] | None:
        """(T_short, T_long, x, dw) at the most negative total-variance step, or None."""
        if not self.bad.any():
            return None
        dw = np.where(self.bad, self.dw, np.nan)
        i, j = np.unravel_index(np.nanargmin(dw), dw.shape)
        return float(self.T_short[i]), float(self.T_long[i]), float(self.x[i, j]), float(self.dw[i, j])


def check_butterfly(
    surface: VolSurface, z_max: float = 4.0, n: int = 401, per_interval: int = 3
) -> ButterflyResult:
    """Density check at every pillar and interior maturity, on ``k in +-z_max * sqrt(w_atm)``."""
    Ts = maturity_grid(surface, per_interval)
    ks, gs, dps, bads_g, bads_p = [], [], [], [], []
    undefined = 0
    for T in Ts:
        k = np.linspace(-1, 1, n) * z_max * np.sqrt(float(surface.total_variance(0.0, T)))
        g = density_g(surface, T, k)
        dp = price_butterfly_density(surface, T, k)
        defined = np.isfinite(g) & np.isfinite(dp)
        undefined += int((~defined).sum())
        peak = np.nanmax(np.abs(dp)) if np.isfinite(dp).any() else 1.0
        ks.append(k), gs.append(g), dps.append(dp)
        bads_g.append(defined & (g < G_TOL))
        bads_p.append(defined & (dp < FLY_TOL * peak))
    return ButterflyResult(Ts, ks, gs, dps, bads_g, bads_p, undefined)


def check_calendar(
    surface: VolSurface,
    coordinate: str = "forward_moneyness",
    z_max: float = 4.0,
    n: int = 401,
    per_interval: int = 3,
) -> CalendarResult:
    """Total variance must not fall with T, at fixed ``k`` or at fixed strike.

    Each interval of the maturity grid gets a band of +-``z_max`` standard deviations of its shorter
    tenor. For the strike coordinate the band is centred between the two forwards and widened by half
    the forward drift, so both maturities are tested around where their own forwards sit.
    """
    if coordinate not in ("forward_moneyness", "strike"):
        raise ValueError("coordinate must be 'forward_moneyness' or 'strike'")
    Ts = maturity_grid(surface, per_interval)
    xs, ws, wl = [], [], []
    for Ta, Tb in zip(Ts[:-1], Ts[1:]):
        half = z_max * np.sqrt(float(surface.total_variance(0.0, Ta)))
        if coordinate == "forward_moneyness":
            x = np.linspace(-half, half, n)
            w_a, w_b = surface.total_variance(x, Ta), surface.total_variance(x, Tb)
        else:
            Fa, Fb = surface.forward(Ta), surface.forward(Tb)
            centre = 0.5 * (np.log(Fa) + np.log(Fb))
            half += 0.5 * abs(np.log(Fb / Fa))
            x = np.exp(np.linspace(centre - half, centre + half, n))
            w_a = surface.total_variance(np.log(x / Fa), Ta)
            w_b = surface.total_variance(np.log(x / Fb), Tb)
        xs.append(x), ws.append(w_a), wl.append(w_b)
    x, w_s, w_l = np.array(xs), np.array(ws), np.array(wl)
    defined = np.isfinite(w_s) & np.isfinite(w_l)
    bad = defined & (w_l - w_s < CAL_TOL * w_s)
    return CalendarResult(coordinate, Ts[:-1], Ts[1:], x, w_s, w_l, bad, int((~defined).sum()))


@dataclass
class ArbitrageReport:
    surface: VolSurface
    butterfly: ButterflyResult
    calendar_k: CalendarResult
    calendar_strike: CalendarResult
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.butterfly.ok and self.calendar_k.ok and self.calendar_strike.ok

    def summary(self) -> str:
        s = self.surface
        lines = []
        b = self.butterfly
        if b.ok:
            lines.append(f"butterfly (density): OK across {len(b.T)} maturities")
        else:
            by_T = {f"{T:.4g}": int(m.sum()) for T, m in zip(b.T, b.bad_g) if m.any()}
            T, k, g = b.worst()
            delta = float(signed_delta(s, T, k))
            lines.append(
                f"butterfly (density): {b.n_violations} violating points, maturities {by_T}; "
                f"worst g={g:.4g} at T={T:.4g}, k={k:+.4f}, delta={delta:+.3f}"
            )
            agree = sum(int((mg & mp).sum()) for mg, mp in zip(b.bad_g, b.bad_price))
            lines.append(f"  price-butterfly detector flags {agree} of the same {b.n_violations} points")
        for res in (self.calendar_k, self.calendar_strike):
            label = {"forward_moneyness": "fixed k=ln(K/F)", "strike": "fixed strike"}[res.coordinate]
            if res.ok:
                lines.append(f"calendar ({label}): OK")
            else:
                Ta, Tb, x, dw = res.worst()
                where = f"k={x:+.4f}" if res.coordinate == "forward_moneyness" else f"K={x:.5f}"
                lines.append(
                    f"calendar ({label}): {res.n_violations} violating cells; worst dw={dw:.3g} "
                    f"between T={Ta:.4g} and T={Tb:.4g} at {where}"
                )
        n_undef = b.n_undefined + self.calendar_k.n_undefined + self.calendar_strike.n_undefined
        if n_undef:
            lines.append(f"undefined (model returned no vol): {n_undef} points skipped")
        return "\n".join(lines + self.notes)


def check_surface(surface: VolSurface, **kw) -> ArbitrageReport:
    return ArbitrageReport(
        surface,
        check_butterfly(surface, **{k: v for k, v in kw.items() if k in ("z_max", "n", "per_interval")}),
        check_calendar(surface, "forward_moneyness", **{k: v for k, v in kw.items() if k in ("z_max", "per_interval")}),
        check_calendar(surface, "strike", **{k: v for k, v in kw.items() if k in ("z_max", "per_interval")}),
    )
