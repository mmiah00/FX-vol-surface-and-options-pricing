"""Stage 3 demo: the surface, then four ways to break it and how each break is detected.

SYNTHETIC DATA: every quote here is illustrative (see fxvol/synthetic.py), not market data.

    python scripts/surface_demo.py            # prints reports, writes figures/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_smiles import (  # noqa: E402  (also applies the shared rcParams)
    AXIS,
    BLUE,
    DPI,
    INK,
    INK2,
    LW,
    MUTED,
    ORANGE,
    OUT,
    RED,
    SURFACE,
    end_label,
)

from fxvol.arbitrage import check_butterfly, check_calendar, check_surface, density_from_g, density_g, price_butterfly_density  # noqa: E402
from fxvol.synthetic import (  # noqa: E402
    PILLARS,
    eurusd_like,
    high_carry_steep_front,
    stale_6m_atm,
    vv_short_end_butterfly,
)

CRITICAL = "#d03b3b"  # status color: reserved for "this is broken"
TENOR_RAMP = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential blue steps 300-700, short -> long
SYNTH = "Synthetic quotes (illustrative, not market data)."


def titled(fig, title, subtitle):
    h = fig.get_figheight()
    fig.text(0.015, 1 - 0.10 / h, title, fontsize=12, fontweight="semibold", ha="left", va="top")
    fig.text(0.015, 1 - 0.40 / h, subtitle, fontsize=8.5, color=INK2, ha="left", va="top", wrap=True)


def finish(fig, name):
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.90 / fig.get_figheight()))
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)


def runs(mask):
    """Contiguous (first, last) index pairs where ``mask`` is True."""
    idx = np.flatnonzero(mask)
    return [(s[0], s[-1]) for s in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(s)]


def pct(ax):
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.1f}")


# ----------------------------------------------------------------------- figure 1
def fig_clean_surface():
    s = eurusd_like()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for (name, T), color in zip(PILLARS.items(), TENOR_RAMP):
        half = 3.0 * np.sqrt(float(s.total_variance(0.0, T)))
        k = np.linspace(-half, half, 300)
        K = s.forward(T) * np.exp(k)
        sl = s.slice(T)
        cd = np.asarray(sl.call_delta(K))
        v = np.asarray(s.vol_k(k, T))
        a1.plot(cd, v, color=color, lw=LW, label=name)
        end_label(a1, cd, v, name)
    a1.invert_xaxis()
    a1.set_xlabel("Call delta of the strike (put at left, call at right)")
    a1.set_ylabel("Implied vol (%)")
    a1.set_title("Smile by tenor, delta space", loc="left", fontsize=10, color=INK2)
    pct(a1)
    a1.legend(title="tenor", loc="upper right", title_fontsize=8.5)

    Tg = np.linspace(0.01, 1.0, 200)
    for kval, label, color in [(-0.05, "k = -0.05", BLUE), (0.0, "k = 0 (forward)", MUTED), (0.10, "k = +0.10", RED)]:
        w = np.array([float(s.total_variance(kval, T)) for T in Tg])
        a2.plot(Tg, w, color=color, lw=LW, label=label)
        a2.plot(s.T, [float(s.total_variance(kval, T)) for T in s.T], "o", ms=6, color=color, mec=SURFACE, mew=1.5, ls="none")
    a2.set_xlabel("Maturity T (years)")
    a2.set_ylabel("Total variance  sigma^2 T")
    a2.set_title("Total variance at fixed k = ln(K/F): non-decreasing in T", loc="left", fontsize=10, color=INK2)
    a2.grid(axis="x", visible=False)
    a2.legend(loc="upper left")
    for ax in (a1, a2):
        ax.grid(axis="x", visible=False)
    titled(fig, "A clean surface: smiles per tenor, and total variance rising with maturity",
           f"{SYNTH} Malz smile per tenor, linear in total variance between pillars (dots = pillars).")
    finish(fig, "surface_clean.png")


# ----------------------------------------------------------------------- figure 2
def fig_butterfly_break():
    vv = vv_short_end_butterfly()
    malz_bf = vv.smiles[0].quote.bf
    from fxvol.synthetic import make_surface
    bf = [0.0015, 0.002, 0.0025, 0.003, 0.0035]
    bf[0] = malz_bf
    malz = make_surface(1.10, "malz", bf=bf)
    T = float(vv.T[0])
    res = check_butterfly(vv)
    bad = res.bad_g[0]
    k = res.k[0]
    F = vv.forward(T)
    K = F * np.exp(k)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for sm, name, color in [(malz, "Malz", BLUE), (vv, "Vanna-Volga", ORANGE)]:
        a1.plot(K, np.asarray(sm.vol_k(k, T)), color=color, lw=LW, label=name)
        a2.plot(K, np.asarray(density_from_g(sm, T, k, density_g(sm, T, k))), color=color, lw=LW, label=name)
    q_vv = np.asarray(density_from_g(vv, T, k, density_g(vv, T, k)))
    a2.plot(K, price_butterfly_density(vv, T, k), ls="none", marker="o", ms=3, markevery=8, color=INK, label="VV via price butterfly")
    from fxvol.arbitrage import signed_delta
    for i0, i1 in runs(bad):
        for ax in (a1, a2):
            ax.axvspan(K[i0], K[i1], color=CRITICAL, alpha=0.14, lw=0)
        i = i0 + int(np.argmin(q_vv[i0 : i1 + 1]))
        d = float(signed_delta(vv, T, k[i]))
        a2.annotate(f"negative density\nnear {abs(d) * 100:.0f}Δ {'put' if d < 0 else 'call'}", (K[i], 0),
                    xytext=(0, -14), textcoords="offset points", ha="center", va="top", fontsize=8.5, color=INK2)
    for ax in (a1, a2):
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Strike (USD per EUR)")
    a2.axhline(0, color=AXIS, lw=0.9)
    a2.set_ylim(-16, 72)
    a1.set_ylabel("Implied vol (%)")
    pct(a1)
    a2.set_ylabel("Risk-neutral density of the strike")
    a1.set_title("1W smile, 1.0-vol-point butterfly", loc="left", fontsize=10, color=INK2)
    a2.set_title("Implied density: Vanna-Volga dips below zero", loc="left", fontsize=10, color=INK2)
    a1.legend(loc="upper center")
    a2.legend(loc="upper left")
    titled(fig, "Where it breaks (butterfly): Vanna-Volga gives a negative density in both 1W wings",
           f"{SYNTH} The same quotes in Malz are arbitrage-free. Shaded = g(k) < 0; the model-free price butterfly "
           "C(K-h) - 2C(K) + C(K+h) (black dots) flags exactly the same points.")
    finish(fig, "arb_butterfly.png")


# ----------------------------------------------------------------------- figure 3
def fig_calendar_break():
    clean, bad = eurusd_like(), stale_6m_atm()
    Tg = np.linspace(0.01, 1.0, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for s, name, color in [(clean, "clean quotes", MUTED), (bad, "6M ATM mis-quoted at 12%", ORANGE)]:
        w = np.array([float(s.total_variance(0.0, T)) for T in Tg])
        a1.plot(Tg, w, color=color, lw=LW, label=name)
        a1.plot(s.T, [float(s.total_variance(0.0, T)) for T in s.T], "o", ms=6, color=color, mec=SURFACE, mew=1.5, ls="none")
    down = np.diff([float(bad.total_variance(0.0, T)) for T in Tg]) < 0
    Tm = 0.5 * (Tg[1:] + Tg[:-1])
    a1.axvspan(Tm[down].min(), Tm[down].max(), color=CRITICAL, alpha=0.12, lw=0)
    a1.set_xlabel("Maturity T (years)")
    a1.set_ylabel("Total variance at the forward (k = 0)")
    a1.set_title("Total variance falls after 6M", loc="left", fontsize=10, color=INK2)
    a1.grid(axis="x", visible=False)
    a1.legend(loc="lower right")

    # the 6M -> 1Y step in total variance, across moneyness
    kk = np.linspace(-0.08, 0.08, 321)
    for s, name, color in [(clean, "clean quotes", MUTED), (bad, "6M ATM mis-quoted at 12%", ORANGE)]:
        a2.plot(kk, s.total_variance(kk, 1.0) - s.total_variance(kk, 0.5), color=color, lw=LW, label=name)
    a2.axhline(0, color=AXIS, lw=0.9)
    step = bad.total_variance(kk, 1.0) - bad.total_variance(kk, 0.5)
    a2.fill_between(kk, step, 0, where=step < 0, color=CRITICAL, alpha=0.15, lw=0)
    a2.set_xlabel("log forward-moneyness k")
    a2.set_ylabel("w(k, 1Y) - w(k, 6M)")
    a2.set_title("6M to 1Y step must be >= 0 at every k; it is < 0 everywhere", loc="left", fontsize=10, color=INK2)
    a2.grid(axis="x", visible=False)
    a2.legend(loc="center right")
    titled(fig, "Where it breaks (calendar): one stale 6M ATM print makes total variance fall between 6M and 1Y",
           f"{SYNTH} Detected cell by cell as w(k, T_long) - w(k, T_short) < 0 (shaded = violating).")
    finish(fig, "arb_calendar.png")


# ----------------------------------------------------------------------- figure 4
def fig_calendar_coordinates():
    s = high_carry_steep_front()
    Tg = np.linspace(0.02, 1.0, 200)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    k_vals = [(-0.04, BLUE), (0.0, MUTED), (0.04, RED)]
    for kval, color in k_vals:
        a1.plot(Tg, [float(s.total_variance(kval, T)) for T in Tg], color=color, lw=LW, label=f"k = {kval:+.2f}")
    F1 = s.forward(1.0)
    for (kval, color) in k_vals:
        K = F1 * np.exp(kval)
        w = np.array([float(s.total_variance(np.log(K / s.forward(T)), T)) for T in Tg])
        a2.plot(Tg, w, color=color, lw=LW, label=f"K = {K:.3f}")
    for ax, t in ((a1, "Fixed k: non-decreasing (arbitrage-free)"), (a2, "Fixed strike: decreases (false alarm)")):
        ax.set_xlabel("Maturity T (years)")
        ax.set_ylabel("Total variance")
        ax.set_title(t, loc="left", fontsize=10, color=INK2)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left")
    lo = min(a1.get_ylim()[0], a2.get_ylim()[0])
    hi = max(a1.get_ylim()[1], a2.get_ylim()[1])
    a1.set_ylim(lo, hi)
    a2.set_ylim(lo, hi)
    titled(fig, "Calendar check coordinate matters: with carry, fixed strike can flag a surface that has no arbitrage",
           f"{SYNTH} ENGINEERED, not realistic: rd - rf = 10%, ATM vol falling like T^-0.4, call-skewed. "
           "The forward drifts through the skew, so a fixed strike sits at a different k at each tenor.")
    finish(fig, "calendar_coordinates.png")


# ----------------------------------------------------------------------- reports
def print_reports():
    scenarios = [
        ("clean EUR/USD-like, Malz", eurusd_like()),
        ("clean EUR/USD-like, Vanna-Volga", eurusd_like("vv")),
        ("Vanna-Volga, 1W butterfly = 1.0 vol pt", vv_short_end_butterfly()),
        ("stale 6M ATM = 12%", stale_6m_atm()),
        ("engineered: 10% carry, steep front, call skew", high_carry_steep_front()),
    ]
    for name, s in scenarios:
        print(f"== {name}")
        print("   " + check_surface(s).summary().replace("\n", "\n   "))
    print("\n== Vanna-Volga: smallest 1W butterfly that breaks the density")
    for bf in (0.0015, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02):
        w = check_butterfly(vv_short_end_butterfly(bf)).worst()
        print(f"   1W BF = {100 * bf:.2f} vol pt: " + ("OK" if w is None else f"VIOLATES, worst g = {w[2]:.3f} at k = {w[1]:+.4f}"))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    print_reports()
    fig_clean_surface()
    fig_butterfly_break()
    fig_calendar_break()
    fig_calendar_coordinates()
    print("\nwrote figures:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
