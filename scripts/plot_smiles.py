"""Stage 2 figures: single-tenor smile in delta space and strike space.

SYNTHETIC DATA: the quotes below are illustrative EUR/USD-like numbers, not market data.

    python scripts/plot_smiles.py            # writes figures/*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fxvol.smile import SmileQuote, build_smile

OUT = Path(__file__).resolve().parents[1] / "figures"

# --- design tokens (dataviz reference palette, light mode) -------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1, 2
RED = "#e34948"  # diverging pole opposite blue
BLUE_RAMP = ["#6da7ec", "#2a78d6", "#104281"]  # sequential blue steps 300 / 450 / 650

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9.5,
        "legend.frameon": True,
        "legend.facecolor": SURFACE,
        "legend.edgecolor": "none",
        "legend.framealpha": 1.0,
        "legend.fontsize": 9,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)
LW = 1.5  # ~2px at the saved resolution
DPI = 200

BASE = dict(S=1.10, T=0.25, rd=0.053, rf=0.035)  # rd = USD, rf = EUR
SUBTITLE = (
    "Synthetic EUR/USD 3M quotes (illustrative, not market data). Spot delta, premium excluded; "
    "ATM = delta-neutral straddle. Grey vertical line = forward."
)


def curve(smile, z_max=2.8, n=400):
    """(strike, vol, spot call delta) on a grid uniform in standardised log-moneyness."""
    q = smile.quote
    z = np.linspace(-z_max, z_max, n)
    K = q.F * np.exp(z * q.atm * np.sqrt(q.T))
    return K, np.asarray(smile.vol(K)), np.asarray(smile.call_delta(K))


def style_axes(ax, xlabel, invert=False):
    ax.set_xlabel(xlabel)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.1f}")
    ax.grid(axis="x", visible=False)
    if invert:
        ax.invert_xaxis()


# (x offset, y offset, horizontal alignment) in points, per node: put / ATM / call.
# Put: above-right (the curve falls away to the right). ATM: below. Call: below-right (the curve rises to the right).
NODE_LABEL_POS = [(6, 8, "left"), (0, -18, "center"), (6, -18, "left")]


def mark_nodes(ax, xs, ys, labels):
    # neutral ink, not a series color: nodes are inputs shared by every model
    ax.plot(xs, ys, "o", ms=6, color=INK, mec=SURFACE, mew=1.5, zorder=5, ls="none")
    for x, y, lab, (dx, dy, ha) in zip(xs, ys, labels, NODE_LABEL_POS):
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8.5, color=INK2)


def end_label(ax, x, y, text, below=False):
    """Direct label at the right-hand end of a line, in ink (never the series color)."""
    ok = np.isfinite(y)
    ax.annotate(text, (x[ok][-1], y[ok][-1]), xytext=(-2, -14 if below else 7), textcoords="offset points", ha="right", fontsize=8.5, color=INK2)


def forward_line(ax, F):
    ax.axvline(F, color=AXIS, lw=0.9, zorder=1)


def titled(fig, title):
    h = fig.get_figheight()
    fig.text(0.015, 1 - 0.10 / h, title, fontsize=12, fontweight="semibold", ha="left", va="top")
    fig.text(0.015, 1 - 0.40 / h, SUBTITLE, fontsize=8.5, color=INK2, ha="left", va="top")


# --- Figure 1: Malz vs Vanna-Volga, both coordinate systems --------------------
def fig_models():
    q = SmileQuote(**BASE, atm=0.075, rr=-0.005, bf=0.0025)
    models = {"Malz (quadratic in delta)": ("malz", BLUE), "Vanna-Volga": ("vv", ORANGE)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for label, (m, color) in models.items():
        sm = build_smile(q, m)
        K, v, cd = curve(sm)
        a1.plot(cd, v, color=color, lw=LW, label=label)
        a2.plot(K, v, color=color, lw=LW, label=label)
        short = label.split(" ")[0]
        end_label(a1, cd, v, short, below=(m == "malz"))
        end_label(a2, K, v, short)
    nodes = sm.nodes
    node_cd = [float(sm.call_delta(k)) for k in nodes.strikes]
    labels = ["25Δ put", "ATM", "25Δ call"]
    for ax in (a1, a2):
        ax.set_ylim(7.2e-2, 9.8e-2)
    mark_nodes(a1, node_cd, nodes.vols, labels)
    mark_nodes(a2, nodes.strikes, nodes.vols, labels)
    forward_line(a2, q.F)
    style_axes(a1, "Call delta of the strike (put at left, call at right)", invert=True)
    style_axes(a2, "Strike (USD per EUR)")
    a1.set_ylabel("Implied vol (%)")
    a1.set_title("Delta space", loc="left", fontsize=10, color=INK2)
    a2.set_title("Strike space", loc="left", fontsize=10, color=INK2)
    a1.legend(loc="upper center")
    a2.legend(loc="upper center")
    titled(fig, "Same three quotes, two interpolators: they agree between the pivots and split in the wings")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / fig.get_figheight()))
    fig.savefig(OUT / "smile_models.png", dpi=DPI)
    plt.close(fig)


# --- Figure 2: what RR and BF do ------------------------------------------------
def fig_rr_bf():
    sweeps = [
        (
            ("RR sweep", "BF = 0.25"),
            [(-0.010, "RR = −1.0", BLUE), (0.0, "RR = 0", MUTED), (0.010, "RR = +1.0", RED)],
            lambda v: dict(rr=v, bf=0.0025),
        ),
        (
            ("BF sweep", "RR = 0"),
            [(0.0, "BF = 0", BLUE_RAMP[0]), (0.005, "BF = 0.5", BLUE_RAMP[1]), (0.010, "BF = 1.0", BLUE_RAMP[2])],
            lambda v: dict(rr=0.0, bf=v),
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for row, ((name, fixed), series, kw) in enumerate(sweeps):
        ad, ak = axes[row]
        for val, label, color in series:
            sm = build_smile(SmileQuote(**BASE, atm=0.075, **kw(val)), "malz")
            K, v, cd = curve(sm)
            ad.plot(cd, v, color=color, lw=LW, label=label)
            ak.plot(K, v, color=color, lw=LW, label=label)
            end_label(ad, cd, v, label)
            end_label(ak, K, v, label)
        q = sm.quote
        forward_line(ak, q.F)
        for ax, xl, inv in ((ad, "Call delta of the strike (put at left, call at right)", True), (ak, "Strike (USD per EUR)", False)):
            style_axes(ax, xl, invert=inv)
            ax.set_ylabel("Implied vol (%)")
        ad.set_title(f"{name} · delta space ({fixed})", loc="left", fontsize=10, color=INK2)
        ak.set_title(f"{name} · strike space", loc="left", fontsize=10, color=INK2)
        ad.legend(loc="upper center")
        ak.legend(loc="upper center")
    titled(fig, "RR tilts the smile; BF lifts both wings and leaves ATM alone (Malz interpolation)")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / fig.get_figheight()))
    fig.savefig(OUT / "rr_bf_intuition.png", dpi=DPI)
    plt.close(fig)


def print_summary():
    q = SmileQuote(**BASE, atm=0.075, rr=-0.005, bf=0.0025)
    print("Synthetic EUR/USD 3M: ATM 7.50, RR25 -0.50, BF25 0.25 (vol pts)\n")
    print(f"{'model':6s} {'strangle':8s} {'b (smile BF)':>13s} {'25dP vol':>9s} {'25dC vol':>9s} {'25dP K':>8s} {'25dC K':>8s}")
    for m in ("malz", "vv"):
        for st in ("smile", "market"):
            s = build_smile(q, m, st)
            print(
                f"{m:6s} {st:8s} {100 * s.bf_smile:13.5f} {100 * s.nodes.vols[0]:9.4f} {100 * s.nodes.vols[2]:9.4f} "
                f"{s.nodes.strikes[0]:8.5f} {s.nodes.strikes[2]:8.5f}"
            )
    print("\nvol at fixed deltas (vol pts)")
    ms, vv = build_smile(q, "malz"), build_smile(q, "vv")
    print(f"{'delta':>6s} {'malz':>8s} {'vv':>8s}")
    for d in (-0.05, -0.10, -0.25, 0.25, 0.10, 0.05):
        print(f"{d:+6.2f} {100 * ms.vol_at_delta(d):8.3f} {100 * vv.vol_at_delta(d):8.3f}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_models()
    fig_rr_bf()
    print_summary()
