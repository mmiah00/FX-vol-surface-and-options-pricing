"""Stage 4: build the daily market-state dataset, audit it, and exercise the surface on real and synthetic days.

    python scripts/build_dataset.py [raw_dir] [out_dir]     # defaults: data/raw, data/processed

Reads the downloaded files in raw_dir as untrusted data (run this with `python -I` if you like; nothing here executes
them). Provenance of every number printed or plotted is REAL, ASSUMED or SYNTHETIC and is labelled as such.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_smiles import BLUE, DPI, INK2, LW, MUTED, ORANGE, OUT, SURFACE  # noqa: E402  (also applies shared rcParams)

from fxvol.arbitrage import check_surface  # noqa: E402
from fxvol.data import build_market_state, estr_crosscheck, forward_realized_vol  # noqa: E402
from fxvol.market import (  # noqa: E402
    AssumedParams,
    atm_over_evz_sensitivity,
    atm_term_structure,
    calibrate_atm_over_evz,
    stress_scenarios,
    surface_from_state,
)
from fxvol.synthetic import PILLARS, T_PILLARS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CRITICAL = "#d03b3b"


def assumed_params(df: pd.DataFrame) -> AssumedParams:
    """Long-run vol from the sample mean of non-stale EVZ (REAL-derived; full-sample, so look-ahead for backtests)."""
    ok = df[~df["evz_stale"]]
    p = replace(AssumedParams(), long_run_vol=float(ok["evz"].mean()))
    c = calibrate_atm_over_evz(p, S=float(df["spot"].mean()), rd=float(df["rd"].mean()), rf=float(df["rf"].mean()))
    return replace(p, atm_over_evz=c)


# -------------------------------------------------------------------- data audit
def report_data(df: pd.DataFrame, raw: Path):
    a = df.attrs["audit"]
    print("== REAL DATA: sources and coverage")
    for k in ("spot", "evz", "sofr", "estr"):
        v = a[k]
        print(f"   {k:5s} {v['first']} -> {v['last']}  obs={v['observations']:5d}  blank={v['blank']:4d}")
    print(f"   joined sample: {a['rows']} rows, {a['first']} -> {a['last']}  (limited by EVZ end and EUR STR start)")
    print(f"   dates dropped: spot-only {a['dates_spot_only']} (no EVZ), EVZ-only {a['dates_evz_only']} (no spot), "
          f"{a['rows_dropped_no_prior_rate']} with no prior EUR STR (pre-2019-10)")
    print(f"   EVZ stale repeats (runs of >=3 identical values): {a['evz_stale_days_flagged']} in full series, "
          f"{a['evz_stale_rows_in_sample']} in sample ({100 * a['evz_stale_rows_in_sample'] / a['rows']:.1f}%)")
    print("   ECB vs FRED mirror of EUR STR:", estr_crosscheck(raw))
    print("\n   ranges in sample:")
    for c, lab in [("spot", "EUR/USD"), ("evz", "EVZ"), ("rd", "USD rate (cc)"), ("rf", "EUR rate (cc)"), ("rv21", "trailing 21d realised vol")]:
        s = df[c].dropna()
        print(f"     {lab:26s} min {s.min():8.4f}  mean {s.mean():8.4f}  max {s.max():8.4f}")
    big = df["logret"].abs().nlargest(3)
    print("   largest |daily log returns|:", ", ".join(f"{d.date()} {r:.4f}" for d, r in big.items()))


def report_implied_vs_realized(df: pd.DataFrame):
    d = df.assign(rv_fwd=forward_realized_vol(df)).dropna(subset=["rv_fwd"])
    d = d[~d["evz_stale"]]
    prem = d["evz"] - d["rv_fwd"]
    print("\n== REAL: EVZ (30d implied) vs realised vol over the following 21 returns, non-stale days")
    print(f"   days {len(d)}; mean EVZ {100 * d.evz.mean():.2f}, mean forward realised {100 * d.rv_fwd.mean():.2f}, "
          f"mean premium {100 * prem.mean():+.2f} vol pts; implied > realised on {100 * (prem > 0).mean():.0f}% of days")
    print(f"   premium by year: " + ", ".join(f"{y}: {100 * g.mean():+.2f}" for y, g in prem.groupby(prem.index.year)))
    print(f"   corr(EVZ, forward realised) = {d['evz'].corr(d['rv_fwd']):.2f}")
    pre = prem[prem.index < "2024-01-01"]
    print(f"   before 2024 only: {len(pre)} days, mean premium {100 * pre.mean():+.2f}, implied > realised on {100 * (pre > 0).mean():.0f}%")
    trailing = (df["evz"] > df["rv21"])[df["rv21"].notna()]
    print(f"   EVZ > trailing 21d realised on {100 * trailing.mean():.0f}% of days; by year: "
          + ", ".join(f"{y}: {100 * g.mean():.0f}%" for y, g in trailing.groupby(trailing.index.year)))
    chg = df["evz"].diff().abs() * 100
    print("   EVZ mean |daily change| (vol pts) by year: " + ", ".join(f"{y}: {g.mean():.2f}" for y, g in chg.groupby(chg.index.year))
          + "  | days with |change| > 2: " + ", ".join(f"{y}: {int((g > 2).sum())}" for y, g in chg.groupby(chg.index.year)))
    return d


# ------------------------------------------------------ surfaces on real and synthetic days
def pick_days(df: pd.DataFrame, step: int = 25) -> list[tuple[str, pd.Timestamp]]:
    ok = df[~df["evz_stale"]]
    picks = [("sampled", d) for d in ok.index[::step]]
    picks += [("max EVZ", ok["evz"].idxmax()), ("min EVZ", ok["evz"].idxmin()), ("largest |return|", df["logret"].abs().idxmax())]
    seen, out = set(), []
    for tag, d in picks:
        if d not in seen:
            seen.add(d)
            out.append((tag, d))
    return out


def report_real_days(df: pd.DataFrame, p: AssumedParams):
    print("\n== ASSUMED surface (REAL spot/rates/EVZ + assumed RR/BF/term structure) vs the Stage 3 arbitrage checks")
    days = pick_days(df)
    for model in ("malz", "vv"):
        bad = []
        for tag, d in days:
            r = df.loc[d]
            rep = check_surface(surface_from_state(r.spot, r.rd, r.rf, r.evz, p, model=model), per_interval=1)
            if not rep.ok:
                bad.append((tag, d, rep))
        print(f"   {model}: {len(days) - len(bad)}/{len(days)} real days arbitrage-free "
              f"(pillars + 1 interior point per interval, +-4 sd band)")
        for tag, d, rep in bad:
            first = rep.summary().splitlines()[0]
            print(f"      {d.date()} ({tag}, EVZ {100 * df.loc[d, 'evz']:.1f}): {first}")


def report_stress(df: pd.DataFrame, p: AssumedParams):
    day = df.loc[(df["evz"] - df["evz"].median()).abs().idxmin()]
    print(f"\n== SYNTHETIC stress cases on the median-EVZ real day {day.name.date()} "
          f"(spot {day.spot}, EVZ {100 * day.evz:.1f}, rd {100 * day.rd:.2f}%, rf {100 * day.rf:.2f}%)")
    for model in ("malz", "vv"):
        print(f"   {model}")
        for name, kw in stress_scenarios(day.spot, day.rd, day.rf, day.evz, p).items():
            rep = check_surface(surface_from_state(model=model, **kw))
            verdict = "OK" if rep.ok else "BREAKS: " + "; ".join(l.strip().split(":")[0] for l in rep.summary().splitlines() if "OK" not in l and ":" in l and not l.startswith("  "))
            print(f"      {name:36s} {verdict}")


# ----------------------------------------------------------------------- figures
def titled(fig, title, subtitle):
    h = fig.get_figheight()
    fig.text(0.015, 1 - 0.10 / h, title, fontsize=12, fontweight="semibold", ha="left", va="top")
    fig.text(0.015, 1 - 0.40 / h, subtitle, fontsize=8.5, color=INK2, ha="left", va="top", wrap=True)


def fig_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    a1, a2, a3 = axes
    a1.plot(df.index, df["spot"], color=BLUE, lw=LW)
    a1.set_ylabel("EUR/USD (USD per EUR)")
    a1.set_title("Spot: FRED DEXUSEU, noon New York fix", loc="left", fontsize=10, color=INK2)

    a2.plot(df.index, 100 * df["evz"], color=ORANGE, lw=LW, label="EVZ (30d implied)")
    a2.plot(df.index, 100 * df["rv21"], color=BLUE, lw=LW, label="21d trailing realised")
    stale = df.index[df["evz_stale"]]
    a2.plot(stale, np.full(len(stale), 2.0), "|", color=MUTED, ms=9, mew=1.2, ls="none", label=f"stale EVZ prints ({len(stale)} days)")
    a2.set_ylabel("Annualised vol (%)")
    a2.set_ylim(0, None)
    frac = float((df["evz"] > df["rv21"])[df["rv21"].notna()].mean())
    a2.axvspan(pd.Timestamp("2024-01-01"), df.index[-1], color=MUTED, alpha=0.10, lw=0)
    a2.annotate("EVZ: stale, then jumps", (pd.Timestamp("2024-01-10"), 19.8), fontsize=8.5, color=INK2, va="top")
    a2.set_title(f"EVZ vs 21d trailing realised vol: EVZ is higher on {100 * frac:.0f}% of days", loc="left", fontsize=10, color=INK2)
    a2.legend(loc="upper center", bbox_to_anchor=(0.42, 1.0))

    a3.plot(df.index, df["sofr_pct"], color=BLUE, lw=LW, label="SOFR (USD)")
    a3.plot(df.index, df["estr_pct"], color=ORANGE, lw=LW, label="EUR STR (EUR)")
    a3.axhline(0, color="#c3c2b7", lw=0.9)
    a3.set_ylabel("Overnight rate (%)")
    a3.set_title("Rates: SOFR (FRED) and EUR STR (ECB); the euro rate is negative until September 2022", loc="left", fontsize=10, color=INK2)
    a3.legend(loc="upper left")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    titled(fig, "Real inputs, 2019-10 to 2025-03",
           "Sources: FRED (DEXUSEU, SOFR, EVZCLS), ECB (EUR STR). The sample ends where EVZ ends on FRED (2025-03-11). "
           "Rates enter as the latest print strictly before each date.")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.90 / fig.get_figheight()))
    fig.savefig(OUT / "data_overview.png", dpi=DPI)
    plt.close(fig)


def fig_regimes(df: pd.DataFrame, p: AssumedParams):
    ok = df[~df["evz_stale"]]
    days = {
        "calm (10th pct EVZ)": (ok["evz"] - ok["evz"].quantile(0.10)).abs().idxmin(),
        "typical (median EVZ)": (ok["evz"] - ok["evz"].median()).abs().idxmin(),
        "stress (max EVZ)": ok["evz"].idxmax(),
    }
    colors = [BLUE, MUTED, ORANGE]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for (name, d), color in zip(days.items(), colors):
        r = df.loc[d]
        s = surface_from_state(r.spot, r.rd, r.rf, r.evz, p)
        label = f"{name}: {d.date()}, EVZ {100 * r.evz:.1f}"
        atm = atm_term_structure(r.evz * p.atm_over_evz, T_PILLARS, p.long_run_vol, p.kappa)
        a1.plot(range(5), 100 * atm, color=color, lw=LW, marker="o", ms=6, mec=SURFACE, mew=1.5, label=label)
        sl = s.slice(0.25)
        half = 3.0 * np.sqrt(float(s.total_variance(0.0, 0.25)))
        k = np.linspace(-half, half, 200)
        K = s.forward(0.25) * np.exp(k)
        a2.plot(np.asarray(sl.call_delta(K)), 100 * np.asarray(s.vol_k(k, 0.25)), color=color, lw=LW, label=name)
    a1.set_xticks(range(5), list(PILLARS))
    a1.set_ylabel("ATM vol (%)")
    a1.set_title("ASSUMED ATM term structure from EVZ", loc="left", fontsize=10, color=INK2)
    a1.legend(loc="center left", fontsize=8)
    a2.set_xlabel("Call delta of the strike (put at left, call at right)")
    a2.invert_xaxis()
    a2.set_ylabel("3M implied vol (%)")
    a2.set_title("ASSUMED 3M smile: RR and BF scale with the regime", loc="left", fontsize=10, color=INK2)
    a2.legend(loc="upper center", fontsize=8)
    for ax in (a1, a2):
        ax.grid(axis="x", visible=False)
    titled(fig, "Three real days, one assumed surface model",
           "Spot, rates and EVZ are REAL on each date; RR, BF, the term-structure shape and the EVZ-to-ATM factor are ASSUMED "
           "(not market quotes). Smiles are Malz.")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.90 / fig.get_figheight()))
    fig.savefig(OUT / "assumed_surface_regimes.png", dpi=DPI)
    plt.close(fig)


# -------------------------------------------------------------------------- main
def main(raw: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    df = build_market_state(raw)
    df.to_csv(out / "market_state.csv")
    report_data(df, raw)
    report_implied_vs_realized(df)
    p = assumed_params(df)
    print(f"\n== ASSUMED parameters: long-run vol {100 * p.long_run_vol:.2f}% (sample mean of non-stale EVZ), kappa {p.kappa}, "
          f"atm_over_evz {p.atm_over_evz:.4f}")
    print("   variance-swap/ATM ratio by vol level:", {f"{100 * k:.0f}%": round(v, 4) for k, v in atm_over_evz_sensitivity(p).items()})
    print("   RR/ATM by pillar:", p.rr_over_atm, " BF/ATM:", p.bf_over_atm)
    report_real_days(df, p)
    report_stress(df, p)
    fig_overview(df)
    fig_regimes(df, p)
    (out / "assumed_params.json").write_text(json.dumps(p.__dict__, indent=2, default=list) + "\n")
    print(f"\nwrote {out / 'market_state.csv'} ({len(df)} rows), assumed_params.json, figures/data_overview.png, figures/assumed_surface_regimes.png")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]) if a else ROOT / "data" / "raw", Path(a[1]) if len(a) > 1 else ROOT / "data" / "processed")
