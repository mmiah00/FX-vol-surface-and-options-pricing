"""Loading, cleaning and aligning the REAL public series.

Sources (see ``scripts/fetch_data.py``)::

    spot  FRED DEXUSEU   USD per EUR, Federal Reserve H.10 noon buying rate (New York), daily
    sofr  FRED SOFR      US secured overnight financing rate, percent, act/360
    estr  ECB  EST       euro short-term rate (EUR STR), percent, act/360
    evz   FRED EVZCLS    CBOE EUR/USD ETF Volatility Index, 30-day implied vol in percent

Everything produced here is *Real* data (cleaned and aligned, not modelled). Assumed and Synthetic
layers live in :mod:`fxvol.market`.

Timing and alignment, stated once
---------------------------------
* The spot is a *noon* New York fix and EVZ is a *close* (about 4pm New York), so on any date EVZ is
  observed a few hours after the spot. That is not corrected.
* A rate dated ``t`` is published on the next business day. Rates are therefore joined **as-of the
  previous available date** (strictly before ``t``), which also reconciles the US and TARGET holiday calendars.
  ``sofr_date`` / ``estr_date`` record which observation was used.
* Rows are kept only where both spot and EVZ exist on the same date. Nothing is forward-filled.

Rates are overnight; using them for every tenor assumes a flat curve (an Assumed simplification, see README).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RATE_MAX_STALENESS = pd.Timedelta(7, unit="D")  # an as-of rate older than this is treated as missing


# ----------------------------------------------------------------------------- parsing


def read_fred(path: str | Path) -> pd.Series:
    """FRED ``fredgraph.csv``: ``observation_date,SERIES``. Missing values are blank (older exports used ``.``)."""
    df = pd.read_csv(path, na_values=["", "."])
    if df.shape[1] != 2 or df.columns[0] != "observation_date":
        raise ValueError(f"{path}: expected columns observation_date,<series>; got {list(df.columns)}")
    s = pd.Series(
        pd.to_numeric(df.iloc[:, 1], errors="raise").to_numpy(),
        index=pd.to_datetime(df["observation_date"], errors="raise"),
        name=df.columns[1],
    )
    if s.index.has_duplicates or not s.index.is_monotonic_increasing:
        raise ValueError(f"{path}: dates must be unique and increasing")
    return s


def read_ecb_estr(path: str | Path) -> pd.Series:
    """ECB data API ``csvdata`` for the euro short-term rate; percent, one value per TARGET business day."""
    df = pd.read_csv(path, usecols=["TIME_PERIOD", "OBS_VALUE", "OBS_STATUS"])
    if not (df["OBS_STATUS"] == "A").all():
        raise ValueError(f"{path}: non-normal OBS_STATUS present: {df['OBS_STATUS'].unique().tolist()}")
    s = pd.Series(df["OBS_VALUE"].astype(float).to_numpy(), index=pd.to_datetime(df["TIME_PERIOD"]), name="estr")
    if s.index.has_duplicates or not s.index.is_monotonic_increasing:
        raise ValueError(f"{path}: dates must be unique and increasing")
    return s


# --------------------------------------------------------------------------- transforms


def act360_pct_to_cc(rate_pct):
    """Overnight act/360 simple rate in percent -> continuously compounded, act/365 (what the pricer uses).

    One day grows by ``1 + r/360``; over a 365-day year that is ``exp(rate_cc)``, so ``rate_cc = 365 ln(1 + r/360)``.
    """
    r = np.asarray(rate_pct, dtype=float) / 100.0
    return 365.0 * np.log1p(r / 360.0)


def flag_stale_runs(s: pd.Series, min_run: int = 3) -> pd.Series:
    """True on every repeat inside a run of ``>= min_run`` identical consecutive values (the first print is kept).

    Index levels print to 2 decimals and move by tenths of a point per day, so a single repeat can be chance;
    a run of three or more is not behaving like a market price.
    """
    s = s.dropna()
    run = (s != s.shift()).cumsum()
    length = s.groupby(run).transform("size")
    position = s.groupby(run).cumcount()
    return (length >= min_run) & (position >= 1)


def _asof(left: pd.DataFrame, series: pd.Series, value_name: str, date_name: str) -> pd.DataFrame:
    """Attach, to each left date, the latest ``series`` value strictly before it (and its source date)."""
    right = series.dropna().rename(value_name).to_frame()
    right[date_name] = right.index
    out = pd.merge_asof(
        left.reset_index(names="date"),
        right.reset_index(names="date"),
        on="date",
        direction="backward",
        allow_exact_matches=False,
        tolerance=RATE_MAX_STALENESS,
    )
    return out.set_index("date")


# ------------------------------------------------------------------------- the pipeline


def estr_crosscheck(raw_dir: str | Path) -> dict:
    """Compare the ECB's own EUR STR with FRED's mirror of it. They should be identical where both exist."""
    raw = Path(raw_dir)
    ecb = read_ecb_estr(raw / "ecb_ESTR.csv")
    fred = read_fred(raw / "fred_ECBESTR_crosscheck.csv").dropna()
    both = pd.concat([ecb, fred], axis=1, keys=["ecb", "fred"]).dropna()
    return {
        "overlap_days": int(len(both)),
        "max_abs_diff_pct": float((both["ecb"] - both["fred"]).abs().max()),
        "dates_only_in_ecb": int(ecb.index.difference(fred.index).size),
        "dates_only_in_fred": int(fred.index.difference(ecb.index).size),
    }


def build_market_state(raw_dir: str | Path, stale_min_run: int = 3) -> pd.DataFrame:
    """Daily REAL market state, one row per date with spot and EVZ.

    Columns: ``spot, evz, evz_stale, sofr_pct, sofr_date, estr_pct, estr_date, rd, rf, logret, gap_days, rv21``.
    ``rd`` is USD (domestic), ``rf`` is EUR (foreign), both continuously compounded and flat across tenors.
    ``logret`` is the log spot return from the previous row (spans weekends and holidays; ``gap_days`` says how many
    calendar days). ``rv21`` is the 21-observation realised vol, annualised with ``sqrt(252)``, as a decimal.
    ``evz`` is in vol points divided by 100, i.e. a decimal vol. A dict describing what was dropped is in
    ``df.attrs["audit"]``.
    """
    raw = Path(raw_dir)
    spot = read_fred(raw / "fred_DEXUSEU.csv")
    evz = read_fred(raw / "fred_EVZCLS.csv")
    sofr = read_fred(raw / "fred_SOFR.csv")
    estr = read_ecb_estr(raw / "ecb_ESTR.csv")

    audit: dict = {
        "spot": _span(spot),
        "evz": _span(evz),
        "sofr": _span(sofr),
        "estr": _span(estr),
    }
    stale = flag_stale_runs(evz, stale_min_run)
    audit["evz_stale_days_flagged"] = int(stale.sum())

    both = pd.concat([spot.rename("spot"), evz.rename("evz")], axis=1)
    audit["dates_union"] = int(len(both))
    audit["dates_spot_only"] = int((both["spot"].notna() & both["evz"].isna()).sum())
    audit["dates_evz_only"] = int((both["spot"].isna() & both["evz"].notna()).sum())
    df = both.dropna().copy()
    df["evz"] = df["evz"] / 100.0
    df["evz_stale"] = stale.reindex(df.index).fillna(False).astype(bool)

    df = _asof(df, sofr, "sofr_pct", "sofr_date")
    df = _asof(df, estr, "estr_pct", "estr_date")
    n_before = len(df)
    df = df.dropna(subset=["sofr_pct", "estr_pct"])
    # in practice these are the 2007-2019 rows that predate the EUR STR (first published 2019-10-01)
    audit["rows_dropped_no_prior_rate"] = n_before - len(df)

    df["rd"] = act360_pct_to_cc(df["sofr_pct"])
    df["rf"] = act360_pct_to_cc(df["estr_pct"])
    df["logret"] = np.log(df["spot"]).diff()
    df["gap_days"] = df.index.to_series().diff().dt.days
    df["rv21"] = df["logret"].rolling(21).std() * np.sqrt(TRADING_DAYS)

    audit["rows"] = int(len(df))
    audit["first"], audit["last"] = str(df.index[0].date()), str(df.index[-1].date())
    audit["evz_stale_rows_in_sample"] = int(df["evz_stale"].sum())
    df.attrs["audit"] = audit
    return df


def _span(s: pd.Series) -> dict:
    v = s.dropna()
    return {
        "first": str(v.index[0].date()),
        "last": str(v.index[-1].date()),
        "observations": int(len(v)),
        "blank": int(s.isna().sum()),
    }


def forward_realized_vol(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Realised vol over the *next* ``window`` returns, annualised with ``sqrt(252)`` (decimal).

    This is what EVZ at date ``t`` is a forecast of, so it is the right thing to compare EVZ with. It uses FUTURE
    returns: for evaluation only, never as an input to anything meant to be known at ``t``.
    """
    r = df["logret"]
    fwd = r[::-1].rolling(window).std()[::-1].shift(-1)  # std of r[t+1 .. t+window]
    return (fwd * np.sqrt(TRADING_DAYS)).rename("rv_fwd")
