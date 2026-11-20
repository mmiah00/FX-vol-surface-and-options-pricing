import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxvol.data import (
    act360_pct_to_cc,
    build_market_state,
    estr_crosscheck,
    flag_stale_runs,
    read_ecb_estr,
    read_fred,
)

REAL_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def write(path, text):
    path.write_text(text)
    return path


# ----------------------------------------------------------------------- parsing


def test_read_fred_treats_blank_and_dot_as_missing(tmp_path):
    p = write(tmp_path / "x.csv", "observation_date,DEXUSEU\n2020-01-02,1.1\n2020-01-03,\n2020-01-06,.\n2020-01-07,1.2\n")
    s = read_fred(p)
    assert s.isna().tolist() == [False, True, True, False]
    assert s.iloc[3] == 1.2


@pytest.mark.parametrize(
    "text",
    [
        "date,v\n2020-01-02,1\n",  # wrong header
        "observation_date,A,B\n2020-01-02,1,2\n",  # wrong shape
        "observation_date,A\n2020-01-02,1\n2020-01-02,2\n",  # duplicate dates
        "observation_date,A\n2020-01-03,1\n2020-01-02,2\n",  # not increasing
        "observation_date,A\n2020-01-02,abc\n",  # non-numeric
    ],
)
def test_read_fred_rejects_malformed_files(tmp_path, text):
    with pytest.raises(ValueError):
        read_fred(write(tmp_path / "x.csv", text))


def test_read_fred_rejects_an_html_error_page(tmp_path):
    with pytest.raises(Exception):
        read_fred(write(tmp_path / "x.csv", "<html><body>403 Forbidden</body></html>"))


def test_read_ecb_estr_parses_and_rejects_bad_status(tmp_path):
    good = "KEY,TIME_PERIOD,OBS_VALUE,OBS_STATUS,EXTRA\nk,2019-10-01,-0.549,A,x\nk,2019-10-02,-0.551,A,x\n"
    s = read_ecb_estr(write(tmp_path / "e.csv", good))
    assert s.tolist() == [-0.549, -0.551]
    with pytest.raises(ValueError):
        read_ecb_estr(write(tmp_path / "b.csv", good.replace(",-0.551,A,", ",-0.551,E,")))


# --------------------------------------------------------------------- transforms


def test_rate_conversion_matches_definition_and_sign():
    pct = np.array([-0.55, 0.0, 1.0, 5.0])
    cc = act360_pct_to_cc(pct)
    x = pct / 100 / 360  # |x| ~ 1e-4: the Taylor series is an independent reference more accurate than log(1 + x)
    np.testing.assert_allclose(cc, 365 * (x - x**2 / 2 + x**3 / 3 - x**4 / 4), rtol=1e-13, atol=0)
    assert cc[1] == 0 and cc[0] < 0 < cc[2] < cc[3]
    # one overnight day at the simple rate grows the same as one day at the cc rate
    assert np.exp(cc[3] / 365) == pytest.approx(1 + 0.05 / 360, rel=1e-14)


def test_flag_stale_runs_flags_repeats_in_runs_of_three_or_more_only():
    s = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0], index=pd.date_range("2020-01-01", periods=10))
    flag = flag_stale_runs(s, min_run=3)
    # (2, 2) and (5, 5) are runs of two: chance-sized, not flagged. The four 3s are flagged after the first print.
    assert flag.tolist() == [False, False, False, False, True, True, True, False, False, False]


def test_flag_stale_runs_ignores_nans():
    s = pd.Series([1.0, np.nan, 1.0, 1.0, 1.0], index=pd.date_range("2020-01-01", periods=5))
    assert flag_stale_runs(s).sum() == 3  # NaN is dropped first, leaving a run of four 1.0: three repeats


# ----------------------------------------------------------------------- pipeline


def make_raw(tmp_path, spot_rows, evz_rows, sofr_rows, estr_rows, crosscheck_rows=None):
    fred = lambda name, rows: write(
        tmp_path / name, f"observation_date,{name}\n" + "".join(f"{d},{'' if v is None else v}\n" for d, v in rows)
    )
    fred("fred_DEXUSEU.csv", spot_rows)
    fred("fred_EVZCLS.csv", evz_rows)
    fred("fred_SOFR.csv", sofr_rows)
    fred("fred_ECBESTR_crosscheck.csv", crosscheck_rows or estr_rows)
    write(
        tmp_path / "ecb_ESTR.csv",
        "KEY,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n" + "".join(f"k,{d},{v},A\n" for d, v in estr_rows),
    )
    return tmp_path


@pytest.fixture
def raw(tmp_path):
    # Thu 2020-01-02 .. Wed 2020-01-08, with a blank spot on Mon 01-06 and a spot-less EVZ print on Tue 01-07
    spot = [("2020-01-02", 1.10), ("2020-01-03", 1.11), ("2020-01-06", None), ("2020-01-07", 1.12), ("2020-01-08", 1.13)]
    evz = [("2020-01-02", 7.0), ("2020-01-03", 7.5), ("2020-01-06", 8.0), ("2020-01-07", 8.0), ("2020-01-08", 8.0)]
    sofr = [("2020-01-01", 1.50), ("2020-01-02", 1.60), ("2020-01-03", 1.70), ("2020-01-06", 1.80), ("2020-01-07", 1.90)]
    estr = [("2020-01-02", -0.45), ("2020-01-03", -0.46), ("2020-01-06", -0.47), ("2020-01-07", -0.48)]
    return make_raw(tmp_path, spot, evz, sofr, estr)


def test_pipeline_drops_dates_without_spot_and_converts_units(raw):
    df = build_market_state(raw)
    assert [d.strftime("%m-%d") for d in df.index] == ["01-03", "01-07", "01-08"]  # 01-02 has no prior EUR STR; 01-06 no spot
    assert df["evz"].tolist() == [0.075, 0.08, 0.08]  # percent -> decimal vol
    assert df["spot"].tolist() == [1.11, 1.12, 1.13]


def test_rates_are_taken_strictly_before_the_row_date(raw):
    df = build_market_state(raw)
    # 01-03 uses the 01-02 prints; 01-07 uses the 01-06 prints (never the same-day value); 01-08 uses 01-07
    assert df["sofr_pct"].tolist() == [1.60, 1.80, 1.90]
    assert df["estr_pct"].tolist() == [-0.45, -0.47, -0.48]
    assert (df.index > df["sofr_date"]).all() and (df.index > df["estr_date"]).all()
    np.testing.assert_allclose(df["rd"], act360_pct_to_cc(df["sofr_pct"]))
    assert df["rf"].iloc[0] < 0  # negative euro rate survives


def test_returns_span_gaps_and_are_labelled(raw):
    df = build_market_state(raw)
    assert np.isnan(df["logret"].iloc[0])
    assert df["logret"].iloc[1] == pytest.approx(np.log(1.12 / 1.11))
    assert df["gap_days"].tolist()[1:] == [4.0, 1.0]  # Fri 01-03 -> Tue 01-07 skips the blank 01-06


def test_audit_reports_what_was_dropped(raw):
    a = build_market_state(raw).attrs["audit"]
    assert a["spot"]["blank"] == 1 and a["evz"]["observations"] == 5
    assert a["dates_evz_only"] == 1  # 01-06: EVZ printed, spot blank
    assert a["rows_dropped_no_prior_rate"] == 1  # 01-02: no EUR STR strictly before it
    assert a["rows"] == 3 and a["first"] == "2020-01-03" and a["last"] == "2020-01-08"


def test_stale_flag_is_carried_into_the_state_table(tmp_path):
    n = 8
    dates = pd.bdate_range("2020-01-02", periods=n).strftime("%Y-%m-%d")
    spot = [(d, 1.1) for d in dates]
    evz = [(d, v) for d, v in zip(dates, [7.0, 7.1, 8.0, 8.0, 8.0, 8.0, 7.2, 7.3])]
    sofr = [(d, 1.0) for d in ["2019-12-31", *dates]]
    estr = [(d, -0.5) for d in ["2019-12-31", *dates]]
    df = build_market_state(make_raw(tmp_path, spot, evz, sofr, estr))
    assert df["evz_stale"].tolist() == [False, False, False, True, True, True, False, False]


def test_rate_older_than_the_staleness_limit_is_missing(tmp_path):
    spot = [("2020-02-03", 1.1), ("2020-02-04", 1.1)]
    evz = [("2020-02-03", 7.0), ("2020-02-04", 7.0)]
    old = [("2020-01-02", 1.0)]  # a month old
    with pytest.raises(IndexError):  # every row loses its rate, so nothing is left to index
        build_market_state(make_raw(tmp_path, spot, evz, old, old))


def test_estr_crosscheck_detects_a_mismatch(tmp_path):
    estr = [("2020-01-02", -0.45), ("2020-01-03", -0.46)]
    mirror = [("2020-01-02", -0.45), ("2020-01-03", -0.47), ("2020-01-06", -0.5)]
    r = make_raw(tmp_path, [("2020-01-02", 1.1)], [("2020-01-02", 7.0)], [("2020-01-01", 1.0)], estr, crosscheck_rows=mirror)
    c = estr_crosscheck(r)
    assert c["overlap_days"] == 2 and c["max_abs_diff_pct"] == pytest.approx(0.01)
    assert c["dates_only_in_fred"] == 1 and c["dates_only_in_ecb"] == 0


# --------------------------------------------------------- real data (if downloaded)

needs_data = pytest.mark.skipif(not (REAL_RAW / "fred_EVZCLS.csv").exists(), reason="run scripts/fetch_data.py first")


@needs_data
def test_real_data_invariants():
    df = build_market_state(REAL_RAW)
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert df[["spot", "evz", "rd", "rf", "sofr_pct", "estr_pct"]].notna().all().all()
    assert df["spot"].between(0.8, 1.7).all()
    assert df["evz"].between(0.03, 0.45).all()
    assert df["rd"].between(-0.01, 0.08).all() and df["rf"].between(-0.01, 0.08).all()
    assert ((df.index.to_series() - df["sofr_date"]).dt.days >= 1).all()
    assert df.index[0] >= pd.Timestamp("2019-10-02")  # EUR STR starts 2019-10-01; a prior-day rate is needed
    assert df["gap_days"].dropna().between(1, 7).all()


@needs_data
def test_real_data_ecb_and_fred_mirror_agree():
    assert estr_crosscheck(REAL_RAW)["max_abs_diff_pct"] == 0.0


@needs_data
def test_manifest_hashes_match_files():
    import hashlib

    manifest = json.loads((REAL_RAW / "manifest.json").read_text())
    assert manifest
    for m in manifest:
        assert hashlib.sha256((REAL_RAW / m["file"]).read_bytes()).hexdigest() == m["sha256"]
