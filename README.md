# FX-vol-surface-and-options-pricing
Built a vol surface from the standard market quotes (ATM, 25-delta risk reversal, 25-delta butterfly) using FX delta conventions, then price with Garman-Kohlhagen.


## Setup

```bash
pip install -e ".[dev]"   # numpy, scipy, pandas, pytest, QuantLib, matplotlib
pytest
```

## Stage 1: core pricer (`fxvol/`)

- `gk.py`: Garman-Kohlhagen `price`, `delta`, `gamma`, `vega`, `theta`, `vanna`, `volga`
  (raw partial derivatives; units documented in the module docstring).
- `delta.py`: `fx_delta` and `delta_to_strike` under all four delta conventions.
  Premium-excluded is closed-form; premium-included uses a bracketed root-finder
  (for calls it returns the larger of the two strikes).
- `conventions.py`: `DeltaConvention(delta_type, premium_included)`.
  **EUR/USD default is spot delta, premium excluded** (`EURUSD_DEFAULT`); every function
  that takes a delta accepts a `convention=` argument to switch.

```python
from fxvol import delta_to_strike, DeltaConvention, DeltaType
K25c = delta_to_strike(0.25, S=1.10, T=0.25, rd=0.053, rf=0.035, sigma=0.08)   # default convention
K25p = delta_to_strike(-0.25, 1.10, 0.25, 0.053, 0.035, 0.08,
                       DeltaConvention(DeltaType.FORWARD, premium_included=True))
```

Validation (`tests/`): put-call parity to a few ulps, every Greek against central finite
differences (bumps scaled to `S*sigma*sqrt(T)`), price/delta/gamma/vega/theta against QuantLib's
analytic European engine, and delta<->strike round trips for all four conventions.

## Stage 2: single-tenor smile (`fxvol/smile.py`)

Quotes (ATM, 25Δ RR, 25Δ BF; vols as decimals) become three nodes (25Δ put, ATM, 25Δ call),
which an interpolator then turns into `vol(K)` and `vol_at_delta(Δ)`.

```python
from fxvol.smile import SmileQuote, build_smile
q = SmileQuote(S=1.10, T=0.25, rd=0.053, rf=0.035, atm=0.075, rr=-0.005, bf=0.0025)
smile = build_smile(q, model="malz", strangle="smile")   # model: "malz" | "vv"; strangle: "smile" | "market"
smile.vol(1.12); smile.vol_at_delta(-0.10)
```

- **ATM** defaults to the delta-neutral straddle (`AtmType.DELTA_NEUTRAL`), switchable to forward.
- **Smile strangle** (default): the quoted BF is used directly as the smile butterfly.
  **Market strangle**: BF is the flat-vol strangle's vol minus ATM; the smile BF is solved by a
  1-D root-find so the strangle reprices (needs a model, so the result is model-dependent).
  On the synthetic EUR/USD 3M example the gap is about 0.004 vol pt (Malz) / 0.005 (VV), and a few
  hundredths of a vol point for ATM 12 / BF 1.0.
- **Malz**: exact quadratic in call delta. **Vanna-Volga**: vega/vanna/volga-matching correction to
  ATM-vol Black prices, inverted back to vols (NaN where no implied vol exists).
- `python scripts/plot_smiles.py` writes `figures/smile_models.png` and `figures/rr_bf_intuition.png`
  (**synthetic quotes**, labelled as such on the plots).


## Stage 3: full surface and arbitrage checks (`fxvol/surface.py`, `fxvol/arbitrage.py`)

Five pillars (1W, 1M, 3M, 6M, 1Y), each with its own `rd`/`rf` and a per-tenor smile (Malz or Vanna-Volga),
joined by **linear interpolation in total variance** `w = sigma^2 T`.

```python
from fxvol.synthetic import eurusd_like          # SYNTHETIC quotes, see fxvol/synthetic.py
from fxvol.arbitrage import check_surface
s = eurusd_like()
s.vol(K=1.12, T=0.4); s.vol_at_delta(-0.25, T=0.4)
print(check_surface(s).summary())
```

- **Interpolation coordinate:** fixed log forward-moneyness `k = ln(K/F_T)`, not fixed strike. The
  no-calendar-arbitrage condition lives in `k`; with carry the forward drifts, so fixed strike is a different
  moneyness at each tenor. Outside the pillars the vol is held flat at fixed `k`.
- **Butterfly (density >= 0)**, two independent detectors that must agree: Gatheral's `g(k)` (from finite
  differences of `w`) and the model-free price butterfly `C(K-h) - 2C(K) + C(K+h)`.
- **Calendar (total variance non-decreasing in T)**, reported both at fixed `k` and at fixed strike.
- Violations carry their location (maturity, `k`, strike, delta); points where a model has no vol (NaN) are
  counted separately, never silently passed.
- `python scripts/surface_demo.py` prints the reports and writes `figures/surface_clean.png`,
  `arb_butterfly.png`, `arb_calendar.png`, `calendar_coordinates.png`. Results are written up in
  [findings.md](findings.md).


## Stage 4: data, and which numbers are real

Every quantity in this project is exactly one of **REAL**, **ASSUMED** or **SYNTHETIC**.

| Quantity | Source | Status | Notes |
|---|---|---|---|
| EUR/USD spot | FRED `DEXUSEU` (Federal Reserve H.10 noon New York buying rate) | **REAL** | USD per EUR, daily, about 10 blank days a year (US holidays) |
| USD rate `rd` | FRED `SOFR` | **REAL** (overnight) | Act/360 simple % converted to continuously compounded act/365. Used as-of the previous available date |
| EUR rate `rf` | ECB `EST` (euro short-term rate) | **REAL** (overnight) | Identical to FRED's mirror of it on all 1,827 overlapping dates (checked). Negative until 2022-09-13 (it first rose to -0.09 on 2022-07-27, and turned positive on 2022-09-14) |
| 30-day implied vol | FRED `EVZCLS` (CBOE EUR/USD ETF Volatility Index) | **REAL** | A VIX-style index from options on an EUR/USD ETF, so closer to a variance-swap level than to ATM vol. Ends 2025-03-11 on FRED; see data quality below |
| Rates across tenors | the overnight rates, held flat | **ASSUMED** | No term structure of rates |
| 1M ATM vol | `EVZ x 0.9614` | **ASSUMED** | The factor makes the model's own 30-day variance-swap vol equal EVZ. Depends on the assumed smile below |
| ATM term structure | Forward variance mean-reverts to the sample mean EVZ (7.75%), `kappa = 2` per year | **ASSUMED** | Calendar-arbitrage-free by construction; `kappa` is a guess |
| 25D RR and BF | Fixed fractions of ATM at each pillar (RR -2.9% to -8.9%, BF 2.2% to 4.4% of ATM) | **ASSUMED** | Plausible EUR/USD magnitudes, **not calibrated to any quote**. They scale with the vol regime because they scale with ATM |
| Stress surfaces (6x skew, 6x butterfly, EVZ 4 and 30, inverted front end) | `fxvol.market.stress_scenarios` on a real day | **SYNTHETIC** | Edge cases; each changes one thing |
| Quotes in the Stage 1-3 demos and tests | `fxvol/synthetic.py` | **SYNTHETIC** | Illustrative only |

Consequence: spot, rates and the level of vol are real. **Anything about skew or the wings is assumed**, so a result
that depends on RR or BF reflects the assumption, not the market.

```bash
python scripts/fetch_data.py                 # downloads data/raw/*.csv + manifest.json (URL, time, SHA-256)
python -I scripts/build_dataset.py           # data/processed/market_state.csv, audit report, two figures
```

Raw and processed CSVs are **not committed** (third-party data; `data/raw/manifest.json` is, so downloads can be verified).
Downloaded files are treated as untrusted: the fetcher only writes them after checking they look like the CSV requested, and
the build is meant to be run with `python -I`.

Alignment rules (details in `fxvol/data.py`): rows need spot and EVZ on the same date; nothing is forward-filled; rates are
the latest print **strictly before** each date; spot is a noon fix while EVZ is a close, a mismatch that is not corrected.

Data quality, found while building this (numbers in [findings.md](findings.md)):

- **The usable window is 2019-10-02 to 2025-03-11** (1,352 days): the EUR STR starts 2019-10-01 and EVZ on FRED ends 2025-03-11, although
  spot and rates run to November 2026. CBOE's own EVZ file returned 403, so the series could not be extended.
- **EVZ has stale prints** (95 sample days in runs of three or more identical values; flagged in `evz_stale`) and from 2024 it updates
  irregularly (stale, then jumps). Treat 2024-2025 EVZ with caution.
- **`long_run_vol` is the full-sample mean of EVZ**, which is look-ahead for any backtest. Use an expanding mean or a fixed value there.
