# Findings

Results from Stage 2 (single-tenor smile) and Stage 3 (full surface and arbitrage checks). All inputs are
**synthetic** and illustrative, not market data.

- EUR/USD-like: S = 1.10, T = 3M, rd = 5.3%, rf = 3.5%, ATM 7.5, RR25 = -0.5, BF25 = 0.25 (vol points).
- Delta convention: spot delta, premium excluded. ATM: delta-neutral straddle.
- Reproduce with `python scripts/plot_smiles.py`.

## Smile strangle vs market strangle

The gap is tiny for EUR/USD-like inputs.

- ATM 7.5 / BF 0.25 at 3M: about 0.004 vol point with Malz and about 0.005 with Vanna-Volga.
- ATM 12 / BF 1.0: a few hundredths of a vol point.
- The market strangle depends on the interpolator, because its repricing condition uses the smile vols at the strangle strikes.

Smile BF recovered by the market-strangle solve, minus the quoted BF (in hundredths of a vol point, so +0.41 = 0.0041 vol pt):

| Tenor | EUR/USD-like, Malz | EUR/USD-like, VV | High-vol (ATM 12, RR -2, BF 1), Malz | High-vol, VV |
|-------|-------------------:|-----------------:|-------------------------------------:|-------------:|
| 1W    | +0.41 | +0.58 | +2.98 | +5.12 |
| 1M    | +0.39 | +0.56 | +2.65 | +4.66 |
| 3M    | +0.36 | +0.52 | +2.17 | +4.02 |
| 1Y    | +0.29 | +0.43 | +1.05 | +2.49 |
| 2Y    | +0.23 | +0.35 | +0.07 | +1.14 |
| 5Y    | +0.09 | +0.17 | -2.28 | -2.08 |

Correction to an earlier claim: I first wrote in the module docstring that the gap grows with tenor. Measuring it showed the opposite. The gap shrinks with tenor and can flip sign at 5Y (high-vol case), so the docstring was fixed.

## Malz vs Vanna-Volga

They agree between roughly 15Δ and 40Δ, then split in the wings.

- **Malz wings are flat.** Vol tends to a constant as the strike goes to 0 or infinity. Here that is 9.0% on the put side and 8.0% on the call side.
- **Vanna-Volga wings keep climbing.** At 5Δ put it gives 9.37% against Malz's 8.78%.
- **Vanna-Volga turns back down** at the very far put wing, to about ATM level, and returns NaN far enough out (about 4.5x the forward at 3M). Treat its far wings as unusable.

Vol at fixed deltas (vol points):

| Delta | Malz  | Vanna-Volga |
|------:|------:|------------:|
| -0.05 | 8.776 | 9.367 |
| -0.10 | 8.551 | 8.926 |
| -0.25 | 8.000 | 8.000 |
| +0.25 | 7.500 | 7.500 |
| +0.10 | 7.746 | 7.960 |
| +0.05 | 7.869 | 8.431 |

The 25Δ rows match by construction, since both models are fitted through those nodes.

## Malz formula check

The Malz fit with forward-delta nodes is exactly the textbook formula, `sigma(x) = ATM - 2 RR (x - 1/2) + 16 BF (x - 1/2)^2`. A test checks this (`test_malz_matches_textbook_formula_with_forward_delta`).

## Stage 2 caveats

- The market-strangle and Vanna-Volga numbers above come from this implementation on synthetic inputs. They have not been compared against an external reference.
- Vanna-Volga weights were verified to reproduce vega, vanna and volga of the target option to about 1e-16.


---

# Stage 3: full surface and arbitrage checks

Surface: pillars 1W, 1M, 3M, 6M, 1Y (EUR/USD-like ATM 6.8 to 7.9, RR -0.2 to -0.7, BF 0.15 to 0.35 vol points,
USD about 5.3% to 4.7%, EUR about 3.55% to 3.0%), linear in total variance between pillars. Quotes are defined in
`fxvol/synthetic.py`; reproduce everything with `python scripts/surface_demo.py`.

## What was checked, and how

| Check | Detector | Tolerance |
|-------|----------|-----------|
| Butterfly (density >= 0) | `g(k) < 0`, with `w'`, `w''` from central differences in `k` | -1e-6 (g is O(1)) |
| Butterfly (independent) | `exp(rd T) [C(K-h) - 2C(K) + C(K+h)] / h^2 < 0` on the surface's own GK prices | -1e-9 of peak density |
| Calendar (fixed k) | `w(k, T_long) - w(k, T_short) < 0` | -1e-9 relative |
| Calendar (fixed strike) | same, at fixed K | -1e-9 relative |

Maturities checked: the 5 pillars plus 3 interior points per interval (17 in all), so breaks created by the
interpolation itself would show up. Calendar bands are +-4 standard deviations of the *shorter* tenor in each
interval, the region where that tenor's options have non-negligible value.

## Verification of the detectors themselves

- `g(k)` equals 1 under flat vol (Black-Scholes), and the density it implies integrates to 1.
- The `g(k)` density and the price-butterfly density agree to about 5e-5 relative on the clean surface. They share
  no algebra, so each validates the other.
- On the Vanna-Volga break below, both detectors flag the identical 37 points, with no points flagged by only one.

## Results by scenario

| Scenario | Butterfly | Calendar, fixed k | Calendar, fixed strike |
|----------|-----------|-------------------|------------------------|
| Clean, Malz | OK | OK | OK |
| Clean, Vanna-Volga | OK | OK | OK |
| Vanna-Volga, 1W BF = 1.0 vol pt | **37 points violate** (only the 1W pillar) | OK | OK |
| 6M ATM mis-quoted at 12% (neighbours 7.3%, 7.9%) | OK | **1062 cells violate** | **1015 cells violate** |
| Engineered: 10% carry, ATM falling like T^-0.4, call skew | OK | OK | **385 cells flag (false alarm)** |

## Where it breaks, and how it was detected

**1. Vanna-Volga produces negative densities.**
- At 1W with BF = 1.0 the density goes negative in *both* wings: near 14 delta put (worst g = -0.078 at k = -0.0145) and near 13 delta call.
- Likely cause (consistent with the plots, not isolated by an experiment): the Vanna-Volga smile overshoots the 25 delta pivots and curls back down, the same behaviour noted in Stage 2. The negative-density regions sit where the smile bends back down.
- The same quotes through Malz are arbitrage-free.
- Threshold: about 0.88 vol point of BF at ATM about 6.8% (1W). It is not specific to 1W: BF = 1.0 breaks Vanna-Volga at every pillar (worst g between -0.067 and -0.082).
- The shaded region in `figures/arb_butterfly.png` is each contiguous run of `g < 0`, not a span from first to last bad strike. The first version shaded one span across the healthy middle and was misleading.

**2. A stale quote breaks calendar arbitrage, in both coordinates.**
- A 12% 6M ATM next to 7.3% (3M) and 7.9% (1Y) makes total variance fall from 6M to 1Y at every `k`, not just at the money.
- Because `w = sigma^2 T`, ATM calendar arbitrage needs the shorter tenor's vol to exceed the longer one's by `sqrt(T_long / T_short)`: about 2.1x for 1W to 1M, 1.4x for 6M to 1Y. A realistic term structure rarely trips this at the money; a bad print or an event-vol spike does.
- Butterfly stays clean: a calendar break is not a butterfly break, so both checks are needed.

**3. The calendar check coordinate matters.**
- With carry, `d_T w` at fixed strike equals `d_T w` at fixed `k` minus `carry * d_k w`. Where the skew is steep and total variance grows slowly, the second term wins and fixed-strike total variance falls even though the surface has no arbitrage.
- The reason: `S_T / F_T` is a martingale, so monotonicity in `k` is the right condition (Gatheral-Jacquier). A fixed strike at zero carry is the same as fixed `k`; with carry it is not.
- Confirmed: with zero carry both coordinates give identical results (tested); with 10% carry the engineered surface passes at fixed `k` and flags at fixed strike.
- The stage prompt says "fixed strike". For EUR/USD-sized carry (about 1.8%) the two coordinates give the same pass/fail verdict on every scenario here (cell counts differ slightly, e.g. 1062 vs 1015 on the stale quote), so a verdict difference only showed up with the high-carry case. The engineered case is not meant to look like a real pair.

## Practical notes

- Malz was never driven into a butterfly violation in these tests, including BF = 3.0 vol points at 1W. I did not look for a Malz break exhaustively, so this is an observation, not a guarantee.
- My first calendar check used one global band sized by the longest tenor. That put 1W options 10+ standard deviations out and reported thousands of violations at **zero carry**, where the two coordinates must agree. They were an artifact of the band (Malz wings are bounded), not arbitrage. Per-interval bands sized by the shorter tenor removed them.
- Vol evaluation was the bottleneck: per-strike root-finding made a surface check run past a two-minute timeout. Vectorised bisection in `implied_vol` and the Malz solver brought a full check on the 5-pillar surface to about 0.7 seconds.

## Stage 3 caveats

- Pillar-to-pillar flat forward variance is an assumption of the interpolation, not a market fact.
- Rates are synthetic and interpolated on integrated rates; no discount-curve bootstrap.
- No external reference surface was available, so detector correctness rests on the internal cross-checks above (two independent butterfly methods, analytic limits, zero-carry agreement).
- Arbitrage checks are static and finite-grid: a clean report means no violation was found on the grid, not a proof.


---

# Stage 4: data

Sample: **2019-10-02 to 2025-03-11, 1,352 daily rows**. Spot, rates and EVZ are real; RR, BF and the term-structure shape are assumed
(see the README table). Reproduce with `python scripts/fetch_data.py` then `python -I scripts/build_dataset.py`.

## What constrained the sample

| Series | Coverage | Observations | Blank |
|--------|----------|-------------:|------:|
| EUR/USD (FRED DEXUSEU) | 1999-01-04 to 2026-11-13 | 6,988 | 282 |
| EVZ (FRED EVZCLS) | 2007-11-01 to **2025-03-11** | 4,360 | 169 |
| SOFR (FRED) | 2018-04-03 to 2026-11-18 | 2,155 | 97 |
| EUR STR (ECB) | **2019-10-01** to 2026-11-18 | 1,828 | 0 |

- The window is the intersection: it starts when the EUR STR starts and ends when EVZ ends. Spot and both rates run to November 2026, but there is no EVZ after 2025-03-11, and forward-filling about 20 months of implied vol would have fabricated it.
- CBOE's own EVZ history (`cdn.cboe.com`) returned HTTP 403. I did not try to get around that, so EVZ could not be extended or cross-checked against a second source.
- Date joins dropped 2,674 dates with spot but no EVZ, 46 with EVZ but no spot, and 2,962 pre-2019-10 rows with no prior EUR STR.

## Data quality findings

**FRED encodes missing values as an empty string, not `.`.** A first count with `grep ',\.$'` reported zero missing values; the files actually have 282 (spot), 169 (EVZ) and 97 (SOFR) blanks. The parser maps both `""` and `"."` to NaN and rejects non-numeric text, duplicate dates and unsorted dates.

**The ECB and FRED copies of the EUR STR are identical.** They agree on all 1,827 overlapping dates (max difference 0.0). The row counts differ for two benign reasons, both checked: FRED's mirror has 34 extra rows that are *blank* (TARGET holidays such as 25 and 26 December, 1 January, Easter), and the ECB file has one date the mirror lacks, 2026-11-18, the newest day.

**EVZ has stale prints.** 192 days repeat the previous day's value in the full series. Runs of three or more identical values are not chance at 2-decimal resolution, and they cluster late: days inside such runs (counting each run's first print) are 65 in 2024, 28 in 2021, 16 in 2025, 15 in 2023 and 5 in 2009. Repeats inside those runs are flagged (`evz_stale`): 99 in the full series, **95 in the sample (7.0%)**.

**From 2024 EVZ updates irregularly, and the stale flag does not capture all of it.**

| Year | Mean abs daily change (vol pts) | Days with abs change > 2 |
|------|-------------------------------:|-------------------------:|
| 2019 | 0.19 | 0 |
| 2020 | 0.39 | 7 |
| 2021 | 0.20 | 0 |
| 2022 | 0.40 | 5 |
| 2023 | 0.23 | 0 |
| 2024 | **0.72** | **26** |
| 2025 | 0.47 | 4 |

The lag-1 autocorrelation of EVZ changes is -0.43 in 2024. A late-2024 window shows the pattern directly: a value held for days, then a catch-up jump (6.29, then 7.36, then 9.97). I did not isolate the cause (a feed problem on FRED's side is the most likely reading). Practical consequence: 2019-2023 EVZ is far cleaner than 2024-2025, and the 2024 numbers below should be read with that in mind.

## Real-data sanity checks

- Ranges are sensible: EUR/USD 0.9616 to 1.2295; EVZ 4.13 to 19.31 (peak on 2020-03-18); USD rate 0.01% to 5.47%; EUR rate -0.60% to 3.97% (continuously compounded).
- The EUR STR stays negative until 2022-09-13 and first turns positive on 2022-09-14 (+0.662). The July 2022 hike only moved it from -0.58 to -0.09, because the first move took the ECB deposit rate to 0%.
- Largest daily moves are real events: 2.2% on 2025-03-05, 1.9% on 2023-12-14, 1.8% on 2020-03-12.
- Weekend and holiday gaps are as expected (1-day gaps 1,051, weekend 234, long weekend 48).
- Every rate used is dated 1 to 4 days before its row, never the same day.

## Implied vs realised vol (all real)

EVZ at `t` forecasts the next 30 days, so it is compared with realised vol over the following 21 returns (non-stale days, 1,244 observations):

- Mean EVZ **7.74**, mean forward realised **7.08**: a premium of **+0.66 vol points**; implied exceeds realised on 64% of days.
- Correlation between EVZ and forward realised vol: **0.55**.
- Premium by year: 2019 +1.31, 2020 +0.88, 2021 +0.55, 2022 +0.13, 2023 +0.10, **2024 +1.89, 2025 -1.86** (2025 is only about 2.5 months).
- Before 2024 only: premium **+0.47**, implied above realised on 63% of days. The 2024 figure is the year with the worst EVZ quality, so I would not read +1.89 as a market fact.

Against *trailing* 21-day realised vol, EVZ is higher on 69% of days overall but the share swings from 100% (2019) to 54% (2023) to 22% (2025). An earlier draft of the overview plot said "EVZ is almost always above trailing realised"; that was wrong and the title now states the measured 69%.

Caveat for Stage 5: the premium is measured with noon-to-noon returns from the Fed H.10 series against an index based on options on a USD-listed EUR/USD ETF, not OTC EUR/USD options. It is a reasonable proxy for the variance risk premium, not the premium a dealer's book would earn.

## The assumed layer

- **EVZ to ATM:** the variance-swap-to-ATM ratio of the assumed smile at 30 days is **1.040**, and it barely moves with vol level (1.0396 at 4% vol to 1.0404 at 30%). One factor, **0.9614**, therefore converts EVZ to 1M ATM on every day. It is a property of the assumed RR/BF, not a measurement.
- **Term structure:** `v(t) = theta + (v0 - theta) exp(-kappa t)` with `theta` = 7.75% squared (sample mean of non-stale EVZ) and `kappa = 2`. Total variance increases with maturity for every input tried (4% to 40% 30-day vol), so the ATM term structure cannot create a calendar break. A 30-day vol below about 0.28 times the long-run level would make forward variance negative; the code rejects it instead of clipping.
- On a high-vol day the term structure inverts: on 2020-03-18 (EVZ 19.3) ATM runs from about 19% at 1W to about 14% at 1Y. This is the mean-reversion assumption, not an observation.
- RR and BF are fixed fractions of ATM, so skew and wings widen automatically in high-vol regimes.

## Arbitrage checks on real days (Stage 3 checks, assumed smile)

- Surfaces built for 54 real days (every 25th non-stale day plus the max-EVZ, min-EVZ and largest-move days): **54 of 54 arbitrage-free with Malz, 54 of 54 with Vanna-Volga** (pillars plus one interior maturity per interval, +-4 sd band).
- That says the assumed RR/BF levels are mild enough not to trouble either model. It does not say the surfaces are right; the smile is assumed.

## Synthetic stress tests (on the real median-EVZ day, 2020-05-28)

| Scenario | Malz | Vanna-Volga |
|----------|------|-------------|
| Base day | OK | OK |
| RR x6 (extreme skew) | OK | **breaks**: butterfly, plus points with no implied vol |
| BF x6 (extreme butterfly) | OK | **breaks**: butterfly |
| EVZ 30 (vol spike) | OK | OK |
| EVZ 4 (vol crush) | OK | OK |
| 1W ATM = 2.5x 1M (inverted front end) | **breaks**: calendar | **breaks**: calendar |

- Malz survives every smile and level stress; only a term-structure inversion breaks it, and that is the case the `sqrt(T_long/T_short)` rule predicts (2.5 against a threshold of 2.1 for 1W to 1M).
- Vanna-Volga breaks twice under extreme skew: negative density where it has a vol, and no implied vol at all in other regions. In an earlier run of the 6x-skew case the two butterfly detectors disagreed on 2 of 20 flagged points (the price butterfly flagged 18). That is at g near -25, where the finite-difference steps are the likely culprit; I did not investigate further.

## Stage 4 caveats

- **No real RR or BF anywhere.** Every skew, wing and smile-dependent conclusion rests on the assumed fractions.
- **Rates are overnight and flat across tenors.** Carry over a one-year option would differ from a term-rate curve.
- **`long_run_vol` uses the whole sample**, including the future relative to most days. Fine for a data audit; wrong for a backtest.
- **Timing mismatch** between the noon spot fix and the 4pm EVZ close, and between US and TARGET holiday calendars (handled by the as-of join), is not corrected.
- **EVZ is an ETF-option index**, not an OTC measure, and its late history is degraded.
- The 54-day arbitrage sample is a sample, not every day.
