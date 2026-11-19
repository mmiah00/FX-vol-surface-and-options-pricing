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
