# Findings

Results from Stage 2 (single-tenor smile). All inputs are **synthetic** and illustrative, not market data.

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

## Caveats

- The market-strangle and Vanna-Volga numbers above come from this implementation on synthetic inputs. They have not been compared against an external reference.
- Vanna-Volga weights were verified to reproduce vega, vanna and volga of the target option to about 1e-16.
