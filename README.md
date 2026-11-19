# FX-vol-surface-and-options-pricing
Built a vol surface from the standard market quotes (ATM, 25-delta risk reversal, 25-delta butterfly) using FX delta conventions, then price with Garman-Kohlhagen.


## Setup

```bash
pip install -e ".[dev]"   # numpy, scipy, pytest, QuantLib, matplotlib
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
