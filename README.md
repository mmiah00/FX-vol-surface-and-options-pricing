# FX-vol-surface-and-options-pricing
Built a vol surface from the standard market quotes (ATM, 25-delta risk reversal, 25-delta butterfly) using FX delta conventions, then price with Garman-Kohlhagen.


## Setup

```bash
pip install -e ".[dev]"   # numpy, scipy, pytest, QuantLib
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
