import itertools

import numpy as np
import pytest

from fxvol import (
    EURUSD_DEFAULT,
    DeltaConvention,
    DeltaType,
    delta_to_strike,
    forward,
    fx_delta,
    gk,
)

CONVENTIONS = [
    DeltaConvention(t, pi) for t, pi in itertools.product(DeltaType, (False, True))
]
DELTAS = [0.10, 0.25, 0.40, 0.50]  # magnitudes; sign added per option type
MARKETS = [
    (1.10, 1 / 52, 0.053, 0.035, 0.07),
    (1.10, 0.25, 0.053, 0.035, 0.08),
    (1.10, 1.0, 0.045, 0.025, 0.09),
    (1.25, 2.0, -0.005, 0.03, 0.12),
    (0.95, 5.0, 0.04, 0.01, 0.15),
]
cid = lambda c: f"{c.delta_type.value}-{'pi' if c.premium_included else 'px'}"


def test_default_is_spot_premium_excluded():
    assert EURUSD_DEFAULT == DeltaConvention(DeltaType.SPOT, False)


@pytest.mark.parametrize("conv", CONVENTIONS, ids=cid)
@pytest.mark.parametrize("S,T,rd,rf,vol", MARKETS)
@pytest.mark.parametrize("mag", DELTAS)
@pytest.mark.parametrize("sign", [1, -1])
def test_round_trip(conv, S, T, rd, rf, vol, mag, sign):
    """delta -> strike -> delta recovers the input, in every convention."""
    target = sign * mag
    scale = np.exp(-rf * T) if conv.delta_type is DeltaType.SPOT else 1.0
    target *= scale  # keep the target inside the attainable range for spot
    try:
        K = delta_to_strike(target, S, T, rd, rf, vol, conv)
    except ValueError:
        # only legitimate for premium-included calls above the delta maximum
        assert conv.premium_included and sign > 0
        return
    back = fx_delta(S, K, T, rd, rf, vol, sign > 0, conv)
    assert back == pytest.approx(target, abs=1e-12)


@pytest.mark.parametrize("S,T,rd,rf,vol", MARKETS)
def test_premium_excluded_forward_closed_form(S, T, rd, rf, vol):
    from scipy.stats import norm

    F = forward(S, T, rd, rf)
    conv = DeltaConvention(DeltaType.FORWARD, False)
    k25c = delta_to_strike(0.25, S, T, rd, rf, vol, conv)
    k25p = delta_to_strike(-0.25, S, T, rd, rf, vol, conv)
    s = vol * np.sqrt(T)
    assert k25c == pytest.approx(F * np.exp(-s * norm.ppf(0.25) + 0.5 * s * s), rel=1e-13)
    assert k25p == pytest.approx(F * np.exp(+s * norm.ppf(0.25) + 0.5 * s * s), rel=1e-13)
    assert k25p < F < k25c


def test_spot_and_forward_agree_up_to_discount_factor():
    S, T, rd, rf, vol = 1.1, 2.0, 0.04, 0.02, 0.1
    for pi in (False, True):
        spot, fwd = DeltaConvention(DeltaType.SPOT, pi), DeltaConvention(DeltaType.FORWARD, pi)
        k1 = delta_to_strike(0.25 * np.exp(-rf * T), S, T, rd, rf, vol, spot)
        k2 = delta_to_strike(0.25, S, T, rd, rf, vol, fwd)
        assert k1 == pytest.approx(k2, rel=1e-13)


def test_spot_premium_excluded_matches_gk_delta():
    S, K, T, rd, rf, vol = 1.1, 1.17, 0.75, 0.05, 0.03, 0.09
    for is_call in (True, False):
        assert fx_delta(S, K, T, rd, rf, vol, is_call, EURUSD_DEFAULT) == pytest.approx(
            gk.delta(S, K, T, rd, rf, vol, is_call), rel=1e-15
        )


@pytest.mark.parametrize("S,K,T,rd,rf,vol", [(1.1, 1.17, 0.75, 0.05, 0.03, 0.09), (1.25, 1.1, 3.0, 0.0, 0.03, 0.12)])
@pytest.mark.parametrize("is_call", [True, False])
def test_premium_included_is_delta_minus_premium_over_spot(S, K, T, rd, rf, vol, is_call):
    """Independent check of the (K/F) N(d2) form: Delta_pi = dV/dS - V/S."""
    expected = gk.delta(S, K, T, rd, rf, vol, is_call) - gk.price(S, K, T, rd, rf, vol, is_call) / S
    got = fx_delta(S, K, T, rd, rf, vol, is_call, DeltaConvention(DeltaType.SPOT, True))
    assert got == pytest.approx(expected, abs=1e-14)
    got_f = fx_delta(S, K, T, rd, rf, vol, is_call, DeltaConvention(DeltaType.FORWARD, True))
    assert got_f == pytest.approx(expected * np.exp(rf * T), abs=1e-14)


def test_premium_included_call_has_two_roots_and_we_return_the_larger():
    from scipy.optimize import brentq

    S, T, rd, rf, vol = 1.1, 1.0, 0.04, 0.02, 0.20
    conv = DeltaConvention(DeltaType.FORWARD, True)
    K_hi = delta_to_strike(0.10, S, T, rd, rf, vol, conv)
    f = lambda K: fx_delta(S, K, T, rd, rf, vol, True, conv) - 0.10
    # a second, much smaller strike has the same delta
    K_lo = brentq(f, 1e-3 * S, 0.5 * S)
    assert K_lo < K_hi
    assert f(K_lo) == pytest.approx(0, abs=1e-12) and f(K_hi) == pytest.approx(0, abs=1e-12)
    # larger root sits where delta is decreasing in K
    h = 1e-6
    assert fx_delta(S, K_hi + h, T, rd, rf, vol, True, conv) < fx_delta(S, K_hi - h, T, rd, rf, vol, True, conv)


def test_premium_included_call_above_max_delta_raises():
    # the max premium-adjusted forward call delta falls as sigma*sqrt(T) grows:
    # ~0.80 at s=0.1, ~0.31 at s=1 (here s = 0.5*sqrt(4) = 1)
    conv = DeltaConvention(DeltaType.FORWARD, True)
    with pytest.raises(ValueError, match="no strike"):
        delta_to_strike(0.4, 1.1, 4.0, 0.04, 0.02, 0.5, conv)
    assert delta_to_strike(0.3, 1.1, 4.0, 0.04, 0.02, 0.5, conv) > 0  # just under the max


def test_premium_included_strikes_ordering():
    """A given delta sits at a lower strike under premium-included than excluded (calls and puts).

    Call: delta_pi(K) < delta_px(K) and delta falls in K, so the same delta is reached at lower K.
    Put:  |delta_pi(K)| > |delta_px(K)| and |put delta| rises in K, so likewise at lower K.
    """
    S, T, rd, rf, vol = 1.1, 1.0, 0.04, 0.02, 0.09
    px = DeltaConvention(DeltaType.SPOT, False)
    pi = DeltaConvention(DeltaType.SPOT, True)
    assert delta_to_strike(0.25, S, T, rd, rf, vol, pi) < delta_to_strike(0.25, S, T, rd, rf, vol, px)
    assert delta_to_strike(-0.25, S, T, rd, rf, vol, pi) < delta_to_strike(-0.25, S, T, rd, rf, vol, px)


def test_vectorised_inputs():
    deltas = np.array([-0.10, -0.25, 0.25, 0.10])
    for conv in CONVENTIONS:
        K = delta_to_strike(deltas, 1.1, 0.5, 0.04, 0.02, 0.08, conv)
        assert K.shape == deltas.shape
        assert np.all(np.diff(K) > 0)
        back = fx_delta(1.1, K, 0.5, 0.04, 0.02, 0.08, deltas > 0, conv)
        np.testing.assert_allclose(back, deltas, atol=1e-12)


def test_zero_delta_rejected():
    with pytest.raises(ValueError):
        delta_to_strike(0.0, 1.1, 1.0, 0.03, 0.01, 0.1)
