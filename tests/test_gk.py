import numpy as np
import pytest

from fxvol import gk

EPS = np.finfo(float).eps


def test_put_call_parity_machine_precision(case):
    S, K, T, rd, rf, vol, _ = case
    c = gk.price(S, K, T, rd, rf, vol, True)
    p = gk.price(S, K, T, rd, rf, vol, False)
    parity = S * np.exp(-rf * T) - K * np.exp(-rd * T)
    scale = S * np.exp(-rf * T) + K * np.exp(-rd * T)
    assert abs((c - p) - parity) <= 8 * EPS * scale


def test_put_call_parity_vectorised():
    K = np.linspace(0.8, 1.4, 101)
    c = gk.price(1.1, K, 0.75, 0.05, 0.03, 0.09, True)
    p = gk.price(1.1, K, 0.75, 0.05, 0.03, 0.09, False)
    parity = 1.1 * np.exp(-0.03 * 0.75) - K * np.exp(-0.05 * 0.75)
    np.testing.assert_allclose(c - p, parity, rtol=0, atol=8 * EPS * 2.5)


def test_call_put_delta_difference_is_foreign_discount(case):
    S, K, T, rd, rf, vol, _ = case
    dc = gk.delta(S, K, T, rd, rf, vol, True)
    dp = gk.delta(S, K, T, rd, rf, vol, False)
    assert dc - dp == pytest.approx(np.exp(-rf * T), abs=8 * EPS)


def _spot_scale(S, T, vol):
    """Natural length scale of spot for this option: S * sigma * sqrt(T).

    Finite-difference bumps are expressed as fractions of this so truncation
    error is uniform across tenors (a fixed 1e-4*S bump is ~1% of the scale at 1W).
    """
    return S * vol * np.sqrt(T)


def test_delta_matches_finite_difference(case):
    S, K, T, rd, rf, vol, is_call = case
    h = 1e-3 * _spot_scale(S, T, vol)
    f = lambda s: gk.price(s, K, T, rd, rf, vol, is_call)
    fd = (f(S + h) - f(S - h)) / (2 * h)
    assert fd == pytest.approx(gk.delta(S, K, T, rd, rf, vol, is_call), rel=1e-5, abs=1e-9)


def test_gamma_matches_finite_difference(case):
    S, K, T, rd, rf, vol, is_call = case
    h = 1e-3 * _spot_scale(S, T, vol)
    f = lambda s: gk.delta(s, K, T, rd, rf, vol, is_call)
    fd = (f(S + h) - f(S - h)) / (2 * h)
    assert fd == pytest.approx(gk.gamma(S, K, T, rd, rf, vol), rel=1e-5, abs=1e-9)


def test_gamma_matches_second_difference_of_price(case):
    S, K, T, rd, rf, vol, is_call = case
    h = 1e-2 * _spot_scale(S, T, vol)
    f = lambda s: gk.price(s, K, T, rd, rf, vol, is_call)
    fd = (f(S + h) - 2 * f(S) + f(S - h)) / h**2
    assert fd == pytest.approx(gk.gamma(S, K, T, rd, rf, vol), rel=1e-3, abs=1e-6)


def test_vega_matches_finite_difference(case):
    S, K, T, rd, rf, vol, is_call = case
    h = 1e-5
    f = lambda v: gk.price(S, K, T, rd, rf, v, is_call)
    fd = (f(vol + h) - f(vol - h)) / (2 * h)
    assert fd == pytest.approx(gk.vega(S, K, T, rd, rf, vol), rel=1e-6, abs=1e-9)


def test_theta_matches_finite_difference(case):
    S, K, T, rd, rf, vol, is_call = case
    h = 1e-5 * T
    f = lambda t: gk.price(S, K, t, rd, rf, vol, is_call)
    fd = -(f(T + h) - f(T - h)) / (2 * h)  # theta = dV/dt = -dV/dT
    assert fd == pytest.approx(gk.theta(S, K, T, rd, rf, vol, is_call), rel=1e-6, abs=1e-8)


def test_vanna_matches_finite_difference(case):
    S, K, T, rd, rf, vol, is_call = case
    h = 1e-5
    # d(delta)/d(sigma)
    f = lambda v: gk.delta(S, K, T, rd, rf, v, is_call)
    fd_delta = (f(vol + h) - f(vol - h)) / (2 * h)
    # d(vega)/dS
    hs = 1e-4 * _spot_scale(S, T, vol)
    g = lambda s: gk.vega(s, K, T, rd, rf, vol)
    fd_vega = (g(S + hs) - g(S - hs)) / (2 * hs)
    v = gk.vanna(S, K, T, rd, rf, vol)
    assert fd_delta == pytest.approx(v, rel=1e-6, abs=1e-9)
    assert fd_vega == pytest.approx(v, rel=1e-6, abs=1e-9)


def test_vanna_matches_mixed_price_difference(case):
    S, K, T, rd, rf, vol, is_call = case
    hs, hv = 1e-2 * _spot_scale(S, T, vol), 1e-2 * vol
    f = lambda s, v: gk.price(s, K, T, rd, rf, v, is_call)
    fd = (f(S + hs, vol + hv) - f(S + hs, vol - hv) - f(S - hs, vol + hv) + f(S - hs, vol - hv)) / (4 * hs * hv)
    assert fd == pytest.approx(gk.vanna(S, K, T, rd, rf, vol), rel=1e-3, abs=1e-5)


def test_volga_matches_finite_difference(case):
    S, K, T, rd, rf, vol, is_call = case
    h = 1e-5
    f = lambda v: gk.vega(S, K, T, rd, rf, v)
    fd = (f(vol + h) - f(vol - h)) / (2 * h)
    assert fd == pytest.approx(gk.volga(S, K, T, rd, rf, vol), rel=1e-6, abs=1e-9)


def test_volga_matches_second_difference_of_price(case):
    S, K, T, rd, rf, vol, is_call = case
    h = 1e-3
    f = lambda v: gk.price(S, K, T, rd, rf, v, is_call)
    fd = (f(vol + h) - 2 * f(vol) + f(vol - h)) / h**2
    assert fd == pytest.approx(gk.volga(S, K, T, rd, rf, vol), rel=1e-3, abs=1e-5)


def test_known_value_atm_forward():
    # ATM-forward call: V = df_d * F * (2 N(s/2) - 1)
    from scipy.stats import norm

    S, T, rd, rf, vol = 1.0, 1.0, 0.03, 0.01, 0.10
    F = S * np.exp((rd - rf) * T)
    expected = np.exp(-rd * T) * F * (2 * norm.cdf(0.5 * vol * np.sqrt(T)) - 1)
    assert gk.price(S, F, T, rd, rf, vol, True) == pytest.approx(expected, rel=1e-14)


def test_reduces_to_black_scholes_when_rf_is_zero():
    from scipy.stats import norm

    S, K, T, r, vol = 100.0, 105.0, 0.5, 0.04, 0.2
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    bs = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    assert gk.price(S, K, T, r, 0.0, vol, True) == pytest.approx(bs, rel=1e-14)


@pytest.mark.parametrize("bad", [dict(T=0.0), dict(sigma=0.0), dict(sigma=-0.1), dict(K=-1.0)])
def test_invalid_inputs_raise(bad):
    args = dict(S=1.1, K=1.1, T=1.0, rd=0.03, rf=0.01, sigma=0.1)
    args.update(bad)
    with pytest.raises(ValueError):
        gk.price(args["S"], args["K"], args["T"], args["rd"], args["rf"], args["sigma"])
