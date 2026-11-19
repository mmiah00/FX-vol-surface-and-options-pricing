import numpy as np
import pytest

from fxvol.surface import TenorQuote, VolSurface
from fxvol.synthetic import T_PILLARS, eurusd_like, make_surface

MODELS = ["malz", "vv"]


@pytest.mark.parametrize("model", MODELS)
def test_surface_reproduces_each_pillar_smile(model):
    s = eurusd_like(model)
    for T, sm in zip(s.T, s.smiles):
        K = np.linspace(sm.nodes.strikes[0], sm.nodes.strikes[2], 7)
        np.testing.assert_allclose(s.vol(K, T), sm.vol(K), rtol=1e-12)


def test_pillar_forwards_and_rates_are_exact_at_pillars():
    s = eurusd_like()
    for q in s.tenors:
        assert s.forward(q.T) == pytest.approx(s.S * np.exp((q.rd - q.rf) * q.T), rel=1e-14)
        assert s.rates(q.T) == pytest.approx((q.rd, q.rf), rel=1e-14)


def test_integrated_rates_interpolate_linearly_between_pillars():
    s = eurusd_like()
    Ta, Tb = s.T[1], s.T[2]
    T = 0.5 * (Ta + Tb)
    rd, rf = s.rates(T)
    qa, qb = s.tenors[1], s.tenors[2]
    assert rd * T == pytest.approx(0.5 * (qa.rd * qa.T + qb.rd * qb.T), rel=1e-14)
    assert rf * T == pytest.approx(0.5 * (qa.rf * qa.T + qb.rf * qb.T), rel=1e-14)


def test_total_variance_is_linear_in_T_at_fixed_k():
    s = eurusd_like()
    k = np.linspace(-0.03, 0.03, 9)
    for j in range(len(s.T) - 1):
        lam = 0.3
        T = s.T[j] + lam * (s.T[j + 1] - s.T[j])
        expected = (1 - lam) * s.pillar_variance(j, k) + lam * s.pillar_variance(j + 1, k)
        np.testing.assert_allclose(s.total_variance(k, T), expected, rtol=1e-13)


def test_interpolating_total_variance_is_not_interpolating_vol():
    """Flat smiles, different level per tenor: vol(T)^2 T is linear, vol(T) itself is not."""
    s1, s2 = 0.06, 0.09
    q = [TenorQuote(0.25, 0.03, 0.03, s1, 0, 0), TenorQuote(1.0, 0.03, 0.03, s2, 0, 0)]
    s = VolSurface(1.0, q)
    T = 0.625  # lam = 0.5
    var_interp = np.sqrt((0.5 * s1**2 * 0.25 + 0.5 * s2**2 * 1.0) / T)
    assert s.vol_k(0.0, T) == pytest.approx(var_interp, rel=1e-12)
    assert abs(var_interp - 0.5 * (s1 + s2)) > 1e-3  # distinguishable from naive vol interpolation


def test_flat_quotes_give_flat_surface_including_extrapolation():
    q = [TenorQuote(T, 0.04, 0.02, 0.08, 0.0, 0.0) for T in (0.1, 0.5, 2.0)]
    s = VolSurface(1.2, q)
    for T in (0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0):
        K = s.forward(T) * np.exp(np.linspace(-0.2, 0.2, 5))
        np.testing.assert_allclose(s.vol(K, T), 0.08, atol=1e-9)


def test_extrapolation_holds_vol_flat_at_fixed_k():
    s = eurusd_like()
    k = np.array([-0.01, 0.0, 0.01])
    np.testing.assert_allclose(s.vol_k(k, s.T[0] / 2), s.vol_k(k, s.T[0]), rtol=1e-13)
    np.testing.assert_allclose(s.vol_k(k, 2 * s.T[-1]), s.vol_k(k, s.T[-1]), rtol=1e-13)


def test_slice_delta_lookup_matches_pillar_quotes():
    s = eurusd_like()
    for q, T in zip(s.tenors, s.T):
        sl = s.slice(T)
        assert sl.vol_at_delta(0.25) == pytest.approx(q.atm + q.bf + q.rr / 2, abs=1e-9)
        assert sl.vol_at_delta(-0.25) == pytest.approx(q.atm + q.bf - q.rr / 2, abs=1e-9)


def test_vol_at_delta_between_pillars_is_between_pillar_values_for_this_surface():
    s = eurusd_like()
    T = 0.375  # between 3M and 6M
    v = s.vol_at_delta(-0.25, T)
    lo, hi = sorted((s.slice(0.25).vol_at_delta(-0.25), s.slice(0.5).vol_at_delta(-0.25)))
    assert lo < v < hi


@pytest.mark.parametrize("tenors", [[], [TenorQuote(0.5, 0, 0, 0.1, 0, 0), TenorQuote(0.5, 0, 0, 0.1, 0, 0)]])
def test_bad_tenors_rejected(tenors):
    with pytest.raises(ValueError):
        VolSurface(1.0, tenors)


def test_unknown_override_rejected():
    with pytest.raises(TypeError):
        make_surface(foo=[1] * 5)


def test_pillars_are_the_standard_five():
    assert len(T_PILLARS) == 5 and T_PILLARS[0] == pytest.approx(7 / 365) and T_PILLARS[-1] == 1.0
