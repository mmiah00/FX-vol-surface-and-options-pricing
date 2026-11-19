import numpy as np
import pytest

from fxvol.arbitrage import (
    check_butterfly,
    check_calendar,
    check_surface,
    density_from_g,
    density_g,
    price_butterfly_density,
)
from fxvol.surface import TenorQuote, VolSurface
from fxvol.synthetic import (
    eurusd_like,
    high_carry_steep_front,
    make_surface,
    stale_6m_atm,
    vv_short_end_butterfly,
)


# ------------------------------------------------------------- detectors are correct


def test_g_is_one_for_flat_variance():
    """Black-Scholes: w' = w'' = 0 and k*w' = 0, so g = 1 and the density is lognormal."""
    q = [TenorQuote(T, 0.03, 0.03, 0.08, 0.0, 0.0) for T in (0.25, 1.0)]
    s = VolSurface(1.0, q)
    k = np.linspace(-0.1, 0.1, 21)
    np.testing.assert_allclose(density_g(s, 0.5, k), 1.0, atol=1e-5)


@pytest.mark.parametrize("model", ["malz", "vv"])
@pytest.mark.parametrize("T", [7 / 365, 0.25, 0.4, 1.0])
def test_g_density_agrees_with_independent_price_butterfly(model, T):
    """Two unrelated routes to the same density: g(k) algebra vs second difference of call prices."""
    s = eurusd_like(model)
    k = np.linspace(-2.5, 2.5, 11) * np.sqrt(float(s.total_variance(0.0, T)))
    q_g = density_from_g(s, T, k, density_g(s, T, k))
    q_p = price_butterfly_density(s, T, k)
    np.testing.assert_allclose(q_g, q_p, rtol=2e-3)


def test_lognormal_density_integrates_to_one_under_flat_vol():
    q = [TenorQuote(T, 0.03, 0.01, 0.1, 0.0, 0.0) for T in (0.25, 1.0)]
    s = VolSurface(1.0, q)
    T = 0.5
    k = np.linspace(-1.2, 1.2, 4001)
    q_k = density_from_g(s, T, k, density_g(s, T, k)) * s.forward(T) * np.exp(k)  # density of k
    assert np.trapezoid(q_k, k) == pytest.approx(1.0, abs=1e-3)


# ----------------------------------------------------------------- clean surfaces pass


@pytest.mark.parametrize("model", ["malz", "vv"])
def test_clean_surface_has_no_violations_in_any_check(model):
    r = check_surface(eurusd_like(model))
    assert r.ok, r.summary()
    assert r.butterfly.n_undefined == 0


# ------------------------------------------------------------------ butterfly breaks


def test_vv_with_large_short_end_butterfly_has_negative_density():
    b = check_butterfly(vv_short_end_butterfly())
    assert not b.ok
    T, k, g = b.worst()
    assert T == pytest.approx(7 / 365)  # only the 1W pillar breaks
    assert k < 0 and g < 0  # put wing


def test_both_butterfly_detectors_flag_exactly_the_same_points():
    b = check_butterfly(vv_short_end_butterfly())
    for bad_g, bad_p in zip(b.bad_g, b.bad_price):
        np.testing.assert_array_equal(bad_g, bad_p)
    assert b.n_violations > 0


def test_negative_g_means_negative_price_density():
    s = vv_short_end_butterfly()
    b = check_butterfly(s)
    for k, g, dp, bad in zip(b.k, b.g, b.density_price, b.bad_g):
        if bad.any():
            assert np.all(dp[bad] < 0)
            assert np.all(g[bad] < 0)


def test_same_quotes_in_malz_are_arbitrage_free():
    bf = [0.0015, 0.002, 0.0025, 0.003, 0.0035]
    bf[0] = 0.01
    assert check_butterfly(make_surface(1.10, "malz", bf=bf)).ok


def test_butterfly_violation_grows_with_the_butterfly_quote():
    worst = []
    for bf in (0.0015, 0.01, 0.02):
        w = check_butterfly(vv_short_end_butterfly(bf)).worst()
        worst.append(0.0 if w is None else w[2])
    assert worst[0] == 0.0 and worst[1] > worst[2]  # g gets more negative


# ------------------------------------------------------------------- calendar breaks


def test_stale_6m_atm_breaks_calendar_in_both_coordinates():
    r = check_surface(stale_6m_atm())
    assert not r.calendar_k.ok and not r.calendar_strike.ok
    assert r.butterfly.ok  # a calendar break is not a butterfly break
    Ta, Tb, _, dw = r.calendar_k.worst()
    assert 0.5 <= Ta < Tb <= 1.0 and dw < 0


def test_calendar_violation_at_the_forward_matches_hand_arithmetic():
    """At k = 0 the 6M and 1Y total variances are the surface's own numbers; the check must agree with them."""
    s = stale_6m_atm(0.12)
    w_6m = float(s.total_variance(0.0, 0.5))
    w_1y = float(s.total_variance(0.0, 1.0))
    assert w_1y < w_6m  # 1Y has LESS total variance than 6M: the definition of calendar arbitrage
    r = check_calendar(s, "forward_moneyness")
    row = int(np.argmin(np.abs(r.T_short - 0.875)))  # last interval before 1Y
    j = int(np.argmin(np.abs(r.x[row])))  # nearest the forward
    assert r.bad[row, j] and r.dw[row, j] < 0


def test_calendar_only_ok_when_every_step_is_nonnegative():
    r = check_calendar(eurusd_like())
    assert r.ok and np.all(r.dw >= 0)


def test_zero_carry_makes_the_two_coordinates_identical():
    zero = [0.03] * 5
    atm = [0.068, 0.070, 0.073, 0.12, 0.079]
    s = make_surface(1.10, "malz", atm=atm, rd=zero, rf=zero)
    ck, cs = check_calendar(s, "forward_moneyness"), check_calendar(s, "strike")
    assert ck.n_violations == cs.n_violations > 0
    np.testing.assert_allclose(ck.dw, cs.dw, rtol=1e-9, atol=1e-15)


def test_fixed_strike_check_false_alarms_when_carry_drifts_the_forward_through_a_skew():
    """No arbitrage in k (where the theory lives), yet total variance falls with T at fixed strike."""
    s = high_carry_steep_front()
    r = check_surface(s)
    assert r.calendar_k.ok and r.butterfly.ok
    assert not r.calendar_strike.ok
    # the identity d_T w|_K = d_T w|_k - carry * d_k w explains it: show the sign on the offending cell
    cs = r.calendar_strike
    Ta, Tb, K, dw = cs.worst()
    assert dw < 0


def test_calendar_coordinate_validated():
    with pytest.raises(ValueError):
        check_calendar(eurusd_like(), "delta")


def test_undefined_points_are_counted_not_flagged():
    # a 40-sigma band pushes VV past the point where any vol reproduces its price
    s = eurusd_like("vv")
    r = check_butterfly(s, z_max=40.0)
    assert r.n_undefined > 0
    for bad, g in zip(r.bad_g, r.g):
        assert not np.any(bad & ~np.isfinite(g))


def test_summary_reports_where_and_how():
    text = check_surface(vv_short_end_butterfly()).summary()
    assert "butterfly" in text and "worst g=" in text and "delta=" in text
    assert "price-butterfly detector flags" in text
    assert "calendar (fixed k" in text and "OK" in text
