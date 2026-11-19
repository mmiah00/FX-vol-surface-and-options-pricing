"""Cross-check against QuantLib's analytic European engine (Garman-Kohlhagen process)."""

import pytest

from fxvol import gk

ql = pytest.importorskip("QuantLib")

# (S, K, T_years, rd, rf, vol): a handful of hand-picked cases
CASES = [
    (1.10, 1.10, 1.0, 0.045, 0.025, 0.09),
    (1.10, 1.15, 0.5, 0.053, 0.035, 0.08),
    (1.10, 1.05, 0.25, 0.053, 0.035, 0.07),
    (1.25, 1.40, 2.0, -0.005, 0.03, 0.12),
    (0.95, 0.80, 5.0, 0.04, 0.01, 0.15),
    (150.0, 145.0, 0.5, 0.001, 0.05, 0.11),
]


def _ql_option(S, K, T, rd, rf, vol, is_call):
    # Act/365F with an expiry exactly T years out so QuantLib's time matches ours
    dc = ql.Actual365Fixed()
    today = ql.Date(19, 11, 2026)
    ql.Settings.instance().evaluationDate = today
    expiry = today + int(round(T * 365))
    T_eff = dc.yearFraction(today, expiry)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry))
    cal = ql.NullCalendar()
    flat = lambda r: ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    vol_h = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, cal, vol, dc))
    process = ql.GarmanKohlagenProcess(ql.QuoteHandle(ql.SimpleQuote(S)), flat(rf), flat(rd), vol_h)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option, T_eff


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("S,K,T,rd,rf,vol", CASES)
def test_matches_quantlib(S, K, T, rd, rf, vol, is_call):
    option, T_eff = _ql_option(S, K, T, rd, rf, vol, is_call)
    args = (S, K, T_eff, rd, rf, vol)
    assert gk.price(*args, is_call) == pytest.approx(option.NPV(), rel=1e-12)
    assert gk.delta(*args, is_call) == pytest.approx(option.delta(), rel=1e-11)
    assert gk.gamma(*args) == pytest.approx(option.gamma(), rel=1e-11)
    assert gk.vega(*args) == pytest.approx(option.vega(), rel=1e-11)
    assert gk.theta(*args, is_call) == pytest.approx(option.theta(), rel=1e-10)
