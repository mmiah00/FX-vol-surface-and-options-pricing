import itertools

import numpy as np
import pytest

from fxvol import DeltaConvention, DeltaType, fx_delta, gk
from fxvol.smile import (
    AtmType,
    SmileQuote,
    atm_strike,
    build_smile,
    smile_strangle_vols,
)

CONVENTIONS = [DeltaConvention(t, pi) for t, pi in itertools.product(DeltaType, (False, True))]
cid = lambda c: f"{c.delta_type.value}-{'pi' if c.premium_included else 'px'}"
MODELS = ["malz", "vv"]


def quote(**kw):
    base = dict(S=1.10, T=0.25, rd=0.053, rf=0.035, atm=0.075, rr=-0.005, bf=0.0025)
    base.update(kw)
    return SmileQuote(**base)


# ------------------------------------------------------------------ quote algebra


@pytest.mark.parametrize("rr,bf", [(-0.005, 0.0025), (0.01, 0.004), (0.0, 0.0)])
def test_smile_strangle_vols_reproduce_rr_and_bf(rr, bf):
    atm = 0.08
    sp, sc = smile_strangle_vols(atm, rr, bf)
    assert sc - sp == pytest.approx(rr, abs=1e-15)
    assert 0.5 * (sc + sp) - atm == pytest.approx(bf, abs=1e-15)


@pytest.mark.parametrize("conv", CONVENTIONS, ids=cid)
@pytest.mark.parametrize("model", MODELS)
def test_nodes_sit_at_wing_delta_and_atm(conv, model):
    q = quote(convention=conv)
    sm = build_smile(q, model)
    kp, ka, kc = sm.nodes.strikes
    sp, sa, sc = sm.nodes.vols
    args = (q.S, q.T, q.rd, q.rf)
    assert fx_delta(*args[:1], kp, *args[1:], sp, False, conv) == pytest.approx(-0.25, abs=1e-12)
    assert fx_delta(q.S, kc, q.T, q.rd, q.rf, sc, True, conv) == pytest.approx(0.25, abs=1e-12)
    # delta-neutral straddle: call delta + put delta = 0 under the same convention
    dsum = fx_delta(q.S, ka, q.T, q.rd, q.rf, sa, True, conv) + fx_delta(q.S, ka, q.T, q.rd, q.rf, sa, False, conv)
    assert dsum == pytest.approx(0.0, abs=1e-13)
    assert kp < ka < kc


def test_forward_atm_strike_is_forward():
    q = quote(atm_type=AtmType.FORWARD)
    assert build_smile(q, "malz").nodes.strikes[1] == pytest.approx(q.F, rel=1e-15)


def test_atm_dns_above_forward_when_premium_excluded_below_when_included():
    args = (1.1, 1.0, 0.04, 0.02, 0.1)
    F = float(gk.forward(*args[:2], *args[2:4]))
    assert atm_strike(*args, convention=DeltaConvention(DeltaType.SPOT, False)) > F
    assert atm_strike(*args, convention=DeltaConvention(DeltaType.SPOT, True)) < F


# ------------------------------------------------------------------------- shared


@pytest.mark.parametrize("conv", CONVENTIONS, ids=cid)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("strangle", ["smile", "market"])
def test_smile_reproduces_its_nodes(conv, model, strangle):
    sm = build_smile(quote(convention=conv), model, strangle)
    np.testing.assert_allclose(sm.vol(sm.nodes.strikes), sm.nodes.vols, atol=1e-10)


@pytest.mark.parametrize("model", MODELS)
def test_smile_strangle_vol_at_delta_returns_input_vols(model):
    q = quote()
    sm = build_smile(q, model)
    assert sm.vol_at_delta(-0.25) == pytest.approx(q.atm + q.bf - q.rr / 2, abs=1e-10)
    assert sm.vol_at_delta(+0.25) == pytest.approx(q.atm + q.bf + q.rr / 2, abs=1e-10)


@pytest.mark.parametrize("model", MODELS)
def test_strike_at_delta_is_consistent_with_vol(model):
    q = quote()
    sm = build_smile(q, model)
    for d in (-0.35, -0.1, 0.1, 0.35):
        K = sm.strike_at_delta(d)
        assert sm.vol(K) == pytest.approx(sm.vol_at_delta(d), abs=1e-10)
        assert fx_delta(q.S, K, q.T, q.rd, q.rf, sm.vol(K), d > 0, q.convention) == pytest.approx(d, abs=1e-10)


@pytest.mark.parametrize("conv", CONVENTIONS, ids=cid)
@pytest.mark.parametrize("model", MODELS)
def test_flat_quotes_give_flat_smile(conv, model):
    sm = build_smile(quote(rr=0.0, bf=0.0, convention=conv), model)
    K = np.linspace(0.95, 1.30, 15)
    np.testing.assert_allclose(sm.vol(K), 0.075, atol=1e-9)


@pytest.mark.parametrize("model", MODELS)
def test_risk_reversal_sign_skews_the_smile(model):
    for rr in (0.006, -0.006):
        sm = build_smile(quote(rr=rr, bf=0.002), model)
        skew = sm.vol_at_delta(0.10) - sm.vol_at_delta(-0.10)
        assert np.sign(skew) == np.sign(rr)


@pytest.mark.parametrize("model", MODELS)
def test_butterfly_lifts_both_wings_and_leaves_atm(model):
    lo, hi = (build_smile(quote(rr=0.0, bf=b), model) for b in (0.001, 0.006))
    for d in (-0.1, 0.1, -0.25, 0.25):
        assert hi.vol_at_delta(d) > lo.vol_at_delta(d)
    assert hi.vol(hi.nodes.strikes[1]) == pytest.approx(lo.vol(lo.nodes.strikes[1]), abs=1e-12)


# --------------------------------------------------------------------------- Malz


def test_malz_matches_textbook_formula_with_forward_delta():
    """Forward/premium-excluded: nodes sit at x = 0.25, 0.5, 0.75 so the fit IS Malz's formula."""
    q = quote(convention=DeltaConvention(DeltaType.FORWARD, False))
    sm = build_smile(q, "malz")
    x = np.linspace(0.02, 0.98, 25)
    malz = q.atm - 2 * q.rr * (x - 0.5) + 16 * q.bf * (x - 0.5) ** 2
    np.testing.assert_allclose(np.polyval(sm.coeffs, x), malz, atol=1e-14)


@pytest.mark.parametrize("conv", CONVENTIONS, ids=cid)
def test_malz_explicit_delta_lookup_matches_generic_fixed_point(conv):
    from fxvol.smile import Smile

    sm = build_smile(quote(convention=conv), "malz")
    for d in (-0.4, -0.15, 0.05, 0.3):
        generic = Smile.vol_at_delta(sm, d)
        assert sm.vol_at_delta(d) == pytest.approx(generic, abs=1e-10)


def test_malz_wings_are_bounded():
    q = quote()
    sm = build_smile(q, "malz")
    far = sm.vol(np.array([0.3, 5.0]))
    # K->0 and K->inf tend to q(1) and q(0)
    assert far[0] == pytest.approx(np.polyval(sm.coeffs, 1.0), abs=1e-6)
    assert far[1] == pytest.approx(np.polyval(sm.coeffs, 0.0), abs=1e-6)


# --------------------------------------------------------------------- Vanna-Volga


def test_vv_weights_replicate_vega_vanna_volga():
    sm = build_smile(quote(), "vv")
    q = quote()
    for K in (1.0, 1.08, 1.2):
        w = sm.weights(K)[:, 0]
        target = np.array([g(q.S, K, q.T, q.rd, q.rf, sm._base) for g in (gk.vega, gk.vanna, gk.volga)])
        np.testing.assert_allclose(sm._A @ w, target, rtol=1e-10, atol=1e-14)


def test_vv_weights_are_unit_vectors_at_pivots():
    sm = build_smile(quote(), "vv")
    np.testing.assert_allclose(sm.weights(sm.nodes.strikes), np.eye(3), atol=1e-9)


def test_vv_preserves_put_call_parity():
    q = quote()
    sm = build_smile(q, "vv")
    K = np.linspace(1.0, 1.25, 11)
    parity = q.S * np.exp(-q.rf * q.T) - K * np.exp(-q.rd * q.T)
    np.testing.assert_allclose(sm.price(K, True) - sm.price(K, False), parity, atol=1e-14)


def test_vv_and_malz_agree_between_pivots_but_differ_in_wings():
    q = quote()
    malz, vv = build_smile(q, "malz"), build_smile(q, "vv")
    for d in (-0.35, -0.15, 0.15, 0.35):
        assert abs(malz.vol_at_delta(d) - vv.vol_at_delta(d)) < 0.0025  # within 0.25 vol pt
    for d in (-0.05, 0.05):  # VV wings are steeper than Malz for this (butterfly > 0) smile
        assert vv.vol_at_delta(d) > malz.vol_at_delta(d) + 0.003


def test_vv_returns_nan_instead_of_garbage_when_no_implied_vol_exists():
    q = quote()  # 3M; K = 5.0 is ~4.5x the forward
    out = build_smile(q, "vv").vol(np.array([q.F, 5.0]))
    assert np.isfinite(out[0])
    assert np.isnan(out[1])


# -------------------------------------------------------------- market vs smile BF


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("conv", CONVENTIONS, ids=cid)
def test_market_strangle_reprices_flat_vol_strangle(model, conv):
    from fxvol.delta import delta_to_strike

    q = quote(convention=conv)
    sm = build_smile(q, model, "market")
    s_st = q.atm + q.bf
    kc = delta_to_strike(0.25, q.S, q.T, q.rd, q.rf, s_st, conv)
    kp = delta_to_strike(-0.25, q.S, q.T, q.rd, q.rf, s_st, conv)
    flat = gk.price(q.S, kc, q.T, q.rd, q.rf, s_st, True) + gk.price(q.S, kp, q.T, q.rd, q.rf, s_st, False)
    smile = gk.price(q.S, kc, q.T, q.rd, q.rf, sm.vol(kc), True) + gk.price(
        q.S, kp, q.T, q.rd, q.rf, sm.vol(kp), False
    )
    assert smile == pytest.approx(flat, rel=1e-11)
    # RR is still honoured exactly at the true 25D points
    assert sm.nodes.vols[2] - sm.nodes.vols[0] == pytest.approx(q.rr, abs=1e-15)


@pytest.mark.parametrize("model", MODELS)
def test_market_strangle_equals_smile_strangle_when_there_is_no_smile(model):
    sm = build_smile(quote(rr=0.0, bf=0.0), model, "market")
    assert sm.bf_smile == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("model", MODELS)
def test_market_strangle_differs_from_smile_strangle_but_only_slightly(model):
    q = quote(atm=0.12, rr=-0.02, bf=0.01, T=0.25)
    sm = build_smile(q, model, "market")
    gap = sm.bf_smile - q.bf
    assert 0 < gap < 0.001  # > 0, but well below 0.1 vol point
    assert build_smile(q, model, "smile").bf_smile == q.bf


def test_bad_options_raise():
    with pytest.raises(ValueError):
        build_smile(quote(), "sabr")
    with pytest.raises(ValueError):
        build_smile(quote(), "malz", "broker")
