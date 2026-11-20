from dataclasses import replace

import numpy as np
import pytest

from fxvol.arbitrage import check_surface
from fxvol.market import (
    T_EVZ,
    AssumedParams,
    atm_over_evz_sensitivity,
    atm_term_structure,
    calibrate_atm_over_evz,
    stress_scenarios,
    surface_from_state,
)
from fxvol.surface import TenorQuote, VolSurface, variance_swap_vol
from fxvol.synthetic import T_PILLARS

DAY = dict(S=1.1069, rd=0.0006, rf=-0.0055, evz=0.073)  # a typical low-rate day: negative euro rate


# ---------------------------------------------------------------- ATM term structure


def test_term_structure_is_pinned_at_the_30_day_vol():
    for a in (0.04, 0.06, 0.0775, 0.12, 0.25):
        assert atm_term_structure(a, [T_EVZ], long_run_vol=0.0775)[0] == pytest.approx(a, rel=1e-13)


def test_term_structure_is_flat_when_30_day_equals_long_run():
    np.testing.assert_allclose(atm_term_structure(0.0775, long_run_vol=0.0775), 0.0775, rtol=1e-13)


@pytest.mark.parametrize("a30", [0.04, 0.06, 0.12, 0.25, 0.40])
def test_total_variance_always_increases_with_maturity(a30):
    T = np.linspace(0.01, 5.0, 300)
    w = atm_term_structure(a30, T, 0.0775, 2.0) ** 2 * T
    assert np.all(np.diff(w) > 0)


def test_term_structure_reverts_towards_the_long_run_level():
    hi = atm_term_structure(0.15, long_run_vol=0.0775)
    lo = atm_term_structure(0.05, long_run_vol=0.0775)
    assert np.all(np.diff(hi) < 0) and np.all(np.diff(lo) > 0)
    assert 0.0775 < hi[-1] < 0.15 and 0.05 < lo[-1] < 0.0775
    # ATM vol at T is the *average* variance over [0, T], so the early excess fades only like 1/T: still 0.5% high at
    # 20 years, but converged at 1000
    assert atm_term_structure(0.15, [1000.0], long_run_vol=0.0775)[0] == pytest.approx(0.0775, abs=2e-4)


def test_too_low_a_front_vol_is_rejected_not_silently_negative():
    with pytest.raises(ValueError):
        atm_term_structure(0.01, long_run_vol=0.0775)


# ------------------------------------------------------------------ variance swap


@pytest.mark.parametrize("vol", [0.04, 0.10, 0.30])
def test_variance_swap_vol_of_a_flat_surface_is_the_flat_vol(vol):
    q = [TenorQuote(T, 0.03, 0.01, vol, 0.0, 0.0) for T in (T_EVZ, 1.0)]
    assert variance_swap_vol(VolSurface(1.2, q), T_EVZ) == pytest.approx(vol, rel=1e-4)


def test_variance_swap_vol_exceeds_atm_when_the_smile_is_convex():
    s = surface_from_state(**DAY)
    assert variance_swap_vol(s, T_EVZ) > s.vol_k(0.0, T_EVZ)


def test_calibration_makes_the_model_variance_swap_equal_evz():
    p = AssumedParams()
    c = calibrate_atm_over_evz(p, **{k: DAY[k] for k in ("S", "rd", "rf")})
    p = replace(p, atm_over_evz=c)
    s = surface_from_state(**DAY, params=p)
    assert variance_swap_vol(s, T_EVZ) == pytest.approx(DAY["evz"], rel=2e-3)
    assert c < 1.0  # ATM vol is below the variance-swap level when the smile is convex


def test_variance_swap_to_atm_ratio_barely_depends_on_the_vol_level():
    r = np.array(list(atm_over_evz_sensitivity(AssumedParams()).values()))
    assert r.max() / r.min() - 1 < 3e-3  # 4% to 30% vol: ratio moves by well under 0.3%


# -------------------------------------------------------------------- the surface


def test_surface_reproduces_assumed_rr_and_bf_fractions_at_each_pillar():
    p = AssumedParams()
    s = surface_from_state(**DAY, params=p)
    atm = atm_term_structure(DAY["evz"], T_PILLARS, p.long_run_vol, p.kappa)
    for j, T in enumerate(T_PILLARS):
        sl = s.slice(float(T))
        rr = sl.vol_at_delta(0.25) - sl.vol_at_delta(-0.25)
        bf = 0.5 * (sl.vol_at_delta(0.25) + sl.vol_at_delta(-0.25)) - atm[j]
        assert rr == pytest.approx(p.rr_over_atm[j] * atm[j], abs=1e-9)
        assert bf == pytest.approx(p.bf_over_atm[j] * atm[j], abs=1e-9)


def test_smile_scales_with_the_vol_regime():
    calm, wild = (surface_from_state(**{**DAY, "evz": e}) for e in (0.05, 0.20))
    s_calm, s_wild = calm.slice(0.25), wild.slice(0.25)
    assert abs(s_wild.vol_at_delta(0.25) - s_wild.vol_at_delta(-0.25)) > abs(s_calm.vol_at_delta(0.25) - s_calm.vol_at_delta(-0.25))


def test_negative_foreign_rate_is_handled():
    s = surface_from_state(**DAY)
    assert s.rates(0.25)[1] < 0
    assert s.forward(1.0) == pytest.approx(DAY["S"] * np.exp((DAY["rd"] - DAY["rf"]) * 1.0), rel=1e-12)
    assert check_surface(s).ok


def test_explicit_atm_override_is_used():
    atm = np.array([0.05, 0.06, 0.07, 0.08, 0.09])
    s = surface_from_state(**DAY, atm=atm)
    assert s.tenors[3].atm == pytest.approx(0.08)


# ------------------------------------------------------------- synthetic stress tests


def _breaks(model, name):
    sc = stress_scenarios(**DAY, params=AssumedParams())[name]
    return not check_surface(surface_from_state(model=model, **sc), per_interval=1).ok


def test_stress_scenarios_cover_the_documented_edge_cases():
    names = set(stress_scenarios(**DAY, params=AssumedParams()))
    assert names == {
        "base day",
        "extreme skew (RR x6)",
        "extreme butterfly (BF x6)",
        "vol spike (EVZ 30)",
        "vol crush (EVZ 4)",
        "inverted front end (1W = 2.5x 1M)",
    }


@pytest.mark.parametrize(
    "name", ["base day", "extreme skew (RR x6)", "extreme butterfly (BF x6)", "vol spike (EVZ 30)", "vol crush (EVZ 4)"]
)
def test_malz_survives_every_stress_except_the_inverted_front_end(name):
    assert not _breaks("malz", name)


def test_inverted_front_end_breaks_calendar_in_malz_as_the_variance_ratio_predicts():
    sc = stress_scenarios(**DAY, params=AssumedParams())["inverted front end (1W = 2.5x 1M)"]
    r = check_surface(surface_from_state(model="malz", **sc), per_interval=1)
    assert r.butterfly.ok and not r.calendar_k.ok
    # 1W/1M vol ratio is 2.5; the threshold is sqrt(T_1M / T_1W)
    assert 2.5 > np.sqrt(T_PILLARS[1] / T_PILLARS[0])


@pytest.mark.parametrize("name", ["extreme skew (RR x6)", "extreme butterfly (BF x6)"])
def test_vanna_volga_breaks_under_extreme_smiles(name):
    assert _breaks("vv", name)
