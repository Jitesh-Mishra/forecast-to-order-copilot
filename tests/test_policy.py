import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f2o.policy import budget_lp, expected_cost, proportional_budget, quantile_at, quantile_fn, simulate, GRID  # noqa


def _econ(price, margin=0.3, spoil=0.05):
    from f2o.policy import economics
    return economics(np.asarray(price, float), {"costs": {"gross_margin": margin, "annual_holding_rate": 0.25,
                                                           "weekly_spoilage_rate": spoil}})


def test_critical_ratio_matches_hand_calc():
    e = _econ([10.0])
    co = 7.0 * (0.25 / 52 + 0.05)
    assert abs(e["cr"][0] - 3.0 / (3.0 + co)) < 1e-12


def test_simulator_accounting():
    D = np.array([[5, 0, 3]], float)
    S = np.array([[4, 4, 4]], float)
    econ = [_econ([10.0])] * 3
    r = simulate(S, D, econ)
    # w1: stock 4, sell 4, lose 1, end 0 | w2: stock 4, sell 0, end 4 | w3: stock 4, sell 3, end 1
    co, cu = econ[0]["co"][0], econ[0]["cu"][0]
    assert abs(r["total_cost"] - (cu * 1 + co * 4 + co * 1)) < 1e-9
    assert abs(r["fill_rate"] - 7 / 8) < 1e-12


def test_quantile_fn_interpolates_and_is_monotone():
    lv = np.array([0.5, 0.9])
    q = np.array([[10.0, 20.0]])
    assert abs(quantile_at(lv, q, np.array([0.7]))[0] - 15.0) < 1e-9
    v = quantile_fn(lv, q, GRID)[0]
    assert np.all(np.diff(v) >= -1e-9) and v[0] >= 0


def test_budget_lp_matches_brute_force():
    rng = np.random.default_rng(0)
    lv = np.array([0.5, 0.8, 0.95])
    q = np.sort(rng.gamma(2, 3, (5, 3)), 1)
    e = _econ(rng.uniform(2, 8, 5))
    S_lp, info = budget_lp(lv, q, e, 0.7)
    assert (S_lp * e["unit_cost"]).sum() <= info["budget"] + 1e-6
    samples = quantile_fn(lv, q, GRID)
    opts = np.stack([np.zeros(5)] + [np.ceil(q[:, k] - 1e-9) for k in range(3)], 1)
    best = np.inf
    for ch in itertools.product(range(4), repeat=5):
        S = opts[np.arange(5), ch]
        if (S * e["unit_cost"]).sum() <= info["budget"] + 1e-9:
            best = min(best, expected_cost(S, samples, e["cu"], e["co"]).sum())
    got = expected_cost(S_lp, samples, e["cu"], e["co"]).sum()
    assert got <= best * 1.10 + 1e-6      # LP relaxation + one rounded SKU: within 10% of the integer optimum
    assert info["lp_expected_cost"] <= best + 1e-6   # relaxation is a lower bound


def test_proportional_respects_budget():
    S = np.array([10.0, 20.0, 5.0]); c = np.array([1.0, 2.0, 3.0])
    out = proportional_budget(S, c, 40.0)
    assert (out * c).sum() <= 40.0 and np.all(out <= S)


def test_features_do_not_peek_past_origin():
    from f2o.data import load_panel
    from f2o.forecast import features_at
    try:
        p = load_panel()
    except FileNotFoundError:
        pytest.skip("data not present")
    o = 250
    base = features_at(p, o, 2)
    p.Y[:, o:] = 999.0                                      # scramble the future
    after = features_at(p, o, 2)
    cols = [c for c in base.columns if c != "lag52"]       # lag52 points to week o+1-52, still the past
    assert np.allclose(base[cols].to_numpy(float), after[cols].to_numpy(float), equal_nan=True)
    assert np.allclose(base["lag52"], after["lag52"])
