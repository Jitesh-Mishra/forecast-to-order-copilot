"""Prescriptive layer: quantile forecasts -> order-up-to levels, budget-constrained LP, replay simulator.

Economics per SKU and week (all assumptions in config.yaml; M5 has no cost data):
  unit_cost = price * (1 - margin)
  underage  cu = margin * price                                       (lost margin per unit short)
  overage   co = unit_cost * (annual_holding_rate / 52 + spoilage)    (per unit left at week end)
  critical ratio CR = cu / (cu + co)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

GRID = (np.arange(200) + 0.5) / 200     # quantile grid used for expected-cost integrals


def economics(price: np.ndarray, cfg: dict) -> dict:
    c = cfg["costs"]
    unit_cost = price * (1 - c["gross_margin"])
    cu = c["gross_margin"] * price
    co = unit_cost * (c["annual_holding_rate"] / 52 + c["weekly_spoilage_rate"])
    return {"unit_cost": unit_cost, "cu": cu, "co": co, "cr": cu / (cu + co)}


def quantile_fn(levels: np.ndarray, qvals: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Piecewise-linear quantile function per SKU. levels [K], qvals [n, K], u [m] -> [n, m].
    Lower tail runs linearly to 0 at u=0; upper tail extrapolates the last segment's slope."""
    lv = np.concatenate([[0.0], levels])
    qv = np.concatenate([np.zeros((len(qvals), 1)), qvals], 1)
    out = np.empty((len(qvals), len(u)))
    for j, uu in enumerate(u):
        if uu <= lv[-1]:
            k = np.searchsorted(lv, uu, side="right") - 1
            k = min(k, len(lv) - 2)
            w = (uu - lv[k]) / (lv[k + 1] - lv[k])
            out[:, j] = qv[:, k] + w * (qv[:, k + 1] - qv[:, k])
        else:
            slope = (qv[:, -1] - qv[:, -2]) / (lv[-1] - lv[-2])
            out[:, j] = qv[:, -1] + slope * (uu - lv[-1])
    return np.maximum(out, 0)


def quantile_at(levels, qvals, alpha: np.ndarray) -> np.ndarray:
    """Per-SKU quantile at per-SKU level alpha."""
    lv = np.concatenate([[0.0], levels])
    qv = np.concatenate([np.zeros((len(qvals), 1)), qvals], 1)
    alpha = np.clip(alpha, 1e-6, 0.999)
    out = np.empty(len(qvals))
    for i, a in enumerate(alpha):
        if a <= lv[-1]:
            k = min(np.searchsorted(lv, a, side="right") - 1, len(lv) - 2)
            w = (a - lv[k]) / (lv[k + 1] - lv[k])
            out[i] = qv[i, k] + w * (qv[i, k + 1] - qv[i, k])
        else:
            out[i] = qv[i, -1] + (qv[i, -1] - qv[i, -2]) / (lv[-1] - lv[-2]) * (a - lv[-1])
    return np.maximum(out, 0)


def expected_cost(S: np.ndarray, samples: np.ndarray, cu, co) -> np.ndarray:
    """E[co (S-D)+ + cu (D-S)+] under the forecast distribution (samples [n, m])."""
    over = np.maximum(S[:, None] - samples, 0).mean(1)
    under = np.maximum(samples - S[:, None], 0).mean(1)
    return co * over + cu * under


# ----------------------------------------------------------------------------- policies -> order-up-to S
def s_quantile_newsvendor(levels, qvals, alpha):
    return np.ceil(quantile_at(levels, qvals, alpha) - 1e-9)


def s_normal_newsvendor(point, sigma, alpha):
    from scipy.stats import norm
    return np.ceil(np.maximum(point + norm.ppf(np.clip(alpha, 1e-6, 1 - 1e-6)) * sigma, 0) - 1e-9)


def s_days_of_cover(ma4, weeks):
    return np.ceil(ma4 * weeks - 1e-9)


# ----------------------------------------------------------------------------- budget LP
def budget_lp(levels, qvals, econ, budget_frac):
    """Choose one service option per SKU to minimise expected cost s.t. total stock value <= budget.

    Options per SKU: S = 0 or S = ceil(Q(alpha)) for alpha in levels. Multiple-choice knapsack, LP
    relaxation solved with HiGHS (at most one fractional SKU, rounded to its cheaper-in-budget option).
    Budget = budget_frac x stock value of the unconstrained cost-minimising choice.
    Returns S, info (budget, shadow price = expected-cost dollars saved per extra budget dollar).
    """
    n = len(qvals)
    samples = quantile_fn(levels, qvals, GRID)
    opts = [np.zeros(n)] + [np.ceil(qvals[:, k] - 1e-9) for k in range(len(levels))]
    S_opt = np.stack(opts, 1)                                     # [n, J]
    J = S_opt.shape[1]
    cost = np.stack([expected_cost(S_opt[:, j], samples, econ["cu"], econ["co"]) for j in range(J)], 1)
    val = S_opt * econ["unit_cost"][:, None]
    j_star = cost.argmin(1)
    B100 = val[np.arange(n), j_star].sum()
    B = budget_frac * B100
    c = cost.ravel()
    A_ub = val.ravel()[None, :]
    rows = np.repeat(np.arange(n), J)
    from scipy.sparse import csr_matrix
    A_eq = csr_matrix((np.ones(n * J), (rows, np.arange(n * J))), shape=(n, n * J))
    res = linprog(c, A_ub=A_ub, b_ub=[B], A_eq=A_eq, b_eq=np.ones(n), bounds=(0, 1), method="highs")
    x = res.x.reshape(n, J)
    choice = x.argmax(1)
    frac = np.where((x > 1e-6).sum(1) > 1)[0]
    for i in frac:   # round fractional SKU to its lowest-value option among those used
        used = np.where(x[i] > 1e-6)[0]
        choice[i] = used[np.argmin(val[i, used])]
    S = S_opt[np.arange(n), choice]
    S_star = S_opt[np.arange(n), j_star]
    return S, {"S_star": S_star, "budget": B, "B100": B100, "lp_expected_cost": float(res.fun),
               "shadow_price": float(-res.ineqlin.marginals[0]), "n_fractional": int(len(frac)),
               "stock_value": float((S * econ["unit_cost"]).sum())}


def proportional_budget(S_star, unit_cost, budget):
    """Scale every SKU's order-up-to level by one factor s, rounded down, with s chosen by bisection
    so the plan spends the same budget as the LP (a fair 'cut everything evenly' baseline)."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if (np.floor(S_star * mid + 1e-9) * unit_cost).sum() <= budget:
            lo = mid
        else:
            hi = mid
    return np.floor(S_star * lo + 1e-9)


# ----------------------------------------------------------------------------- replay simulator
def simulate(S_weeks: np.ndarray, D_weeks: np.ndarray, econ_weeks: list[dict]) -> dict:
    """Lost-sales periodic review, lead time 1 week. S_weeks, D_weeks: [n, T].
    Each week: stock up to S (excess carried over, never returned), sell min(D, stock),
    unmet demand is lost (cost cu), units left at week end cost co."""
    n, T = D_weeks.shape
    inv_end = np.zeros(n)
    tot = {"over": 0.0, "under": 0.0, "units_demand": 0.0, "units_sold": 0.0, "stock_value": 0.0}
    per_sku_cost = np.zeros(n)
    for t in range(T):
        e = econ_weeks[t]
        start = np.maximum(inv_end, S_weeks[:, t])
        sold = np.minimum(D_weeks[:, t], start)
        lost = D_weeks[:, t] - sold
        inv_end = start - sold
        oc, uc = e["co"] * inv_end, e["cu"] * lost
        per_sku_cost += oc + uc
        tot["over"] += oc.sum(); tot["under"] += uc.sum()
        tot["units_demand"] += D_weeks[:, t].sum(); tot["units_sold"] += sold.sum()
        tot["stock_value"] += (start * e["unit_cost"]).sum()
    return {"total_cost": tot["over"] + tot["under"], "overage_cost": tot["over"], "lost_margin": tot["under"],
            "fill_rate": tot["units_sold"] / max(tot["units_demand"], 1),
            "avg_stock_value": tot["stock_value"] / T, "per_sku_cost": per_sku_cost}
