"""End-to-end fast proof: forecasts -> policies -> budget LP -> replay, over 2 backtest folds + 1 locked test.

python -m f2o.run --folds backtest     # tuning / sanity, test untouched
python -m f2o.run --folds test         # the single locked evaluation on d_1914..d_1941
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import date

import numpy as np
import pandas as pd

from .data import REPORTS, intermittency, load_config, load_panel
from .forecast import baseline_point, build_rows, train_lgb, tsb_forecast
from .policy import (budget_lp, economics, proportional_budget, quantile_fn, s_days_of_cover,
                     s_normal_newsvendor, s_quantile_newsvendor, simulate, GRID, expected_cost)

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


def sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "uncommitted"


def pinball(y, q, a):
    d = y - q
    return np.mean(np.maximum(a * d, (a - 1) * d))


def run_fold(p, cfg, o, tag, scaled=True):
    H, levels = cfg["split"]["horizon"], np.array(cfg["policy"]["quantiles"])
    log(f"== fold {tag}: test weeks {o}..{o+H-1} (d_{p.week_start_day[o]}..d_{p.week_start_day[o+H-1]+6})")
    model = train_lgb(p, o, levels, H, seed=cfg["seed"], scaled=scaled, log=log)

    # evaluation SKUs: >= 13 weeks of history, priced in all test weeks, non-zero MASE scale
    Y = p.Y
    scale = np.array([np.abs(np.diff(Y[i, p.first_week[i]:o])).mean() if o - p.first_week[i] > 1 else 0
                      for i in range(len(Y))])
    ev = (o - p.first_week >= 13) & ~np.isnan(p.P[:, o:o + H]).any(1) & (scale > 0)
    idx = np.where(ev)[0]
    D = Y[idx, o:o + H]                                                    # actual weekly sales [n, H]

    # ---------- accuracy at a fixed origin, h = 1..4
    rows = build_rows(p, [o], range(1, H + 1), need_target=False)
    rows = rows[rows["sku"].isin(idx)]
    qp = model.predict(rows)
    Q = np.zeros((len(idx), H, len(levels)))
    pos = {s: k for k, s in enumerate(idx)}
    for (s, h), qq in zip(rows[["sku", "h"]].to_numpy(), qp):
        Q[pos[s], h - 1] = qq
    tsb = tsb_forecast(p, o, H)[idx]
    pts = {"naive": np.stack([baseline_point(p, o, h)["naive"][idx] for h in range(1, H + 1)], 1),
           "snaive52": np.stack([baseline_point(p, o, h)["snaive52"][idx] for h in range(1, H + 1)], 1),
           "ma4": np.stack([baseline_point(p, o, h)["ma4"][idx] for h in range(1, H + 1)], 1),
           "tsb": tsb, "lightgbm_p50": Q[:, :, list(levels).index(0.5)]}
    sc = scale[idx][:, None]
    sig26 = Y[idx, o - 26:o].std(1)
    from scipy.stats import norm
    seg = intermittency(Y, p.first_week, o)["class"].to_numpy()[idx]
    acc = []
    for name, f in pts.items():
        mase_sku = (np.abs(D - f) / sc).mean(1)
        r = {"fold": tag, "model": name, "n_sku": len(idx), "mase_mean": mase_sku.mean(),
             "mase_median": np.median(mase_sku)}
        for s_ in ("smooth", "erratic", "intermittent", "lumpy"):
            r[f"mase_{s_}"] = mase_sku[seg == s_].mean() if (seg == s_).any() else np.nan
        acc.append(r)
    for k, a in enumerate(levels):
        acc_q = {"fold": tag, "model": f"lightgbm_q{a}", "n_sku": len(idx),
                 "pinball_scaled": pinball(D / sc, Q[:, :, k] / sc, a), "coverage": (D <= Q[:, :, k]).mean()}
        acc.append(acc_q)
    for a in (0.9,):
        qn = np.maximum(tsb + norm.ppf(a) * sig26[:, None], 0)
        acc.append({"fold": tag, "model": f"tsb_normal_q{a}", "n_sku": len(idx),
                    "pinball_scaled": pinball(D / sc, qn / sc, a), "coverage": (D <= qn).mean()})

    # ---------- rolling h=1 forecasts for the ordering policy (re-forecast weekly with new actuals)
    Qr = np.zeros((len(idx), H, len(levels)))
    tsb_r, ma4_r = np.zeros((len(idx), H)), np.zeros((len(idx), H))
    sig_r = np.zeros((len(idx), H))
    for t in range(H):
        r1 = build_rows(p, [o + t], [1], need_target=False)
        r1 = r1.set_index("sku").reindex(idx)
        Qr[:, t] = model.predict(r1.reset_index())
        tsb_r[:, t] = tsb_forecast(p, o + t, 1)[idx, 0]
        ma4_r[:, t] = Y[idx, o + t - 4:o + t].mean(1)
        sig_r[:, t] = Y[idx, o + t - 26:o + t].std(1)
    econ = [economics(p.P[idx, o + t], cfg) for t in range(H)]

    def sim(S):
        return simulate(S, D, econ)

    res, frontier, plans = [], [], []
    S_q = np.stack([s_quantile_newsvendor(levels, Qr[:, t], econ[t]["cr"]) for t in range(H)], 1)
    S_n = np.stack([s_normal_newsvendor(tsb_r[:, t], sig_r[:, t], econ[t]["cr"]) for t in range(H)], 1)
    S_d = np.stack([s_days_of_cover(ma4_r[:, t], cfg["policy"]["days_of_cover_weeks"]) for t in range(H)], 1)
    per_sku = {}
    for name, S in [("quantile_newsvendor_lgb", S_q), ("normal_newsvendor_tsb", S_n),
                    (f"days_of_cover_{cfg['policy']['days_of_cover_weeks']}w_ma4", S_d)]:
        r = sim(S); per_sku[name] = r.pop("per_sku_cost")
        res.append({"fold": tag, "policy": name, "setting": "cost-based CR" if "newsvendor" in name else "default", **r})
    for a in np.round(np.arange(0.5, 0.995, 0.02), 3):
        for name, S in [("quantile_newsvendor_lgb",
                         np.stack([s_quantile_newsvendor(levels, Qr[:, t], np.full(len(idx), a)) for t in range(H)], 1)),
                        ("normal_newsvendor_tsb",
                         np.stack([s_normal_newsvendor(tsb_r[:, t], sig_r[:, t], np.full(len(idx), a)) for t in range(H)], 1))]:
            r = sim(S); r.pop("per_sku_cost")
            frontier.append({"fold": tag, "policy": name, "knob": a, **r}); plans.append((name, a, S))
    for k in np.round(np.arange(0.5, 4.01, 0.1), 2):
        Sd = np.stack([s_days_of_cover(ma4_r[:, t], k) for t in range(H)], 1)
        r = sim(Sd); r.pop("per_sku_cost")
        frontier.append({"fold": tag, "policy": "days_of_cover_ma4", "knob": k, **r}); plans.append(("days_of_cover_ma4", k, Sd))

    # ---------- sensitivity: margin x spoilage (costs change; the CR-based plans are recomputed)
    sens = []
    for m in (0.2, 0.3, 0.4):
        for sp in (0.01, 0.05, 0.10):
            c2 = {**cfg, "costs": {**cfg["costs"], "gross_margin": m, "weekly_spoilage_rate": sp}}
            e2 = [economics(p.P[idx, o + t], c2) for t in range(H)]
            Sq2 = np.stack([s_quantile_newsvendor(levels, Qr[:, t], e2[t]["cr"]) for t in range(H)], 1)
            Sn2 = np.stack([s_normal_newsvendor(tsb_r[:, t], sig_r[:, t], e2[t]["cr"]) for t in range(H)], 1)
            row = {"fold": tag, "margin": m, "spoilage": sp, "cr_median": float(np.median(e2[0]["cr"]))}
            for nm, S in [("qnv", Sq2), ("nnv", Sn2), ("doc2w", S_d)]:
                r = simulate(S, D, e2); row[f"{nm}_cost"], row[f"{nm}_fill"] = r["total_cost"], r["fill_rate"]
            fr2 = []
            for nm, kn, S in plans:
                r = simulate(S, D, e2); fr2.append({"fold": tag, "policy": nm, "knob": kn, **{k: v for k, v in r.items() if k != "per_sku_cost"}})
            mt2 = matched(pd.DataFrame(fr2), fills=(0.95,)).set_index("policy")["total_cost"]
            for nm in mt2.index:
                row[f"{nm}_cost_at_95fill"] = mt2[nm]
            # best days-of-cover in hindsight (k chosen on the test outcome; an upper bound for that baseline)
            row["doc_best_k_cost"] = min(f["total_cost"] for f in fr2 if f["policy"] == "days_of_cover_ma4")
            sens.append(row)

    # ---------- budget-constrained LP vs proportional scale-down
    bud = []
    for frac in cfg["budget"]["levels"] + [1.0]:
        S_lp, S_pr, infos = np.zeros_like(D), np.zeros_like(D), []
        for t in range(H):
            S_t, info = budget_lp(levels, Qr[:, t], econ[t], frac)
            S_lp[:, t] = S_t
            S_pr[:, t] = proportional_budget(info.pop("S_star"), econ[t]["unit_cost"], info["budget"])
            infos.append(info)
        r_lp, r_pr = sim(S_lp), sim(S_pr)
        r_lp.pop("per_sku_cost"); r_pr.pop("per_sku_cost")
        bud.append({"fold": tag, "budget_frac": frac, "lp_total_cost": r_lp["total_cost"],
                    "prop_total_cost": r_pr["total_cost"], "lp_fill": r_lp["fill_rate"], "prop_fill": r_pr["fill_rate"],
                    "lp_stock_value": r_lp["avg_stock_value"], "prop_stock_value": r_pr["avg_stock_value"],
                    "budget_avg": np.mean([i["budget"] for i in infos]),
                    "shadow_price_avg": np.mean([i["shadow_price"] for i in infos]),
                    "lp_vs_prop_pct": 100 * (r_lp["total_cost"] / r_pr["total_cost"] - 1)})
        log(f"  budget {frac:.0%}: LP {r_lp['total_cost']:.0f} vs prop {r_pr['total_cost']:.0f} "
            f"(shadow {bud[-1]['shadow_price_avg']:.3f})")

    pd.DataFrame(sens).to_csv(REPORTS / f"sensitivity_{tag}.csv", index=False, float_format="%.5g")
    segdf = pd.DataFrame({"fold": tag, "sku": p.ids[idx], "class": seg, **{f"cost_{k}": v for k, v in per_sku.items()}})
    return acc, res, frontier, bud, segdf, {"n_sku": int(len(idx)), "iters": {str(k): v for k, v in model.iters.items()},
                                            "cr_median": float(np.median(econ[0]["cr"]))}


def matched(front: pd.DataFrame, fills=(0.90, 0.93, 0.95, 0.97)):
    """Total cost of each policy family interpolated at common fill rates (and fill at common stock value)."""
    out = []
    for (fold, pol), g in front.groupby(["fold", "policy"]):
        g = g.sort_values("fill_rate")
        for f in fills:
            if g["fill_rate"].min() <= f <= g["fill_rate"].max():
                out.append({"fold": fold, "policy": pol, "fill_rate": f,
                            "total_cost": np.interp(f, g["fill_rate"], g["total_cost"]),
                            "avg_stock_value": np.interp(f, g["fill_rate"], g["avg_stock_value"])})
    return pd.DataFrame(out)


def main(which="backtest", scaled=False):
    cfg = load_config()
    p = load_panel(cfg)
    folds = ([(o, f"backtest_{o}") for o in cfg["split"]["backtest_origins"]] if which == "backtest"
             else [(cfg["split"]["test_origin"], "test")])
    A, R, F, B, SEG, META = [], [], [], [], [], {}
    for o, tag in folds:
        a, r, f, b, s, m = run_fold(p, cfg, o, tag, scaled)
        A += a; R += r; F += f; B += b; SEG.append(s); META[tag] = m
    sfx = "backtest" if which == "backtest" else "test"
    acc, res, fr, bud = map(pd.DataFrame, (A, R, F, B))
    mt = matched(fr)
    for df, name in [(acc, "forecast_accuracy"), (res, "policy_results"), (fr, "frontier"), (bud, "budget"),
                     (mt, "matched_fill"), (pd.concat(SEG), "per_sku_cost")]:
        df.to_csv(REPORTS / f"{name}_{sfx}.csv", index=False, float_format="%.5g")
    (REPORTS / f"meta_{sfx}.json").write_text(json.dumps(META, indent=2))
    with open(REPORTS / "experiments.csv", "a") as fh:
        if fh.tell() == 0:
            fh.write("date,commit,folds,config,lgb_mase_mean,best_baseline,best_baseline_mase,qnv_cost,doc_cost,notes\n")
        for tag in META:
            a = acc[(acc.fold == tag) & acc.mase_mean.notna()].set_index("model")["mase_mean"]
            bb = a.drop("lightgbm_p50").idxmin()
            rr = res[res.fold == tag].set_index("policy")["total_cost"]
            fh.write(f"{date.today()},{sha()},{tag},scaled_target={scaled},{a['lightgbm_p50']:.4f},{bb},{a[bb]:.4f},"
                     f"{rr.iloc[0]:.0f},{rr.iloc[2]:.0f},\n")
    if which == "test":
        tr = REPORTS / "test_runs.json"
        d = json.loads(tr.read_text()) if tr.exists() else {"final_test_runs": 0, "runs": []}
        d["final_test_runs"] += 1; d["runs"].append({"date": str(date.today()), "commit": sha()})
        tr.write_text(json.dumps(d, indent=2))
    print(acc[acc.mase_mean.notna()][["fold", "model", "mase_mean", "mase_median", "mase_smooth", "mase_intermittent", "mase_lumpy"]].round(4).to_string())
    print(acc[acc.coverage.notna()][["fold", "model", "pinball_scaled", "coverage"]].round(4).to_string())
    print(res.drop(columns=[]).round(3).to_string())
    print(mt.round(1).to_string())
    print(bud.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", choices=["backtest", "test"], default="backtest")
    ap.add_argument("--scaled-target", action="store_true")
    a = ap.parse_args()
    main(a.folds, scaled=a.scaled_target)
