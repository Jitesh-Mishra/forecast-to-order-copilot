"""Forecasters: naive / seasonal naive / 4-week mean / TSB baselines and LightGBM direct quantiles.

Every forecast made at origin o uses weeks < o only; week o + h - 1 is the target for horizon h.
"""
from __future__ import annotations

import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import Panel

FEATS = ["h", "lag1", "lag2", "lag3", "lag4", "lag8", "lag52", "rm4", "rm8", "rm13", "rm26", "rm52", "rstd13",
         "zero13", "nzmean13", "since_sale", "age", "price", "price_ratio13", "price_chg1", "snap_ca", "event",
         "month", "woy", "dept", "item"]
CATS = ["dept", "item"]


def _hist(Y, o, k):
    return Y[:, o - k:o]


def features_at(p: Panel, o: int, h: int) -> pd.DataFrame:
    """Feature block for all SKUs at origin o (data through week o-1), horizon h (target week o+h-1)."""
    Y, P = p.Y, p.P
    tw = o + h - 1
    h13 = _hist(Y, o, 13)
    nz = np.where(h13 > 0, h13, np.nan)
    last_sale = np.where((Y[:, :o] > 0).any(1), o - 1 - np.argmax((Y[:, :o] > 0)[:, ::-1], axis=1), -999)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pr13 = np.nanmean(P[:, o - 13:o], axis=1)
        nzm = np.nanmean(nz, axis=1)
    price_t = P[:, tw] if tw < p.n_weeks else P[:, o - 1]
    cal = p.cal.loc[min(tw, p.n_weeks - 1)]
    n = len(Y)
    return pd.DataFrame({
        "h": np.full(n, h), "lag1": Y[:, o - 1], "lag2": Y[:, o - 2], "lag3": Y[:, o - 3], "lag4": Y[:, o - 4],
        "lag8": Y[:, o - 8], "lag52": Y[:, tw - 52],
        "rm4": _hist(Y, o, 4).mean(1), "rm8": _hist(Y, o, 8).mean(1), "rm13": h13.mean(1),
        "rm26": _hist(Y, o, 26).mean(1), "rm52": _hist(Y, o, 52).mean(1), "rstd13": h13.std(1),
        "zero13": (h13 == 0).mean(1), "nzmean13": np.nan_to_num(nzm), "since_sale": o - 1 - last_sale,
        "age": o - p.first_week, "price": price_t, "price_ratio13": price_t / pr13,
        "price_chg1": price_t / P[:, o - 1], "snap_ca": np.full(n, cal["snap_ca"]),
        "event": np.full(n, cal["event"]), "month": np.full(n, cal["month"]), "woy": np.full(n, cal["woy"]),
        "dept": pd.Categorical(p.dept).codes, "item": np.arange(n),
    })


def build_rows(p: Panel, origins, hs, need_target=True):
    blocks = []
    for o in origins:
        for h in hs:
            f = features_at(p, o, h)
            f["sku"], f["origin"] = np.arange(len(p.Y)), o
            tw = o + h - 1
            f["y"] = p.Y[:, tw] if tw < p.n_weeks else np.nan
            # usable: launched >= 13 weeks before origin and on sale in the target week
            ok = (o - p.first_week >= 13) & ~np.isnan(f["price"].to_numpy())
            blocks.append(f[ok])
    df = pd.concat(blocks, ignore_index=True)
    if need_target:
        df = df[df["y"].notna()]
    return df


class LGBQuantiles:
    """One LightGBM per quantile on raw weekly units. scaled=True divides the target by an origin-known
    level (rm13 + 1), which preserves quantiles; on both backtest folds it was slightly worse (MASE and
    simulated cost), so raw units is the default (docs/decisions.md D4)."""

    PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=2)

    def __init__(self, quantiles, seed=42, scaled=False):
        self.q, self.seed, self.scaled = list(quantiles), seed, scaled

    def _scale(self, df):
        return (df["rm13"].to_numpy() + 1.0) if self.scaled else np.ones(len(df))

    def fit(self, tr: pd.DataFrame, va: pd.DataFrame, log=print):
        self.models, self.iters = {}, {}
        str_, sva = self._scale(tr), self._scale(va)
        dtr = lgb.Dataset(tr[FEATS], tr["y"] / str_, categorical_feature=CATS, free_raw_data=False)
        dva = lgb.Dataset(va[FEATS], va["y"] / sva, reference=dtr, categorical_feature=CATS)
        for a in self.q:
            t0 = time.time()
            m = lgb.train({**self.PARAMS, "objective": "quantile", "alpha": a, "metric": "quantile", "seed": self.seed},
                          dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            self.models[a], self.iters[a] = m, m.best_iteration
            log(f"  lgb q{a}: iters={m.best_iteration} ({time.time()-t0:.0f}s)")
        return self

    def predict(self, df) -> np.ndarray:
        s = self._scale(df)
        out = np.stack([self.models[a].predict(df[FEATS], num_iteration=self.iters[a]) * s for a in self.q], 1)
        return np.maximum.accumulate(np.clip(out, 0, None), axis=1)   # non-negative, non-crossing


def train_lgb(p: Panel, fold_origin: int, quantiles, horizon=4, first_origin=120, val_weeks=8, seed=42,
              scaled=False, log=print):
    """Train on targets before fold_origin - val_weeks, early-stop on the val_weeks before fold_origin."""
    rows = build_rows(p, range(first_origin, fold_origin), range(1, horizon + 1))
    tw = rows["origin"] + rows["h"] - 1
    tr, va = rows[tw < fold_origin - val_weeks], rows[(tw >= fold_origin - val_weeks) & (tw < fold_origin)]
    log(f"  lgb rows train={len(tr):,} val={len(va):,}")
    return LGBQuantiles(quantiles, seed, scaled).fit(tr, va, log)


# ----------------------------------------------------------------------------- baselines
def baseline_point(p: Panel, o: int, h: int) -> dict:
    Y = p.Y
    tw = o + h - 1
    return {"naive": Y[:, o - 1], "snaive52": Y[:, tw - 52], "ma4": Y[:, o - 4:o].mean(1)}


def tsb_forecast(p: Panel, o: int, horizon: int, alpha_d=0.1, alpha_p=0.1) -> np.ndarray:
    """TSB (statsforecast) fitted on each SKU's history before o. Returns [sku, horizon]."""
    from statsforecast import StatsForecast
    from statsforecast.models import TSB
    rows = []
    for i in range(len(p.Y)):
        st = min(p.first_week[i], o - 1)
        y = p.Y[i, st:o]
        rows.append(pd.DataFrame({"unique_id": i, "ds": np.arange(st, o), "y": y}))
    df = pd.concat(rows)
    sf = StatsForecast(models=[TSB(alpha_d=alpha_d, alpha_p=alpha_p)], freq=1, n_jobs=1)
    fc = sf.forecast(df=df, h=horizon)
    fc = fc.sort_values(["unique_id", "ds"])
    return fc["TSB"].to_numpy().reshape(len(p.Y), horizon)
