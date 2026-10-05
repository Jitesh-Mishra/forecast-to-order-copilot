"""Load the M5 subset and bucket it into Monday-to-Sunday weeks ending on d_1941."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def load_config(path=ROOT / "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


@dataclass
class Panel:
    ids: np.ndarray          # SKU ids
    item: np.ndarray
    dept: np.ndarray
    Y: np.ndarray            # weekly units sold [sku, week]
    P: np.ndarray            # mean shelf price in the week [sku, week], nan before launch
    cal: pd.DataFrame        # weekly calendar features, index = week
    first_week: np.ndarray   # first week with a sale, per SKU
    week_start_day: np.ndarray  # d-number of each week's first day

    @property
    def n_weeks(self):
        return self.Y.shape[1]


def load_panel(cfg: dict | None = None) -> Panel:
    cfg = cfg or load_config()
    w0 = cfg["split"]["week_start_day"]
    s = pd.read_parquet(DATA / "sales.parquet")
    cal = pd.read_parquet(DATA / "calendar.parquet")
    pr = pd.read_parquet(DATA / "prices.parquet")
    dcols = [f"d_{i}" for i in range(w0, 1942)]
    n_weeks = len(dcols) // 7
    daily = s[dcols].to_numpy(dtype=np.float64)
    Y = daily.reshape(len(s), n_weeks, 7).sum(2)

    # daily price via the Walmart week of each day, then mean over our Monday-Sunday week
    cal = cal.assign(dnum=np.arange(1, len(cal) + 1))  # row i is d_{i+1} (2011-01-29 = d_1)
    day_wk = cal.set_index("dnum").loc[range(w0, 1942), "wm_yr_wk"].to_numpy()
    piv = pr.pivot_table(index=["item_id"], columns="wm_yr_wk", values="sell_price")
    piv = piv.reindex(s["item_id"].to_numpy())
    dp = piv.reindex(columns=day_wk).to_numpy()  # [sku, day]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-nan weeks before launch stay nan
        P = np.nanmean(dp.reshape(len(s), n_weeks, 7), axis=2)

    c = cal.set_index("dnum").loc[range(w0, 1942)]
    wk = np.repeat(np.arange(n_weeks), 7)
    wcal = pd.DataFrame({
        "week": wk,
        "snap_ca": c["snap_CA"].to_numpy(),
        "event": c["event_name_1"].notna().to_numpy().astype(int),
        "month": pd.to_datetime(c["date"]).dt.month.to_numpy(),
        "woy": pd.to_datetime(c["date"]).dt.isocalendar().week.to_numpy().astype(int),
    }).groupby("week").agg(snap_ca=("snap_ca", "sum"), event=("event", "sum"), month=("month", "first"),
                           woy=("woy", "first"))
    first = np.where((Y > 0).any(1), (Y > 0).argmax(1), n_weeks)
    ids = (s["item_id"] + "_" + s["store_id"] + "_evaluation").to_numpy()   # Kaggle-style id
    return Panel(ids, s["item_id"].to_numpy(), s["dept_id"].to_numpy(), Y, P, wcal, first,
                 w0 + 7 * np.arange(n_weeks))


def intermittency(Y: np.ndarray, first: np.ndarray, end: int) -> pd.DataFrame:
    """ADI / CV^2 classification (Syntetos-Boylan) on weeks [first, end)."""
    rows = []
    for i in range(len(Y)):
        y = Y[i, first[i]:end]
        nz = y[y > 0]
        adi = len(y) / max(len(nz), 1)
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0
        cls = ("smooth" if adi < 1.32 and cv2 < 0.49 else "erratic" if adi < 1.32 else
               "intermittent" if cv2 < 0.49 else "lumpy")
        rows.append((adi, cv2, cls))
    return pd.DataFrame(rows, columns=["adi", "cv2", "class"])
