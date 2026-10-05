"""Rebuild data/ from the M5 Forecasting Accuracy CSVs (Kaggle, or Nixtla's GitHub mirror of m5.zip).

python scripts/prepare_data.py /path/to/m5_csvs [intermittent_ids.txt]
Default subset: store CA_1, category FOODS (1,437 SKUs).
"""
import os
import sys

import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "."
s = pd.read_csv(os.path.join(src, "sales_train_evaluation.csv"))
if len(sys.argv) > 2:
    ids = set(open(sys.argv[2]).read().split())
    s = s[s["id"].isin(ids)]
else:
    s = s[(s.store_id == "CA_1") & (s.cat_id == "FOODS")]
os.makedirs("data", exist_ok=True)
s.drop(columns=[c for c in ["id"] if c in s.columns]).to_parquet("data/sales.parquet", index=False)
p = pd.read_csv(os.path.join(src, "sell_prices.csv")).merge(s[["store_id", "item_id"]].drop_duplicates())
p.to_parquet("data/prices.parquet", index=False)
c = pd.read_csv(os.path.join(src, "calendar.csv"))
c.drop(columns=[x for x in ["d"] if x in c.columns]).to_parquet("data/calendar.parquet", index=False)
print(len(s), "SKUs")
