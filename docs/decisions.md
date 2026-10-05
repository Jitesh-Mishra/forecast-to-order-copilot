# Decisions

**D1 Data: M5 via Nixtla's GitHub mirror, subset CA_1 FOODS (2026-10-05)**
The workspace can't reach Kaggle, but `raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip` (the official M5 files) is reachable. `intermittent_ids.txt` from the M5 benchmark wasn't available, so I used the brief's fallback: store CA_1, FOODS, 1,437 SKUs. At weekly level most are smooth (912) rather than intermittent (369) or lumpy (78), so this is a mixed set, not the intermittent bottom level.

**D2 Weekly buckets Monday to Sunday from d_3**
This makes the last 4 weeks exactly d_1914 to d_1941 (the M5 evaluation window). The 2 earlier origins (weeks 265 and 269, i.e. d_1858 and d_1886) are the backtests. The brief's d_1857/d_1885 were rounded to week boundaries.

**D3 Test lock**
All modelling choices were made on the two backtests. The test fold ran once (`reports/test_runs.json`).

**D4 LightGBM on raw units, not scaled targets**
I tried dividing the target by an origin-known level (rm13 + 1). That preserves quantiles and should balance SKUs in the loss. On both backtests it was slightly worse: MASE 0.949 vs 0.938 and 0.885 vs 0.879, and simulated cost was higher too. Raw units became the default (`reports/experiments.csv` rows 1 to 4).

**D5 Practitioner baseline = 2 weeks of the 4-week moving average**
The brief said 2 weeks of cover on the seasonal-naive mean. At weekly SKU level, the 52-week seasonal naive is the worst forecaster here (MASE 1.54), which would make the baseline a straw man. The 4-week average is what a planner actually uses, and it is the strongest point baseline (MASE 1.05). I also report a days-of-cover sweep, and its best k in hindsight, so the baseline is never under-tuned.

**D6 Costs include spoilage**
With only 25%/yr holding, the critical ratio is about 0.99 and every policy becomes "stock everything". FOODS spoil, so `co` includes 5% of unit cost per week of excess (assumption), giving CR about 0.89. Margin and spoilage are swept in the sensitivity table.

**D7 Weekly re-forecast (rolling h = 1) drives the policy**
Accuracy is reported at a fixed origin for h = 1 to 4 (standard). The ordering policy re-forecasts each week with the new actuals, as a real replenishment system does. Lead time is 1 week (order at the end of week t−1, arrives at the start of week t).

**D8 Matched-fill comparison**
Policies trade cost for service, so comparing them at their own settings mixes the two. Each family's knob is swept (service level for the newsvendors, k for days-of-cover), and cost is interpolated at common fill rates (90/93/95/97%).

**D9 Fair proportional baseline for the budget**
An even scale-down with floor rounding under-spent the budget, which flattered the LP (−12% to −17% on backtests). The proportional factor is now found by bisection so both plans spend the same budget. That shrank the LP's edge to −9% to −12%, the honest number.

**D10 The budget LP is myopic and expected-cost based**
At a 90% budget it loses slightly to the even cut on the test (+1.3%) and on one backtest it nearly ties (−0.2%). It optimises expected cost under the forecast, and the upper quantiles under-cover (P90 = 87%), so when the budget barely binds, its reallocations aren't worth it. I'm reporting this rather than tuning it away on the test.
