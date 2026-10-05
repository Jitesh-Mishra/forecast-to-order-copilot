# Probe prep (answers come only from what this run did)

Run `python -m f2o.run --folds backtest` and `pytest -q` yourself before an interview. Don't rerun `--folds test`; it's locked.

## How the code works, in 5 lines
1. `data.py` buckets the M5 CA_1 FOODS daily sales into Monday-to-Sunday weeks so the last 4 are the M5 evaluation window. It averages Walmart's weekly prices into those weeks.
2. `forecast.py` builds features at an origin using only earlier weeks. It trains one LightGBM per quantile (P30 to P98) and runs the baselines (naive, seasonal naive, 4-week mean, TSB via statsforecast).
3. `policy.py` turns price into economics (`cu` = lost margin, `co` = holding + spoilage) and sets the order-up-to level at the critical-ratio quantile. Under a budget it solves a multiple-choice LP with HiGHS and reads the shadow price from the dual.
4. `simulate()` replays the held-out weeks: stock up to S, sell min(demand, stock), lose the rest, carry leftovers forward, and charge `co` on leftovers and `cu` on lost units.
5. `run.py` does this for 2 backtest folds and the locked test, then writes accuracy, policy, frontier, matched-fill, budget and sensitivity tables to `reports/`.

## Probes

**Q: Your forecast is only 6% better on MASE. Why should I care?**
- 15s: Because the money is in the decision, not the point forecast. A 6% MASE gain became a 26% cost reduction at the same 95% fill rate, because the policy uses the whole distribution and each SKU's economics.
- 60s: Days-of-cover buys service by over-stocking everything, so its cost curve explodes above 96% fill. The quantile newsvendor stocks deep only where shortages are expensive and demand is predictable. Against a normal-distribution newsvendor, the gain is 5% on smooth SKUs but 22 to 27% on intermittent and erratic ones, where the normal is the wrong shape.
- Evidence: `reports/matched_fill_test.csv`, `reports/per_sku_cost_test.csv`, README section 3.

**Q: The costs are made up. How do you know the conclusion isn't an artefact?**
- I swept margin 20/30/40% × spoilage 1/5/10% per week. The policy wins all 9 scenarios, even against days-of-cover with k tuned on the test outcome. The gap narrows to −7.7% when spoilage is high and margin thin, and I say so in the README.
- Evidence: `reports/sensitivity_test.csv`.

**Q: Why not compare policies at their default settings?**
- Because they sit at different fill rates. I sweep each policy's knob and compare cost at matched fill rates (90/93/95/97%). The recommended policy is cheapest at every one.

**Q: What does the shadow price mean?**
- It's the dual of the budget constraint: expected weekly cost saved per extra dollar of inventory budget. At 80% budget λ = $0.063, at 70% it's $0.128. It tells finance where the cut stops being cheap.

**Q: The LP loses at 90% budget. Isn't that a bug?**
- No. It optimises *expected* cost under the forecast, and the upper quantiles under-cover (P90 = 87.2%). When the budget barely binds, the reallocations it makes are worth less than the forecast thinks. The two backtests showed the same pattern (−0.2% and −1.8%). Fixes: recalibrate the tails (conformal) and make the LP aware of on-hand stock.
- Evidence: `reports/budget_test.csv`, `reports/budget_backtest.csv`, docs/decisions.md D10.

**Q: How did you avoid leakage and overfitting the test?**
- Features at origin o use weeks < o only. A unit test scrambles all future weeks and asserts the features don't change. Target-week price and calendar are used, because promotions and prices are planned in advance (standard M5 practice). Every choice (raw vs scaled target, baselines) was made on 2 backtest folds. The test ran once.

**Q: Why was the proportional baseline changed?**
- The first version rounded down and under-spent the budget, which made the LP look 3 to 5 points better than it was. I made both plans spend the same budget before reporting. The LP edge fell to about 9 to 12% on backtests and 4 to 7% on test.

**Q: Why LightGBM and not DeepAR or Croston?**
- Fast, quantile-native and strong on tabular features. TSB stands in for the intermittent-demand family. Plugging in the M5 benchmark's Croston/DeepAR forecasts to test whether the MASE winner is the cost winner is in the parking lot.

**Q: What would change your recommendation?**
- Real margins and spoilage where spoilage is high and margins thin (the gap shrinks to 8% there), longer or variable lead times, and stockout-censored demand on fast sellers.
