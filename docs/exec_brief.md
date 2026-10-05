# Executive brief: replacing days-of-cover ordering for grocery

**To:** VP Supply Chain · **Scope:** 1,437 FOODS SKUs, one store (public M5 data, Walmart CA_1) · **Test:** 4 held-out weeks, replayed against actual sales

## Recommendation
Replace the "order up to two weeks of average sales" rule with **service-optimised ordering**. Each SKU's order target comes from its forecast demand distribution and its own cost of running short versus over-stocking. Under an inventory budget, use the optimiser to decide **which SKUs absorb the cut**, instead of cutting all of them evenly.

## What it is worth (simulated, under stated cost assumptions)

| | Current rule | Recommended | Change |
|---|---:|---:|---:|
| Cost of shortages + spoilage/holding, 4 weeks | $12,240 | $7,919 | **−35%** |
| Average inventory dollars | $84.4K | $64.7K | **−23%** |
| Unit fill rate | 96.3% | 96.0% | −0.3 pts |
| Cost at the *same* 95% fill rate | $10,614 | $7,891 | **−26%** |

- It holds in all 9 margin and spoilage scenarios tested, including against the current rule tuned with hindsight (−8% to −46%).
- The biggest gains are on sporadic and erratic items (−22% to −27% versus a textbook normal-distribution method). Steady sellers gain about 5%.

## If the inventory budget is cut

| Budget | Even cut | Optimised cut | Saving | Value of one more budget dollar |
|---|---:|---:|---:|---:|
| 70% | $13,151 | $12,181 | −7.4% | 12.8¢ per week |
| 80% | $9,818 | $9,408 | −4.2% | 6.3¢ per week |
| 90% | $8,258 | $8,365 | +1.3% | 1.6¢ per week |

**Read:** cuts down to about 90% of the optimal plan cost little, and below 80% they get expensive fast. If a cut is needed, the optimiser should decide where it lands. A shallow 10% cut doesn't need it.

## Risks
1. The costs are assumptions (30% margin, 25%/yr holding, 5%/week spoilage). The direction is robust; the dollar amounts need real margins and shrink data.
2. History records sales, not demand, so past stockouts hide some demand.
3. Lead time is fixed at one week here. Longer lead times need one more modelling step before rollout.

## Next steps (2 to 3 weeks)
1. A pilot on one department with real margins and spoilage, run as shadow orders next to the current rule for 4 weeks.
2. A planner copilot that answers "why this order?" and "what if the budget drops 10%?" from the same numbers, with human approval for overrides.
3. A go/no-go on cost at matched fill rate, measured on the pilot.
