"""reports/figures/{frontier,budget}-{light,dark}.png from the locked test-fold reports."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP, OUT = ROOT / "reports", ROOT / "reports" / "figures"
fr = pd.read_csv(REP / "frontier_test.csv")
pol = pd.read_csv(REP / "policy_results_test.csv")
bud = pd.read_csv(REP / "budget_test.csv")
NAMES = [("days_of_cover_ma4", "Days of cover (baseline)", "days_of_cover_2.0w_ma4"),
         ("normal_newsvendor_tsb", "Normal newsvendor on TSB", "normal_newsvendor_tsb"),
         ("quantile_newsvendor_lgb", "Quantile newsvendor on LightGBM", "quantile_newsvendor_lgb")]
THEMES = {"light": dict(bg="#ffffff", ink="#1f2328", ink2="#59636e", grid="#e6e9ed",
                        cols=["#2a78d6", "#eb6834", "#1baf7a"]),
          "dark": dict(bg="#0d1117", ink="#e6edf3", ink2="#9198a1", grid="#21262d",
                       cols=["#3987e5", "#d95926", "#199e70"])}

for theme, th in THEMES.items():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": th["ink2"], "axes.labelcolor": th["ink"],
                         "xtick.color": th["ink2"], "ytick.color": th["ink2"], "text.color": th["ink"]})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=th["bg"], gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for (key, label, pkey), col in zip(NAMES, th["cols"]):
        g = fr[(fr.policy == key) & (fr.fill_rate >= 0.86)].sort_values("fill_rate")
        ax.plot(100 * g.fill_rate, g.total_cost / 1000, color=col, lw=2, label=label)
        op = pol[pol.policy == pkey].iloc[0]
        ax.scatter([100 * op.fill_rate], [op.total_cost / 1000], s=60, color=col, edgecolor=th["bg"], linewidth=2, zorder=5)
    ax.axvline(95, color=th["ink2"], lw=1, ls="--")
    ax.text(95.15, ax.get_ylim()[1] * 0.97, "matched 95% fill", color=th["ink2"], va="top", fontsize=9)
    ax.set_title("Simulated cost vs fill rate (dots = each policy's own setting)", loc="left", color=th["ink"])
    ax.set_xlabel("unit fill rate, %"); ax.set_ylabel("total cost over 4 weeks, $K (simulated)")
    ax.legend(frameon=False, loc="upper left", labelcolor=th["ink"])
    ax = axes[1]
    b = bud[bud.budget_frac < 1].copy()
    x = range(len(b))
    ax.bar([i - 0.2 for i in x], b.prop_total_cost / 1000, width=0.38, color=th["cols"][0], label="Proportional cut")
    ax.bar([i + 0.2 for i in x], b.lp_total_cost / 1000, width=0.38, color=th["cols"][2], label="Budget LP")
    for i, r in enumerate(b.itertuples()):
        ax.text(i + 0.2, r.lp_total_cost / 1000 + 0.15, f"{r.lp_vs_prop_pct:+.1f}%", ha="center", fontsize=9, color=th["ink"])
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{int(100*f)}% budget\nλ = {s:.3f}" for f, s in zip(b.budget_frac, b.shadow_price_avg)])
    ax.set_title("Inventory budget: LP vs cutting every SKU evenly", loc="left", color=th["ink"])
    ax.set_ylabel("total cost over 4 weeks, $K (simulated)")
    ax.legend(frameon=False, loc="upper right", labelcolor=th["ink"])
    for a in axes:
        a.set_facecolor(th["bg"]); a.grid(axis="y", color=th["grid"], lw=0.8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / f"decision-{theme}.png", dpi=150, facecolor=th["bg"])
print("ok")
