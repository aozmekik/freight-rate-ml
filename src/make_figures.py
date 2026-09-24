"""Generate the report figures in the shared single-accent style."""

from pathlib import Path

import numpy as np
import pandas as pd

import plotstyle
from plotstyle import ACCENT, GRAY

plotstyle.apply()
import matplotlib.pyplot as plt

ASSETS = Path("report_assets")
ASSETS.mkdir(exist_ok=True)

df = pd.read_csv("data/train_test.csv", parse_dates=["date"])
df["rpm"] = df["posted_rate"] / df["distance"]
clean = df[df["rpm"].between(0.5, 10.0)]

# ------------------------------------------------ 1. rate vs distance
fig, ax = plt.subplots(figsize=(6.1, 2.9))
markers = {"Dry Van": "o", "Reefer": "^", "Flatbed": "s"}
for equip, group in clean.groupby("equipment"):
    ax.scatter(group["distance"], group["posted_rate"], s=2.5, alpha=0.18,
               color=ACCENT, marker=markers[equip], label=equip,
               linewidths=0, rasterized=True)
ax.set(xlabel="Distance (miles)", ylabel="Posted rate (USD)",
       title="Posted rate versus distance, by equipment type")
leg = ax.legend(markerscale=3.5, loc="upper left", handletextpad=0.2,
                borderaxespad=0.1)
for handle in leg.legend_handles:
    handle.set_alpha(0.9)
fig.savefig(ASSETS / "rate_vs_distance.png")
plt.close(fig)

# ------------------------------------------------ 2. monthly trend
monthly = clean.groupby(clean["date"].dt.to_period("M")).agg(
    rpm=("rpm", "mean"), market=("market_index", "mean"))
x = range(len(monthly))
labels = [p.strftime("%b") for p in monthly.index]
fig, ax = plt.subplots(figsize=(6.1, 2.7))
ax2 = ax.twinx()
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color("#9DAFB3")
ax2.grid(False)
l1, = ax.plot(x, monthly["rpm"], color=ACCENT, marker="o", markersize=3.5,
              linewidth=1.8, label="Rate per mile (left)")
l2, = ax2.plot(x, monthly["market"], color=ACCENT, marker="o", markersize=3.5,
               linewidth=1.4, linestyle=(0, (4, 2)), markerfacecolor="white",
               label="Market index (right)")
ax.set_xticks(list(x), labels)
ax.set_ylabel("Rate per mile (USD)")
ax2.set_ylabel("Market index")
ax.set_ylim(2.0, 2.62)      # headroom so the legend clears both series
ax2.set_ylim(0.80, 1.52)
ax.set_title("Monthly rate per mile tracks the market index")
ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left",
          ncol=2, columnspacing=1.2, handlelength=2.6)
fig.savefig(ASSETS / "monthly_trend.png")
plt.close(fig)

# ------------------------------------------------ 3. rpm distribution
fig, ax = plt.subplots(figsize=(6.1, 2.5))
ax.hist(df["rpm"], bins=np.arange(0, 16.1, 0.08), color=ACCENT, alpha=0.9,
        linewidth=0)
for cut in (0.5, 10.0):
    ax.axvline(cut, color=GRAY, linestyle=(0, (4, 2)), linewidth=1.1)
ax.annotate("removed\n(0.3% of rows)", xy=(10.25, 4200), fontsize=8.5,
            color=GRAY)
ax.set(xlim=(0, 16), xlabel="Rate per mile (USD)", ylabel="Number of loads",
       title="Rate-per-mile distribution with outlier cutoffs (dashed)")
fig.savefig(ASSETS / "rpm_distribution.png")
plt.close(fig)

print("figures written to report_assets/")
