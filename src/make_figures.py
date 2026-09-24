"""Generate the EDA figures used in report.pdf."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ASSETS = Path("report_assets")
ASSETS.mkdir(exist_ok=True)
COLOR = "#064A56"

df = pd.read_csv("data/train_test.csv", parse_dates=["date"])
df["rpm"] = df["posted_rate"] / df["distance"]
clean = df[df["rpm"].between(0.5, 10.0)]

# 1. Monthly rate-per-mile vs market index
monthly = clean.groupby(clean["date"].dt.to_period("M")).agg(
    rpm=("rpm", "mean"), market=("market_index", "mean"))
fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
ax2 = ax.twinx()
ax.plot(monthly.index.astype(str), monthly["rpm"], color=COLOR, marker="o",
        linewidth=2, label="Mean rate per mile ($)")
ax2.plot(monthly.index.astype(str), monthly["market"], color="#C0392B",
         marker="s", linewidth=2, linestyle="--", label="Market index")
ax.set_ylabel("Rate per mile ($)")
ax2.set_ylabel("Market index", color="#C0392B")
ax.set_title("Rate per mile tracks the market index over time")
ax.tick_params(axis="x", rotation=35)
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(ASSETS / "monthly_trend.png")
plt.close(fig)

# 2. Rate vs distance, colored by equipment
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
for equip, group in clean.groupby("equipment"):
    ax.scatter(group["distance"], group["posted_rate"], s=3, alpha=0.2, label=equip)
ax.set(xlabel="Distance (miles)", ylabel="Posted rate ($)",
       title="Posted rate vs distance by equipment type")
ax.legend(markerscale=4, fontsize=8)
fig.tight_layout()
fig.savefig(ASSETS / "rate_vs_distance.png")
plt.close(fig)

# 3. Rate-per-mile distribution with outlier cutoffs
fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
ax.hist(df["rpm"], bins=200, color=COLOR, alpha=0.85)
for cut in (0.5, 10.0):
    ax.axvline(cut, color="#C0392B", linestyle="--", linewidth=1.2)
ax.set(xlabel="Rate per mile ($)", ylabel="Loads",
       title="Rate-per-mile distribution — dashed lines mark removed outliers (152 rows)")
ax.set_xlim(0, 16)
fig.tight_layout()
fig.savefig(ASSETS / "rpm_distribution.png")
plt.close(fig)

print("figures written to report_assets/")
