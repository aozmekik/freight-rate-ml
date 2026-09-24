"""Build report.pdf from the training outputs and scorer chart."""

import json
from pathlib import Path

from fpdf import FPDF

ASSETS = Path("report_assets")
CHART = Path("scorer_results/candidate_december.png")
metrics = json.loads((ASSETS / "metrics.json").read_text())["holdout_metrics"]


class Report(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 9)
        self.set_text_color(6, 74, 86)
        self.cell(0, 8, "Freight Rate Prediction - ML Assessment", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def h1(self, text):
        self.set_font("helvetica", "B", 15)
        self.set_text_color(6, 74, 86)
        self.multi_cell(0, 8, text)
        self.ln(1)

    def h2(self, text):
        self.set_font("helvetica", "B", 12)
        self.set_text_color(6, 74, 86)
        self.multi_cell(0, 7, text)
        self.ln(1)

    def body(self, text):
        self.set_font("helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.2, text)
        self.ln(1.5)

    def bullet(self, text):
        self.set_font("helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.2, "-  " + text)
        self.ln(0.5)

    def figure(self, path, w=175):
        self.image(str(path), x=(210 - w) / 2, w=w)
        self.ln(3)


pdf = Report("P", "mm", "A4")
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(17, 14, 17)

# ---------------------------------------------------------------- page 1
pdf.add_page()
pdf.h1("Freight Rate Prediction")
pdf.body(
    "Goal: predict the posted rate for 12,000 truckload freight loads dated "
    "November-December 2025 (data/validation.csv), given 48,000 labeled loads "
    "from January-October 2025 (data/train_test.csv), plus a fixed daily "
    "December 2025 rate chart for a reference lane (Lexington to Fort Wayne, "
    "360 mi, Dry Van, 32,000 lb)."
)
pdf.body(
    "Solution: a LightGBM gradient-boosted tree ensemble trained on "
    "log(posted_rate) with an L1 objective, validated with a strict "
    "time-based holdout (train Jan-Sep, validate Oct), then refit on all ten "
    "months across 5 seeds whose predictions are averaged."
)

pdf.h2("Key findings from data exploration")
pdf.bullet(
    "Distance dominates: correlation with posted_rate is 0.91, and rate per "
    "mile falls as distance grows (long hauls price lower per mile).")
pdf.bullet(
    "Rates are multiplicative - rate per mile is roughly log-normal with a "
    "median of $2.15/mi - so the model predicts log(rate).")
pdf.bullet(
    "The market_index feature drives the time pattern: daily mean rate per "
    "mile correlates 0.58 with market_index, and it shows a weekly sawtooth "
    "(rates rise through the week, dip on weekends).")
pdf.bullet(
    "Equipment matters: Reefer (+13%) and Flatbed (+8%) price above Dry Van "
    "per mile. Heavier loads price slightly higher per mile.")
pdf.bullet(
    "quote_signal carries only a weak, mostly incremental signal (row-level "
    "correlation with rate per mile ~0.06).")
pdf.bullet(
    "The validation set introduces 8 cities never seen in training (e.g. "
    "Chicago, Charlotte, San Diego) and 1,461 loads on unseen routes - so "
    "the model leans on coordinates, not memorized lanes.")
pdf.figure(ASSETS / "rate_vs_distance.png")

# ---------------------------------------------------------------- page 2
pdf.add_page()
pdf.h2("Data quality issues and how they were handled")
pdf.bullet(
    "Outliers: 152 training rows (0.3%) have implausible rate-per-mile values "
    "(< $0.50/mi or > $10/mi, e.g. a $25,533 load). These were dropped from "
    "training; the L1 objective adds further robustness.")
pdf.bullet(
    "Missing values: weight is missing for 300 rows and market_index for 374 "
    "(~0.7% each, spread evenly across months - consistent with missing at "
    "random). LightGBM handles NaNs natively, so no imputation was needed.")
pdf.bullet(
    "Coordinates are constant per city, so they act as a geographic encoding "
    "and generalize to the 8 unseen validation cities.")
pdf.bullet(
    "load_id values are unique and disjoint between train and validation; "
    "no leakage through identifiers.")
pdf.figure(ASSETS / "rpm_distribution.png")
pdf.figure(ASSETS / "monthly_trend.png")

# ---------------------------------------------------------------- page 3
pdf.add_page()
pdf.h2("Validation approach and train/test split")
pdf.body(
    "The development data (Jan-Oct 2025) was split by time, not randomly: "
    "training on January-September and validating on October - the month "
    "immediately before the November-December prediction window. This mirrors "
    "the real task, where the future must be predicted from the past. A random "
    "split would leak lane- and market-level patterns across the boundary and "
    "report over-optimistic metrics."
)
pdf.body(
    "October holdout results (4,831 loads):"
)
pdf.set_font("courier", "", 10)
for line in [
    f"  MAE        ${metrics['MAE']:,.2f}",
    f"  RMSE       ${metrics['RMSE']:,.2f}   (inflated by a handful of extreme loads)",
    f"  MAPE       {metrics['MAPE'] * 100:.2f}%",
    f"  Median APE {metrics['MedianAPE'] * 100:.2f}%",
    f"  R2         {metrics['R2']:.4f}",
]:
    pdf.cell(0, 5.5, line)
    pdf.ln()
pdf.ln(2)
pdf.body(
    "Model selection (objective, target transform, feature set, tree size) was "
    "done exclusively on this holdout. The final model was then refit on all "
    "ten months with the early-stopped iteration count, using 5 seeds averaged "
    "at inference for stability."
)
pdf.h2("Model choice")
pdf.body(
    "Gradient-boosted trees (LightGBM) were chosen because the relationships "
    "are non-linear and full of interactions (lane x equipment x market), the "
    "data is tabular and mid-sized, native categorical support avoids fragile "
    "one-hot encoding of 64 cities, and NaNs are handled natively. Predicting "
    "log(rate) with an L1 objective matched the multiplicative error structure "
    "and outperformed raw-target, L2, Huber and Tweedie variants on the "
    "holdout. The ~4,000-value route string was tested and dropped - it "
    "overfit relative to city + coordinate features. A month feature was "
    "also deliberately excluded: November/December never appear in training, "
    "and month was confirmed to hurt holdout performance; the model instead "
    "uses market_index plus day-of-week to capture time effects that "
    "generalize to unseen months."
)
pdf.body(
    "Features: distance and log(distance), weight, equipment, pickup/delivery "
    "city (categorical), pickup/delivery coordinates, market_index, "
    "quote_signal, and day-of-week."
)
pdf.figure(ASSETS / "feature_importance.png")

# ---------------------------------------------------------------- page 4
pdf.add_page()
pdf.h2("Holdout fit")
pdf.figure(ASSETS / "predicted_vs_actual.png", w=115)
pdf.h2("December 2025 prediction chart (scorer output)")
pdf.body(
    "The December chart inputs contain no market_index, quote_signal or "
    "coordinates. Coordinates were attached from the city lookup; market_index "
    "and quote_signal were estimated per December date from the daily means "
    "observed in the validation set (which covers Dec 2025). The resulting "
    "curve reproduces the weekly sawtooth of market_index, with a "
    "Christmas-week peak - consistent with the route's history (32 training "
    "loads on this lane, mean rate $857)."
)
pdf.figure(CHART, w=168)
pdf.h2("Reproduce")
pdf.set_font("courier", "", 9)
for line in [
    "python -m pip install -r requirements.txt",
    "python src/train.py      # holdout metrics + final models",
    "python src/predict.py    # validation_predictions.csv + December inputs",
    "python score.py --predictions validation_predictions.csv \\",
    "    --december-predictions data/december_chart_inputs.csv",
]:
    pdf.cell(0, 5.2, line)
    pdf.ln()

pdf.output("report.pdf")
print("wrote report.pdf")
