# Freight Rate Prediction Challenge

Predicts posted rates for 12,000 truckload loads (Nov–Dec 2025) from 48,000
labeled loads (Jan–Oct 2025), and produces the fixed December 2025 rate chart.

See `freight-rate-ml-assessment.pdf` for the assessment instructions and
`report.pdf` for the full write-up (exploration findings, data-quality
handling, validation strategy, model choice, results).

## Approach in short

- **Model:** LightGBM on `log(posted_rate)` with an L1 objective; final model
  is a 5-seed ensemble averaged in log space.
- **Features:** distance, log(distance), weight, equipment, pickup/delivery
  city (native categorical), pickup/delivery coordinates, `market_index`,
  `quote_signal`, day-of-week. The ~4,000-value route string and a month
  feature were tested and deliberately excluded (both overfit / don't
  generalize to unseen months).
- **Validation:** strict time-based split — train Jan–Sep, validate on
  October, the month right before the Nov–Dec prediction window. Holdout:
  MAE ≈ $118, MAPE ≈ 4.6%, R² ≈ 0.84. The final model is then refit on all
  ten months.
- **Data quality:** 152 training rows with implausible rate-per-mile
  (<$0.50/mi or >$10/mi) dropped; missing `weight`/`market_index` (~0.7%
  each) left as NaN for LightGBM to handle natively.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python src/train.py       # holdout metrics, figures, final models in model/
python src/predict.py     # validation_predictions.csv + fills data/december_chart_inputs.csv
python src/make_figures.py
python score.py --predictions validation_predictions.csv \
    --december-predictions data/december_chart_inputs.csv
pdflatex report.tex       # optional: rebuilds report.pdf (needs a LaTeX install)
```

The scorer validates both files and creates
`scorer_results/candidate_december.png`.

## Repository layout

```
data/                       input CSVs (train/validation/templates)
src/features.py             shared feature engineering (train/serve identical)
src/plotstyle.py            shared figure style (single accent color)
src/train.py                time-split validation + final model training
src/predict.py              validation + December predictions
src/make_figures.py         EDA figures for the report
experiments.py              one-off model-config comparison (exploratory)
report.tex                  LaTeX source of the assessment report
model/                      trained LightGBM boosters + meta.json
report_assets/              metrics.json + figures
scorer_results/             candidate_december.png
validation_predictions.csv  final submission file (load_id,predicted_rate)
report.pdf                  assessment report
```
