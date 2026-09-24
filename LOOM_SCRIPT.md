# Loom walkthrough script (2–3 minutes)

## 1. Key findings from exploring the data (~30s)

- 48,000 labeled loads, Jan–Oct 2025; the 12,000 validation loads are Nov–Dec.
- Distance dominates pricing (correlation 0.91), and pricing is
  multiplicative — rate per mile is roughly log-normal around $2.15/mi, with
  long hauls cheaper per mile.
- `market_index` drives the time pattern and has a weekly sawtooth; Reefer and
  Flatbed price above Dry Van per mile; heavier loads price slightly higher.
- The validation set introduces 8 unseen cities and ~1,500 loads on unseen
  routes — the model has to generalize geographically.

## 2. Data-quality issues and fixes (~30s)

- 152 rows (0.3%) with impossible rate-per-mile (<$0.50 or >$10/mi) — dropped
  from training.
- `weight` and `market_index` are missing for ~0.7% of rows, evenly across
  months — left as NaN; LightGBM handles missing values natively.
- Verified load IDs are unique and disjoint between train and validation.

## 3. Model choice (~30s)

- LightGBM gradient-boosted trees on `log(posted_rate)` with an L1 objective:
  handles non-linear lane × equipment × market interactions, native
  categoricals and NaNs, and the log target matches the multiplicative error
  structure. It beat raw-target, L2, Huber and Tweedie variants.
- Route string and month features were tested and removed — both overfit and
  don't generalize to unseen routes/months; `market_index` + day-of-week carry
  the time signal instead.

## 4. Training and validation approach (~40s)

- Strict time-based split: train Jan–Sep, validate on October — the month
  right before the prediction window. A random split would leak lane and
  market patterns and overstate accuracy.
- October holdout: MAE ≈ $118, MAPE ≈ 4.6%, median APE ≈ 2.7%, R² ≈ 0.84.
- Final model refit on all ten months with the early-stopped iteration count;
  5 seeds averaged for stability.

## 5. Code walkthrough (~40s)

- `src/features.py` — one shared feature function used by both training and
  serving, so there's no train/serve skew.
- `src/train.py` — split, early stopping, metrics, figures, then the final
  5-seed refit saved to `model/`.
- `src/predict.py` — predicts all 12,000 validation loads into
  `validation_predictions.csv`; for the December chart inputs (which lack
  `market_index`/coordinates) it attaches city coordinates and estimates each
  December day's market signal from the validation set's December rows.
- Finish by showing `scorer_results/candidate_december.png`: the weekly
  sawtooth with a Christmas-week peak, around $790–820 for the reference lane
  (training history on that lane averaged $857).
