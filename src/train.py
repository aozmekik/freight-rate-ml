"""Train the freight rate model.

Validation strategy
-------------------
Time-based holdout that mirrors the real task: train on Jan-Sep 2025 and
validate on October 2025, the month immediately preceding the Nov/Dec 2025
prediction window. A random split would leak route-level and market-level
patterns across the boundary and overstate performance.

After model selection, the final model is refit on ALL of Jan-Oct with the
early-stopped iteration count, using several seeds whose predictions are
averaged at inference time.

Outputs
-------
model/final_seed<N>.txt   LightGBM boosters (final, trained on all data)
model/meta.json           feature list, config, best iteration
report_assets/metrics.json        October holdout metrics
report_assets/feature_importance.png
report_assets/predicted_vs_actual.png
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import CATEGORICAL_FEATURES, FEATURES, TARGET, build_features, clean_training

DATA = Path("data/train_test.csv")
MODEL_DIR = Path("model")
ASSETS = Path("report_assets")

SEEDS = [7, 21, 42, 84, 168]

PARAMS = dict(
    objective="l1",          # robust to the heavy right tail of posted_rate
    learning_rate=0.05,
    num_leaves=511,
    min_data_in_leaf=30,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbose=-1,
    n_jobs=8,
)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    ape = np.abs(err) / y_true
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAPE": float(ape.mean()),
        "MedianAPE": float(np.median(ape)),
        "R2": float(1 - (err**2).sum() / ((y_true - y_true.mean()) ** 2).sum()),
    }


def train_one(train_df: pd.DataFrame, valid_df: pd.DataFrame | None, seed: int,
              num_boost_round: int) -> lgb.Booster:
    params = {**PARAMS, "seed": seed}
    cats = [c for c in CATEGORICAL_FEATURES if c in FEATURES]
    dtrain = lgb.Dataset(train_df[FEATURES], np.log(train_df[TARGET]),
                         categorical_feature=cats)
    callbacks = []
    valid_sets = None
    if valid_df is not None:
        valid_sets = [lgb.Dataset(valid_df[FEATURES], np.log(valid_df[TARGET]),
                                  reference=dtrain)]
        callbacks.append(lgb.early_stopping(200, verbose=False))
    return lgb.train(params, dtrain, num_boost_round=num_boost_round,
                     valid_sets=valid_sets, callbacks=callbacks)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)

    df = clean_training(build_features(pd.read_csv(DATA, parse_dates=["date"])))
    holdout = df.date.dt.month == 10
    train_df, valid_df = df[~holdout], df[holdout]
    print(f"Train Jan-Sep: {len(train_df):,} rows | Holdout Oct: {len(valid_df):,} rows")

    model = train_one(train_df, valid_df, seed=SEEDS[0], num_boost_round=4000)
    best_iter = model.best_iteration
    pred = np.exp(model.predict(valid_df[FEATURES], num_iteration=best_iter))
    scores = metrics(valid_df[TARGET].to_numpy(), pred)
    print(f"Best iteration: {best_iter}")
    print("October holdout:", {k: round(v, 4) for k, v in scores.items()})
    (ASSETS / "metrics.json").write_text(json.dumps(
        {"split": "train=2025-01..2025-09, holdout=2025-10",
         "best_iteration": best_iter, "holdout_metrics": scores}, indent=2))

    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    imp.plot.barh(ax=ax, color="#064A56")
    ax.set_title("Feature importance (gain)")
    fig.tight_layout()
    fig.savefig(ASSETS / "feature_importance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    ax.scatter(valid_df[TARGET], pred, s=3, alpha=0.25, color="#064A56")
    lim = (0, float(max(valid_df[TARGET].quantile(0.999), np.quantile(pred, 0.999))))
    ax.plot(lim, lim, color="#C0392B", linewidth=1)
    ax.set(xlabel="Actual rate ($)", ylabel="Predicted rate ($)",
           title="October holdout: predicted vs actual", xlim=lim, ylim=lim)
    fig.tight_layout()
    fig.savefig(ASSETS / "predicted_vs_actual.png")
    plt.close(fig)

    # Final model: refit on ALL of Jan-Oct with the early-stopped round count.
    for seed in SEEDS:
        final = train_one(df, None, seed=seed, num_boost_round=best_iter)
        final.save_model(str(MODEL_DIR / f"final_seed{seed}.txt"))
    (MODEL_DIR / "meta.json").write_text(json.dumps(
        {"features": FEATURES,
         "categorical": [c for c in CATEGORICAL_FEATURES if c in FEATURES],
         "params": PARAMS, "seeds": SEEDS, "num_iterations": best_iter,
         "log_target": True}, indent=2))
    print(f"Saved {len(SEEDS)} final models (trained on all {len(df):,} rows) to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
