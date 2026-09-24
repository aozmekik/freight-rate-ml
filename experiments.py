"""One-off experiment: compare model configs on a time-based holdout.

Train Jan-Sep 2025, validate on October 2025 (adjacent to the Nov/Dec
prediction window). Not part of the final pipeline - see train.py.
"""

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from features import CATEGORICAL_FEATURES, FEATURES, TARGET, build_features, clean_training


def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs(y_true - y_pred) / y_true)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)


def run(name, features, log_target=True, extra_params=None, train_df=None, valid_df=None):
    global train, valid
    if train_df is None:
        train_df = train
    if valid_df is None:
        valid_df = valid
    params = dict(
        objective="regression",
        learning_rate=0.05,
        num_leaves=255,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbose=-1,
        n_jobs=8,
    )
    if extra_params:
        params.update(extra_params)
    cats = [c for c in CATEGORICAL_FEATURES if c in features]
    ytr = np.log(train_df[TARGET]) if log_target else train_df[TARGET]
    dtrain = lgb.Dataset(train_df[features], ytr, categorical_feature=cats)
    dvalid = lgb.Dataset(valid_df[features], np.log(valid_df[TARGET]) if log_target else valid_df[TARGET])
    model = lgb.train(
        params, dtrain, num_boost_round=4000, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    pred = model.predict(valid_df[features])
    if log_target:
        pred = np.exp(pred)
    m = evaluate(valid_df[TARGET].values, pred)
    print(f"{name:40s} best_iter={model.best_iteration:5d}  " +
          "  ".join(f"{k}={v:,.4f}" for k, v in m.items()))
    return m


df = build_features(pd.read_csv("data/train_test.csv", parse_dates=["date"]))
df = clean_training(df)
train = df[df.date.dt.month <= 9].copy()
valid = df[df.date.dt.month == 10].copy()
print(f"train={len(train)}  valid={len(valid)}")

run("baseline (all features, log target)", FEATURES)
run("raw target", FEATURES, log_target=False)
run("no route categorical", [f for f in FEATURES if f != "route"])
run("no lat/lon", [f for f in FEATURES if not f.endswith(("_lat", "_lon"))])
run("no dow", [f for f in FEATURES if f != "dow"])
run("l1 objective", FEATURES, extra_params=dict(objective="l1"))
run("tweedie", FEATURES, log_target=False, extra_params=dict(objective="tweedie", tweedie_variance_power=1.2))

# month feature: helps on Oct holdout? (Nov/Dec are unseen at final-train time,
# so this is a leakage-risk probe only)
train_m = train.assign(month=train.date.dt.month.astype("category"))
valid_m = valid.assign(month=valid.date.dt.month.astype("category"))
run("with month (probe only)", FEATURES + ["month"], train_df=train_m, valid_df=valid_m)
