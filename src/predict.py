"""Generate final predictions.

1. Predicts all 12,000 loads in data/validation.csv and writes
   validation_predictions.csv (load_id,predicted_rate), preserving the
   template's load_id order.
2. Fills the predicted_rate column of data/december_chart_inputs.csv.

The December chart inputs carry no market_index / quote_signal / coordinates,
so they are enriched with:
  - city coordinates from the train+validation lookup (exact, constant per city)
  - market_index / quote_signal estimated per December date as the daily mean
    observed in validation.csv (which covers Dec 2025); fallbacks use the most
    recent 30 training days.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, build_features, city_coordinates

DATA = Path("data")
MODEL_DIR = Path("model")


def load_models() -> tuple[list[lgb.Booster], dict]:
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    models = [lgb.Booster(model_file=str(MODEL_DIR / f"final_seed{s}.txt"))
              for s in meta["seeds"]]
    return models, meta


def predict(models: list[lgb.Booster], frame: pd.DataFrame) -> np.ndarray:
    """Average log-rate predictions across seeds, then exponentiate."""
    log_preds = np.mean([m.predict(frame[FEATURES]) for m in models], axis=0)
    return np.exp(log_preds)


def enrich_december(december: pd.DataFrame, train: pd.DataFrame,
                    validation: pd.DataFrame) -> pd.DataFrame:
    out = december.copy()
    out["date"] = pd.to_datetime(out["date"])

    coords = city_coordinates(train, validation)
    for role in ("pickup", "delivery"):
        out[f"{role}_lat"] = out[role].map(coords["lat"])
        out[f"{role}_lon"] = out[role].map(coords["lon"])

    # Daily signal estimates for each December date from validation.csv,
    # falling back to the last 30 training days.
    val_dec = validation[validation["date"].dt.month == 12]
    recent = train[train["date"] >= train["date"].max() - pd.Timedelta(days=29)]
    for signal in ("market_index", "quote_signal"):
        daily = val_dec.groupby(val_dec["date"].dt.normalize())[signal].mean()
        fallback = recent[signal].mean()
        out[signal] = out["date"].dt.normalize().map(daily).fillna(fallback)
    return out


def main() -> None:
    models, meta = load_models()
    n_iter = meta["num_iterations"]
    del n_iter  # boosters saved after refit are used in full

    train = build_features(pd.read_csv(DATA / "train_test.csv", parse_dates=["date"]))
    validation = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])

    # --- validation predictions -------------------------------------------------
    val_features = build_features(validation)
    val_pred = predict(models, val_features)
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    rates = pd.Series(val_pred, index=validation["load_id"])
    template["predicted_rate"] = template["load_id"].map(rates).round(2)
    assert template["predicted_rate"].notna().all() and (template["predicted_rate"] > 0).all()
    template.to_csv("validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv ({len(template):,} rows, "
          f"mean=${template['predicted_rate'].mean():,.2f})")

    # --- December chart inputs ---------------------------------------------------
    december = pd.read_csv(DATA / "december_chart_inputs.csv")
    enriched = enrich_december(december, train, validation.assign(
        date=pd.to_datetime(validation["date"])))
    dec_pred = predict(models, build_features(enriched))
    december["predicted_rate"] = np.round(dec_pred, 2)
    december.to_csv(DATA / "december_chart_inputs.csv", index=False)
    print(f"Filled {len(december)} December rows "
          f"(range ${dec_pred.min():,.2f} - ${dec_pred.max():,.2f})")


if __name__ == "__main__":
    main()
