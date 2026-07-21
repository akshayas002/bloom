"""
Bloom — Cycle Prediction Model Training
========================================

v2 — fixes a data-leakage bug from the previous version (see ML_NOTES.md
for the full diagnosis). Summary: the old training data included the
CURRENT cycle's own completed length as a feature, which is only knowable
after the fact. The model had learned to compute
`days_until = cycle_length - days_since_last_period` — pure arithmetic on
its own answer key — which is why it reported an implausible ~0.1-day MAE.

What changed:
  - The leaking feature is gone. Features now come exclusively from
    app.ml.feature_spec, which encodes ONLY information available before
    the current cycle ends (historical averages, same-day logs).
  - Validation is grouped by user_id (GroupKFold + a held-out group split),
    so a user's own cycles can never appear in both train and test. Random
    row-level splitting would let the model "cheat" by seeing other cycles
    from the same person and inferring their personal baseline.
  - The reported MAE now comes from a genuinely held-out set of users the
    model has never seen in any form, not from cross-validation on rows
    that share users with the training set.

Trains a stacked ensemble (GradientBoosting + RandomForest -> Ridge) plus
quantile models for a confidence interval, same modelling approach as
before — only the inputs and the way we validate them have changed.
"""

import os, warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.metrics import mean_absolute_error

from app.ml.feature_spec import (
    MOOD_MAP, FLOW_MAP, SYMPTOM_MAP, STRESS_MAP, SLEEP_MAP, EXERCISE_MAP,
    FEATURE_COLS, IRREGULAR_THRESHOLD,
)

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models_ml")
CSV_PATH   = os.path.join(MODELS_DIR, "menstrual_cycle_dataset.csv")


def load_and_engineer(path: str) -> pd.DataFrame:
    """Load the CSV and encode categoricals via the shared feature_spec maps.

    Expected columns: user_id, age, BMI, days_since_last_period,
    avg_previous_cycle, cycle_variation, mood, flow_intensity, symptom,
    stress_level, sleep, exercise, target_days_to_next_period.

    Deliberately absent: any column describing the CURRENT cycle's own
    completed length. If you swap in a real dataset, do not add one back —
    that's the leak this version fixes.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df["mood_enc"]     = df["mood"].map(MOOD_MAP).fillna(1)
    df["flow_enc"]     = df["flow_intensity"].map(FLOW_MAP).fillna(0)
    df["symptom_enc"]  = df["symptom"].map(SYMPTOM_MAP).fillna(0)
    df["stress_enc"]   = df["stress_level"].map(STRESS_MAP).fillna(1)
    df["sleep_enc"]    = df["sleep"].map(SLEEP_MAP).fillna(1)
    df["exercise_enc"] = df["exercise"].map(EXERCISE_MAP).fillna(1)

    df["avg_previous"] = df["avg_previous_cycle"]
    df["cycle_variation"] = df["cycle_variation"].fillna(2.0)

    # Irregularity / risk flags derived from HISTORICAL average only —
    # never from the current (not-yet-known) cycle length.
    df["is_irregular"]   = (df["avg_previous"] > IRREGULAR_THRESHOLD).astype(int)
    df["high_variation"] = (df["cycle_variation"] > 5).astype(int)
    df["bmi_risk"]        = df["BMI"].apply(lambda x: 1 if (x < 18.5 or x > 30) else 0)

    df["target"] = df["target_days_to_next_period"].clip(0, 60)
    df = df.dropna(subset=["target"] + FEATURE_COLS)
    return df


def build_stacked_model() -> StackingRegressor:
    base_estimators = [
        ("gb", GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            min_samples_leaf=5, random_state=42
        )),
        ("rf", RandomForestRegressor(
            n_estimators=200, max_depth=8,
            min_samples_leaf=4, random_state=42,
            n_jobs=-1
        )),
    ]
    meta = Ridge(alpha=1.0)
    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta,
        cv=5, n_jobs=-1,
        passthrough=True,
    )


def train():
    print("🌸 Loading dataset…")
    df = load_and_engineer(CSV_PATH)
    print(f"   {len(df)} rows, {df['user_id'].nunique()} users, {len(FEATURE_COLS)} features")

    X = df[FEATURE_COLS].values
    y = df["target"].values
    groups = df["user_id"].values

    # ── Held-out USER split — these users are invisible until final eval ──────
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train    = groups[train_idx]

    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("🌸 Training stacked ensemble (GB + RF → Ridge)…")
    model = build_stacked_model()
    model.fit(X_train_sc, y_train)

    # Grouped CV on the training split only — no user ever appears in both
    # the fit fold and the validation fold within a given split.
    cv = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(
        build_stacked_model(), X_train_sc, y_train, groups=groups_train,
        cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    cv_mae = -cv_scores.mean()
    print(f"   Grouped 5-fold CV MAE (train users only): {cv_mae:.2f} days (±{cv_scores.std():.2f})")

    # Genuinely held-out users the model has never touched in any form.
    test_pred = model.predict(X_test_sc)
    holdout_mae = mean_absolute_error(y_test, test_pred)
    print(f"   Held-out user MAE ({len(test_idx)} rows, {df['user_id'].iloc[test_idx].nunique()} unseen users): {holdout_mae:.2f} days")

    # ── Refit on ALL data for the shipped model ────────────────────────────────
    print("🌸 Refitting on full dataset for the production model…")
    scaler_full = StandardScaler()
    X_full_sc = scaler_full.fit_transform(X)
    final_model = build_stacked_model()
    final_model.fit(X_full_sc, y)

    print("🌸 Training quantile models for confidence intervals…")
    q10 = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=200,
                                     learning_rate=0.05, max_depth=3, random_state=42)
    q90 = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=200,
                                     learning_rate=0.05, max_depth=3, random_state=42)
    q10.fit(X_full_sc, y)
    q90.fit(X_full_sc, y)

    irr_mask = df["is_irregular"] == 1
    if irr_mask.sum() > 30:
        print(f"🌸 Training irregular-cycle sub-model ({irr_mask.sum()} samples)…")
        irr_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        )
        irr_model.fit(X_full_sc[irr_mask.values], y[irr_mask.values])
    else:
        irr_model = final_model

    bundle = {
        "model":       final_model,
        "irr_model":   irr_model,
        "q10":         q10,
        "q90":         q90,
        "scaler":      scaler_full,
        "feature_cols": FEATURE_COLS,
        "mae":         round(holdout_mae, 2),          # honest, unseen-user MAE
        "cv_mae":      round(cv_mae, 2),
        "irregular_threshold": IRREGULAR_THRESHOLD,
        "trained_at":  pd.Timestamp.now().isoformat(),
        "trained_on":  "synthetic_v2_no_leakage",
    }

    out_path = os.path.join(MODELS_DIR, "bloom_model.pkl")
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"✅ Model saved → {out_path}")
    print(f"   Reported MAE (held-out users): ±{holdout_mae:.1f} days")
    print(f"   (Grouped CV MAE on training users only: ±{cv_mae:.1f} days)")

    return bundle


if __name__ == "__main__":
    train()
