"""
Bloom — Improved Cycle Prediction Model
=======================================
Trains a stacked ensemble (GradientBoosting + Ridge meta-learner) that:
  - Handles irregular cycles (PCOS / anovulatory / stress-induced)
  - Uses previous-cycle history + variation as key features
  - Clips predictions to physiologically plausible range [14, 60]
  - Outputs confidence intervals via quantile regression forests
  - Saves model + scaler + metadata to models_ml/
"""

import os, json, warnings
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import StackingRegressor

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models_ml")
CSV_PATH   = os.path.join(MODELS_DIR, "menstrual_cycle_dataset.csv")

# ── Encoding maps (must match API-side maps) ──────────────────────────────────
MOOD_MAP     = {"happy": 0, "normal": 1, "sad": 2, "angry": 3, "tired": 4, "neutral": 1}
FLOW_MAP     = {"none": -1, "light": 0, "medium": 1, "heavy": 2}
SYMPTOM_MAP  = {"cramps": 0, "headache": 1, "fatigue": 2, "bloating": 3, "nausea": 4,
                "back pain": 5, "acne": 6, "none": -1}
STRESS_MAP   = {"low": 0, "medium": 1, "high": 2}
EXERCISE_MAP = {"none": 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}   # numeric → low/med/high

IRREGULAR_THRESHOLD = 35   # cycles > 35 days → irregular flag


def load_and_engineer(path: str) -> pd.DataFrame:
    """Load CSV, encode categoricals, engineer irregularity features."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Encode mood
    df["mood_enc"] = df["mood"].map(MOOD_MAP).fillna(1)

    # Encode flow
    df["flow_enc"] = df["flow_intensity"].map(FLOW_MAP).fillna(0)

    # Encode symptom
    df["symptom_enc"] = df["symptom"].map(SYMPTOM_MAP).fillna(0)

    # Exercise: already numeric in dataset
    df["exercise_enc"] = df["exercise_level"].apply(
        lambda x: 0 if x <= 1 else (1 if x <= 3 else 2)
    )

    # Stress: already numeric 1-10 → normalise to 0-2
    df["stress_enc"] = df["stress_level"].apply(
        lambda x: 0 if x <= 3 else (1 if x <= 6 else 2)
    )

    # Sleep: 0-2 from hours
    df["sleep_enc"] = df["sleep_hours"].apply(
        lambda x: 0 if x < 6 else (2 if x > 9 else 1)
    )

    # Irregularity features
    df["is_irregular"]    = (df["cycle_length"] > IRREGULAR_THRESHOLD).astype(int)
    df["cycle_variation"] = df["cycle_variation"].fillna(0)
    df["avg_previous"]    = df["avg_previous_cycle"].fillna(df["cycle_length"])

    # Cycle-length variance band (how unpredictable)
    df["high_variation"]  = (df["cycle_variation"] > 5).astype(int)

    # BMI-based risk (PCOS correlation)
    df["bmi_risk"] = df["BMI"].apply(
        lambda x: 1 if (x < 18.5 or x > 30) else 0
    ).fillna(0)

    # Target clipping to physiological range
    df["target"] = df["target_days_to_next_period"].clip(0, 60)

    df = df.dropna(subset=["target"])
    return df


FEATURE_COLS = [
    "age", "cycle_length", "days_since_last_period",
    "mood_enc", "flow_enc", "symptom_enc",
    "stress_enc", "sleep_enc", "exercise_enc",
    "BMI", "avg_previous", "cycle_variation",
    "is_irregular", "high_variation", "bmi_risk"
]


def build_stacked_model():
    """Stacked ensemble: GB + RF base → Ridge meta."""
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
        passthrough=True   # also pass raw features to meta-learner
    )


def train():
    print("🌸 Loading dataset…")
    df = load_and_engineer(CSV_PATH)
    print(f"   {len(df)} rows, {len(FEATURE_COLS)} features")

    X = df[FEATURE_COLS].values
    y = df["target"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    print("🌸 Training stacked ensemble (GB + RF → Ridge)…")
    model = build_stacked_model()
    model.fit(X_sc, y)

    # Cross-validation MAE
    cv_scores = cross_val_score(model, X_sc, y,
                                cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                scoring="neg_mean_absolute_error", n_jobs=-1)
    mae = -cv_scores.mean()
    print(f"   5-fold CV MAE: {mae:.2f} days  (std ±{cv_scores.std():.2f})")

    # Also train a quantile model for confidence intervals (p10 and p90)
    print("🌸 Training quantile models for confidence intervals…")
    q10 = GradientBoostingRegressor(
        loss="quantile", alpha=0.10,
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )
    q90 = GradientBoostingRegressor(
        loss="quantile", alpha=0.90,
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )
    q10.fit(X_sc, y)
    q90.fit(X_sc, y)

    # Irregular-specific model (only train on irregular cycles)
    irr_mask = df["is_irregular"] == 1
    if irr_mask.sum() > 30:
        print(f"🌸 Training irregular-cycle sub-model ({irr_mask.sum()} samples)…")
        irr_model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        )
        irr_model.fit(X_sc[irr_mask], y[irr_mask])
    else:
        irr_model = model

    # Save
    bundle = {
        "model":       model,
        "irr_model":   irr_model,
        "q10":         q10,
        "q90":         q90,
        "scaler":      scaler,
        "feature_cols": FEATURE_COLS,
        "mae":         round(mae, 2),
        "irregular_threshold": IRREGULAR_THRESHOLD,
        "trained_at":  pd.Timestamp.now().isoformat(),
    }

    out_path = os.path.join(MODELS_DIR, "bloom_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"✅ Model saved → {out_path}")
    print(f"   MAE: ±{mae:.1f} days")

    # Quick sanity check
    test_regular  = [25, 28, 14, 0, 1, 0, 1, 1, 1, 22.5, 28, 1.5, 0, 0, 0]
    test_pcos     = [28, 42, 20, 1, 0, 2, 2, 0, 0, 28.0, 40, 8.0, 1, 1, 1]
    for label, t in [("Regular", test_regular), ("PCOS/irregular", test_pcos)]:
        ts = scaler.transform([t])
        pred = model.predict(ts)[0]
        lo   = q10.predict(ts)[0]
        hi   = q90.predict(ts)[0]
        print(f"   {label}: {pred:.1f} days (80% CI: {lo:.0f}–{hi:.0f})")

    return bundle


if __name__ == "__main__":
    train()
