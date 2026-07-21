"""
Bloom ML Predictor Service
Wraps the trained model bundle, handles irregular cycles,
computes confidence intervals, and derives PCOS risk scores.

v2: no longer accepts a "current cycle length" argument. The old signature
took `cycle_length` and used it both as a feature and to decide irregularity
— but the *current* cycle's length isn't known until it's over, so it can
never legitimately be an input to a prediction made mid-cycle. Every place
that used to read `cycle_length` now reads `avg_previous` (the user's
rolling historical average, computed by cycle_service from their own past
logs). See ML_NOTES.md for why this matters and app.ml.feature_spec for the
shared feature contract this file and train_model.py both use.
"""

import os, pickle, logging
import numpy as np
from datetime import date, timedelta
from typing import Optional, Tuple, List

from app.ml.feature_spec import (
    build_feature_vector, IRREGULAR_THRESHOLD, DAYS_UNTIL_MIN, DAYS_UNTIL_MAX,
)

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models_ml", "bloom_model.pkl")

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                _bundle = pickle.load(f)
            logger.info(f"✅ Bloom model loaded (held-out MAE: ±{_bundle.get('mae','?')} days)")
        except Exception as e:
            logger.warning(f"⚠️  Model not loaded: {e}")
            _bundle = {}
    return _bundle


def score_pcos_risk(
    avg_cycle_length: float,
    cycle_variation: float,
    bmi: float,
    symptoms: List[str],
    stress: str,
) -> Tuple[str, float, List[str]]:
    """
    Return (risk_level, score_0_to_1, reasons).
    risk_level: 'low' | 'moderate' | 'high'

    Uses the user's rolling historical average cycle length, not the
    in-progress current cycle (which isn't finished yet).
    """
    score   = 0.0
    reasons = []

    if avg_cycle_length > 35:
        score += 0.35; reasons.append(f"Average cycle length {avg_cycle_length:.0f} days (>35)")
    elif avg_cycle_length > 32:
        score += 0.15; reasons.append(f"Slightly long average cycle ({avg_cycle_length:.0f} days)")

    if cycle_variation > 7:
        score += 0.25; reasons.append(f"High cycle variation (±{cycle_variation:.1f} days)")
    elif cycle_variation > 4:
        score += 0.10

    if bmi > 30:
        score += 0.15; reasons.append(f"BMI {bmi:.1f} (obesity is a PCOS risk factor)")
    elif bmi < 17:
        score += 0.10; reasons.append(f"Low BMI {bmi:.1f} can disrupt cycles")

    pcos_symptoms = {"acne", "hair loss", "fatigue", "bloating", "mood swings"}
    matched = pcos_symptoms & set(symptoms)
    if len(matched) >= 2:
        score += 0.15; reasons.append(f"Multiple PCOS-linked symptoms: {', '.join(matched)}")
    elif len(matched) == 1:
        score += 0.05

    if stress == "high":
        score += 0.10; reasons.append("Chronic high stress disrupts the HPG axis")

    score = min(score, 1.0)
    level = "high" if score >= 0.5 else ("moderate" if score >= 0.25 else "low")
    return level, round(score, 2), reasons


def predict(
    age: float,
    days_since_last: float,
    mood: str = "neutral",
    flow: str = "medium",
    symptom: str = "cramps",
    stress: str = "medium",
    sleep: str = "normal",
    exercise: str = "okay",
    bmi: float = 22.5,
    avg_previous: float = 28.0,
    cycle_variation: float = 2.0,
    symptoms: Optional[List[str]] = None,
) -> dict:
    """
    Run the ML model and return a rich prediction dict.
    Falls back to heuristic if model is unavailable.

    `avg_previous` — the user's rolling average cycle length computed from
    their own logged history — stands in for "expected cycle length"
    everywhere. There is no `cycle_length` argument: the current cycle's
    real length isn't known until it ends, so it can't be an input.
    """
    bundle = _load()
    feats  = build_feature_vector(age, days_since_last, mood, flow, symptom,
                                   stress, sleep, exercise, bmi,
                                   avg_previous, cycle_variation)
    is_irregular = avg_previous > IRREGULAR_THRESHOLD

    if bundle and "model" in bundle:
        scaler = bundle["scaler"]
        Xs     = scaler.transform([feats])

        mdl    = bundle["irr_model"] if is_irregular else bundle["model"]
        pred   = float(np.clip(mdl.predict(Xs)[0], DAYS_UNTIL_MIN, DAYS_UNTIL_MAX))
        lo     = float(np.clip(bundle["q10"].predict(Xs)[0], DAYS_UNTIL_MIN, pred))
        hi     = float(np.clip(bundle["q90"].predict(Xs)[0], pred, DAYS_UNTIL_MAX))
        mae    = bundle.get("mae", 3.0)
    else:
        # Heuristic fallback
        pred = max(DAYS_UNTIL_MIN, float(avg_previous) - float(days_since_last))
        lo   = max(DAYS_UNTIL_MIN, pred - 5)
        hi   = min(DAYS_UNTIL_MAX, pred + 5)
        mae  = 5.0

    days_until = int(round(pred))
    next_date  = date.today() + timedelta(days=days_until)

    # Fertile window: ovulation ≈ avg_previous - 14 from last period start
    ovulation_day   = int(avg_previous) - 14
    fertile_start_d = days_until - (int(avg_previous) - ovulation_day + 5)
    fertile_end_d   = days_until - (int(avg_previous) - ovulation_day - 1)
    fertile_start   = date.today() + timedelta(days=max(0, fertile_start_d))
    fertile_end     = date.today() + timedelta(days=max(0, fertile_end_d))

    # PCOS risk — based on the historical average, not the in-progress cycle
    pcos_level, pcos_score, pcos_reasons = score_pcos_risk(
        avg_previous, cycle_variation, bmi, symptoms or [], stress
    )

    # Phase inference
    cycle_day = int(days_since_last) + 1
    if cycle_day <= 5:
        phase = "menstrual"
    elif cycle_day <= int(avg_previous) - 14 - 2:
        phase = "follicular"
    elif cycle_day <= int(avg_previous) - 14 + 3:
        phase = "ovulation"
    else:
        phase = "luteal"

    return {
        "days_until":       days_until,
        "next_period_date": next_date.isoformat(),
        "confidence_lo":    int(round(lo)),
        "confidence_hi":    int(round(hi)),
        "mae_days":         mae,
        "is_irregular":     is_irregular,
        "cycle_phase":      phase,
        "cycle_day":        cycle_day,
        "fertile_start":    fertile_start.isoformat(),
        "fertile_end":      fertile_end.isoformat(),
        "pcos_risk_level":  pcos_level,
        "pcos_risk_score":  pcos_score,
        "pcos_reasons":     pcos_reasons,
    }
