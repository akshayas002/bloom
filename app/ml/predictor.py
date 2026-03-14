"""
Bloom ML Predictor Service
Wraps the trained model bundle, handles irregular cycles,
computes confidence intervals, and derives PCOS risk scores.
"""

import os, pickle, logging
import numpy as np
from datetime import date, timedelta
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models_ml", "bloom_model.pkl")

MOOD_MAP     = {"happy": 0, "normal": 1, "sad": 2, "angry": 3, "tired": 4, "neutral": 1}
FLOW_MAP     = {"none": -1, "light": 0, "medium": 1, "heavy": 2}
SYMPTOM_MAP  = {"cramps": 0, "headache": 1, "fatigue": 2, "bloating": 3,
                "nausea": 4, "back pain": 5, "acne": 6, "none": -1}
STRESS_MAP   = {"low": 0, "medium": 1, "high": 2}
SLEEP_MAP    = {"low": 0, "normal": 1, "high": 2}
EXERCISE_MAP = {"bad": 0, "okay": 1, "good": 2}

IRREGULAR_THRESHOLD = 35
PHYS_MIN, PHYS_MAX  = 14, 60     # physiological bounds

_bundle = None

def _load():
    global _bundle
    if _bundle is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                _bundle = pickle.load(f)
            logger.info(f"✅ Bloom model loaded (MAE: ±{_bundle.get('mae','?')} days)")
        except Exception as e:
            logger.warning(f"⚠️  Model not loaded: {e}")
            _bundle = {}
    return _bundle


def _build_features(
    age: float,
    cycle_length: float,
    days_since_last: float,
    mood: str = "neutral",
    flow: str = "medium",
    symptom: str = "cramps",
    stress: str = "medium",
    sleep: str = "normal",
    exercise: str = "okay",
    bmi: float = 22.5,
    avg_previous: Optional[float] = None,
    cycle_variation: float = 2.0,
) -> List[float]:
    """Build 15-feature vector matching training."""
    avg_prev = avg_previous if avg_previous is not None else cycle_length
    is_irr   = int(cycle_length > IRREGULAR_THRESHOLD)
    high_var  = int(cycle_variation > 5)
    bmi_risk  = int(bmi < 18.5 or bmi > 30)

    return [
        age, cycle_length, days_since_last,
        MOOD_MAP.get(mood, 1),
        FLOW_MAP.get(flow, 0),
        SYMPTOM_MAP.get(symptom, 0),
        STRESS_MAP.get(stress, 1),
        SLEEP_MAP.get(sleep, 1),
        EXERCISE_MAP.get(exercise, 1),
        bmi,
        avg_prev,
        cycle_variation,
        is_irr,
        high_var,
        bmi_risk,
    ]


def score_pcos_risk(
    cycle_length: float,
    cycle_variation: float,
    avg_previous: float,
    bmi: float,
    symptoms: List[str],
    stress: str,
) -> Tuple[str, float, List[str]]:
    """
    Return (risk_level, score_0_to_1, reasons).
    risk_level: 'low' | 'moderate' | 'high'
    """
    score   = 0.0
    reasons = []

    if cycle_length > 35:
        score += 0.35; reasons.append(f"Cycle length {cycle_length:.0f} days (>35)")
    elif cycle_length > 32:
        score += 0.15; reasons.append(f"Slightly long cycle ({cycle_length:.0f} days)")

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
    cycle_length: float,
    days_since_last: float,
    mood: str = "neutral",
    flow: str = "medium",
    symptom: str = "cramps",
    stress: str = "medium",
    sleep: str = "normal",
    exercise: str = "okay",
    bmi: float = 22.5,
    avg_previous: Optional[float] = None,
    cycle_variation: float = 2.0,
    symptoms: Optional[List[str]] = None,
) -> dict:
    """
    Run the ML model and return a rich prediction dict.
    Falls back to heuristic if model is unavailable.
    """
    bundle = _load()
    feats  = _build_features(age, cycle_length, days_since_last,
                              mood, flow, symptom, stress, sleep, exercise,
                              bmi, avg_previous, cycle_variation)
    is_irregular = cycle_length > IRREGULAR_THRESHOLD

    if bundle and "model" in bundle:
        scaler = bundle["scaler"]
        Xs     = scaler.transform([feats])

        # Choose model: irregular-specific or general
        mdl    = bundle["irr_model"] if is_irregular else bundle["model"]
        pred   = float(np.clip(mdl.predict(Xs)[0], PHYS_MIN, PHYS_MAX))
        lo     = float(np.clip(bundle["q10"].predict(Xs)[0], PHYS_MIN, pred))
        hi     = float(np.clip(bundle["q90"].predict(Xs)[0], pred, PHYS_MAX))
        mae    = bundle.get("mae", 2.0)
    else:
        # Heuristic fallback
        pred = max(PHYS_MIN, float(cycle_length) - float(days_since_last))
        lo   = max(PHYS_MIN, pred - 5)
        hi   = min(PHYS_MAX, pred + 5)
        mae  = 5.0

    days_until = int(round(pred))
    next_date  = date.today() + timedelta(days=days_until)

    # Fertile window: ovulation ≈ cycle_length - 14 from last period start
    approx_cycle = avg_previous or cycle_length
    ovulation_day   = int(approx_cycle) - 14
    fertile_start_d = days_until - (int(approx_cycle) - ovulation_day + 5)
    fertile_end_d   = days_until - (int(approx_cycle) - ovulation_day - 1)
    fertile_start   = date.today() + timedelta(days=max(0, fertile_start_d))
    fertile_end     = date.today() + timedelta(days=max(0, fertile_end_d))

    # PCOS risk
    pcos_level, pcos_score, pcos_reasons = score_pcos_risk(
        cycle_length, cycle_variation, avg_previous or cycle_length,
        bmi, symptoms or [], stress
    )

    # Phase inference
    cycle_day = int(days_since_last) + 1
    if cycle_day <= 5:
        phase = "menstrual"
    elif cycle_day <= int(approx_cycle) - 14 - 2:
        phase = "follicular"
    elif cycle_day <= int(approx_cycle) - 14 + 3:
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
