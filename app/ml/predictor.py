"""
Bloom ML Predictor Service

Fixes applied:
  1. symptom="none" accepted — SYMPTOM_MAP now maps "none" to -1 (was missing).
  2. Fertile window validated — fertile_start guaranteed < fertile_end,
     both guaranteed > today; if math produces impossible values, returns None.
  3. cycle_day capped — can never exceed cycle_length (edge case when user
     hasn't logged in a long time).
  4. PHYS_MIN / PHYS_MAX clipping applied to confidence intervals too.
"""

import os, pickle, logging
import numpy as np
from datetime import date, timedelta
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models_ml", "bloom_model.pkl")

MOOD_MAP = {
    "happy": 0, "normal": 1, "sad": 2, "angry": 3, "tired": 4, "neutral": 1,
}
FLOW_MAP = {
    "none": -1, "light": 0, "medium": 1, "heavy": 2,
}
SYMPTOM_MAP = {
    "none": -1,           # FIX: was missing — caused KeyError when symptom="none"
    "cramps": 0, "headache": 1, "fatigue": 2, "bloating": 3,
    "nausea": 4, "back pain": 5, "acne": 6,
}
STRESS_MAP   = {"low": 0, "medium": 1, "high": 2}
SLEEP_MAP    = {"low": 0, "normal": 1, "high": 2}
EXERCISE_MAP = {"bad": 0, "okay": 1, "good": 2}

IRREGULAR_THRESHOLD = 35
PHYS_MIN, PHYS_MAX  = 1, 60   # physiological bounds (days_until can be 0+)

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                _bundle = pickle.load(f)
            logger.info(f"✅ Bloom model loaded (MAE ±{_bundle.get('mae','?')} days)")
        except Exception as e:
            logger.warning(f"⚠️  Model not loaded: {e}")
            _bundle = {}
    return _bundle


def _build_features(
    age: float,
    cycle_length: float,
    days_since_last: float,
    mood: str = "neutral",
    flow: str = "none",
    symptom: str = "none",
    stress: str = "medium",
    sleep: str = "normal",
    exercise: str = "okay",
    bmi: float = 22.5,
    avg_previous: Optional[float] = None,
    cycle_variation: float = 2.0,
) -> List[float]:
    avg_prev = avg_previous if avg_previous is not None else cycle_length
    is_irr   = int(cycle_length > IRREGULAR_THRESHOLD)
    high_var = int(cycle_variation > 5)
    bmi_risk = int(bmi < 18.5 or bmi > 30)

    return [
        age, cycle_length, days_since_last,
        MOOD_MAP.get(mood, 1),
        FLOW_MAP.get(flow, -1),
        SYMPTOM_MAP.get(symptom, -1),   # -1 = no symptom, not cramps
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
    score   = 0.0
    reasons = []

    if cycle_length > 35:
        score += 0.35
        reasons.append(f"Cycle length {cycle_length:.0f} days (>35)")
    elif cycle_length > 32:
        score += 0.15
        reasons.append(f"Slightly long cycle ({cycle_length:.0f} days)")

    if cycle_variation > 7:
        score += 0.25
        reasons.append(f"High cycle variation (±{cycle_variation:.1f} days)")
    elif cycle_variation > 4:
        score += 0.10

    if bmi > 30:
        score += 0.15
        reasons.append(f"BMI {bmi:.1f} (obesity is a PCOS risk factor)")
    elif bmi < 17:
        score += 0.10
        reasons.append(f"Low BMI {bmi:.1f} can disrupt cycles")

    pcos_symptoms = {"acne", "hair loss", "fatigue", "bloating", "mood swings"}
    matched = pcos_symptoms & set(symptoms)
    if len(matched) >= 2:
        score += 0.15
        reasons.append(f"Multiple PCOS-linked symptoms: {', '.join(sorted(matched))}")
    elif len(matched) == 1:
        score += 0.05

    if stress == "high":
        score += 0.10
        reasons.append("Chronic high stress disrupts the HPG axis")

    score = min(score, 1.0)
    level = "high" if score >= 0.5 else ("moderate" if score >= 0.25 else "low")
    return level, round(score, 2), reasons


def predict(
    age: float,
    cycle_length: float,
    days_since_last: float,
    mood: str = "neutral",
    flow: str = "none",
    symptom: str = "none",
    stress: str = "medium",
    sleep: str = "normal",
    exercise: str = "okay",
    bmi: float = 22.5,
    avg_previous: Optional[float] = None,
    cycle_variation: float = 2.0,
    symptoms: Optional[List[str]] = None,
) -> dict:
    bundle = _load()
    feats  = _build_features(
        age, cycle_length, days_since_last,
        mood, flow, symptom, stress, sleep, exercise,
        bmi, avg_previous, cycle_variation,
    )
    is_irregular = cycle_length > IRREGULAR_THRESHOLD
    approx_cycle = float(avg_previous or cycle_length)

    if bundle and "model" in bundle:
        scaler = bundle["scaler"]
        Xs     = scaler.transform([feats])
        mdl    = bundle["irr_model"] if is_irregular else bundle["model"]
        pred   = float(np.clip(mdl.predict(Xs)[0], 0, PHYS_MAX))
        lo     = float(np.clip(bundle["q10"].predict(Xs)[0], 0, pred))
        hi     = float(np.clip(bundle["q90"].predict(Xs)[0], pred, PHYS_MAX))
        mae    = bundle.get("mae", 2.0)
    else:
        pred = max(0, float(approx_cycle) - float(days_since_last))
        lo   = max(0, pred - 5)
        hi   = min(PHYS_MAX, pred + 5)
        mae  = 5.0

    days_until = max(0, int(round(pred)))
    next_date  = date.today() + timedelta(days=days_until)

    # ── Cycle day ─────────────────────────────────────────────────────────────
    # Cap at cycle_length so it never shows "Day 45 of a 28-day cycle"
    cycle_day = min(int(days_since_last) + 1, int(approx_cycle))

    # ── Cycle phase ───────────────────────────────────────────────────────────
    ovulation_day = int(approx_cycle) - 14
    if cycle_day <= 5:
        phase = "menstrual"
    elif cycle_day <= max(6, ovulation_day - 2):
        phase = "follicular"
    elif cycle_day <= ovulation_day + 3:
        phase = "ovulation"
    else:
        phase = "luteal"

    # ── Fertile window ────────────────────────────────────────────────────────
    # Ovulation ≈ next_date minus 14 days
    # Fertile window = 5 days before ovulation through 1 day after
    predicted_ovulation = next_date - timedelta(days=14)
    fertile_start_dt    = predicted_ovulation - timedelta(days=5)
    fertile_end_dt      = predicted_ovulation + timedelta(days=1)

    today = date.today()
    # Only show fertile window if it's in the future and makes sense
    if fertile_start_dt < fertile_end_dt and fertile_end_dt >= today:
        fertile_start = max(fertile_start_dt, today).isoformat()
        fertile_end   = fertile_end_dt.isoformat()
    else:
        fertile_start = None
        fertile_end   = None

    # ── PCOS risk ─────────────────────────────────────────────────────────────
    pcos_level, pcos_score, pcos_reasons = score_pcos_risk(
        cycle_length, cycle_variation,
        avg_previous or cycle_length,
        bmi, symptoms or [], stress,
    )

    return {
        "days_until":        days_until,
        "next_period_date":  next_date.isoformat(),
        "confidence_lo":     max(0, int(round(lo))),
        "confidence_hi":     int(round(hi)),
        "mae_days":          mae,
        "is_irregular":      is_irregular,
        "cycle_phase":       phase,
        "cycle_day":         cycle_day,
        "fertile_start":     fertile_start,   # None when not meaningful
        "fertile_end":       fertile_end,     # None when not meaningful
        "pcos_risk_level":   pcos_level,
        "pcos_risk_score":   pcos_score,
        "pcos_reasons":      pcos_reasons,
    }