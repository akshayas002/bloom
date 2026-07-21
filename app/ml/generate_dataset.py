"""
Bloom — synthetic training data generator.
============================================

We don't have access to a real clinical menstrual-cycle dataset, so this
script builds a synthetic one. It exists so the training pipeline is
honest, reproducible, and inspectable — every assumption below is a
deliberate, documented modelling choice, not a black box.

Modelling approach
------------------
- Each simulated *user* has a persistent "archetype" that sets their true
  underlying cycle-length distribution:
    * regular         (65%): mean ~N(28, 2) days, low cycle-to-cycle noise
    * irregular_pcos  (20%): mean ~N(40, 5) days, high noise, skews to
                              higher BMI and more PCOS-associated symptoms
                              (acne, fatigue, bloating) — mirrors the
                              correlations already assumed in
                              predictor.score_pcos_risk()
    * stress_variable (15%): mean ~N(30, 3) days, variability driven by
                              simulated stress spikes rather than a fixed
                              per-user noise level
- Cycle-to-cycle length is generated with a light AR(1) term (each cycle's
  deviation from the user's baseline is correlated with the previous
  cycle's deviation) since real cycles drift gradually rather than
  resetting randomly every month.
- For each completed cycle (after the user's first), we simulate a random
  "check-in day" partway through that cycle and compute features using
  ONLY information that would have existed as of that day:
    * avg_previous / cycle_variation are rolling stats over the last up to
      6 PRIOR completed cycles — computed with the exact same formula
      cycle_service._previous_cycle_lengths() uses in production (mean,
      and max-min for variation), so training and serving see the same
      quantity under the same name.
    * days_since_last_period is the elapsed days as of the check-in.
    * mood / flow / symptom / stress / sleep / exercise are sampled for
      that specific day, weighted toward what's realistic for the
      inferred cycle phase (e.g. period-flow + cramps near day 1-5,
      mood dips / bloating in the luteal window).
  The cycle's OWN completed length is never a feature — only the label.
- Label: target_days_to_next_period = (full length of the current cycle)
  minus (the check-in day). This is the one place the "future" appears,
  and it's supposed to — it's what we're trying to predict, not a feature.

This generator is deliberately simple and documented rather than
maximally realistic. Swap in a real dataset by pointing CSV_PATH at it —
train_model.py doesn't care how the CSV was produced, only that it has
these columns.
"""

import os
import numpy as np
import pandas as pd

MOODS     = ["happy", "sad", "angry", "tired", "neutral"]
SYMPTOMS  = ["cramps", "headache", "fatigue", "bloating", "nausea", "back pain", "acne"]
PCOS_SYMPTOMS = ["acne", "fatigue", "bloating"]

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models_ml")
OUT_PATH   = os.path.join(MODELS_DIR, "menstrual_cycle_dataset.csv")

N_USERS       = 600
MIN_CYCLES    = 8
MAX_CYCLES    = 14
HISTORY_WINDOW = 6   # mirrors cycle_service._previous_cycle_lengths(n=6)

rng = np.random.default_rng(42)


def _make_user(user_id: int) -> dict:
    roll = rng.random()
    if roll < 0.65:
        archetype = "regular"
        mean_len  = np.clip(rng.normal(28, 2), 24, 33)
        noise_sd  = rng.uniform(1, 3)
        bmi       = np.clip(rng.normal(22, 3), 17, 32)
    elif roll < 0.85:
        archetype = "irregular_pcos"
        mean_len  = np.clip(rng.normal(40, 5), 33, 55)
        noise_sd  = rng.uniform(5, 12)
        bmi       = np.clip(rng.normal(27, 5), 18, 40)
    else:
        archetype = "stress_variable"
        mean_len  = np.clip(rng.normal(30, 3), 25, 38)
        noise_sd  = rng.uniform(3, 7)
        bmi       = np.clip(rng.normal(23, 4), 17, 34)

    age = np.clip(rng.normal(29, 6), 18, 45)
    return dict(user_id=user_id, archetype=archetype, mean_len=mean_len,
                noise_sd=noise_sd, bmi=bmi, age=age)


def _simulate_cycle_lengths(user: dict) -> list:
    """AR(1)-correlated cycle lengths around the user's baseline."""
    n = int(rng.integers(MIN_CYCLES, MAX_CYCLES + 1))
    lengths, prev_dev = [], 0.0
    for _ in range(n):
        dev = 0.4 * prev_dev + rng.normal(0, user["noise_sd"])
        length = int(round(np.clip(user["mean_len"] + dev, 14, 65)))
        lengths.append(length)
        prev_dev = dev
    return lengths


def _phase_for_day(day: int, cycle_len: int) -> str:
    ov = cycle_len - 14
    if day <= 5:
        return "menstrual"
    if day <= ov - 2:
        return "follicular"
    if day <= ov + 3:
        return "ovulation"
    return "luteal"


def _sample_day_log(phase: str, archetype: str, rng) -> dict:
    """Sample mood/flow/symptom/stress/sleep/exercise for one check-in day."""
    if phase == "menstrual":
        flow    = rng.choice(["light", "medium", "heavy"], p=[0.35, 0.4, 0.25])
        symptom = rng.choice(["cramps", "headache", "back pain", "fatigue"],
                              p=[0.5, 0.2, 0.2, 0.1])
        mood    = rng.choice(["tired", "sad", "neutral", "happy"], p=[0.35, 0.25, 0.3, 0.1])
    elif phase == "luteal":
        flow = "none"
        pool = PCOS_SYMPTOMS if archetype == "irregular_pcos" else SYMPTOMS
        symptom = rng.choice(pool)
        mood = rng.choice(["sad", "angry", "tired", "neutral", "happy"],
                           p=[0.25, 0.2, 0.25, 0.2, 0.1])
    else:  # follicular / ovulation
        flow = "none"
        symptom = rng.choice(["none", "cramps", "acne", "headache"], p=[0.55, 0.15, 0.15, 0.15])
        mood = rng.choice(["happy", "neutral", "tired", "sad"], p=[0.4, 0.35, 0.15, 0.1])

    stress_base = {"regular": 0.3, "stress_variable": 0.55, "irregular_pcos": 0.45}[archetype]
    stress = rng.choice(["low", "medium", "high"],
                         p=[1 - stress_base - 0.15, 0.35, stress_base - 0.2] if stress_base > 0.35
                           else [0.45, 0.4, 0.15])
    sleep    = rng.choice(["low", "normal", "high"], p=[0.25, 0.6, 0.15])
    exercise = rng.choice(["bad", "okay", "good"], p=[0.3, 0.45, 0.25])

    return dict(mood=mood, flow_intensity=flow, symptom=symptom,
                stress_level=stress, sleep=sleep, exercise=exercise)


def generate() -> pd.DataFrame:
    rows = []
    for uid in range(N_USERS):
        user    = _make_user(uid)
        lengths = _simulate_cycle_lengths(user)

        for i in range(1, len(lengths)):          # need >=1 prior cycle for history
            history = lengths[max(0, i - HISTORY_WINDOW):i]   # prior cycles only
            avg_previous    = float(np.mean(history))
            cycle_variation = float(max(history) - min(history)) if len(history) > 1 else 2.0

            cur_len = lengths[i]
            checkin_day = int(rng.integers(1, cur_len))        # 1 .. cur_len-1
            phase = _phase_for_day(checkin_day, cur_len)
            day_log = _sample_day_log(phase, user["archetype"], rng)

            rows.append({
                "user_id":                 uid,
                "age":                     round(user["age"], 1),
                "BMI":                     round(user["bmi"], 1),
                "days_since_last_period":  checkin_day,
                "avg_previous_cycle":      round(avg_previous, 2),
                "cycle_variation":         round(cycle_variation, 2),
                "mood":                    day_log["mood"],
                "flow_intensity":          day_log["flow_intensity"],
                "symptom":                 day_log["symptom"],
                "stress_level":            day_log["stress_level"],
                "sleep":                   day_log["sleep"],
                "exercise":                day_log["exercise"],
                "target_days_to_next_period": cur_len - checkin_day,
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Generated {len(df)} rows from {N_USERS} synthetic users → {OUT_PATH}")
    print(df["target_days_to_next_period"].describe())
