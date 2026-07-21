"""
Bloom ML — single source of truth for the cycle-prediction feature contract.

Why this file exists
---------------------
The previous version of this model had two separate, hand-maintained copies
of the encoding maps and feature order: one in train_model.py, one in
predictor.py. They could (and did) drift apart. Worse, the training data
included the CURRENT cycle's own completed length as an input feature —
a value that, by definition, isn't known until after the period has already
started. Predicting "days until next period" from a feature that already
encodes almost the entire answer is why the old model reported an MAE of
~0.1 days: it wasn't predicting anything, it was doing arithmetic on its own
answer key. See ML_NOTES.md for the full write-up with numbers.

The fix has two parts:
  1. Remove the leaking feature. Only backward-looking / same-day
     information is allowed in — never the current cycle's own outcome.
  2. Make that feature contract impossible to duplicate-and-drift: both
     train_model.py and predictor.py import everything from here.
"""

from typing import List, Tuple

MOOD_MAP     = {"happy": 0, "normal": 1, "sad": 2, "angry": 3, "tired": 4, "neutral": 1}
FLOW_MAP     = {"none": -1, "light": 0, "medium": 1, "heavy": 2}
SYMPTOM_MAP  = {"cramps": 0, "headache": 1, "fatigue": 2, "bloating": 3,
                "nausea": 4, "back pain": 5, "acne": 6, "none": -1}
STRESS_MAP   = {"low": 0, "medium": 1, "high": 2}
SLEEP_MAP    = {"low": 0, "normal": 1, "high": 2}
EXERCISE_MAP = {"bad": 0, "okay": 1, "good": 2}

IRREGULAR_THRESHOLD = 35        # avg cycle length (days) beyond which we call it "irregular"
PHYS_MIN, PHYS_MAX  = 14, 60    # physiologically plausible CYCLE-LENGTH bounds

# Bounds for the actual model OUTPUT ("days until next period"), which is a
# different quantity from cycle length. A check-in the day before a period
# starts has days_until == 0/1 — that's normal, not out of range. Clipping
# this output to PHYS_MIN (14) would force every near-term prediction to
# say "at least 14 days," which is wrong for roughly half of real check-ins
# (anyone logging more than ~halfway through their cycle). See ML_NOTES.md.
DAYS_UNTIL_MIN, DAYS_UNTIL_MAX = 0, 60

# The ONE feature order both training and serving use. Change it here only.
FEATURE_COLS = [
    "age", "days_since_last_period",
    "mood_enc", "flow_enc", "symptom_enc",
    "stress_enc", "sleep_enc", "exercise_enc",
    "BMI", "avg_previous", "cycle_variation",
    "is_irregular", "high_variation", "bmi_risk",
]


def encode_categoricals(
    mood: str, flow: str, symptom: str, stress: str, sleep: str, exercise: str
) -> Tuple[int, int, int, int, int, int]:
    return (
        MOOD_MAP.get(mood, 1),
        FLOW_MAP.get(flow, 0),
        SYMPTOM_MAP.get(symptom, 0),
        STRESS_MAP.get(stress, 1),
        SLEEP_MAP.get(sleep, 1),
        EXERCISE_MAP.get(exercise, 1),
    )


def build_feature_vector(
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
) -> List[float]:
    """Build the feature vector in the exact order FEATURE_COLS defines.

    Deliberately takes NO "current cycle length" argument. That value is
    only knowable once the cycle is over, so it can never be a legitimate
    input — only historical averages (avg_previous) and same-day/backward-
    looking information belong here.
    """
    mood_enc, flow_enc, symptom_enc, stress_enc, sleep_enc, exercise_enc = \
        encode_categoricals(mood, flow, symptom, stress, sleep, exercise)

    is_irregular   = int(avg_previous > IRREGULAR_THRESHOLD)
    high_variation = int(cycle_variation > 5)
    bmi_risk       = int(bmi < 18.5 or bmi > 30)

    return [
        age, days_since_last,
        mood_enc, flow_enc, symptom_enc,
        stress_enc, sleep_enc, exercise_enc,
        bmi, avg_previous, cycle_variation,
        is_irregular, high_variation, bmi_risk,
    ]
