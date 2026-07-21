"""
Cycle analytics service.

Fixes applied:
  1. _last_period_log()         → returns cycle START (first day of bleeding run),
                                   not the most recent bleeding day.
  2. _previous_cycle_lengths()  → uses cycle starts only (same fix), so gaps
                                   are true cycle lengths (~28 days, not ~25).
  3. build_chart_data()         → proper calendar-month arithmetic instead of
                                   approximate timedelta(days=i*30).
  4. get_prediction()           → explicit "insufficient data" guard for new
                                   users; returns has_data=False flag to UI.
  5. get_prediction()           → default symptom is None/"none", not "cramps".
  6. build_user_context()       → aggregates symptoms/mood from last 30 days
                                   using actual logs, not the most recent single log.
"""

from calendar import monthrange
from datetime import date, timedelta
from typing import List, Dict, Optional
from collections import Counter, defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.db_models import CycleLog, User
from app.ml.predictor import predict as ml_predict


PERIOD_FLOWS = ("light", "medium", "heavy")

# Minimum days gap to treat two bleeding logs as the SAME period
# (anything less = same bleed, not a new cycle)
SAME_PERIOD_GAP_DAYS = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _month_label(d: date) -> str:
    return d.strftime("%b %Y")


def _subtract_months(d: date, months: int) -> date:
    """Subtract `months` from date using proper calendar arithmetic."""
    month = d.month - months
    year  = d.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    day   = min(d.day, monthrange(year, month)[1])
    return d.replace(year=year, month=month, day=day)


def _detect_cycle_starts(bleeding_dates: List[date]) -> List[date]:
    """
    Given a sorted list of bleeding log dates, return only the FIRST day
    of each distinct bleeding run (= cycle start dates).

    e.g. [Jan 5, Jan 6, Jan 7, Feb 3, Feb 4] → [Jan 5, Feb 3]
    """
    if not bleeding_dates:
        return []
    sorted_dates = sorted(bleeding_dates)
    starts = [sorted_dates[0]]
    for prev, curr in zip(sorted_dates, sorted_dates[1:]):
        if (curr - prev).days > SAME_PERIOD_GAP_DAYS:
            starts.append(curr)
    return starts


async def _get_cycle_starts(db: AsyncSession, user_id: int) -> List[date]:
    """Fetch all bleeding log dates and return only cycle start dates."""
    result = await db.execute(
        select(CycleLog.log_date)
        .where(and_(
            CycleLog.user_id == user_id,
            CycleLog.flow_intensity.in_(PERIOD_FLOWS),
        ))
        .order_by(CycleLog.log_date)
    )
    all_bleeding_dates = [r[0] for r in result.fetchall()]
    return _detect_cycle_starts(all_bleeding_dates)


async def _last_period_start(db: AsyncSession, user_id: int) -> Optional[date]:
    """Return the most recent cycle START date (first day of last period)."""
    starts = await _get_cycle_starts(db, user_id)
    return starts[-1] if starts else None


async def _cycle_lengths_from_starts(db: AsyncSession, user_id: int) -> List[int]:
    """
    Return actual cycle lengths in days (gap between consecutive cycle starts).
    Requires at least 2 cycle start dates.
    """
    starts = await _get_cycle_starts(db, user_id)
    if len(starts) < 2:
        return []
    return [(starts[i + 1] - starts[i]).days for i in range(len(starts) - 1)]


# ── Prediction ────────────────────────────────────────────────────────────────

async def get_prediction(db: AsyncSession, user: User) -> dict:
    """
    Build a prediction dict.

    Returns has_data=False when there is insufficient logged history
    so the dashboard can show a gentle prompt instead of a fake prediction.
    """
    last_start   = await _last_period_start(db, user.id)
    cycle_lengths = await _cycle_lengths_from_starts(db, user.id)

    # ── Determine if we have enough data ─────────────────────────────────────
    has_data = last_start is not None

    if last_start:
        days_since = (date.today() - last_start).days
    else:
        # No period logged yet — use half of avg_cycle as a neutral assumption
        # but flag it so the UI knows this is just a seed value
        days_since = int((user.avg_cycle or 28) // 2)

    if cycle_lengths:
        avg_prev  = sum(cycle_lengths) / len(cycle_lengths)
        cycle_var = float(max(cycle_lengths) - min(cycle_lengths)) if len(cycle_lengths) > 1 else 2.0
    else:
        avg_prev  = float(user.avg_cycle or 28)
        cycle_var = 2.0

    # ── Pull most recent log for lifestyle features ───────────────────────────
    recent_result = await db.execute(
        select(CycleLog)
        .where(CycleLog.user_id == user.id)
        .order_by(CycleLog.log_date.desc())
        .limit(1)
    )
    recent   = recent_result.scalars().first()
    mood     = recent.mood           if recent else "neutral"
    flow     = recent.flow_intensity if recent else "none"
    stress   = recent.stress         if recent else "medium"
    sleep    = recent.sleep          if recent else "normal"
    exercise = recent.exercise       if recent else "okay"
    symptoms = recent.symptoms       if recent else []

    # FIX: do NOT default to "cramps" — pass "none" when there are no symptoms
    # so the ML model isn't incorrectly told the user has cramps.
    symptom = symptoms[0] if symptoms else "none"

    result = ml_predict(
        age             = float(user.age or 25),
        days_since_last = float(days_since),
        mood=mood, flow=flow, symptom=symptom,
        stress=stress, sleep=sleep, exercise=exercise,
        bmi             = float(user.bmi or 22.5),
        avg_previous    = float(avg_prev),
        cycle_variation = float(cycle_var),
        symptoms        = symptoms,
    )

    # Attach metadata the dashboard needs
    result["has_data"]          = has_data
    result["cycle_count"]       = len(cycle_lengths)       # how many full cycles observed
    result["avg_cycle_observed"] = round(avg_prev, 1)
    return result


# ── Calendar events ───────────────────────────────────────────────────────────

async def build_calendar_events(db: AsyncSession, user: User, prediction: dict) -> List[dict]:
    result = await db.execute(
        select(CycleLog)
        .where(CycleLog.user_id == user.id)
        .order_by(CycleLog.log_date)
    )
    logs   = result.scalars().all()
    events = []

    intensity_colors = {"light": "#FFB3AD", "medium": "#FF6F61", "heavy": "#CC3B30"}
    mood_emojis = {"happy": "😊", "sad": "😢", "angry": "😠", "tired": "😴", "neutral": "😐"}

    for log in logs:
        ds   = log.log_date.isoformat()
        syms = log.symptoms  # always a list (SymptomList TypeDecorator)

        if log.flow_intensity in PERIOD_FLOWS:
            events.append({
                "title":  f"🩸 Period ({log.flow_intensity})",
                "start":  ds,
                "color":  intensity_colors.get(log.flow_intensity, "#FF6F61"),
                "extendedProps": {"type": "period", "cycle_day": log.cycle_day},
            })

        if syms:
            events.append({
                "title":  "🔵 " + ", ".join(syms[:2]),
                "start":  ds,
                "color":  "#87CEEB",
                "extendedProps": {"type": "symptoms", "all_symptoms": syms},
            })

        if log.mood:
            events.append({
                "title":  f"{mood_emojis.get(log.mood, '')} {log.mood.capitalize()}",
                "start":  ds,
                "color":  "#C08081",
                "extendedProps": {"type": "mood"},
            })

    # Predicted period — only add if we have real data
    next_date_str = prediction.get("next_period_date")
    if next_date_str and prediction.get("has_data", True):
        try:
            nd = date.fromisoformat(next_date_str)
            events.append({
                "title":   "🔮 Predicted Period",
                "start":   nd.isoformat(),
                "end":     (nd + timedelta(days=5)).isoformat(),
                "display": "background",
                "color":   "#FF6F61",
                "extendedProps": {"type": "predicted"},
            })

            lo = prediction.get("confidence_lo", 0)
            hi = prediction.get("confidence_hi", 0)
            if isinstance(lo, int) and isinstance(hi, int) and hi > lo:
                lo_d = date.today() + timedelta(days=lo)
                hi_d = date.today() + timedelta(days=hi)
                events.append({
                    "title":   "📊 Likely window",
                    "start":   lo_d.isoformat(),
                    "end":     (hi_d + timedelta(days=1)).isoformat(),
                    "display": "background",
                    "color":   "rgba(255,111,97,0.20)",
                    "extendedProps": {"type": "confidence"},
                })
        except (ValueError, TypeError):
            pass  # skip bad date rather than crash

    fs = prediction.get("fertile_start")
    fe = prediction.get("fertile_end")
    if fs and fe and fs != fe:
        try:
            fs_d = date.fromisoformat(fs)
            fe_d = date.fromisoformat(fe)
            if fs_d < fe_d:
                events.append({
                    "title":   "🌺 Fertile Window",
                    "start":   fs,
                    "end":     fe,
                    "display": "background",
                    "color":   "#C08081",
                    "extendedProps": {"type": "fertile"},
                })
        except (ValueError, TypeError):
            pass

    return events


# ── Chart data ────────────────────────────────────────────────────────────────

async def build_chart_data(db: AsyncSession, user: User) -> dict:
    """
    Fetch all logs from the past 6 calendar months in one query,
    then group by month using proper month arithmetic (not 30-day approximation).
    """
    today  = date.today()
    cutoff = _subtract_months(today.replace(day=1), 5)

    result = await db.execute(
        select(CycleLog).where(and_(
            CycleLog.user_id == user.id,
            CycleLog.log_date >= cutoff,
        )).order_by(CycleLog.log_date)
    )
    all_logs = result.scalars().all()

    by_month: Dict[str, list] = defaultdict(list)
    for log in all_logs:
        by_month[_month_label(log.log_date)].append(log)

    # Build exactly 6 calendar-month labels using proper arithmetic
    labels = [
        _month_label(_subtract_months(today.replace(day=1), i))
        for i in range(5, -1, -1)
    ]

    mood_num  = {"happy": 5, "neutral": 3, "tired": 2, "sad": 1, "angry": 1}
    flow_rank = {"none": 0, "light": 1, "medium": 2, "heavy": 3}

    cramps, headache, fatigue, bloating, nausea, mood_scores, flow_dom = \
        [], [], [], [], [], [], []

    for label in labels:
        logs = by_month.get(label, [])

        def cnt(sym, _logs=logs):
            return sum(1 for l in _logs if sym in (l.symptoms or []))

        cramps.append(cnt("cramps"))
        headache.append(cnt("headache"))
        fatigue.append(cnt("fatigue"))
        bloating.append(cnt("bloating"))
        nausea.append(cnt("nausea"))

        avg_mood = round(
            sum(mood_num.get(l.mood, 3) for l in logs) / len(logs), 1
        ) if logs else 0
        mood_scores.append(avg_mood)

        if logs:
            flows = [l.flow_intensity for l in logs if l.flow_intensity]
            dom   = max(flows, key=lambda f: flow_rank.get(f, 0)) if flows else "none"
        else:
            dom = "none"
        flow_dom.append(dom)

    return {
        "labels":   labels,
        "cramps":   cramps,
        "headache": headache,
        "fatigue":  fatigue,
        "bloating": bloating,
        "nausea":   nausea,
        "mood":     mood_scores,
        "flow":     flow_dom,
    }


# ── User context for AI chat ──────────────────────────────────────────────────

async def build_user_context(db: AsyncSession, user: User, prediction: dict) -> dict:
    """
    Aggregate the last 30 days of logs to derive top symptoms and dominant mood
    rather than only looking at the single most recent log.
    """
    cutoff = date.today() - timedelta(days=30)
    result = await db.execute(
        select(CycleLog).where(and_(
            CycleLog.user_id == user.id,
            CycleLog.log_date >= cutoff,
        )).order_by(CycleLog.log_date.desc())
    )
    recent_logs = result.scalars().all()

    all_syms  = []
    all_moods = []
    for l in recent_logs:
        all_syms  += (l.symptoms or [])
        if l.mood:
            all_moods.append(l.mood)

    top_syms = [s for s, _ in Counter(all_syms).most_common(3)]
    dom_mood = Counter(all_moods).most_common(1)[0][0] if all_moods else None

    return {
        "name":              user.name,
        "age":               user.age,
        "avg_cycle":         user.avg_cycle,
        "avg_cycle_observed": prediction.get("avg_cycle_observed", user.avg_cycle),
        "bmi":               user.bmi,
        "is_irregular":      user.is_irregular,
        "has_data":          prediction.get("has_data", False),
        "cycle_count":       prediction.get("cycle_count", 0),
        "prediction":        prediction,
        "recent_symptoms":   top_syms,
        "dominant_mood":     dom_mood,
    }