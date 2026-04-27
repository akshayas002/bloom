"""
Cycle analytics service — prediction, calendar events, chart data,
and user-data context for AI chat.
"""

import json
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional
from collections import Counter

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, extract, func, or_

from app.models.db_models import CycleLog, User
from app.ml.predictor import predict as ml_predict, score_pcos_risk


PERIOD_FLOWS = ("light", "medium", "heavy")


# ── Utility ───────────────────────────────────────────────────────────────────
def _symptoms(log: CycleLog) -> List[str]:
    syms = log.symptoms
    if isinstance(syms, list):
        return syms
    try:
        return json.loads(syms or "[]")
    except Exception:
        return []


async def _last_period_log(db: AsyncSession, user_id: int) -> Optional[CycleLog]:
    result = await db.execute(
        select(CycleLog)
        .where(and_(CycleLog.user_id == user_id,
                    CycleLog.flow_intensity.in_(PERIOD_FLOWS)))
        .order_by(CycleLog.log_date.desc())
        .limit(1)
    )
    return result.scalars().first()


async def _previous_cycle_lengths(db: AsyncSession, user_id: int, n: int = 6) -> List[int]:
    """Return list of the last n inter-period distances in days."""
    result = await db.execute(
        select(CycleLog.log_date)
        .where(and_(CycleLog.user_id == user_id,
                    CycleLog.flow_intensity.in_(PERIOD_FLOWS)))
        .order_by(CycleLog.log_date.desc())
        .limit(n + 1)
    )
    dates = [r[0] for r in result.fetchall()]
    if len(dates) < 2:
        return []
    lengths = [(dates[i] - dates[i+1]).days for i in range(len(dates)-1)]
    return lengths


# ── Prediction ────────────────────────────────────────────────────────────────
async def get_prediction(db: AsyncSession, user: User) -> dict:
    last_log = await _last_period_log(db, user.id)
    if last_log:
        days_since = (date.today() - last_log.log_date).days
    else:
        days_since = int(user.avg_cycle // 2)

    prev_lengths = await _previous_cycle_lengths(db, user.id)
    avg_prev     = sum(prev_lengths) / len(prev_lengths) if prev_lengths else user.avg_cycle
    cycle_var    = (max(prev_lengths) - min(prev_lengths)) if len(prev_lengths) > 1 else 2.0

    # Gather most recent log's symptom/mood data
    recent_result = await db.execute(
        select(CycleLog)
        .where(CycleLog.user_id == user.id)
        .order_by(CycleLog.log_date.desc())
        .limit(1)
    )
    recent = recent_result.scalars().first()
    mood     = recent.mood     if recent else "neutral"
    flow     = recent.flow_intensity if recent else "none"
    stress   = recent.stress   if recent else "medium"
    sleep    = recent.sleep    if recent else "normal"
    exercise = recent.exercise if recent else "okay"
    symptoms = _symptoms(recent) if recent else []
    symptom  = symptoms[0] if symptoms else "cramps"

    result = ml_predict(
        age             = float(user.age or 25),
        cycle_length    = float(user.avg_cycle or 28),
        days_since_last = float(days_since),
        mood=mood, flow=flow, symptom=symptom,
        stress=stress, sleep=sleep, exercise=exercise,
        bmi             = float(user.bmi or 22.5),
        avg_previous    = float(avg_prev),
        cycle_variation = float(cycle_var),
        symptoms        = symptoms,
    )
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

    for log in logs:
        ds   = log.log_date.isoformat()
        syms = _symptoms(log)

        if log.flow_intensity in PERIOD_FLOWS:
            intensity_colors = {"light": "#FFB3AD", "medium": "#FF6F61", "heavy": "#CC3B30"}
            events.append({
                "title":  f"🩸 Period ({log.flow_intensity})",
                "start":  ds,
                "color":  intensity_colors.get(log.flow_intensity, "#FF6F61"),
                "extendedProps": {"type": "period", "cycle_day": log.cycle_day}
            })

        if syms:
            events.append({
                "title": "🔵 " + ", ".join(syms[:2]),
                "start": ds,
                "color": "#87CEEB",
                "extendedProps": {"type": "symptoms", "all_symptoms": syms}
            })

        mood_emojis = {"happy":"😊","sad":"😢","angry":"😠","tired":"😴","neutral":"😐"}
        if log.mood:
            events.append({
                "title": f"{mood_emojis.get(log.mood,'')} {log.mood.capitalize()}",
                "start": ds,
                "color": "#C08081",
                "extendedProps": {"type": "mood"}
            })

    # Predicted period window
    next_date = prediction.get("next_period_date")
    if next_date:
        nd = date.fromisoformat(next_date)
        events.append({
            "title":   "🔮 Predicted Period",
            "start":   nd.isoformat(),
            "end":     (nd + timedelta(days=5)).isoformat(),
            "display": "background",
            "color":   "#FF6F61",
            "extendedProps": {"type": "predicted"}
        })
        # Confidence range (lighter shade)
        lo_d = date.today() + timedelta(days=prediction.get("confidence_lo", 0))
        hi_d = date.today() + timedelta(days=prediction.get("confidence_hi", 0))
        events.append({
            "title":   "📊 Likely window",
            "start":   lo_d.isoformat(),
            "end":     (hi_d + timedelta(days=1)).isoformat(),
            "display": "background",
            "color":   "rgba(255,111,97,0.20)",
            "extendedProps": {"type": "confidence"}
        })

    # Fertile window
    fs = prediction.get("fertile_start")
    fe = prediction.get("fertile_end")
    if fs and fe:
        events.append({
            "title":   "🌺 Fertile Window",
            "start":   fs,
            "end":     fe,
            "display": "background",
            "color":   "#C08081",
            "extendedProps": {"type": "fertile"}
        })

    return events


# ── Chart data ────────────────────────────────────────────────────────────────
async def build_chart_data(db: AsyncSession, user: User) -> dict:
    labels, cramps, headache, fatigue, bloating, nausea, mood_scores, flow_dom = \
        [], [], [], [], [], [], [], []

    mood_num = {"happy": 5, "neutral": 3, "tired": 2, "sad": 1, "angry": 1}
    flow_rank = {"none": 0, "light": 1, "medium": 2, "heavy": 3}

    for i in range(5, -1, -1):
        target = (date.today().replace(day=1) - timedelta(days=i * 30))
        labels.append(target.strftime("%b %Y"))

        result = await db.execute(
            select(CycleLog).where(
                and_(CycleLog.user_id == user.id,
                     extract("month", CycleLog.log_date) == target.month,
                     extract("year",  CycleLog.log_date) == target.year)
            )
        )
        logs = result.scalars().all()

        def cnt(sym):
            return sum(1 for l in logs if sym in _symptoms(l))

        cramps.append(cnt("cramps"))
        headache.append(cnt("headache"))
        fatigue.append(cnt("fatigue"))
        bloating.append(cnt("bloating"))
        nausea.append(cnt("nausea"))

        avg_mood = round(sum(mood_num.get(l.mood, 3) for l in logs) / len(logs), 1) if logs else 0
        mood_scores.append(avg_mood)

        # Dominant flow
        if logs:
            flows = [l.flow_intensity for l in logs if l.flow_intensity]
            dom = max(flows, key=lambda f: flow_rank.get(f, 0)) if flows else "none"
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


# ── User context for AI ───────────────────────────────────────────────────────
async def build_user_context(db: AsyncSession, user: User, prediction: dict) -> dict:
    """Assemble rich context dict for AI personalisation."""
    # Recent symptoms (last 30 days)
    cutoff = date.today() - timedelta(days=30)
    result = await db.execute(
        select(CycleLog).where(
            and_(CycleLog.user_id == user.id,
                 CycleLog.log_date >= cutoff)
        ).order_by(CycleLog.log_date.desc())
    )
    recent_logs = result.scalars().all()

    all_syms   = []
    all_moods  = []
    for l in recent_logs:
        all_syms  += _symptoms(l)
        if l.mood:
            all_moods.append(l.mood)

    top_syms = [s for s, _ in Counter(all_syms).most_common(3)]
    dom_mood = Counter(all_moods).most_common(1)[0][0] if all_moods else None

    return {
        "name":             user.name,
        "age":              user.age,
        "avg_cycle":        user.avg_cycle,
        "bmi":              user.bmi,
        "is_irregular":     user.is_irregular,
        "prediction":       prediction,
        "recent_symptoms":  top_syms,
        "dominant_mood":    dom_mood,
    }
