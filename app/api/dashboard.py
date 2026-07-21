"""Dashboard, log, predict, chart, calendar and kits routes."""

import json
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, delete

from app.core.database import get_db
from app.models.db_models import User, CycleLog
from app.schemas.schemas import PredictRequest
from app.services.cycle_service import (
    get_prediction, build_calendar_events, build_chart_data, build_user_context,
)
from app.utils.utils import flash, get_flashes, get_current_user

router    = APIRouter()
templates = Jinja2Templates(directory="templates")

# Safe fallback prediction for new users / errors
_EMPTY_PREDICTION = {
    "days_until":        None,
    "next_period_date":  None,
    "cycle_day":         1,
    "cycle_phase":       "follicular",
    "pcos_risk_level":   "low",
    "pcos_risk_score":   0.0,
    "pcos_reasons":      [],
    "is_irregular":      False,
    "confidence_lo":     None,
    "confidence_hi":     None,
    "fertile_start":     None,   # None prevents date.fromisoformat() crash
    "fertile_end":       None,
    "has_data":          False,
    "cycle_count":       0,
    "avg_cycle_observed": 28.0,
}
_EMPTY_CHART = {
    "labels": [], "cramps": [], "headache": [],
    "fatigue": [], "bloating": [], "nausea": [],
    "mood": [], "flow": [],
}


@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse("/dashboard", status_code=303)
    return RedirectResponse("/login", status_code=303)


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    try:
        prediction = await get_prediction(db, user)
        cal_events = await build_calendar_events(db, user, prediction)
        chart_data = await build_chart_data(db, user)
    except Exception as e:
        print(f"[dashboard] prediction error: {e}")
        prediction = dict(_EMPTY_PREDICTION)
        cal_events = []
        chart_data = dict(_EMPTY_CHART)

    result = await db.execute(
        select(CycleLog)
        .where(CycleLog.user_id == user.id)
        .order_by(CycleLog.log_date.desc())
        .limit(10)
    )
    recent_logs = [
        {
            "id":             l.id,
            "log_date":       l.log_date,
            "flow_intensity": l.flow_intensity,
            "mood":           l.mood,
            "symptoms_list":  l.symptoms or [],
            "stress":         l.stress,
            "sleep":          l.sleep,
            "exercise":       l.exercise,
            "notes":          l.notes,
            "cycle_day":      l.cycle_day,
        }
        for l in result.scalars().all()
    ]

    phase = prediction.get("cycle_phase", "follicular")
    phase_tips = {
        "menstrual":  "You're in your menstrual phase. Rest, stay warm, and eat iron-rich foods. 🌹",
        "follicular": "Energy is rising in your follicular phase! Great for strength training and new projects. 🌱",
        "ovulation":  "You're near ovulation — peak energy and confidence! Great for important meetings. ✨",
        "luteal":     "You're in the luteal phase. Prioritise sleep, reduce caffeine, and eat magnesium-rich foods. 🌙",
    }
    avg_cycle = user.avg_cycle or 28
    cycle_day = prediction.get("cycle_day", 1)
    progress  = min(100, int((cycle_day / avg_cycle) * 100))

    return templates.TemplateResponse(request, "dashboard.html", {
        "user":            user,
        "flashes":         get_flashes(request),
        "prediction":      prediction,
        "has_data":        prediction.get("has_data", False),
        "days_until":      prediction.get("days_until"),        # None = not enough data
        "next_date":       prediction.get("next_period_date"),  # None-safe in template
        "cycle_day":       cycle_day,
        "cycle_phase":     phase.capitalize(),
        "progress":        progress,
        "pcos_risk":       prediction.get("pcos_risk_level", "low"),
        "pcos_score":      prediction.get("pcos_risk_score", 0),
        "pcos_reasons":    prediction.get("pcos_reasons", []),
        "is_irregular":    prediction.get("is_irregular", False),
        "confidence_lo":   prediction.get("confidence_lo"),
        "confidence_hi":   prediction.get("confidence_hi"),
        "wellness_tip":    phase_tips.get(phase, phase_tips["follicular"]),
        "fertile_start":   prediction.get("fertile_start"),     # None = hide in template
        "fertile_end":     prediction.get("fertile_end"),
        "cycle_count":     prediction.get("cycle_count", 0),
        "avg_cycle_observed": prediction.get("avg_cycle_observed", user.avg_cycle),
        "recent_logs":     recent_logs,
        "calendar_events": json.dumps(cal_events),
        "chart_data":      json.dumps(chart_data),
    })


@router.get("/log", response_class=HTMLResponse)
async def log_page(request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "log.html", {
        "user":    user,
        "flashes": get_flashes(request),
        "today":   date.today().isoformat(),
    })


@router.post("/log")
async def log_submit(
    request:        Request,
    db:             AsyncSession = Depends(get_db),
    log_date:       str  = Form(...),
    flow_intensity: str  = Form("none"),
    mood:           str  = Form("neutral"),
    stress:         str  = Form("medium"),
    sleep:          str  = Form("normal"),
    exercise:       str  = Form("okay"),
    notes:          str  = Form(""),
):
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form     = await request.form()
    symptoms = list(form.getlist("symptoms"))
    ld       = datetime.strptime(log_date, "%Y-%m-%d").date()

    # Find most recent cycle START before this date
    result = await db.execute(
        select(CycleLog).where(and_(
            CycleLog.user_id == user.id,
            CycleLog.flow_intensity.in_(("light", "medium", "heavy")),
            CycleLog.log_date < ld,
        )).order_by(CycleLog.log_date.desc()).limit(1)
    )
    last = result.scalars().first()
    if last:
        days_since = (ld - last.log_date).days
        cycle_day  = days_since + 1
    else:
        days_since = None
        cycle_day  = 1

    existing = await db.execute(
        select(CycleLog).where(and_(CycleLog.user_id == user.id, CycleLog.log_date == ld))
    )
    log_obj = existing.scalars().first()

    if log_obj:
        log_obj.flow_intensity  = flow_intensity
        log_obj.mood            = mood
        log_obj.symptoms        = symptoms
        log_obj.stress          = stress
        log_obj.sleep           = sleep
        log_obj.exercise        = exercise
        log_obj.notes           = notes
        log_obj.cycle_day       = cycle_day
        log_obj.days_since_last = days_since
    else:
        new_log = CycleLog(
            user_id=user.id, log_date=ld,
            flow_intensity=flow_intensity, mood=mood,
            symptoms=symptoms, stress=stress,
            sleep=sleep, exercise=exercise,
            notes=notes, cycle_day=cycle_day,
            days_since_last=days_since,
        )
        db.add(new_log)

    await db.commit()
    flash(request, "Log saved! 🌸", "success")
    return RedirectResponse("/dashboard", status_code=303)


@router.get("/logs", response_class=HTMLResponse)
async def all_logs(request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    result = await db.execute(
        select(CycleLog)
        .where(CycleLog.user_id == user.id)
        .order_by(CycleLog.log_date.desc())
    )
    logs = [
        {
            "id":             l.id,
            "log_date":       l.log_date,
            "flow_intensity": l.flow_intensity,
            "mood":           l.mood,
            "symptoms_list":  l.symptoms or [],
            "stress":         l.stress,
            "sleep":          l.sleep,
            "exercise":       l.exercise,
            "notes":          l.notes,
            "cycle_day":      l.cycle_day,
        }
        for l in result.scalars().all()
    ]
    return templates.TemplateResponse(request, "logs.html", {
        "user":    user,
        "flashes": get_flashes(request),
        "logs":    logs,
    })


@router.post("/logs/delete/{log_id}")
async def delete_log(log_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)
    await db.execute(
        delete(CycleLog).where(and_(CycleLog.id == log_id, CycleLog.user_id == user.id))
    )
    await db.commit()
    flash(request, "Log deleted.", "info")
    return RedirectResponse("/logs", status_code=303)


@router.get("/kits", response_class=HTMLResponse)
async def kits_page(request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)
    try:
        prediction  = await get_prediction(db, user)
        next_period = prediction.get("next_period_date")
    except Exception:
        next_period = None
    return templates.TemplateResponse(request, "kits.html", {
        "user":        user,
        "flashes":     get_flashes(request),
        "next_period": next_period,
    })


@router.post("/api/predict")
async def predict_api(request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    raw = await request.json()
    try:
        data = PredictRequest(**raw)
    except Exception as e:
        return JSONResponse({"error": f"Invalid input: {e}"}, status_code=422)

    from app.ml.predictor import predict as ml_predict
    result = ml_predict(
        age             = float(data.age             or user.age or 25),
        days_since_last = float(data.days_since_last or 14),
        mood            = data.mood,
        flow            = data.flow,
        symptom         = data.symptom or "none",
        stress          = data.stress,
        sleep           = data.sleep,
        exercise        = data.exercise,
        bmi             = float(data.bmi             or user.bmi or 22.5),
        # cycle_length is a legacy field name — treated as a fallback
        # historical average, never as the (unknowable mid-cycle) current
        # cycle's own length.
        avg_previous    = float(data.avg_previous or data.cycle_length or user.avg_cycle or 28),
        cycle_variation = float(data.cycle_variation),
        symptoms        = data.symptoms,
    )
    pcos = result["pcos_risk_level"]
    nd   = result["next_period_date"]
    msg  = (
        f"Your next period is predicted in {result['days_until']} days ({nd}). "
        f"80% confidence window: {result['confidence_lo']}–{result['confidence_hi']} days."
    )
    if result["is_irregular"]:
        msg += " ⚠️ Irregular cycle — keep logging for accuracy."
    if pcos in ("moderate", "high"):
        msg += f" 🔔 PCOS risk: {pcos.upper()} — consider consulting a gynecologist."
    return JSONResponse({**result, "message": msg})


@router.get("/api/chart")
async def chart_api(request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    if not user:
        return JSONResponse({}, status_code=401)
    return JSONResponse(await build_chart_data(db, user))


@router.get("/api/calendar")
async def calendar_api(request: Request, db: AsyncSession = Depends(get_db)):
    user = await get_current_user(request, db)
    if not user:
        return JSONResponse([], status_code=401)
    pred = await get_prediction(db, user)
    return JSONResponse(await build_calendar_events(db, user, pred))