"""Dashboard, log, predict, chart, calendar, and kits routes."""

import json
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, delete

from app.core.database import get_db
from app.models.db_models import User, CycleLog, ChatMessage
from app.schemas.schemas import PredictRequest
from app.services.cycle_service import (
    get_prediction, build_calendar_events, build_chart_data, build_user_context
)

router    = APIRouter()
templates = Jinja2Templates(directory="templates")

# ── Auth helpers ──────────────────────────────────────────────────────────────
def flash(request: Request, msg: str, cat: str = "info"):
    request.session.setdefault("_flashes", []).append((cat, msg))

def get_flashes(request: Request):
    return request.session.pop("_flashes", [])

async def require_user(request: Request, db: AsyncSession) -> Optional[User]:
    uid = request.session.get("user_id")
    if not uid:
        return None
    result = await db.execute(select(User).where(User.id == uid))
    return result.scalars().first()


# ── Root redirect ─────────────────────────────────────────────────────────────
@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse("/dashboard", status_code=303)
    return RedirectResponse("/login", status_code=303)


# ── Dashboard ─────────────────────────────────────────────────────────────────
@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: AsyncSession = Depends(get_db)):
    user = await require_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    prediction   = await get_prediction(db, user)
    cal_events   = await build_calendar_events(db, user, prediction)
    chart_data   = await build_chart_data(db, user)

    result = await db.execute(
        select(CycleLog)
        .where(CycleLog.user_id == user.id)
        .order_by(CycleLog.log_date.desc())
        .limit(10)
    )
    logs = result.scalars().all()
    recent_logs = []
    for l in logs:
        d = {
            "id":             l.id,
            "log_date":       l.log_date,
            "flow_intensity": l.flow_intensity,
            "mood":           l.mood,
            "symptoms_list":  l.symptoms if isinstance(l.symptoms, list) else (json.loads(l.symptoms or "[]")),
            "stress":         l.stress,
            "sleep":          l.sleep,
            "exercise":       l.exercise,
            "notes":          l.notes,
            "cycle_day":      l.cycle_day,
        }
        recent_logs.append(d)

    phase = prediction.get("cycle_phase", "follicular")
    phase_tips = {
        "menstrual":  "You're in your menstrual phase. Rest, stay warm, and eat iron-rich foods like spinach. A heating pad is your best friend right now. 🌹",
        "follicular": "Energy is rising in your follicular phase! Great time for strength training, new projects, and socialising. Your oestrogen is climbing. 🌱",
        "ovulation":  "You're near ovulation — peak energy and confidence! Ideal for important meetings, dates, and intense workouts. ✨",
        "luteal":     "You're in the luteal phase. Prioritise sleep, reduce caffeine, and enjoy magnesium-rich foods like dark chocolate and nuts. 🌙",
    }

    days_until = prediction.get("days_until", 0)
    avg_cycle  = user.avg_cycle or 28
    progress   = min(100, int((prediction.get("cycle_day", 1) / avg_cycle) * 100))

    return templates.TemplateResponse("dashboard.html", {
        "request":         request,
        "user":            user,
        "flashes":         get_flashes(request),
        "prediction":      prediction,
        "days_until":      days_until,
        "next_date":       prediction.get("next_period_date", ""),
        "cycle_day":       prediction.get("cycle_day", 1),
        "cycle_phase":     phase.capitalize(),
        "progress":        progress,
        "pcos_risk":       prediction.get("pcos_risk_level", "low"),
        "pcos_score":      prediction.get("pcos_risk_score", 0),
        "pcos_reasons":    prediction.get("pcos_reasons", []),
        "is_irregular":    prediction.get("is_irregular", False),
        "confidence_lo":   prediction.get("confidence_lo", days_until),
        "confidence_hi":   prediction.get("confidence_hi", days_until),
        "wellness_tip":    phase_tips.get(phase, phase_tips["follicular"]),
        "fertile_start":   prediction.get("fertile_start", ""),
        "fertile_end":     prediction.get("fertile_end", ""),
        "recent_logs":     recent_logs,
        "calendar_events": json.dumps(cal_events),
        "chart_data":      json.dumps(chart_data),
    })


# ── Log Entry — GET ───────────────────────────────────────────────────────────
@router.get("/log", response_class=HTMLResponse)
async def log_page(request: Request, db: AsyncSession = Depends(get_db)):
    user = await require_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("log.html", {
        "request": request,
        "user":    user,
        "flashes": get_flashes(request),
        "today":   date.today().isoformat(),
    })


# ── Log Entry — POST ──────────────────────────────────────────────────────────
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
    user = await require_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    form     = await request.form()
    symptoms = list(form.getlist("symptoms"))
    ld       = datetime.strptime(log_date, "%Y-%m-%d").date()

    result = await db.execute(
        select(CycleLog)
        .where(and_(CycleLog.user_id == user.id,
                    CycleLog.flow_intensity.in_(("light", "medium", "heavy")),
                    CycleLog.log_date < ld))
        .order_by(CycleLog.log_date.desc())
        .limit(1)
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
        log_obj = CycleLog(
            user_id=user.id, log_date=ld,
            flow_intensity=flow_intensity, mood=mood,
            symptoms=symptoms, stress=stress,
            sleep=sleep, exercise=exercise,
            notes=notes, cycle_day=cycle_day,
            days_since_last=days_since,
        )
        db.add(log_obj)

    await db.commit()
    flash(request, "Log saved! 🌸", "success")
    return RedirectResponse("/dashboard", status_code=303)


# ── All Logs ──────────────────────────────────────────────────────────────────
@router.get("/logs", response_class=HTMLResponse)
async def all_logs(request: Request, db: AsyncSession = Depends(get_db)):
    user = await require_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)

    result = await db.execute(
        select(CycleLog)
        .where(CycleLog.user_id == user.id)
        .order_by(CycleLog.log_date.desc())
    )
    rows = result.scalars().all()
    logs = []
    for l in rows:
        logs.append({
            "id":             l.id,
            "log_date":       l.log_date,
            "flow_intensity": l.flow_intensity,
            "mood":           l.mood,
            "symptoms_list":  l.symptoms if isinstance(l.symptoms, list) else json.loads(l.symptoms or "[]"),
            "stress":         l.stress,
            "sleep":          l.sleep,
            "exercise":       l.exercise,
            "notes":          l.notes,
            "cycle_day":      l.cycle_day,
        })

    return templates.TemplateResponse("logs.html", {
        "request": request,
        "user":    user,
        "flashes": get_flashes(request),
        "logs":    logs,
    })


# ── Delete Log ────────────────────────────────────────────────────────────────
@router.post("/logs/delete/{log_id}")
async def delete_log(log_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    user = await require_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)
    await db.execute(
        delete(CycleLog).where(and_(CycleLog.id == log_id, CycleLog.user_id == user.id))
    )
    await db.commit()
    flash(request, "Log deleted.", "info")
    return RedirectResponse("/logs", status_code=303)


# ── Kits page ─────────────────────────────────────────────────────────────────
@router.get("/kits", response_class=HTMLResponse)
async def kits_page(request: Request, db: AsyncSession = Depends(get_db)):
    user = await require_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=303)
    prediction  = await get_prediction(db, user)
    next_period = prediction.get("next_period_date", "")
    return templates.TemplateResponse("kits.html", {
        "request":     request,
        "user":        user,
        "flashes":     get_flashes(request),
        "next_period": next_period,
    })


# ── Predict API ───────────────────────────────────────────────────────────────
@router.post("/api/predict")
async def predict_api(request: Request, db: AsyncSession = Depends(get_db)):
    user = await require_user(request, db)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    data = await request.json()

    from app.ml.predictor import predict as ml_predict
    result = ml_predict(
        age             = float(data.get("age",             user.age or 25)),
        cycle_length    = float(data.get("cycle_length",    user.avg_cycle or 28)),
        days_since_last = float(data.get("last_period_day", 14)),
        mood            = data.get("mood",     "neutral"),
        flow            = data.get("flow",     "medium"),
        symptom         = data.get("symptom",  "cramps"),
        stress          = data.get("stress",   "medium"),
        sleep           = data.get("sleep",    "normal"),
        exercise        = data.get("exercise", "okay"),
        bmi             = float(data.get("bmi",          user.bmi or 22.5)),
        avg_previous    = float(data.get("avg_previous",  user.avg_cycle or 28)),
        cycle_variation = float(data.get("cycle_variation", 2.0)),
        symptoms        = data.get("symptoms", []),
    )

    pcos = result["pcos_risk_level"]
    msg  = f"Your next period is predicted in {result['days_until']} days ({result['next_period_date']})."
    msg += f" 80% confidence window: {result['confidence_lo']}–{result['confidence_hi']} days."
    if result["is_irregular"]:
        msg += " ⚠️ Your cycle is irregular — predictions may vary. Keep logging for accuracy."
    if pcos in ("moderate", "high"):
        msg += f" 🔔 PCOS risk: {pcos.upper()} — consider consulting a gynecologist."

    return JSONResponse({**result, "message": msg})


# ── Chart API ─────────────────────────────────────────────────────────────────
@router.get("/api/chart")
async def chart_api(request: Request, db: AsyncSession = Depends(get_db)):
    user = await require_user(request, db)
    if not user:
        return JSONResponse({}, status_code=401)
    return JSONResponse(await build_chart_data(db, user))


# ── Calendar API ──────────────────────────────────────────────────────────────
@router.get("/api/calendar")
async def calendar_api(request: Request, db: AsyncSession = Depends(get_db)):
    user = await require_user(request, db)
    if not user:
        return JSONResponse([], status_code=401)
    pred = await get_prediction(db, user)
    return JSONResponse(await build_calendar_events(db, user, pred))