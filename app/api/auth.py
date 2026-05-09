"""Auth routes — register, login, logout.

Fixes vs original:
  - Server-side validation added (email regex, name length, password min, age/cycle bounds)
  - bcrypt hashing via security.py
  - flash() and get_flashes() imported from shared utils (not duplicated here)
"""

import re
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.security import hash_password, verify_password
from app.models.db_models import User
from app.utils.utils import flash, get_flashes

router    = APIRouter()
templates = Jinja2Templates(directory="templates")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ── Register ──────────────────────────────────────────────────────────────────

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("auth.html", {
        "request": request,
        "mode":    "register",
        "flashes": get_flashes(request),
    })


@router.post("/register")
async def register(
    request:      Request,
    name:         str   = Form(...),
    email:        str   = Form(...),
    password:     str   = Form(...),
    age:          int   = Form(25),
    avg_cycle:    float = Form(28.0),
    bmi:          float = Form(22.5),
    is_irregular: bool  = Form(False),
    db: AsyncSession = Depends(get_db),
):
    email = email.strip().lower()
    name  = name.strip()

    # ── Validation (was completely absent in the original) ────────────────────
    if len(name) < 2:
        flash(request, "Name must be at least 2 characters.", "error")
        return RedirectResponse("/register", status_code=303)

    if not EMAIL_RE.match(email):
        flash(request, "Please enter a valid email address.", "error")
        return RedirectResponse("/register", status_code=303)

    if len(password) < 6:
        flash(request, "Password must be at least 6 characters.", "error")
        return RedirectResponse("/register", status_code=303)

    if not (10 <= age <= 60):
        flash(request, "Age must be between 10 and 60.", "error")
        return RedirectResponse("/register", status_code=303)

    if not (14.0 <= avg_cycle <= 60.0):
        flash(request, "Average cycle must be between 14 and 60 days.", "error")
        return RedirectResponse("/register", status_code=303)

    # ── Duplicate email check ─────────────────────────────────────────────────
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalars().first():
        flash(request, "An account with this email already exists.", "error")
        return RedirectResponse("/register", status_code=303)

    user = User(
        name=name,
        email=email,
        password=hash_password(password),   # bcrypt — not SHA-256
        age=age,
        avg_cycle=avg_cycle,
        bmi=bmi,
        is_irregular=is_irregular,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    request.session["user_id"]   = user.id
    request.session["user_name"] = user.name
    flash(request, f"Welcome to Bloom, {user.name}! 🌸", "success")
    return RedirectResponse("/dashboard", status_code=303)


# ── Login ─────────────────────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse("/dashboard", status_code=303)
    return templates.TemplateResponse("auth.html", {
        "request": request,
        "mode":    "login",
        "flashes": get_flashes(request),
    })


@router.post("/login")
async def login(
    request:  Request,
    email:    str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    email  = email.strip().lower()
    result = await db.execute(select(User).where(User.email == email))
    user   = result.scalars().first()

    if not user or not verify_password(user.password, password):
        flash(request, "Invalid email or password.", "error")
        return RedirectResponse("/login", status_code=303)

    request.session["user_id"]   = user.id
    request.session["user_name"] = user.name
    flash(request, f"Welcome back, {user.name}! 🌸", "success")
    return RedirectResponse("/dashboard", status_code=303)


# ── Logout ────────────────────────────────────────────────────────────────────

@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)
