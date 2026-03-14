"""Auth routes — register, login, logout."""

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.security import hash_password, verify_password
from app.models.db_models import User

router    = APIRouter()
templates = Jinja2Templates(directory="templates")


def flash(request: Request, message: str, category: str = "info"):
    request.session.setdefault("_flashes", []).append((category, message))


def get_flashes(request: Request):
    flashes = request.session.pop("_flashes", [])
    return flashes


# ── Register ──────────────────────────────────────────────────────────────────
@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("auth.html",
        {"request": request, "mode": "register", "flashes": get_flashes(request)})


@router.post("/register")
async def register(
    request: Request,
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
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalars().first():
        flash(request, "Email already registered.", "error")
        return RedirectResponse("/register", status_code=303)

    user = User(name=name.strip(), email=email,
                password=hash_password(password),
                age=age, avg_cycle=avg_cycle,
                bmi=bmi, is_irregular=is_irregular)
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
    return templates.TemplateResponse("auth.html",
        {"request": request, "mode": "login", "flashes": get_flashes(request)})


@router.post("/login")
async def login(
    request:  Request,
    email:    str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    email = email.strip().lower()
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
