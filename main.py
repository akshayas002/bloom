"""
Bloom — Menstrual Health AI Assistant
FastAPI application entry point.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.core.config import get_settings
from app.core.database import create_tables
from app.api.auth      import router as auth_router
from app.api.dashboard import router as dash_router
from app.api.chat      import router as chat_router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger   = logging.getLogger("bloom")
settings = get_settings()

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌸 Bloom starting up…")
    os.makedirs("instance",   exist_ok=True)
    os.makedirs("models_ml",  exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js",  exist_ok=True)
    await create_tables()
    logger.info("✅ Database tables ready")
    yield
    logger.info("🌸 Bloom shutting down")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Bloom — Menstrual Health AI",
    description = "Cycle tracking, PCOS awareness, and AI-powered insights",
    version     = "2.0.0",
    lifespan    = lifespan,
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    SessionMiddleware,
    secret_key   = settings.SECRET_KEY,
    session_cookie = "bloom_session",
    https_only   = False,   # set True in production behind HTTPS
    same_site    = "lax",
)

# ── Static files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Templates ─────────────────────────────────────────────────────────────────
templates = Jinja2Templates(directory="templates")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(dash_router)
app.include_router(chat_router)

# ── Global 404 ───────────────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found(request: Request, exc):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
