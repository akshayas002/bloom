"""Bloom — FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.core.config import get_settings
from app.core.database import create_tables
from app.api import auth, dashboard, chat

settings  = get_settings()
templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY,
    # FIX: https_only=False always — Render terminates TLS at the proxy level.
    # The app itself receives plain HTTP from the proxy, so https_only=True
    # would cause the session cookie to never be set → infinite redirect loop.
    https_only=False,
    same_site=settings.SESSION_COOKIE_SAMESITE,
)

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    pass

app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(chat.router)


@app.exception_handler(404)
async def not_found(request: Request, exc):
    return templates.TemplateResponse(request, "404.html", {}, status_code=404)