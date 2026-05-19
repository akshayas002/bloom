"""
Shared utilities used across all API routes.

Fixes: flash() and get_current_user() were duplicated in auth.py and
dashboard.py with identical code. Now there's one source of truth.
"""

from typing import Optional
from fastapi import Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.db_models import User


# ── Flash messaging ───────────────────────────────────────────────────────────

def flash(request: Request, message: str, category: str = "info") -> None:
    """Append a flash message to the session."""
    request.session.setdefault("_flashes", []).append((category, message))


def get_flashes(request: Request) -> list:
    """Pop and return all flash messages from the session."""
    return request.session.pop("_flashes", [])


# ── Auth helpers ──────────────────────────────────────────────────────────────

async def get_current_user(request: Request, db: AsyncSession) -> Optional[User]:
    """Return the authenticated User or None (non-raising)."""
    uid = request.session.get("user_id")
    if not uid:
        return None
    result = await db.execute(select(User).where(User.id == uid))
    return result.scalars().first()


async def require_user(request: Request, db: AsyncSession) -> User:
    """Return the authenticated User or raise 401."""
    user = await get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
