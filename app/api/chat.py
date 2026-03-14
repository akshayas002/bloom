"""Chat API — data-aware Bloom AI endpoint."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.db_models import User, ChatMessage
from app.services.ai_service import ask_bloom
from app.services.cycle_service import get_prediction, build_user_context

router = APIRouter()


async def _get_user(request: Request, db: AsyncSession) -> User | None:
    uid = request.session.get("user_id")
    if not uid:
        return None
    r = await db.execute(select(User).where(User.id == uid))
    return r.scalars().first()


@router.post("/api/chat")
async def chat(request: Request, db: AsyncSession = Depends(get_db)):
    user = await _get_user(request, db)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    body   = await request.json()
    prompt = body.get("message", "").strip()
    if not prompt:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # Fetch conversation history (last 14 messages)
    hist_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.user_id == user.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(14)
    )
    history = [
        {"role": m.role, "content": m.content}
        for m in reversed(hist_result.scalars().all())
    ]

    # Build live user context (prediction + recent symptoms)
    try:
        prediction   = await get_prediction(db, user)
        user_context = await build_user_context(db, user, prediction)
    except Exception:
        user_context = {"name": user.name, "avg_cycle": user.avg_cycle}

    response = await ask_bloom(prompt, history, user_context)

    # Persist messages
    db.add(ChatMessage(user_id=user.id, role="user",      content=prompt))
    db.add(ChatMessage(user_id=user.id, role="assistant", content=response))
    await db.commit()

    return JSONResponse({"response": response})
