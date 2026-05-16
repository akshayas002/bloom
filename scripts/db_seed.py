#!/usr/bin/env python3
"""
db_seed.py — Seed Bloom with sample data for local development.
Creates 2 users with 90 days of realistic cycle logs each.

Usage:
  python scripts/db_seed.py
  DATABASE_URL=postgresql+asyncpg://... python scripts/db_seed.py
"""
import asyncio, os, sys, random
from datetime import date, timedelta, datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DEBUG", "true")

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from app.core.database import Base
from app.models.db_models import User, CycleLog, ChatMessage
from app.core.security import hash_password
import app.models.db_models  # noqa

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./instance/bloom.db")
MOODS     = ["happy", "sad", "angry", "tired", "neutral"]
STRESSES  = ["low", "medium", "high"]
SLEEPS    = ["low", "normal", "high"]
EXERCISES = ["bad", "okay", "good"]
SYMPTOMS  = ["cramps", "headache", "fatigue", "bloating", "nausea", "back pain", "acne"]


async def seed():
    if "sqlite" in DATABASE_URL:
        os.makedirs("instance", exist_ok=True)
    connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
    engine  = create_async_engine(DATABASE_URL, echo=False, connect_args=connect_args)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    users_data = [
        dict(name="Priya Sharma", email="priya@example.com", age=26, avg_cycle=28.0, bmi=22.1, is_irregular=False),
        dict(name="Aisha Khan",   email="aisha@example.com", age=30, avg_cycle=38.0, bmi=24.5, is_irregular=True),
    ]

    async with Session() as db:
        for ud in users_data:
            existing = await db.execute(select(User).where(User.email == ud["email"]))
            if existing.scalars().first():
                print(f"  ⏭️  {ud['name']} already exists, skipping.")
                continue

            user = User(
                name=ud["name"], email=ud["email"],
                password=hash_password("password123"),
                age=ud["age"], avg_cycle=ud["avg_cycle"],
                bmi=ud["bmi"], is_irregular=ud["is_irregular"],
            )
            db.add(user)
            await db.flush()

            avg_cycle    = int(ud["avg_cycle"])
            period_start = date.today() - timedelta(days=90)
            current      = period_start
            period_dates = []
            while current <= date.today():
                period_dates.append(current)
                current += timedelta(days=avg_cycle + random.randint(-3, 3))

            period_day_set = set()
            for pd_start in period_dates:
                for offset in range(random.randint(3, 6)):
                    period_day_set.add(pd_start + timedelta(days=offset))

            for i in range(90):
                d         = date.today() - timedelta(days=89 - i)
                is_period = d in period_day_set
                flow      = random.choice(["light", "medium", "heavy"]) if is_period else "none"
                syms      = random.sample(SYMPTOMS, random.randint(1, 3)) if is_period \
                            else (random.sample(SYMPTOMS, 1) if random.random() < 0.3 else [])
                prev      = [p for p in sorted(period_day_set) if p < d]
                last_p    = max(prev) if prev else None
                days_since = (d - last_p).days if last_p else None
                cycle_day  = (days_since + 1) if days_since is not None else 1

                db.add(CycleLog(
                    user_id=user.id, log_date=d,
                    flow_intensity=flow, mood=random.choice(MOODS),
                    symptoms=syms, stress=random.choice(STRESSES),
                    sleep=random.choice(SLEEPS), exercise=random.choice(EXERCISES),
                    notes="", cycle_day=cycle_day, days_since_last=days_since,
                ))

            if ud["email"] == "priya@example.com":
                for role, content in [
                    ("user",      "When is my next period?"),
                    ("assistant", "Based on your data, your next period is expected in about 12 days. 🌸"),
                    ("user",      "I have bad cramps, what helps?"),
                    ("assistant", "Heating pad, gentle yoga, hydration, and ibuprofen at first sign. 💕"),
                ]:
                    db.add(ChatMessage(user_id=user.id, role=role, content=content))

            await db.commit()
            print(f"  ✅ {ud['name']} — 90 days of logs seeded")

    await engine.dispose()
    print("\n🌸 Done! Login with:")
    for ud in users_data:
        print(f"   {ud['email']}  /  password123")


if __name__ == "__main__":
    asyncio.run(seed())
