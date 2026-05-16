#!/usr/bin/env python3
"""
db_init.py — Initialize the Bloom database (SQLite or PostgreSQL/Supabase).

Usage:
  python scripts/db_init.py
  DATABASE_URL=postgresql+asyncpg://... python scripts/db_init.py
"""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DEBUG", "true")

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from app.core.database import Base
import app.models.db_models  # noqa

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./instance/bloom.db")


async def init_db():
    print(f"\n📦 Database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
    if "sqlite" in DATABASE_URL:
        os.makedirs("instance", exist_ok=True)

    connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
    engine = create_async_engine(DATABASE_URL, echo=False, connect_args=connect_args)

    print("🔨 Creating tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with engine.connect() as conn:
        if "sqlite" in DATABASE_URL:
            result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"))
        else:
            result = await conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"))
        tables = [r[0] for r in result.fetchall()]

    await engine.dispose()
    print(f"✅ Tables: {tables}")
    print("🌸 Database ready!\n")


if __name__ == "__main__":
    asyncio.run(init_db())
