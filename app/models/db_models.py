"""
SQLAlchemy ORM models for Bloom.

Fixes vs original:
  - bcrypt passwords (handled in security.py)
  - SymptomList TypeDecorator: symptoms column always returns a Python list
  - datetime.now(timezone.utc) instead of deprecated datetime.utcnow
  - DateTime(timezone=True) columns throughout
"""

import json
from datetime import datetime, timezone, date

from sqlalchemy import (
    Column, Integer, Float, String, Text,
    Boolean, Date, DateTime, ForeignKey, TypeDecorator,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


# ── SymptomList TypeDecorator ─────────────────────────────────────────────────
# Stores a Python list as a JSON string. On read, always returns a list.
# Eliminates all the isinstance(symptoms, list) else json.loads(...) hedges
# that were scattered across 4 different files in the original code.

class SymptomList(TypeDecorator):
    """Persist symptom lists as JSON; always deserialise to a Python list."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return "[]"
        if isinstance(value, list):
            return json.dumps(value)
        return value  # already a JSON string

    def process_result_value(self, value, dialect):
        if not value:
            return []
        try:
            result = json.loads(value)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, TypeError):
            return []


def _utcnow():
    return datetime.now(timezone.utc)


# ── Models ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id           = Column(Integer, primary_key=True, index=True)
    name         = Column(String(120), nullable=False)
    email        = Column(String(200), unique=True, nullable=False, index=True)
    password     = Column(String(300), nullable=False)   # bcrypt hash
    age          = Column(Integer, default=25)
    avg_cycle    = Column(Float,   default=28.0)
    bmi          = Column(Float,   default=22.5)
    is_irregular = Column(Boolean, default=False)
    created_at   = Column(DateTime(timezone=True), default=_utcnow)

    logs     = relationship("CycleLog",    back_populates="user", cascade="all, delete-orphan")
    messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")


class CycleLog(Base):
    __tablename__ = "cycle_logs"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    log_date        = Column(Date,    nullable=False, default=date.today)
    flow_intensity  = Column(String(20),  default="none")     # none/light/medium/heavy
    mood            = Column(String(30),  default="neutral")
    symptoms        = Column(SymptomList, default=list)        # always a Python list on read
    stress          = Column(String(20),  default="medium")
    sleep           = Column(String(20),  default="normal")
    exercise        = Column(String(20),  default="okay")
    notes           = Column(Text,        default="")
    cycle_day       = Column(Integer,     default=1)
    days_since_last = Column(Integer,     nullable=True)
    created_at      = Column(DateTime(timezone=True), default=_utcnow)

    user = relationship("User", back_populates="logs")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role       = Column(String(20))   # "user" | "assistant"
    content    = Column(Text)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    user = relationship("User", back_populates="messages")
