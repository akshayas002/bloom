"""SQLAlchemy ORM models for Bloom."""

from datetime import datetime, date
from sqlalchemy import (Column, Integer, Float, String, Text,
                        Boolean, Date, DateTime, ForeignKey, JSON)
from sqlalchemy.orm import relationship
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String(120), nullable=False)
    email      = Column(String(200), unique=True, nullable=False, index=True)
    password   = Column(String(300), nullable=False)
    age        = Column(Integer,  default=25)
    avg_cycle  = Column(Float,    default=28.0)
    bmi        = Column(Float,    default=22.5)
    is_irregular = Column(Boolean, default=False)   # user-confirmed
    created_at = Column(DateTime, default=datetime.utcnow)

    logs     = relationship("CycleLog",    back_populates="user", cascade="all, delete-orphan")
    messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")


class CycleLog(Base):
    __tablename__ = "cycle_logs"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    log_date       = Column(Date,    nullable=False, default=date.today)
    flow_intensity = Column(String(20),  default="none")   # none/light/medium/heavy
    mood           = Column(String(30),  default="neutral")
    symptoms       = Column(JSON,        default=list)      # list of strings
    stress         = Column(String(20),  default="medium")
    sleep          = Column(String(20),  default="normal")
    exercise       = Column(String(20),  default="okay")
    notes          = Column(Text,        default="")
    cycle_day      = Column(Integer,     default=1)
    # Computed / derived
    days_since_last = Column(Integer,    nullable=True)
    created_at     = Column(DateTime,    default=datetime.utcnow)

    user = relationship("User", back_populates="logs")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role       = Column(String(20))    # user | assistant
    content    = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="messages")
