"""Pydantic schemas for request/response validation."""

from __future__ import annotations
from datetime import date, datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Auth ──────────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name:         str   = Field(..., min_length=2, max_length=100)
    email:        str   = Field(..., min_length=5, max_length=200)   # plain str — no email-validator dep
    password:     str   = Field(..., min_length=6)
    age:          int   = Field(default=25, ge=10, le=60)
    avg_cycle:    float = Field(default=28.0, ge=14, le=60)
    bmi:          float = Field(default=22.5, ge=10, le=60)
    is_irregular: bool  = False


class LoginRequest(BaseModel):
    email:    str
    password: str


class UserOut(BaseModel):
    id:           int
    name:         str
    email:        str
    age:          int
    avg_cycle:    float
    bmi:          float
    is_irregular: bool
    created_at:   datetime

    class Config:
        from_attributes = True


# ── Cycle Log ─────────────────────────────────────────────────────────────────
class LogCreate(BaseModel):
    log_date:       date
    flow_intensity: str      = "none"
    mood:           str      = "neutral"
    symptoms:       List[str] = []
    stress:         str      = "medium"
    sleep:          str      = "normal"
    exercise:       str      = "okay"
    notes:          str      = ""

    @field_validator("flow_intensity")
    @classmethod
    def val_flow(cls, v):
        assert v in ("none", "light", "medium", "heavy"), "Invalid flow"
        return v

    @field_validator("mood")
    @classmethod
    def val_mood(cls, v):
        assert v in ("happy", "sad", "angry", "tired", "neutral"), "Invalid mood"
        return v


class LogOut(BaseModel):
    id:              int
    user_id:         int
    log_date:        date
    flow_intensity:  str
    mood:            str
    symptoms:        List[str]
    stress:          str
    sleep:           str
    exercise:        str
    notes:           str
    cycle_day:       int
    days_since_last: Optional[int]
    created_at:      datetime

    class Config:
        from_attributes = True


# ── Predict ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    age:             Optional[float] = None
    cycle_length:    Optional[float] = None
    days_since_last: Optional[float] = None
    mood:            str   = "neutral"
    flow:            str   = "medium"
    symptom:         str   = "cramps"
    stress:          str   = "medium"
    sleep:           str   = "normal"
    exercise:        str   = "okay"
    bmi:             Optional[float] = None
    avg_previous:    Optional[float] = None
    cycle_variation: float = 2.0
    symptoms:        List[str] = []


class PredictOut(BaseModel):
    days_until:       int
    next_period_date: str
    confidence_lo:    int
    confidence_hi:    int
    mae_days:         float
    is_irregular:     bool
    cycle_phase:      str
    cycle_day:        int
    fertile_start:    str
    fertile_end:      str
    pcos_risk_level:  str
    pcos_risk_score:  float
    pcos_reasons:     List[str]
    message:          str


# ── Chat ──────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatOut(BaseModel):
    response: str


# ── Chart / Calendar ──────────────────────────────────────────────────────────
class ChartData(BaseModel):
    labels:   List[str]
    cramps:   List[int]
    headache: List[int]
    fatigue:  List[int]
    bloating: List[int]
    nausea:   List[int]
    mood:     List[float]
    flow:     List[str]
