"""
Bloom AI Chat Service

Fixes:
  1. GROQ_API_KEY read fresh from os.environ inside _get_groq() — not from
     module-level cached settings (which may be empty if env vars load after import).
  2. _groq singleton reset on failure — a transient error no longer locks out
     Groq for the entire process lifetime.
  3. Startup connectivity test logs clearly whether Groq is working.
"""

import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy Groq client — initialised on first chat request, not at import time
_groq       = None
_groq_tried = False   # True after first attempt, prevents retry spam


def _get_groq():
    global _groq, _groq_tried

    # Already working
    if _groq is not None:
        return _groq

    # Read key fresh from environment every call until we succeed
    # This ensures Render env vars (loaded after import) are picked up
    api_key = (
        os.environ.get("GROQ_API_KEY", "").strip()
        or _safe_settings_key()
    )

    if not api_key:
        if not _groq_tried:
            logger.warning("⚠️  GROQ_API_KEY not set — Bloom AI using fallback responses")
            _groq_tried = True
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        # Quick validation — list models to confirm the key works
        client.models.list()
        _groq = client
        logger.info("✅ Groq AI connected successfully")
        return _groq
    except Exception as e:
        # Don't cache None on failure — retry next request in case it was transient
        logger.error(f"❌ Groq connection failed: {e}")
        return None


def _safe_settings_key() -> str:
    """Read from pydantic settings as fallback — may be empty on first import."""
    try:
        from app.core.config import get_settings
        return get_settings().GROQ_API_KEY or ""
    except Exception:
        return ""


def _groq_model() -> str:
    model = os.environ.get("GROQ_MODEL", "").strip()
    if model:
        return model
    try:
        from app.core.config import get_settings
        return get_settings().GROQ_MODEL or "llama3-8b-8192"
    except Exception:
        return "llama3-8b-8192"


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Bloom 🌸, a warm, knowledgeable, and empathetic AI health companion
specialised in female menstrual health. Your role:

1. Answer questions about menstrual cycles, PCOS, PCOD, symptoms, mood, fertility, and wellness.
2. Reference the user's PERSONAL HEALTH DATA in the context block — give specific, personalised
   advice (e.g. "Since your cycle is 38 days, which is irregular...").
3. Be concise (3-5 sentences), warm, and non-judgmental.
4. For irregular cycles, acknowledge variability and explain why predictions may shift.
5. For high PCOS risk, gently suggest professional consultation while still being helpful.
6. NEVER diagnose. Always recommend a doctor for medical decisions.
7. If the user seems distressed, prioritise emotional support.

Format: Plain text with occasional emojis. No markdown headers. Short paragraphs.
"""


def build_context_block(user_data: dict) -> str:
    if not user_data:
        return ""

    lines = ["\n── USER HEALTH CONTEXT ──"]
    lines.append(f"Name: {user_data.get('name','User')}, Age: {user_data.get('age','?')}")
    lines.append(f"Average cycle: {user_data.get('avg_cycle_observed') or user_data.get('avg_cycle','?')} days (observed)")
    lines.append(f"BMI: {user_data.get('bmi','?')}")
    lines.append(f"Irregular cycles: {'Yes' if user_data.get('is_irregular') else 'No'}")
    lines.append(f"Cycles logged: {user_data.get('cycle_count', 0)}")
    lines.append(f"Sufficient data: {'Yes' if user_data.get('has_data') else 'No — new user'}")

    pred = user_data.get("prediction") or {}
    if pred and user_data.get("has_data"):
        nd = pred.get("next_period_date")
        if nd:
            lines.append(
                f"Next period: in {pred.get('days_until','?')} days ({nd}), "
                f"80% CI: {pred.get('confidence_lo','?')}–{pred.get('confidence_hi','?')} days"
            )
        lines.append(f"Current phase: {pred.get('cycle_phase','?')} (day {pred.get('cycle_day','?')})")
        fs = pred.get("fertile_start")
        fe = pred.get("fertile_end")
        if fs and fe:
            lines.append(f"Fertile window: {fs} to {fe}")
        pcos = pred.get("pcos_risk_level", "low")
        lines.append(f"PCOS risk: {pcos.upper()}")
        reasons = pred.get("pcos_reasons", [])
        if reasons:
            lines.append(f"PCOS indicators: {'; '.join(reasons)}")

    recent = user_data.get("recent_symptoms")
    if recent:
        lines.append(f"Most frequent symptoms (last 30 days): {', '.join(recent)}")

    mood = user_data.get("dominant_mood")
    if mood:
        lines.append(f"Dominant mood this month: {mood}")

    lines.append("── END CONTEXT ──")
    return "\n".join(lines)


# ── Main chat function ────────────────────────────────────────────────────────

async def ask_bloom(
    prompt: str,
    history: List[Dict[str, str]],
    user_data: Optional[dict] = None,
) -> str:
    groq = _get_groq()

    if groq:
        context_block = build_context_block(user_data or {})
        system_msg    = SYSTEM_PROMPT + context_block

        messages = [{"role": "system", "content": system_msg}]
        for h in history[-12:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = groq.chat.completions.create(
                model=_groq_model(),
                messages=messages,
                max_tokens=600,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            # Don't permanently disable — just fall back this request
            return smart_fallback(prompt, user_data or {})

    return smart_fallback(prompt, user_data or {})


# ── Smart fallback ────────────────────────────────────────────────────────────

def smart_fallback(prompt: str, user_data: dict) -> str:
    p          = prompt.lower()
    name       = user_data.get("name", "")
    avg_cycle  = user_data.get("avg_cycle_observed") or user_data.get("avg_cycle", 28)
    is_irr     = user_data.get("is_irregular", False)
    pred       = user_data.get("prediction") or {}
    pcos_level = pred.get("pcos_risk_level", "low")
    days_until = pred.get("days_until")
    has_data   = user_data.get("has_data", False)

    greeting = f"{name}, " if name else ""

    if any(w in p for w in ["next period", "when", "predict", "due", "how long"]):
        if has_data and days_until is not None:
            base = f"{greeting}based on your logged cycles, your next period is expected in about {days_until} days"
            if is_irr:
                base += (f". Your cycles vary, so the window is "
                         f"{pred.get('confidence_lo','?')}–{pred.get('confidence_hi','?')} days — "
                         "Bloom adjusts as you keep logging 🌸")
            return base + ". Keep logging daily for even more accuracy! 📅"
        return (f"{greeting}log your period start dates in Bloom and the app will predict your next cycle! "
                "The more you log, the more personalised and accurate your predictions become 📅")

    if any(w in p for w in ["pcos", "pcod", "polycystic", "irregular"]):
        base = ("PCOS (Polycystic Ovary Syndrome) is a hormonal condition that can cause irregular cycles, "
                "excess androgens (leading to acne, hair changes), and sometimes ovarian cysts. ")
        if pcos_level in ("moderate", "high"):
            base += (f"Based on your data, there are some indicators worth discussing with a gynecologist — "
                     "they can check hormone levels and do an ultrasound. ")
        base += "Diet rich in whole foods, regular moderate exercise, and stress management all help significantly. 🌸"
        return base

    if any(w in p for w in ["cramp", "pain", "dysmenorrhea", "hurt"]):
        return (f"{greeting}for cramp relief: 🌡️ heating pad on your lower abdomen, "
                "🧘 gentle yoga (child's pose, cat-cow), 💧 stay hydrated, "
                "🫚 anti-inflammatory foods like ginger and turmeric. "
                "Ibuprofen 400mg at the first sign works well. "
                "If cramps are debilitating, please see a doctor — it could be endometriosis. 💕")

    if any(w in p for w in ["mood", "sad", "anxious", "irritable", "emotional", "pms", "pmdd"]):
        phase = pred.get("cycle_phase", "") if pred else ""
        tip   = ""
        if phase in ("luteal", "menstrual"):
            tip = f" You're in the {phase} phase, when mood dips are most common due to hormone shifts."
        return (f"{greeting}mood changes across your cycle are driven by estrogen and progesterone affecting serotonin.{tip} "
                "Magnesium glycinate, regular sleep, and gentle movement are evidence-backed mood supporters. "
                "You're doing great by tracking 🌷")

    if any(w in p for w in ["fertile", "ovulation", "ovulate", "conception", "pregnant", "ttc"]):
        fs = pred.get("fertile_start") if pred else None
        fe = pred.get("fertile_end")   if pred else None
        window = f"{fs} to {fe}" if fs and fe else f"around day {int(avg_cycle) - 14} of your cycle"
        return (f"Your fertile window is roughly {window} 🌺 "
                "Ovulation typically occurs about 14 days before your next period. "
                "Tracking basal body temperature or using OPK strips adds even more precision!")

    if any(w in p for w in ["irregular", "missed", "late", "skip", "spotting"]):
        return (f"{greeting}irregular cycles are more common than people think — stress, weight changes, "
                "thyroid issues, or PCOS can all shift your timing. "
                f"Bloom's predictions account for variability in your ~{avg_cycle:.0f}-day average cycle. "
                "If you miss 3+ periods or cycles are consistently outside 21–45 days, see a gynecologist. 📋")

    if any(w in p for w in ["bloat", "water retention", "swollen"]):
        return ("Period bloating is caused by hormonal changes affecting water retention and digestion. "
                "Reduce sodium, drink more water, try magnesium-rich foods, and take gentle walks. "
                "It usually peaks on days 1–2 and fades quickly 🫖")

    if any(w in p for w in ["stress", "overwhelm", "burnout"]):
        return (f"{greeting}high stress raises cortisol, which can suppress LH and delay ovulation — "
                "actually shifting your cycle length. "
                "Mindfulness, 7–9h sleep, and even 10 minutes of daily deep breathing "
                "can meaningfully lower cortisol and help regulate your cycle 🧘")

    if any(w in p for w in ["phase", "cycle phase", "what phase", "where am i"]):
        phase = pred.get("cycle_phase", "") if pred else ""
        tips  = {
            "menstrual":  "rest, warmth, iron-rich foods, gentle movement 🌹",
            "follicular": "rising energy — great for strength training and new projects 🌱",
            "ovulation":  "peak energy and confidence — great for presentations and social events ✨",
            "luteal":     "wind-down time — prioritise sleep, reduce caffeine, eat magnesium-rich foods 🌙",
        }
        if phase in tips:
            return f"{greeting}you're in your {phase} phase — focus on {tips[phase]}"
        return f"{greeting}log your cycle dates and Bloom will identify your current phase! 🌸"

    # Generic helpful response — NOT the old "add your GROQ_API_KEY" message
    return (f"Hi{', ' + name if name else ''}! I'm Bloom 🌸 "
            "I can help with cycle questions, PCOS/PCOD info, cramp relief, mood support, "
            "fertile window timing, and more. What would you like to know?")