"""
Bloom AI Chat Service
=====================
Data-aware: injects the user's actual cycle data, symptoms, PCOS risk,
and prediction into every Groq prompt so answers are personalised.
Falls back to rich rule-based responses if Groq is unavailable.
"""

import logging
from typing import List, Dict, Optional
from app.core.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()

# Lazy-load Groq client
_groq = None

def _get_groq():
    global _groq
    if _groq is None and settings.GROQ_API_KEY:
        try:
            from groq import Groq
            _groq = Groq(api_key=settings.GROQ_API_KEY)
            logger.info("✅ Groq AI connected")
        except Exception as e:
            logger.warning(f"Groq unavailable: {e}")
    return _groq


SYSTEM_PROMPT = """You are Bloom 🌸, a warm, knowledgeable, and empathetic AI health companion
specialised in female menstrual health. Your role:

1. Answer questions about menstrual cycles, PCOS, PCOD, symptoms, mood, fertility, and wellness.
2. Reference the user's PERSONAL HEALTH DATA provided in the context block below — use it to give 
   specific, personalised advice (e.g. "Since your cycle is 38 days, which is irregular...").
3. Be concise (3-5 sentences for most replies), warm, and non-judgmental.
4. For irregular cycles, always acknowledge the variability and explain why predictions may shift.
5. For high PCOS risk, gently suggest professional consultation while still being helpful.
6. NEVER diagnose. Always recommend a doctor for medical decisions.
7. If the user seems distressed, prioritise emotional support before information.

Format: Plain text with occasional emojis. No markdown headers. Short paragraphs.
"""


def build_context_block(user_data: dict) -> str:
    """Build a personalised context string injected into every system message."""
    if not user_data:
        return ""

    lines = ["\n── USER HEALTH CONTEXT ──"]

    # Basic profile
    lines.append(f"Name: {user_data.get('name','User')}, Age: {user_data.get('age','?')}")
    lines.append(f"Average cycle: {user_data.get('avg_cycle','?')} days")
    lines.append(f"BMI: {user_data.get('bmi','?')}")
    lines.append(f"Irregular cycles: {'Yes' if user_data.get('is_irregular') else 'No'}")

    # Prediction
    pred = user_data.get("prediction")
    if pred:
        lines.append(f"Next period prediction: in {pred.get('days_until','?')} days "
                     f"({pred.get('next_period_date','?')}), "
                     f"80% confidence window: {pred.get('confidence_lo','?')}–{pred.get('confidence_hi','?')} days")
        lines.append(f"Current cycle phase: {pred.get('cycle_phase','?')} (day {pred.get('cycle_day','?')})")
        lines.append(f"Fertile window: {pred.get('fertile_start','?')} to {pred.get('fertile_end','?')}")
        pcos = pred.get("pcos_risk_level","low")
        lines.append(f"PCOS risk: {pcos.upper()}")
        if pred.get("pcos_reasons"):
            lines.append(f"PCOS indicators: {'; '.join(pred['pcos_reasons'])}")

    # Recent symptoms
    recent = user_data.get("recent_symptoms")
    if recent:
        lines.append(f"Most frequent recent symptoms: {', '.join(recent)}")

    # Mood trend
    mood = user_data.get("dominant_mood")
    if mood:
        lines.append(f"Dominant mood this month: {mood}")

    lines.append("── END CONTEXT ──")
    return "\n".join(lines)


async def ask_bloom(
    prompt: str,
    history: List[Dict[str, str]],
    user_data: Optional[dict] = None,
) -> str:
    """
    Query Bloom AI with full user context.
    Falls back to rule-based smart_fallback if Groq is unavailable.
    """
    groq = _get_groq()
    if groq:
        context_block = build_context_block(user_data or {})
        system_msg    = SYSTEM_PROMPT + context_block

        messages = [{"role": "system", "content": system_msg}]
        # Keep last 12 turns
        for h in history[-12:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = groq.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                max_tokens=600,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq error: {e}")

    return smart_fallback(prompt, user_data or {})


def smart_fallback(prompt: str, user_data: dict) -> str:
    """Rich rule-based fallback with personalisation where possible."""
    p          = prompt.lower()
    name       = user_data.get("name", "")
    avg_cycle  = user_data.get("avg_cycle", 28)
    is_irr     = user_data.get("is_irregular", False)
    pred       = user_data.get("prediction", {})
    pcos_level = pred.get("pcos_risk_level", "low") if pred else "low"
    days_until = pred.get("days_until") if pred else None

    greeting   = f"{name}, " if name else ""

    # Next period
    if any(w in p for w in ["next period","when","predict","due","how long"]):
        if days_until is not None:
            base = f"{greeting}based on your data, your next period is expected in about {days_until} days"
            if is_irr:
                base += (f". Since your cycles can vary, the window is "
                         f"{pred.get('confidence_lo','?')}–{pred.get('confidence_hi','?')} days — "
                         "irregular cycles are totally normal and Bloom adjusts as you keep logging 🌸")
            return base + ". Keep logging daily for even more accurate predictions! 📅"
        return ("Your next period prediction shows on your Bloom dashboard 📅 "
                "The more you log, the more personalised the prediction becomes!")

    # PCOS
    if any(w in p for w in ["pcos","pcod","polycystic","irregular"]):
        base = ("PCOS (Polycystic Ovary Syndrome) is a hormonal condition that can cause irregular cycles, "
                "excess androgens (leading to acne, hair changes), and sometimes ovarian cysts. ")
        if pcos_level in ("moderate","high") and avg_cycle:
            base += (f"Your cycle of ~{avg_cycle:.0f} days with some variation suggests it's worth "
                     "discussing with a gynecologist — they can check hormone levels and do an ultrasound. ")
        base += "Diet rich in whole foods, regular moderate exercise, and stress management all help significantly. 🌸"
        return base

    # Cramps / pain
    if any(w in p for w in ["cramp","pain","dysmenorrhea","hurt"]):
        return (f"{greeting}cramp relief toolkit: 🌡️ heating pad on lower abdomen (best evidence!), "
                "🧘 gentle yoga (child's pose, cat-cow), 💧 hydration, 🫚 anti-inflammatory foods "
                "(ginger, turmeric, omega-3). Ibuprofen 400mg taken at the first sign works well. "
                "If cramps are debilitating or worsening, please see a doctor — it could be endometriosis. 💕")

    # Mood / emotions
    if any(w in p for w in ["mood","sad","anxious","irritable","emotional","pms","pmdd"]):
        phase = pred.get("cycle_phase","") if pred else ""
        tip   = ""
        if phase in ("luteal","menstrual"):
            tip = f" You're currently in the {phase} phase, when mood dips are most common due to progesterone/estrogen drops."
        return (f"{greeting}mood changes across your cycle are driven by shifting hormones affecting serotonin and dopamine.{tip} "
                "Tracking your mood in Bloom helps you anticipate low days. Magnesium glycinate, regular sleep, "
                "and gentle movement are evidence-backed mood supporters. You're doing great by tracking 🌷")

    # Fertile window
    if any(w in p for w in ["fertile","ovulation","ovulate","conception","pregnant","ttc"]):
        fs = pred.get("fertile_start","?") if pred else "?"
        fe = pred.get("fertile_end","?")   if pred else "?"
        return (f"Your fertile window is roughly {fs} to {fe} 🌺 "
                "Ovulation typically occurs about 14 days before your next period — "
                f"with a {avg_cycle:.0f}-day cycle, that's around day {int(avg_cycle)-14}. "
                "Bloom marks this in mauve on your calendar. Tracking basal body temperature "
                "or using OPK strips adds even more precision for conception planning!")

    # Irregular
    if any(w in p for w in ["irregular","missed","late","skip","spotting"]):
        return (f"{greeting}irregular cycles are more common than people think — stress, weight changes, "
                "thyroid issues, PCOS, or even intense exercise can shift your timing. "
                f"Your Bloom predictions account for variability in your ~{avg_cycle:.0f}-day average cycle. "
                "If you miss 3+ periods or have cycles consistently outside 21-45 days, see a gynecologist. 📋")

    # Bloating
    if any(w in p for w in ["bloat","water retention","swollen"]):
        return ("Period bloating is caused by prostaglandins and hormonal water retention. "
                "Reduce sodium and processed foods a week before your period, drink more water (paradoxically helps!), "
                "try magnesium-rich foods, and gentle walks. It usually peaks day 1-2 and fades quickly 🫖")

    # Stress
    if any(w in p for w in ["stress","work","overwhelm","burnout"]):
        return (f"{greeting}high stress raises cortisol, which can suppress LH and delay ovulation — "
                "actually shifting your cycle length! This is a real, documented effect. "
                "Mindfulness, adequate sleep (7-9h), and even 10 minutes of daily deep breathing "
                "can meaningfully lower cortisol and help regulate your cycle 🧘")

    # Phase-based advice
    phase = pred.get("cycle_phase","") if pred else ""
    if any(w in p for w in ["phase","cycle phase","where am i","what phase"]):
        tips  = {
            "menstrual":  "rest, warmth, iron-rich foods, gentle movement. Your body is doing hard work! 🌹",
            "follicular": "rising energy! Great for strength training, creative projects, socialising. 🌱",
            "ovulation":  "peak energy and confidence. Excellent for presentations, dates, intense workouts. ✨",
            "luteal":     "wind-down phase. Prioritise sleep, reduce caffeine, eat complex carbs and magnesium. 🌙",
        }
        if phase in tips:
            return f"{greeting}you're in your {phase} phase — focus on {tips[phase]}"

    return (f"Hi{', '+name if name else ''}! I'm Bloom 🌸 I'm here to help with menstrual health, "
            "PCOS, symptom management, mood support, and more. "
            "For full personalised AI responses, add your GROQ_API_KEY to .env. "
            "What would you like to know?")
