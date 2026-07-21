# 🌸 Bloom — Menstrual Health Tracker

FastAPI + async SQLAlchemy + Supabase/SQLite + Groq AI

---

## Quick Start (Local Dev)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env — set SECRET_KEY (see file for command)
# Set GROQ_API_KEY if you want AI chat (free at console.groq.com)

# 3. Create the database
python scripts/db_init.py

# 4. (Optional) seed sample data
python scripts/db_seed.py
# → priya@example.com / password123
# → aisha@example.com / password123

# 5. Run
uvicorn main:app --reload
# Open: http://localhost:8000
```

---

## Production (Render + Supabase)

1. Create a project at supabase.com
2. Get your connection string: **Settings → Database → Session mode**
3. Change `postgresql://` → `postgresql+asyncpg://`
4. Set in Render environment variables:
   ```
   DATABASE_URL = postgresql+asyncpg://...
   SECRET_KEY   = <random 64 char hex>
   DEBUG        = false
   GROQ_API_KEY = <your key>
   ```
5. Run migrations once: `DATABASE_URL=... alembic upgrade head`
6. Push to GitHub → Render auto-deploys

---

## Project Structure

```
bloom/
├── main.py                        ← FastAPI app entry point
├── requirements.txt
├── alembic.ini                    ← Migration config
├── .env.example                   ← Copy to .env
│
├── app/
│   ├── api/
│   │   ├── auth.py                ← Register / Login / Logout
│   │   ├── chat.py                ← POST /api/chat
│   │   └── dashboard.py           ← Dashboard + all routes
│   ├── core/
│   │   ├── config.py              ← Settings from .env
│   │   ├── database.py            ← Async SQLAlchemy engine
│   │   └── security.py            ← bcrypt hashing
│   ├── models/db_models.py        ← ORM: User, CycleLog, ChatMessage
│   ├── schemas/schemas.py         ← Pydantic validation
│   ├── services/
│   │   ├── ai_service.py          ← Groq AI + fallback
│   │   └── cycle_service.py       ← Predictions, charts, calendar
│   ├── ml/
│   │   ├── feature_spec.py        ← Single source of truth: encodings + feature order
│   │   ├── predictor.py           ← ML model + PCOS risk
│   │   ├── train_model.py         ← Model training script (grouped validation)
│   │   └── generate_dataset.py    ← Synthetic training data generator
│   └── utils/utils.py             ← Shared flash + auth helpers
│
├── migrations/                    ← Alembic migrations
├── scripts/
│   ├── db_init.py                 ← Quick DB setup
│   └── db_seed.py                 ← Dev seed data
├── templates/                     ← Jinja2 HTML templates
├── static/                        ← CSS, JS, images
└── models_ml/                     ← bloom_model.pkl (gitignored), ML_NOTES.md
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET/POST | `/register` | Register |
| GET/POST | `/login` | Login |
| GET | `/logout` | Logout |
| GET | `/dashboard` | Main dashboard |
| GET/POST | `/log` | Daily log |
| GET | `/logs` | All logs |
| POST | `/logs/delete/{id}` | Delete log |
| GET | `/kits` | Kits page |
| POST | `/api/chat` | AI chat |
| POST | `/api/predict` | ML prediction |
| GET | `/api/chart` | Chart data |
| GET | `/api/calendar` | Calendar events |

---

## Train the ML Model

```bash
# Option A — use your own data: place menstrual_cycle_dataset.csv in models_ml/
#   (see app/ml/train_model.py -> load_and_engineer() for the expected schema)

# Option B — no dataset handy: generate a documented synthetic one
python -m app.ml.generate_dataset
#   -> models_ml/menstrual_cycle_dataset.csv

python -m app.ml.train_model
# Saves models_ml/bloom_model.pkl, prints grouped-CV and held-out-user MAE
```
Without the model, heuristic fallback runs automatically.

**Note:** v1 of this model had a data-leakage bug (the current cycle's own
completed length was used as a training feature, which isn't knowable until
the cycle is over — see `models_ml/ML_NOTES.md` for the full diagnosis,
the numbers that exposed it, and the fix). `app/ml/feature_spec.py` is now
the single source of truth for the feature contract used by both training
and inference.

---

## Security Notes

- Passwords hashed with **bcrypt** (not SHA-256)
- `SECRET_KEY` must be set — app refuses to start in production without it
- `.env` and `models_ml/bloom_model.pkl` are gitignored
- All DB deletes are CASCADE-safe (delete user → deletes all logs + messages)
