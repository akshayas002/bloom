# 🌸 Bloom  — Menstrual Health AI Assistant

A full-stack **FastAPI + PostgreSQL** web app for intelligent menstrual health tracking — with PCOS detection, irregular-cycle support, AI-powered chat, and an improved ML prediction model.

---

## ✨ Features

| Feature | Detail |
|---|---|
| ⚡ **FastAPI backend** | Async Python, auto-generated `/api/docs`, faster than Flask |
| 🐘 **PostgreSQL support** | Production-ready DB with async SQLAlchemy ORM |
| 🤖 **Data-aware AI chat** | Bloom AI knows your cycle phase, PCOS risk, symptoms, next prediction |
| 🔮 **Improved ML model** | Stacked GB + RF → Ridge, 15 features, confidence intervals, irregular-specific sub-model |
| 📊 **PCOS risk scoring** | Weighted multi-factor score with plain-English reasons |
| 📅 **Irregular cycle support** | Wider confidence windows, irregular-specific model, adaptive prediction |
| 🩺 **Confidence intervals** | 80% prediction window shown on dashboard and calendar |
| 🌺 **Fertile window** | Dynamically computed per-user based on actual cycle length |

---

## 🏗️ Project Structure

```
bloom/
├── main.py                        ← FastAPI app + middleware + lifespan
├── requirements.txt
├── Procfile                       ← Railway/Heroku deploy
├── .env.example
│
├── app/
│   ├── api/
│   │   ├── auth.py                ← /register /login /logout
│   │   ├── dashboard.py           ← /dashboard /log /logs /api/predict /api/chart /api/calendar
│   │   └── chat.py                ← /api/chat (data-aware AI)
│   ├── core/
│   │   ├── config.py              ← Pydantic Settings (env vars)
│   │   ├── database.py            ← Async SQLAlchemy engine + Base
│   │   └── security.py            ← Password hashing
│   ├── models/
│   │   └── db_models.py           ← User, CycleLog, ChatMessage ORM models
│   ├── schemas/
│   │   └── schemas.py             ← Pydantic request/response schemas
│   ├── services/
│   │   ├── cycle_service.py       ← Prediction, calendar, chart, AI context
│   │   └── ai_service.py          ← Data-aware Bloom AI + smart fallback
│   └── ml/
│       ├── train_model.py         ← Re-train the ML model
│       └── predictor.py           ← Prediction service (wraps trained model)
│
├── models_ml/
│   ├── bloom_model.pkl            ← Trained stacked ensemble
│   ├── cycle_model.pkl            ← Original model (backup)
│   └── menstrual_cycle_dataset.csv
│
├── templates/
│   ├── base.html                  ← Navbar + chatbot widget
│   ├── auth.html                  ← Login & Register
│   ├── dashboard.html             ← Full dashboard
│   ├── log.html                   ← Daily log entry
│   ├── logs.html                  ← History view
│   └── 404.html
│
└── static/css/
    └── bloom.css
```

---

## 🚀 Quick Start (Local — SQLite, zero config)

```bash
cd bloom
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Optionally add GROQ_API_KEY for full AI chat

python main.py
# → http://localhost:8000
# → API docs: http://localhost:8000/api/docs
```

---

## 🐘 Switch to PostgreSQL

1. Install Postgres and create a database:
   ```sql
   CREATE DATABASE bloom_db;
   ```

2. Install the async driver:
   ```bash
   pip install asyncpg
   ```

3. Update `.env`:
   ```
   DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/bloom_db
   ```

4. Restart — tables are auto-created on startup via `create_tables()`.

---

## 🌐 Deploy to Railway (recommended — free tier)

1. Push to GitHub
2. [railway.app](https://railway.app) → New Project → from GitHub
3. Add a **PostgreSQL** plugin → Railway auto-sets `DATABASE_URL`
4. Add env vars: `SECRET_KEY`, `GROQ_API_KEY`
5. Railway detects `Procfile` — you're live! 🎉

---

## 🌐 Deploy to Render

1. New Web Service → connect repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker`
4. Add env vars in dashboard

---

## 🤖 Re-train the ML Model

```bash
python -m app.ml.train_model
# or
python app/ml/train_model.py
```

Outputs `models_ml/bloom_model.pkl` with:
- **Stacked ensemble**: GradientBoosting + RandomForest → Ridge meta-learner
- **15 features** including cycle history, variation, BMI, irregularity flags
- **Quantile models** for 10th/90th percentile confidence intervals
- **Irregular sub-model** trained specifically on cycles >35 days

---

## 🤖 AI Chatbot

Bloom uses **Groq LLaMA-3 8B** and injects your actual data into every prompt:
- Current cycle phase and day
- Next period prediction with confidence window
- PCOS risk level and reasons
- Most frequent recent symptoms
- Dominant mood this month

Get a free key at [console.groq.com](https://console.groq.com) and add it to `.env`.
Without a key, Bloom falls back to rich personalised rule-based responses.

---

## 🎨 Design System

| Token | Value |
|---|---|
| Coral primary | `#FF6F61` |
| Mauve accent | `#C08081` |
| Blush background | `#fef6f6` |
| Heading font | Quicksand |
| Body font | Poppins |
