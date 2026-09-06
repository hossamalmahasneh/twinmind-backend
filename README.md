# TwinMind Backend

FastAPI backend for the TwinMind cognitive journaling MVP.

## MVP capabilities

- Unified entries: journal, email, meeting minutes, thoughts, emotions, insights, voice-note transcripts, calendar context and other impactful entries.
- Automatic Arabic/English detection.
- Sentiment and emotion analysis.
- Constructive negativity-response suggestions.
- Low-risk well-being nudges.
- Weekly reflections.
- Memory recall over stored entries.
- OpenAI-powered analysis when `OPENAI_API_KEY` is configured, with a deterministic fallback for local demos.

> TwinMind is a reflection and decision-support tool, not a medical or mental-health diagnostic system.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

Set `OPENAI_API_KEY` in your environment for LLM analysis. Without it, the API remains runnable using its basic fallback analyzer.

## API

- `GET /health`
- `POST /entries`
- `GET /entries`
- `POST /reflection`
- `POST /recall`

Interactive API docs are available at `/docs` while the server is running.

## Production note

SQLite is intentionally used for the MVP. Before multi-user production deployment, migrate persistence to PostgreSQL/Supabase, add authentication and per-user row-level isolation, encrypt sensitive data, and implement explicit connector consent/retention controls.
