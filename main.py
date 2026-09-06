import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DB_PATH = os.getenv("TWINMIND_DB", "twinmind.db")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
client = OpenAI() if OpenAI and os.getenv("OPENAI_API_KEY") else None

app = FastAPI(title="TwinMind API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENTRY_TYPES = Literal[
    "journal", "email", "meeting_minutes", "thought", "emotion",
    "insight", "voice_note", "calendar", "other"
]

class EntryCreate(BaseModel):
    content: str = Field(min_length=1, max_length=50000)
    entry_type: ENTRY_TYPES = "journal"
    source: str = "manual"
    title: str | None = None
    preferred_language: Literal["en", "ar"] | None = None

class RecallRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    language: Literal["en", "ar"] = "en"

class ReflectionRequest(BaseModel):
    language: Literal["en", "ar"] = "en"
    days: int = Field(default=7, ge=1, le=30)


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                entry_type TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT,
                language TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                emotions TEXT,
                tags TEXT,
                wellbeing_suggestion TEXT,
                negativity_suggestion TEXT
            )
            """
        )


init_db()


def detect_language(text: str) -> str:
    return "ar" if re.search(r"[\u0600-\u06FF]", text) else "en"


def heuristic_analysis(text: str, language: str) -> dict:
    lower = text.lower()
    negative_en = ["stress", "stressed", "anxious", "angry", "tired", "frustrated", "worried", "overwhelmed", "bad", "fail"]
    positive_en = ["great", "good", "happy", "relieved", "success", "excellent", "grateful", "positive", "progress"]
    negative_ar = ["قلق", "متوتر", "غاضب", "تعب", "مرهق", "محبط", "سيئ", "فشل", "ضغط"]
    positive_ar = ["سعيد", "ممتاز", "جيد", "مرتاح", "نجاح", "ممتن", "تقدم", "إيجابي"]
    neg = sum(1 for w in (negative_ar if language == "ar" else negative_en) if w in text if language == "ar") if language == "ar" else sum(1 for w in negative_en if w in lower)
    pos = sum(1 for w in (positive_ar if language == "ar" else positive_en) if w in text if language == "ar") if language == "ar" else sum(1 for w in positive_en if w in lower)
    score = max(-1.0, min(1.0, (pos - neg) / max(1, pos + neg)))
    label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    if language == "ar":
        emotions = ["ضغط"] if neg else (["إيجابية"] if pos else ["محايد"])
        wellbeing = "خصص فترة قصيرة للتعافي أو إعادة ترتيب الأولويات إذا كان الضغط مستمراً." if neg else "حافظ على الإيقاع الحالي وسجل ما ساعدك اليوم."
        negativity = "قبل الرد على أي تواصل سلبي، افصل بين الوقائع والانطباع واكتب رداً هادئاً يركز على الحل." if neg else "لا توجد إشارة سلبية قوية حالياً؛ استمر في مراقبة الأنماط دون المبالغة في تفسيرها."
    else:
        emotions = ["stress"] if neg else (["positive"] if pos else ["neutral"])
        wellbeing = "Consider a short recovery block or reprioritization if the pressure persists." if neg else "Maintain the current rhythm and note what contributed to a constructive day."
        negativity = "Before responding to negative communication, separate facts from interpretation and draft a calm, solution-focused response." if neg else "No strong negative signal is present; keep monitoring patterns without over-interpreting them."
    return {
        "summary": text[:280] + ("..." if len(text) > 280 else ""),
        "sentiment_label": label,
        "sentiment_score": score,
        "emotions": emotions,
        "tags": [],
        "wellbeing_suggestion": wellbeing,
        "negativity_suggestion": negativity,
    }


def ai_analysis(text: str, language: str, entry_type: str) -> dict:
    if not client:
        return heuristic_analysis(text, language)
    output_language = "Arabic" if language == "ar" else "English"
    prompt = f"""You are the TwinMind analysis pipeline. Analyze the entry safely and conservatively.
Entry type: {entry_type}
Output language: {output_language}

Return ONLY valid JSON with these keys:
summary (string), sentiment_label (positive|neutral|negative), sentiment_score (number -1 to 1),
emotions (array of short strings), tags (array of short strings), wellbeing_suggestion (string), negativity_suggestion (string).

Rules:
- Preserve the user's meaning and language.
- Sentiment is not a mental-health diagnosis.
- Avoid medical claims, diagnosis, or alarmist language.
- Well-being suggestions must be low-risk and practical.
- Negativity suggestions should reduce reactive communication, encourage fact-checking, reframing, boundaries, or constructive dialogue; never suppress legitimate concerns.

ENTRY:
{text}
"""
    try:
        response = client.responses.create(model=MODEL, input=prompt)
        data = json.loads(response.output_text)
        return data
    except Exception:
        return heuristic_analysis(text, language)


def row_to_dict(row):
    item = dict(row)
    for key in ("emotions", "tags"):
        try:
            item[key] = json.loads(item.get(key) or "[]")
        except Exception:
            item[key] = []
    return item


@app.get("/")
def root():
    return {"name": "TwinMind API", "version": "0.1.0", "status": "ok", "ai_enabled": bool(client)}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/entries")
def create_entry(entry: EntryCreate):
    language = entry.preferred_language or detect_language(entry.content)
    analysis = ai_analysis(entry.content, language, entry.entry_type)
    entry_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        conn.execute(
            """INSERT INTO entries
            (id, created_at, entry_type, source, title, language, content, summary,
             sentiment_label, sentiment_score, emotions, tags, wellbeing_suggestion, negativity_suggestion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry_id, created_at, entry.entry_type, entry.source, entry.title, language,
                entry.content, analysis.get("summary", ""), analysis.get("sentiment_label", "neutral"),
                float(analysis.get("sentiment_score", 0)), json.dumps(analysis.get("emotions", []), ensure_ascii=False),
                json.dumps(analysis.get("tags", []), ensure_ascii=False), analysis.get("wellbeing_suggestion", ""),
                analysis.get("negativity_suggestion", ""),
            ),
        )
    return {"id": entry_id, "created_at": created_at, "language": language, **analysis}


@app.get("/entries")
def list_entries(limit: int = 50):
    limit = max(1, min(limit, 200))
    with connect() as conn:
        rows = conn.execute("SELECT * FROM entries ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [row_to_dict(r) for r in rows]


@app.post("/reflection")
def reflection(request: ReflectionRequest):
    since = (datetime.now(timezone.utc) - timedelta(days=request.days)).isoformat()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM entries WHERE created_at >= ? ORDER BY created_at", (since,)).fetchall()
    entries = [row_to_dict(r) for r in rows]
    if not entries:
        return {"reflection": "لا توجد مدخلات كافية بعد." if request.language == "ar" else "There are not enough entries yet.", "entry_count": 0}
    avg = sum(float(e.get("sentiment_score") or 0) for e in entries) / len(entries)
    context = "\n\n".join(f"[{e['entry_type']}] {e.get('summary') or e['content'][:500]}" for e in entries[-30:])
    if client:
        lang = "Arabic" if request.language == "ar" else "English"
        prompt = f"""Create a concise TwinMind weekly reflection in {lang} from the entries below.
Include: major themes, emotional/sentiment pattern, decisions or commitments, constructive risks to watch, and 2 practical well-being/communication suggestions.
Do not diagnose mental-health conditions. Distinguish observations from inference.
Entries:\n{context}"""
        try:
            text = client.responses.create(model=MODEL, input=prompt).output_text
        except Exception:
            text = ""
    else:
        text = ""
    if not text:
        if request.language == "ar":
            text = f"تم تحليل {len(entries)} مدخلاً. متوسط مؤشر المشاعر {avg:.2f}. راقب المواضيع المتكررة، وافصل بين الوقائع والانطباعات قبل اتخاذ قرارات مهمة، وخصص فترات تعافٍ قصيرة بعد الأحداث عالية الضغط."
        else:
            text = f"Analyzed {len(entries)} entries. Average sentiment score: {avg:.2f}. Watch recurring themes, separate facts from interpretation before important decisions, and schedule short recovery periods after high-pressure events."
    return {"reflection": text, "entry_count": len(entries), "average_sentiment": round(avg, 3)}


@app.post("/recall")
def recall(request: RecallRequest):
    tokens = [t for t in re.findall(r"\w+", request.query.lower()) if len(t) > 2][:8]
    with connect() as conn:
        rows = conn.execute("SELECT * FROM entries ORDER BY created_at DESC LIMIT 200").fetchall()
    entries = [row_to_dict(r) for r in rows]
    ranked = []
    for e in entries:
        hay = (e.get("content") or "").lower() + " " + (e.get("summary") or "").lower()
        score = sum(hay.count(t) for t in tokens)
        if score:
            ranked.append((score, e))
    ranked.sort(key=lambda x: x[0], reverse=True)
    matches = [e for _, e in ranked[:8]] or entries[:5]
    if not matches:
        return {"answer": "لا توجد ذاكرة كافية بعد." if request.language == "ar" else "There is not enough memory yet.", "matches": []}
    context = "\n".join(f"{e['created_at']} | {e['entry_type']} | {e.get('summary') or e['content'][:700]}" for e in matches)
    answer = None
    if client:
        lang = "Arabic" if request.language == "ar" else "English"
        prompt = f"""Answer the user's recall question in {lang} using ONLY the supplied TwinMind memory excerpts.
If the memories do not support a conclusion, say so clearly. Distinguish facts from inferred patterns.
Question: {request.query}\nMemories:\n{context}"""
        try:
            answer = client.responses.create(model=MODEL, input=prompt).output_text
        except Exception:
            pass
    if not answer:
        answer = ("أقرب المدخلات ذات الصلة معروضة أدناه." if request.language == "ar" else "The most relevant stored entries are shown below.")
    return {"answer": answer, "matches": matches}
