import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

DB_PATH = os.getenv("TWINMIND_DB", "twinmind.db")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

app = FastAPI(title="TwinMind API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

ENTRY_TYPES = Literal["journal","email","meeting_minutes","thought","emotion","insight","voice_note","calendar","other"]

class EntryCreate(BaseModel):
    content: str = Field(min_length=1, max_length=50000)
    entry_type: ENTRY_TYPES = "journal"
    source: str = "manual"
    title: str | None = None
    preferred_language: Literal["en","ar"] | None = None
    openai_api_key: str | None = Field(default=None, exclude=True)

class RecallRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    language: Literal["en","ar"] = "en"
    openai_api_key: str | None = Field(default=None, exclude=True)

class ReflectionRequest(BaseModel):
    language: Literal["en","ar"] = "en"
    days: int = Field(default=7, ge=1, le=30)
    openai_api_key: str | None = Field(default=None, exclude=True)

def get_client(key=None):
    api_key = (key or os.getenv("OPENAI_API_KEY") or "").strip()
    return OpenAI(api_key=api_key) if OpenAI and api_key else None

def connect():
    conn=sqlite3.connect(DB_PATH); conn.row_factory=sqlite3.Row; return conn

def init_db():
    with connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, entry_type TEXT NOT NULL, source TEXT NOT NULL, title TEXT, language TEXT NOT NULL, content TEXT NOT NULL, summary TEXT, sentiment_label TEXT, sentiment_score REAL, emotions TEXT, tags TEXT, wellbeing_suggestion TEXT, negativity_suggestion TEXT)""")
init_db()

def detect_language(text): return "ar" if re.search(r"[\u0600-\u06FF]",text) else "en"

def heuristic_analysis(text, language):
    lower=text.lower(); neg_en=["stress","stressed","anxious","angry","tired","frustrated","worried","overwhelmed","bad","fail"]; pos_en=["great","good","happy","relieved","success","excellent","grateful","positive","progress"]
    neg_ar=["قلق","متوتر","غاضب","تعب","مرهق","محبط","سيئ","فشل","ضغط"]; pos_ar=["سعيد","ممتاز","جيد","مرتاح","نجاح","ممتن","تقدم","إيجابي"]
    neg=sum(w in text for w in neg_ar) if language=="ar" else sum(w in lower for w in neg_en); pos=sum(w in text for w in pos_ar) if language=="ar" else sum(w in lower for w in pos_en)
    score=max(-1.0,min(1.0,(pos-neg)/max(1,pos+neg))); label="positive" if score>.2 else "negative" if score<-.2 else "neutral"
    if language=="ar":
        emotions=["ضغط"] if neg else (["إيجابية"] if pos else ["محايد"]); wellbeing="خصص فترة قصيرة للتعافي أو إعادة ترتيب الأولويات إذا كان الضغط مستمراً." if neg else "حافظ على الإيقاع الحالي وسجل ما ساعدك اليوم."; negativity="قبل الرد على أي تواصل سلبي، افصل بين الوقائع والانطباع واكتب رداً هادئاً يركز على الحل." if neg else "لا توجد إشارة سلبية قوية حالياً؛ استمر في مراقبة الأنماط دون المبالغة في تفسيرها."
    else:
        emotions=["stress"] if neg else (["positive"] if pos else ["neutral"]); wellbeing="Consider a short recovery block or reprioritization if the pressure persists." if neg else "Maintain the current rhythm and note what contributed to a constructive day."; negativity="Before responding to negative communication, separate facts from interpretation and draft a calm, solution-focused response." if neg else "No strong negative signal is present; keep monitoring patterns without over-interpreting them."
    return {"summary":text[:280]+("..." if len(text)>280 else ""),"sentiment_label":label,"sentiment_score":score,"emotions":emotions,"tags":[],"wellbeing_suggestion":wellbeing,"negativity_suggestion":negativity}

def ai_analysis(text,language,entry_type,key=None):
    client=get_client(key)
    if not client: return heuristic_analysis(text,language)
    outlang="Arabic" if language=="ar" else "English"
    prompt=f'''You are the TwinMind analysis pipeline. Analyze safely and conservatively. Entry type: {entry_type}. Output language: {outlang}. Return ONLY valid JSON with keys summary, sentiment_label (positive|neutral|negative), sentiment_score (-1 to 1), emotions (array), tags (array), wellbeing_suggestion, negativity_suggestion. Sentiment is not a diagnosis. Avoid medical claims. Suggestions must be low-risk, practical and constructive. ENTRY:\n{text}'''
    try: return json.loads(client.responses.create(model=MODEL,input=prompt).output_text)
    except Exception: return heuristic_analysis(text,language)

def row_to_dict(row):
    item=dict(row)
    for key in ("emotions","tags"):
        try: item[key]=json.loads(item.get(key) or "[]")
        except Exception: item[key]=[]
    return item

@app.get("/")
def root(): return {"name":"TwinMind API","version":"0.2.0","status":"ok","server_ai_enabled":bool(os.getenv("OPENAI_API_KEY"))}
@app.get("/health")
def health(): return {"status":"healthy"}

@app.post("/entries")
def create_entry(entry: EntryCreate):
    language=entry.preferred_language or detect_language(entry.content); analysis=ai_analysis(entry.content,language,entry.entry_type,entry.openai_api_key); entry_id=str(uuid.uuid4()); created_at=datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        conn.execute("INSERT INTO entries (id,created_at,entry_type,source,title,language,content,summary,sentiment_label,sentiment_score,emotions,tags,wellbeing_suggestion,negativity_suggestion) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(entry_id,created_at,entry.entry_type,entry.source,entry.title,language,entry.content,analysis.get("summary",""),analysis.get("sentiment_label","neutral"),float(analysis.get("sentiment_score",0)),json.dumps(analysis.get("emotions",[]),ensure_ascii=False),json.dumps(analysis.get("tags",[]),ensure_ascii=False),analysis.get("wellbeing_suggestion",""),analysis.get("negativity_suggestion","")))
    return {"id":entry_id,"created_at":created_at,"language":language,**analysis}

@app.get("/entries")
def list_entries(limit:int=50):
    with connect() as conn: rows=conn.execute("SELECT * FROM entries ORDER BY created_at DESC LIMIT ?",(max(1,min(limit,200)),)).fetchall()
    return [row_to_dict(r) for r in rows]

@app.post("/reflection")
def reflection(request: ReflectionRequest):
    since=(datetime.now(timezone.utc)-timedelta(days=request.days)).isoformat()
    with connect() as conn: rows=conn.execute("SELECT * FROM entries WHERE created_at >= ? ORDER BY created_at",(since,)).fetchall()
    entries=[row_to_dict(r) for r in rows]
    if not entries: return {"reflection":"لا توجد مدخلات كافية بعد." if request.language=="ar" else "There are not enough entries yet.","entry_count":0}
    avg=sum(float(e.get("sentiment_score") or 0) for e in entries)/len(entries); context="\n\n".join(f"[{e['entry_type']}] {e.get('summary') or e['content'][:500]}" for e in entries[-30:]); client=get_client(request.openai_api_key); text=""
    if client:
        lang="Arabic" if request.language=="ar" else "English"; prompt=f"Create a concise TwinMind weekly reflection in {lang}. Include major themes, emotional/sentiment pattern, decisions/commitments, constructive risks, and 2 practical well-being/communication suggestions. Do not diagnose. Distinguish observations from inference. Entries:\n{context}"
        try: text=client.responses.create(model=MODEL,input=prompt).output_text
        except Exception: pass
    if not text: text=(f"تم تحليل {len(entries)} مدخلاً. متوسط مؤشر المشاعر {avg:.2f}. راقب المواضيع المتكررة وافصل بين الوقائع والانطباعات قبل القرارات المهمة." if request.language=="ar" else f"Analyzed {len(entries)} entries. Average sentiment score: {avg:.2f}. Watch recurring themes and separate facts from interpretation before important decisions.")
    return {"reflection":text,"entry_count":len(entries),"average_sentiment":round(avg,3)}

@app.post("/recall")
def recall(request: RecallRequest):
    tokens=[t for t in re.findall(r"\w+",request.query.lower()) if len(t)>2][:8]
    with connect() as conn: rows=conn.execute("SELECT * FROM entries ORDER BY created_at DESC LIMIT 200").fetchall()
    entries=[row_to_dict(r) for r in rows]; ranked=[]
    for e in entries:
        hay=(e.get("content") or "").lower()+" "+(e.get("summary") or "").lower(); score=sum(hay.count(t) for t in tokens)
        if score: ranked.append((score,e))
    ranked.sort(key=lambda x:x[0],reverse=True); matches=[e for _,e in ranked[:8]] or entries[:5]
    if not matches: return {"answer":"لا توجد ذاكرة كافية بعد." if request.language=="ar" else "There is not enough memory yet.","matches":[]}
    context="\n".join(f"{e['created_at']} | {e['entry_type']} | {e.get('summary') or e['content'][:700]}" for e in matches); client=get_client(request.openai_api_key); answer=None
    if client:
        lang="Arabic" if request.language=="ar" else "English"; prompt=f"Answer in {lang} using ONLY these TwinMind memories. If unsupported, say so. Distinguish facts from inferred patterns. Question: {request.query}\nMemories:\n{context}"
        try: answer=client.responses.create(model=MODEL,input=prompt).output_text
        except Exception: pass
    return {"answer":answer or ("أقرب المدخلات ذات الصلة معروضة أدناه." if request.language=="ar" else "The most relevant stored entries are shown below."),"matches":matches}
