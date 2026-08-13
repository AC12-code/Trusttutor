"""FastAPI surface — shows the tutor could slot into a web/WhatsApp backend.

  GET  /                                         -> chat + voice web UI
  GET  /health
  POST /ingest {"url": "https://youtu.be/..."}  -> (re)loads a lesson from any YouTube URL
  POST /ask    {"question": "..."}               -> grounded answer + citations (chat)
  POST /speak  {"text": "..."}                   -> synthesize arbitrary text to audio/wav
  POST /ask/speak {"question": "..."}            -> ask + speak in one call, audio/wav
  POST /eval                                     -> runs the harness, returns metrics

Run:  uvicorn app.api:app --reload
"""
from __future__ import annotations
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .ingest import load_from_file, load_from_youtube
from .retrieve import chunk_segments, Retriever
from .tutor import Tutor
from .llm import LLMError
from . import eval_harness
from . import speech

app = FastAPI(title="TrustTutor", version="0.1.0")

_web_dir = Path(__file__).parent / "web"

# Load a lesson at startup: LESSON_FILE (default sample) or LESSON_YOUTUBE.
_lesson_file = os.getenv("LESSON_FILE", "data/sample_transcript.json")
_lesson_yt = os.getenv("LESSON_YOUTUBE")

segs = load_from_youtube(_lesson_yt) if _lesson_yt else load_from_file(_lesson_file)
_retriever = Retriever(chunk_segments(segs))
_tutor = Tutor(_retriever)
_lesson_source = _lesson_yt or _lesson_file


class AskIn(BaseModel):
    question: str


class IngestIn(BaseModel):
    url: str


class SpeakIn(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def index():
    return (_web_dir / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(_retriever.chunks), "lesson_source": _lesson_source}


@app.post("/ingest")
def ingest(body: IngestIn):
    """Load any YouTube URL as the active lesson, replacing the previous one."""
    global _retriever, _tutor, _lesson_source
    try:
        segs = load_from_youtube(body.url)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    _retriever = Retriever(chunk_segments(segs))
    _tutor = Tutor(_retriever)
    _lesson_source = body.url
    return {"status": "ok", "segments": len(segs), "chunks": len(_retriever.chunks)}


@app.post("/ask")
def ask(body: AskIn):
    try:
        return _tutor.ask(body.question)
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/speak")
def speak(body: SpeakIn):
    """Synthesize arbitrary text (e.g. an answer already fetched via /ask) to audio."""
    wav_bytes = speech.synthesize_to_wav(body.text)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/ask/speak")
def ask_speak(body: AskIn):
    """Ask + speak in a single call — same grounded answer as /ask, as audio/wav."""
    try:
        result = _tutor.ask(body.question)
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))
    wav_bytes = speech.synthesize_to_wav(result["answer"])
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/eval")
def run_eval():
    qs = eval_harness.load_questions(os.getenv("EVAL_FILE", "data/eval_questions.json"))
    rep = eval_harness.run(_tutor, qs)
    return rep.metrics()
