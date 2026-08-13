# TrustTutor

A chatbot that turns any YouTube video into an interactive, voice-enabled tutor —
grounded strictly in that video's own transcript, with a built-in eval harness
that measures hallucination instead of just hoping it's low.

Paste a URL, ask questions, get a summary, and either read the answers or have
them spoken back to you. It only answers from what the video actually said,
cites the timestamp it used, and refuses when the video doesn't cover your
question — instead of confidently making something up.

Stack: **FastAPI · LangGraph · FAISS / TF-IDF · Python**

---

## What it does

- **Ask questions about any YouTube video.** Paste a link, it pulls the
  transcript, and you can chat with it about the content.
- **Grounded answers only.** It's prompted to use nothing but the retrieved
  transcript context and refuses when the answer isn't there — no
  hallucinated confidence.
- **Timestamp citations.** Every in-video answer points back to where it came
  from, e.g. `[02:15]`.
- **Voice, both ways.** Type or click the mic to ask (browser-native speech
  recognition), and have any answer read back out loud (offline
  text-to-speech — no API key needed).
- **Measurable, not just demo-able.** An eval harness scores a naive LLM
  baseline against the grounded tutor on hallucination rate, refusal
  accuracy, and citation coverage, and writes an HTML report.

## Architecture

```
YouTube URL / transcript ──▶ ingest ──▶ chunk (+timestamps) ──▶ Retriever (tfidf|faiss)
                                                                      │
                                              LangGraph:  retrieve ──▶ grade ──┬─▶ answer (+citations)
                                                                               └─▶ refuse
                                                                               │
                                                                   (optional) speak (TTS)
```

- **Retrieval** defaults to TF-IDF (no model download, never stalls a live
  demo). Flip `RETRIEVER=faiss` for the sentence-transformers + FAISS path.
- **Grounding gate** is a cheap retrieval-score check that refuses obviously
  off-topic questions before any LLM call; an optional LLM grader
  (`USE_LLM_GRADER=true`) adds a stricter sufficiency check on top.
- **LLM layer is provider-agnostic** — Anthropic, OpenAI, or Google Gemini
  behind one interface, or `mock` for offline wiring tests with no key at all.

## Quickstart

<img width="962" height="857" alt="image" src="https://github.com/user-attachments/assets/8ea99a8a-63c3-4e14-bd8c-e976202f532c" />


```bash
git clone <your-repo-url> trusttutor
cd trusttutor
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_PROVIDER and the matching API key (see below)
```

Run the eval (the measurable, no-browser-needed demo):

```bash
python -m scripts.run_eval        # writes report.html — open it in a browser
```

Run the chat + voice web app:

```bash
uvicorn app.api:app --reload
```

Then open **http://127.0.0.1:8000** — paste a YouTube URL under "Load a
lesson", hit Load, and start asking questions. Or drive it via curl:

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"url":"https://youtu.be/VIDEO_ID"}'
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What is this video about?"}'
```

No API key handy? `LLM_PROVIDER=mock python -m scripts.run_eval` exercises
the full wiring offline (behavioral only — mock can't judge real relevance,
so grounding quality only shows up with a real model).

> **Note:** `data/eval_questions.json` is written for `data/sample_transcript.json`
> (a FastAPI dependency-injection lesson). Running `scripts.run_eval --youtube <url>`
> swaps the lesson but not the questions, so the metrics won't mean much unless
> you also write matching questions for that video. For exploring a real video
> interactively, use the web app / `/ask` instead — no matching question file
> needed there.

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai` \| `google` \| `mock` |
| `MODEL` | `claude-3-5-sonnet-20241022` | Set to a model your key can access. For Google, prefer an `-latest` alias (e.g. `gemini-flash-latest`) — pinned versions get deprecated or hit tight free-tier daily caps. |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` | — | Only the one matching `LLM_PROVIDER` is required. |
| `RETRIEVER` | `tfidf` | `tfidf` (default, instant) \| `faiss` (needs `sentence-transformers`/`faiss-cpu`, see requirements.txt) |
| `TOP_K` | `4` | Chunks retrieved per question |
| `CHUNK_WORDS` | `80` | Words per transcript chunk |
| `GROUNDING_THRESHOLD` | `0.08` | Raise to refuse more aggressively |
| `USE_LLM_GRADER` | `false` | `true` adds a stricter LLM sufficiency check |
| `LESSON_FILE` | `data/sample_transcript.json` | Default lesson the API loads at startup |
| `LESSON_YOUTUBE` | — | Set instead of `LESSON_FILE` to boot straight from a video |
| `WEBSHARE_PROXY_USERNAME` / `WEBSHARE_PROXY_PASSWORD` | — | Optional. YouTube rate-limits/blocks an IP after repeated caption fetches — set these (a [Webshare](https://www.webshare.io) account) to route around it |
| `HTTP_PROXY_URL` / `HTTPS_PROXY_URL` | — | Alternative to Webshare — any HTTP/HTTPS proxy you already have |

## API reference

| Route | Purpose |
|---|---|
| `GET /` | Chat + voice web UI |
| `GET /health` | Status check |
| `POST /ingest` `{"url": "..."}` | Load any YouTube URL as the active lesson |
| `POST /ask` `{"question": "..."}` | Grounded chat answer + citations |
| `POST /speak` `{"text": "..."}` | Synthesize any text to `audio/wav` |
| `POST /ask/speak` `{"question": "..."}` | Ask + speak in one call |
| `POST /eval` | Run the eval harness, return metrics as JSON |

## Reading the eval

| metric | meaning | good |
|---|---|---|
| naive hallucination rate | adversarial questions the plain LLM still answered | high (the problem) |
| grounded hallucination | adversarial questions the tutor still answered | ~0% (the fix) |
| refusal accuracy | adversarial questions correctly refused | high |
| citation coverage | in-lesson answers carrying a timestamp | high |

With a real model, grounded hallucination should sit near zero while the
naive baseline answers almost everything — including what it can't know.
That delta is the point.

## Voice notes

- **Speaking back (TTS)** uses `pyttsx3`, which runs fully offline via your
  OS's built-in voices (SAPI5 on Windows, NSSpeechSynthesizer on macOS,
  espeak on Linux). No API key, no network call.
- **Speaking to it (STT)** uses the browser's native `SpeechRecognition` API
  — free, built into Chrome/Edge, zero server-side setup. Other browsers
  (Firefox, Safari) don't support it, so the mic button disables itself
  there.

## Possible improvements

- Hybrid retrieval (BM25 + embeddings) with a rerank pass, for longer or
  messier lectures than TF-IDF handles well.
- Stream answers token-by-token instead of waiting for the full response.
- Multi-video / playlist ingestion instead of one lesson at a time.
- Persist grounding decisions into a per-lesson "answerability map."
- Turn the eval into a CI gate so answer quality can't silently regress.
- Cloud neural TTS as an alternative to the offline voices, for higher audio
  quality when a network call is acceptable.

## Layout

```
app/       config, llm, ingest, retrieve, tutor (LangGraph), eval_harness, speech, api, web/
data/      sample_transcript.json, eval_questions.json
scripts/   run_eval.py
```

> Built with public/sample data only.
