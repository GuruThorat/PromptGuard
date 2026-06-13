"""PromptGuard firewall proxy.

Every prompt is classified by the detector before anything else happens. Malicious
prompts are blocked (never reach the model); benign prompts are forwarded to Ollama.
Every decision is logged to SQLite for the dashboard.

Config via env vars:
  OLLAMA_URL    (default http://127.0.0.1:11434)  -- on UTM/M2, point this at your Mac
                host, e.g. http://192.168.64.1:11434 (see README).
  OLLAMA_MODEL  (default llama3.2:3b)

If Ollama is unreachable the proxy returns a clearly-labelled echo stub, so the firewall
and dashboard still work without any LLM backend installed.

Run:  .venv/bin/uvicorn proxy.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import config  # noqa: E402
from detector.predict import Detector  # noqa: E402

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
DEFENSE = os.environ.get("PROMPTGUARD_DEFENSE", "0").lower() not in ("0", "false", "")

app = FastAPI(title="PromptGuard Firewall", version="1.0")
detector = None


class ChatRequest(BaseModel):
    prompt: str


@app.on_event("startup")
def _load():
    global detector
    detector = Detector()
    if DEFENSE:
        from detector.defenses import NormalizingDetector
        detector = NormalizingDetector(detector)


def call_llm(prompt: str) -> str:
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": OLLAMA_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")
    except Exception as e:
        return (f"[PromptGuard echo -- no Ollama backend reachable at {OLLAMA_URL} "
                f"({type(e).__name__})]  You said: {prompt}")


@app.get("/health")
def health():
    return {"status": "ok",
            "detector_backend": detector.backend if detector else None,
            "ollama_url": OLLAMA_URL, "ollama_model": OLLAMA_MODEL}


@app.post("/chat")
def chat(req: ChatRequest):
    assert detector is not None
    t0 = time.time()
    label, score = detector.predict(req.prompt)
    blocked = label == "malicious"
    response = None if blocked else call_llm(req.prompt)
    latency = (time.time() - t0) * 1000.0
    config.log_request(req.prompt, label, score, blocked, latency,
                       source="proxy", llm_response=response)
    return {
        "blocked": blocked,
        "verdict": label,
        "score": round(score, 4),
        "latency_ms": round(latency, 1),
        "response": ("[BLOCKED by PromptGuard: prompt classified as malicious]"
                     if blocked else response),
    }
