"""Shared config, paths, and small helpers for PromptGuard.

Every entry-point script inserts the repo root on sys.path and imports this module,
so all components agree on where data, models, results, and the request log live.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ARTIFACT_DIR = ROOT / "detector" / "artifacts"

TRAIN_PARQUET = DATA_DIR / "train.parquet"
TEST_PARQUET = DATA_DIR / "test.parquet"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = RESULTS_DIR / "metrics.json"
REDTEAM_PATH = RESULTS_DIR / "redteam.json"
DB_PATH = DATA_DIR / "requests.db"

LABEL_BENIGN, LABEL_MALICIOUS = 0, 1
LABEL_NAMES = {0: "benign", 1: "malicious"}

for _d in (DATA_DIR, RESULTS_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, prompt TEXT, verdict TEXT, score REAL,
            blocked INTEGER, latency_ms REAL, source TEXT, llm_response TEXT
        )"""
    )
    conn.commit()
    return conn


def log_request(prompt, verdict, score, blocked, latency_ms,
                source="proxy", llm_response=None, path: Path = DB_PATH) -> None:
    conn = get_db(path)
    conn.execute(
        "INSERT INTO requests (ts, prompt, verdict, score, blocked, latency_ms, source, llm_response) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (now_iso(), prompt, verdict, float(score), int(blocked), float(latency_ms), source, llm_response),
    )
    conn.commit()
    conn.close()


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion. Returns (point, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))
