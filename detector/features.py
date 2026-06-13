"""Feature backends for the detector.

Primary backend = MiniLM sentence-embeddings (sentence-transformers/all-MiniLM-L6-v2):
~80 MB, CPU-friendly on ARM, sub-10 ms inference. If sentence-transformers / torch
isn't importable (e.g. the wheel failed to install on this ARM VM), the trainer
transparently falls back to a TF-IDF backend so the pipeline still completes.
"""
from __future__ import annotations

import numpy as np

EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedder = None


def minilm_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def get_embedder(name: str = EMBEDDER_NAME):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(name)
    return _embedder


def embed(texts, name: str = EMBEDDER_NAME) -> np.ndarray:
    model = get_embedder(name)
    return np.asarray(
        model.encode(list(texts), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    )
