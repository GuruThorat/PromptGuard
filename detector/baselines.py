"""Non-ML and external baselines, for honest comparison against the trained detector.

- KeywordGuard: a naive regex blocklist. This is the "what most people would hack
  together" baseline that your ML model should beat (especially on recall, since
  obfuscated attacks dodge keywords).
- load_prompt_guard(): optional loader for Meta's Llama-Prompt-Guard-2-86M. The model
  is GATED on Hugging Face -- request access first; the function returns None and a
  hint if it can't be loaded, so nothing else breaks.
"""
from __future__ import annotations

import re

KEYWORDS = [
    r"ignore (all )?(previous|prior|above) (instructions|rules|prompts?)",
    r"disregard (the )?(above|previous|prior)",
    r"system prompt",
    r"\bDAN\b",
    r"developer mode",
    r"no restrictions",
    r"jailbreak",
    r"do anything now",
    r"reveal your (instructions|prompt|guidelines|configuration)",
    r"bypass",
    r"without any filter",
    r"pretend (you|to)",
    r"\bact as\b",
    r"forget (your|the) (rules|instructions)",
]
_PAT = re.compile("|".join(KEYWORDS), re.IGNORECASE)


class KeywordGuard:
    """Naive blocklist baseline: score = 1.0 if any attack keyword is present, else 0.0."""

    def score(self, text: str) -> float:
        return 1.0 if _PAT.search(text or "") else 0.0

    def predict(self, text: str):
        s = self.score(text)
        return ("malicious" if s >= 0.5 else "benign"), s


def load_prompt_guard(model_id: str = "meta-llama/Llama-Prompt-Guard-2-86M"):
    """Return a scorer(text) -> P(malicious) for Meta's guard model, or None if unavailable."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()

        def score(text: str) -> float:
            with torch.no_grad():
                inp = tok(text, return_tensors="pt", truncation=True, max_length=512)
                logits = model(**inp).logits
                return float(torch.softmax(logits, dim=-1)[0, -1])

        return score
    except Exception as e:  # gated / not installed / no access
        print(f"[info] {model_id} unavailable ({type(e).__name__}). "
              f"Request gated access at https://huggingface.co/{model_id} then `huggingface-cli login`.")
        return None
