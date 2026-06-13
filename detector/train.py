"""Train the PromptGuard detector and benchmark it against baselines.

Backend (auto-detected, override with --backend):
  minilm : MiniLM embeddings + {LogisticRegression, MLP, LightGBM}, pick best by F1.
  tfidf  : word + char TF-IDF + LogisticRegression (no torch needed).

Always also scores the KeywordGuard baseline (and, if you have access, Meta's
Prompt-Guard-2 via --prompt-guard) so the metrics table tells an honest story.
Writes results/metrics.json and detector/artifacts/model.joblib.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import config  # noqa: E402
from detector import features  # noqa: E402
from detector.baselines import KeywordGuard, load_prompt_guard  # noqa: E402


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def evaluate(name, y_true, proba, thr=0.5):
    pred = (np.asarray(proba) >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, proba))
    except Exception:
        auc = float("nan")
    return {
        "model": name, "precision": float(p), "recall": float(r), "f1": float(f),
        "accuracy": float(accuracy_score(y_true, pred)), "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "minilm", "tfidf"], default="auto")
    ap.add_argument("--prompt-guard", action="store_true",
                    help="also benchmark Meta Llama-Prompt-Guard-2-86M (requires gated access)")
    args = ap.parse_args()

    train = pd.read_parquet(config.TRAIN_PARQUET)
    test = pd.read_parquet(config.TEST_PARQUET)
    Xtr_txt, ytr = train.text.tolist(), train.label.values
    Xte_txt, yte = test.text.tolist(), test.label.values

    backend = args.backend
    if backend == "auto":
        backend = "minilm" if features.minilm_available() else "tfidf"
    print(f"Backend: {backend}  |  train={len(train)}  test={len(test)}")

    results = []
    artifact = {"backend": backend, "threshold": 0.5, "label_map": config.LABEL_NAMES}

    if backend == "minilm":
        print("Embedding train/test with MiniLM ...")
        Xtr = features.embed(Xtr_txt)
        Xte = features.embed(Xte_txt)
        candidates = {
            "logreg": LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced"),
            "mlp": MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                                 early_stopping=True, random_state=42),
        }
        try:
            from lightgbm import LGBMClassifier
            candidates["lightgbm"] = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                                    n_jobs=-1, verbosity=-1)
        except Exception:
            print("[info] lightgbm unavailable -> skipping")
        fitted = {}
        for name, clf in candidates.items():
            clf.fit(Xtr, ytr)
            fitted[name] = clf
            results.append(evaluate(name, yte, _scores(clf, Xte)))
        best = max(results, key=lambda r: r["f1"])
        artifact.update(embedder_name=features.EMBEDDER_NAME,
                        clf=fitted[best["model"]], best_model=best["model"])
    else:
        candidates = {
            "tfidf_word_logreg": Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                          sublinear_tf=True, max_features=50000)),
                ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))]),
            "tfidf_char_logreg": Pipeline([
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                          min_df=2, max_features=50000)),
                ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))]),
        }
        fitted = {}
        for name, pipe in candidates.items():
            pipe.fit(Xtr_txt, ytr)
            fitted[name] = pipe
            results.append(evaluate(name, yte, _scores(pipe, Xte_txt)))
        best = max(results, key=lambda r: r["f1"])
        artifact.update(pipeline=fitted[best["model"]], best_model=best["model"])

    # --- baselines ---
    kw = KeywordGuard()
    results.append(evaluate("keyword_baseline", yte, [kw.score(t) for t in Xte_txt]))
    if args.prompt_guard:
        pg = load_prompt_guard()
        if pg is not None:
            print("Scoring with Prompt-Guard-2-86M (slow on CPU) ...")
            results.append(evaluate("prompt_guard_2_86m", yte, [pg(t) for t in Xte_txt]))

    joblib.dump(artifact, config.MODEL_PATH)
    metrics = {
        "backend": backend, "best_model": artifact["best_model"],
        "n_train": int(len(train)), "n_test": int(len(test)),
        "train_malicious_frac": float(np.mean(ytr)), "results": results,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("\n=== Test metrics (threshold 0.5) ===")
    for r in sorted(results, key=lambda r: -r["f1"]):
        print(f"  {r['model']:>20}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
              f"F1={r['f1']:.3f}  AUC={r['roc_auc']:.3f}")
    print(f"\nBest = '{artifact['best_model']}'  ->  {config.MODEL_PATH}")
    print(f"Metrics -> {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
