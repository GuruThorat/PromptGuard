#!/usr/bin/env bash
# End-to-end build: prepare data -> train detector -> red-team the guard.
# Run from the repo root. Override the interpreter with PY=... if needed.
set -e
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"

echo "==> [1/3] Preparing data"
"$PY" data/prepare.py

echo "==> [2/3] Training detector"
"$PY" detector/train.py

echo "==> [3/3] Red-teaming the guard"
"$PY" redteam/run_eval.py

echo
echo "Done. Results in results/metrics.json and results/redteam.json"
echo "Start the firewall :  $PY -m uvicorn proxy.app:app --host 0.0.0.0 --port 8000"
echo "Start the dashboard:  .venv/bin/streamlit run dashboard/app.py"
