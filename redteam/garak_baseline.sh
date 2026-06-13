#!/usr/bin/env bash
# Optional: run NVIDIA garak (an off-the-shelf LLM vulnerability scanner) as an
# industry-standard adversarial baseline alongside the custom mutators.
# garak is heavy; this is not required for the core project to run.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

"$PY" -m pip install -q garak

# garak probes are aimed at a generator. To test the GUARD specifically, expose the
# firewall proxy (uvicorn on :8000) and point garak's REST generator at /chat, treating
# a non-blocked response as a bypass. See garak docs for the rest.json generator schema:
#   https://github.com/NVIDIA/garak
echo "garak installed. Suggested probes against your guard wrapper:"
echo "  $PY -m garak --model_type rest -G rest.json --probes dan,promptinject,encoding,leakreplay"
echo "Write rest.json so that 'uri' = http://127.0.0.1:8000/chat and the response"
echo "trigger treats blocked=false as a successful bypass."
