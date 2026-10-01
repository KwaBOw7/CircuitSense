#!/usr/bin/env bash
# One-command start: creates a venv, installs deps, serves http://127.0.0.1:8000 (HOST / PORT env vars override)
set -e
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
exec python -m uvicorn circuitsense.api:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
