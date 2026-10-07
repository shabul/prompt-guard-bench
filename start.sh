#!/usr/bin/env bash
set -euo pipefail
MODEL="${CLEF_MODEL:-Cloudflare/clef-flash}"
echo "Starting $MODEL with Transformers/MPS …"
export CLEF_MODEL="$MODEL"
PYTHON="${PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then PYTHON=".venv/bin/python"; fi
"$PYTHON" transformers_server.py
