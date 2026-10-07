#!/usr/bin/env bash
set -euo pipefail
MODEL="${CLEF_MODEL:-Cloudflare/clef-flash}"
if [ -d "/Users/Shabul/model-weights/Cloudflare/clef-flash" ]; then MODEL="${CLEF_MODEL:-/Users/Shabul/model-weights/Cloudflare/clef-flash}"; fi
MODEL="${CLEF_MODEL:-/Users/Shabul/model-weights/Cloudflare/clef-flash-mlx-4bit}"
PYTHON="${PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then PYTHON=".venv/bin/python"; fi
echo "Starting quantized MLX model $MODEL …"
"$PYTHON" -m mlx_lm.server --model "$MODEL" --host 127.0.0.1 --port 8000 --trust-remote-code --decode-concurrency 2 --chat-template-args '{"enable_thinking":false}' --max-tokens 512 > mlx.log 2>&1 &
MLX_PID=$!
trap 'kill $MLX_PID 2>/dev/null || true' EXIT
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 2; done
echo "Model ready. Playground: http://127.0.0.1:3000"
"$PYTHON" server.py
