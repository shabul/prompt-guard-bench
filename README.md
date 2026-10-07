# clef local playground

A small local UI for `Cloudflare/clef`, powered by the model's OpenAI-compatible vLLM server.

## Run

The model is large (about 55 GB on Hugging Face) and needs a compatible local accelerator/runtime.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install vllm
chmod +x start.sh
./start.sh
```

Open http://127.0.0.1:3000. To use a different local model path, set `CLEF_MODEL` before starting.

## Benchmark prompt-injection detection

```bash
.venv/bin/python prepare_dataset.py
.venv/bin/python evaluate_clef.py
```

This creates a deterministic 80/20 split with seed 42 and writes precision, recall, F1, accuracy, confusion matrix, and per-example latency to `evaluation_results.json`.
