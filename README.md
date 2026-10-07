# Gemma 1B prompt-injection benchmark

Frozen Gemma 3 1B IT hidden-state classifier evaluated on prompt-injection data.

For production handoff, prerequisites, deployment guidance, API design, threshold policy, and validation requirements, see [SHIPMENT_GUIDE.md](SHIPMENT_GUIDE.md).

For the model-to-decision design, see [APPROACH.md](APPROACH.md).

The AG News multi-class experiment is documented in [AGNEWS_REPORT.md](AGNEWS_REPORT.md).

## Run

The model weights are expected at `/Users/Shabul/model-weights/Google/gemma-3-1b-it`.

```bash
.venv/bin/python prepare_gemma1b_dataset.py
.venv/bin/python evaluate_gemma1b.py
```

Evaluate the external Giskard attack set:

```bash
.venv/bin/python evaluate_giskard.py
```

## Benchmark prompt-injection detection

```bash
.venv/bin/python prepare_dataset.py
.venv/bin/python evaluate_clef.py
```

This creates a deterministic 80/20 split with seed 42 and writes precision, recall, F1, accuracy, confusion matrix, and per-example latency to `evaluation_results.json`.
