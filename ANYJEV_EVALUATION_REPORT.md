# AnyJev + Gemma 3 270M IT prompt-injection evaluation

Run date: 2026-10-07  
Model: `google/gemma-3-270m-it`  
Decision layer: AnyJev `Decider`, `L0`, canonical ordering  
Backend: Transformers `HFBackend` on Apple M5 MPS  
Numeric type: `float32`  
Dataset: `deepset/prompt-injections`  
Language filter: `langdetect`, seed 42, English-dominant rows only  
Split: stratified 80/20; benchmarked held-out test split  
Test rows: 71 (31 injection, 40 safe)  
Dataset SHA-256: `61cf99f4ba2ee4b48e9192c3f2db32505bd0bb98b2f46f38638e5a12e5466bc0`

## Results

| Metric | Result |
|---|---:|
| Accuracy | 46.48% |
| Precision | 44.62% |
| Recall | 93.55% |
| F1 | 60.42% |
| Mean latency, run 2 | 57.4 ms |
| P50 latency, run 2 | 54.3 ms |
| P95 latency, run 2 | 66.3 ms |

### Confusion matrix

|  | Predicted safe | Predicted injection |
|---|---:|---:|
| Actual safe | 4 | 36 |
| Actual injection | 2 | 29 |

## Interpretation

The AnyJev/Gemma configuration strongly favors `INJECTION`: it catches 29 of 31 injections, but also flags 36 of 40 safe examples. The result is high recall but low precision and accuracy. This is a valid, auditable baseline—not a claim that the small base model is production-ready.

AnyJev returned finite probabilities for every example. A float16 MPS smoke test produced NaNs, so the benchmark intentionally uses float32 and fails closed on non-finite distributions. A second complete run reproduced the same classification counts and dataset hash; only latency varied. First-pass latency was 63.0 ms mean / 93.9 ms P95.

The two false negatives are listed in [ANYJEV_FALSE_NEGATIVES.md](ANYJEV_FALSE_NEGATIVES.md). Raw probabilities and per-example timings are in the generated, gitignored `anyjev_evaluation_results.json` file.
