# Gemma 270M vs 1B: AnyJev-style L2 surrogate

Both models used the identical experiment:

- 150 stratified calibration examples for the frozen hidden-state ridge head
- 100 held-out test examples
- 101 untouched OOB examples
- Same English-dominant dataset and SHA-256: `7d6f0858ab4b16d60cbe103f634634ff08b379f989e2636176b9e490cdefd803`
- Float32 on Apple M5 MPS
- Fixed ridge threshold shown below; no test/OOB labels used for fitting

## Threshold 0.5

| Model | Split | Accuracy | Precision | Recall | F1 | Mean latency |
|---|---|---:|---:|---:|---:|---:|
| Gemma 270M | Test | 63.00% | 54.32% | 100.00% | 70.40% | 35.4 ms |
| Gemma 270M | OOB | 61.39% | 53.01% | 100.00% | 69.29% | 29.1 ms |
| Gemma 1B | Test | 57.00% | 50.57% | 100.00% | 67.18% | 85.2 ms |
| Gemma 1B | OOB | 58.42% | 51.16% | 100.00% | 67.69% | 71.2 ms |

## Threshold sweep

### Test set

| Model | Threshold | Precision | Accuracy | Recall |
|---|---:|---:|---:|---:|
| 270M | 0.6 | 81.13% | 89.00% | 97.73% |
| 270M | 0.7 | 94.12% | 86.00% | 72.73% |
| 1B | 0.6 | 93.33% | 95.00% | 95.45% |
| 1B | 0.7 | 100.00% | 92.00% | 81.82% |

### OOB set

| Model | Threshold | Precision | Accuracy | Recall |
|---|---:|---:|---:|---:|
| 270M | 0.6 | 86.96% | 90.10% | 90.91% |
| 270M | 0.7 | 100.00% | 78.22% | 50.00% |
| 1B | 0.6 | 87.50% | 92.08% | 95.45% |
| 1B | 0.7 | 100.00% | 87.13% | 70.45% |

## Takeaway

Gemma 1B is materially better at the practical 0.6 operating point: 93.33% test precision / 95.00% accuracy and 87.50% OOB precision / 92.08% accuracy. It is slower, averaging roughly 2–2.5x the 270M latency. Threshold 0.6 is the strongest balanced candidate from this comparison; threshold 0.7 maximizes precision but sacrifices recall.

These are AnyJev-style L2 surrogate results, not official AnyJev L2 results, because the installed AnyJev API does not expose the official hidden-state head methods.
