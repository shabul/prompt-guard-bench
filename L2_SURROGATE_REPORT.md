# AnyJev-style L2 surrogate: Gemma 3 270M IT

This experiment uses a frozen Gemma model and a small closed-form ridge head over the final hidden state. It is intentionally called an AnyJev-style L2 surrogate because the installed AnyJev API does not expose the official L2 head methods.

## Data separation

All 351 English-dominant rows were split once, stratified by label:

| Set | Rows | Safe | Injection | Use |
|---|---:|---:|---:|---|
| Calibration | 150 | 84 | 66 | Fit ridge head only |
| Test | 100 | 56 | 44 | Primary held-out evaluation |
| OOB | 101 | 57 | 44 | Second untouched audit set |

Dataset SHA-256: `7d6f0858ab4b16d60cbe103f634634ff08b379f989e2636176b9e490cdefd803`

## Method

- Base model: `google/gemma-3-270m-it`
- Device: Apple M5 MPS
- Numeric type: float32
- Features: final non-padding hidden state
- Head: closed-form ridge regression, regularization 1.0
- Decision threshold: fixed 0.5
- No test or OOB labels used during fitting

## Results

| Metric | Test | OOB |
|---|---:|---:|
| Accuracy | 63.00% | 61.39% |
| Precision | 54.32% | 53.01% |
| Recall | 100.00% | 100.00% |
| F1 | 70.40% | 69.29% |
| Mean latency | 35.4 ms | 29.1 ms |

### Confusion matrix

| Set | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| Test | 44 | 37 | 0 | 19 |
| OOB | 44 | 39 | 0 | 18 |

## Interpretation

The head generalizes consistently across test and OOB and catches every injection in both sets. However, it predicts injection for most safe prompts, so it is not suitable for a balanced production detector at the fixed 0.5 threshold. The next responsible step would be threshold selection using only a calibration validation subset, followed by a fresh test/OOB evaluation.
