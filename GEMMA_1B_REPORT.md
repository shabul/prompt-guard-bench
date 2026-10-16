# Gemma 3 1B IT prompt-injection experiment

This branch contains only the Gemma 3 1B experiment. The model is frozen; a closed-form ridge head is trained on final hidden states from 150 labeled calibration examples.

These are historical, exploratory results. A 2026-10-16 local rerun with Python 3.12.1, PyTorch 2.14.1, Transformers 5.20.0, Datasets 5.1.0, and Apple M5 MPS reproduced all displayed threshold confusion counts. The sigmoid of a ridge output is an uncalibrated score. Threshold 0.6 was selected after viewing the evaluation splits, so the numbers below are not confirmatory estimates at a preselected operating point.

## Data

- Source: `deepset/prompt-injections`
- English-dominant rows: 351
- Calibration: 150 (84 safe / 66 injection)
- Test: 100 (56 safe / 44 injection)
- Second same-source holdout (named `oob` in stored data): 101 (57 safe / 44 injection)
- All splits are stratified and disjoint.

## Results

| Threshold | Test precision | Test accuracy | Test recall | OOB precision | OOB accuracy | OOB recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 50.57% | 57.00% | 100.00% | 51.16% | 58.42% | 100.00% |
| 0.6 | 93.33% | 95.00% | 95.45% | 87.50% | 92.08% | 95.45% |
| 0.7 | 100.00% | 92.00% | 81.82% | 100.00% | 87.13% | 70.45% |
| 0.8 | no positives | 56.00% | 0.00% | no positives | 56.44% | 0.00% |
| 0.9 | no positives | 56.00% | 0.00% | no positives | 56.44% | 0.00% |

## Historical operating point

Threshold **0.6** was selected after inspecting these results. At that threshold, the test split showed 93.33% precision, 95.00% accuracy, and 95.45% recall; the second holdout showed 87.50% precision, 92.08% accuracy, and 95.45% recall. Both evaluation splits came from deepset, so neither establishes external distribution transfer.

Reported mean model-forward latency was 85.2 ms on test and 71.2 ms on the second holdout on Apple M5 MPS. Tokenization and full request time were not included in those historical figures.
