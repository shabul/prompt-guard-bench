# Gemma 3 1B IT prompt-injection experiment

This branch contains only the Gemma 3 1B experiment. The model is frozen; a closed-form ridge head is trained on final hidden states from 150 labeled calibration examples.

## Data

- Source: `deepset/prompt-injections`
- English-dominant rows: 351
- Calibration: 150 (84 safe / 66 injection)
- Test: 100 (56 safe / 44 injection)
- OOB: 101 (57 safe / 44 injection)
- All splits are stratified and disjoint.

## Results

| Threshold | Test precision | Test accuracy | Test recall | OOB precision | OOB accuracy | OOB recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 50.57% | 57.00% | 100.00% | 51.16% | 58.42% | 100.00% |
| 0.6 | 93.33% | 95.00% | 95.45% | 87.50% | 92.08% | 95.45% |
| 0.7 | 100.00% | 92.00% | 81.82% | 100.00% | 87.13% | 70.45% |
| 0.8 | no positives | 56.00% | 0.00% | no positives | 56.44% | 0.00% |
| 0.9 | no positives | 56.00% | 0.00% | no positives | 56.44% | 0.00% |

## Recommended operating point

Threshold **0.6** is the best balanced choice: 93.33% test precision, 95.00% test accuracy, and 95.45% test recall. OOB performance remains strong at 87.50% precision, 92.08% accuracy, and 95.45% recall.

Mean hidden-state extraction latency was 85.2 ms on test and 71.2 ms on OOB on Apple M5 MPS.
