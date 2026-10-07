# Gemma 3 1B evaluation on Giskard prompt injections

The Giskard CSV contains 35 English attack examples: 20 Jailbreak and 15 Hijacking attacks. It contains no safe/negative examples.

The Gemma 1B hidden-state ridge head was fitted only on the existing 150-row deepset calibration set. The Giskard rows were used strictly as an external positive-only OOD evaluation set.

## Results

| Threshold | Detected | Recall |
|---:|---:|---:|
| 0.5 | 35/35 | 100.00% |
| 0.6 | 35/35 | 100.00% |
| 0.7 | 34/35 | 97.14% |
| 0.8 | 5/35 | 14.29% |
| 0.9 | 0/35 | 0.00% |

Precision, accuracy, and F1 are intentionally not reported: with no negative examples, they are not identifiable. Mean inference latency was 339.03 ms per prompt; P95 latency was 770.22 ms. The generated `giskard_results.json` contains per-row probabilities and exact counts.
