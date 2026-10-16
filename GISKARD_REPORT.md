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

Precision, accuracy, and F1 are intentionally not reported: with no negative examples, they are not identifiable. Historical mean model-forward latency was 339.03 ms per prompt. The previously reported P95 of 770.22 ms used an off-by-one nearest-rank index for 35 rows and should not be cited. A 2026-10-16 rerun using the saved head on Apple M5 MPS measured mean forward latency 344.39 ms and correct nearest-rank P95 774.55 ms. Timings vary by run and exclude tokenization and scoring. The evaluator writes per-row uncalibrated scores and exact counts to `giskard_results.json`.
