# clef-flash prompt-injection evaluation

Run date: 2026-10-07  
Model: `Cloudflare/clef-flash`, converted to 4-bit MLX  
Runtime: MLX on Apple M5 GPU  
Dataset: `deepset/prompt-injections`  
Filter: English-dominant rows using `langdetect`, seed 42  
Split: stratified 80/20, seed `42`

## Results

The filter retained 351 of 662 examples. The test set contained 71 examples: 40 safe and 31 injection.

| Metric | Result |
|---|---:|
| Accuracy | 74.65% |
| Precision | 100.00% |
| Recall | 41.94% |
| F1 | 59.09% |
| Mean latency | 358.0 ms |
| P50 latency | 338.3 ms |
| P95 latency | 416.0 ms |

### Confusion matrix

|  | Predicted safe | Predicted injection |
|---|---:|---:|
| Actual safe | 40 | 0 |
| Actual injection | 18 | 13 |

## Interpretation

The model remained conservative: it produced no false positives, but missed 18 of 31 injections. Removing non-English rows did not improve recall; the remaining misses are mostly indirect role-play, context-switching, persona, and instruction-following attacks.

Latency remained stable on the quantized MLX runtime. This is a zero-shot benchmark, not a fine-tuned classifier evaluation.

See [FALSE_NEGATIVES.md](FALSE_NEGATIVES.md) for the review set. Raw per-example results are in the generated, gitignored `evaluation_results.json` file.
