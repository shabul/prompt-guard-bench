# clef-flash prompt-injection evaluation

Run date: 2026-10-07  
Model: `Cloudflare/clef-flash`, 4-bit MLX conversion  
Runtime: MLX on Apple M5 GPU  
Dataset: `deepset/prompt-injections`  
Filter: English-dominant rows using `langdetect`, seed 42  
Evaluation scope: all 351 retained English-dominant rows

## Results

| Metric | Result |
|---|---:|
| Accuracy | 79.77% |
| Precision | 98.82% |
| Recall | 54.55% |
| F1 | 70.29% |
| Mean latency | 364.1 ms |
| P50 latency | 343.3 ms |
| P95 latency | 463.2 ms |

### Confusion matrix

|  | Predicted safe | Predicted injection |
|---|---:|---:|
| Actual safe | 196 | 1 |
| Actual injection | 70 | 84 |

## Interpretation

This run evaluates all 351 English-dominant examples rather than only a 20% holdout. The model remains highly conservative: it has excellent precision (98.82%) and only one false positive, but misses 70 of 154 injection examples, resulting in 54.55% recall.

Compared with the 71-example English-only holdout, the larger evaluation gives a more stable estimate. Recall is essentially unchanged, suggesting the main limitation is model behavior rather than test-set size or non-English inputs. The misses are primarily indirect role-play, context switching, persona changes, and requests that attempt to redefine the assistant’s task.

Latency remains practical on the quantized MLX runtime: 364 ms mean and 463 ms at P95. This is a zero-shot benchmark, not a fine-tuned classifier evaluation.

See [FALSE_NEGATIVES.md](FALSE_NEGATIVES.md) for the full false-negative review set. Raw per-example results are in the generated, gitignored `evaluation_results.json` file.
