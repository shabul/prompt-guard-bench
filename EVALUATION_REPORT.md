# clef-flash prompt-injection evaluation

Run date: 2026-10-07  
Model: `Cloudflare/clef-flash`, converted to 4-bit MLX  
Runtime: MLX on Apple M5 GPU  
Test set: 133 examples from `deepset/prompt-injections`  
Split: stratified 80/20, seed `42`

## Results

| Metric | Result |
|---|---:|
| Accuracy | 81.95% |
| Precision | 100.00% |
| Recall | 54.72% |
| F1 | 70.73% |
| Mean latency | 345.0 ms |
| P50 latency | 322.0 ms |
| P95 latency | 407.3 ms |

### Confusion matrix

|  | Predicted safe | Predicted injection |
|---|---:|---:|
| Actual safe | 80 | 0 |
| Actual injection | 24 | 29 |

## Interpretation

The model was conservative: it produced no false positives, giving perfect precision, but missed 24 of 53 injections. The main opportunity is improving recall, potentially with a stronger classification prompt, constrained decoding, or a dedicated prompt-injection classifier.

Latency was measured end-to-end through the local HTTP API, including request handling and generation of up to four classification tokens. The 4-bit MLX model used approximately 5.2 GB peak memory in a smoke test and was responsive on the Apple M5. These results are a zero-shot benchmark, not a fine-tuned classifier evaluation.

Raw per-example results are in the generated, gitignored `evaluation_results.json` file.
