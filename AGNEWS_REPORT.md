# Gemma 3 1B multi-class evaluation on AG News

This branch extends the frozen Gemma representation approach from binary prompt-injection detection to four-class topic classification:

```text
Gemma final hidden state → normalization → ridge logits → softmax probabilities
```

Classes: World, Sports, Business, and Science/Technology.

Run the experiment:

```bash
.venv/bin/python prepare_agnews.py
.venv/bin/python evaluate_agnews.py
```

The calibration split contains 200 examples per class. The official AG News test split is used only for evaluation. Exact metrics, per-class scores, the confusion matrix, and latency are written to `agnews_results.json` after the run.

The multi-class head is trained with one-hot labels using the same closed-form ridge procedure as the binary experiment. Gemma remains frozen; only the small classification head is fitted.

## Results

Evaluation completed on Apple MPS using float32 Gemma 3 1B IT.

| Metric | Score |
|---|---:|
| Accuracy | 56.16% |
| Macro precision | 58.23% |
| Macro recall | 56.16% |
| Macro F1 | 56.38% |
| Top-3 accuracy | 90.18% |
| Mean latency | 766.22 ms/example |
| P95 latency | 3024.96 ms/example |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| World | 60.93% | 50.32% | 55.12% | 1900 |
| Sports | 73.95% | 59.32% | 65.83% | 1900 |
| Business | 51.50% | 47.79% | 49.58% | 1900 |
| Science/Technology | 46.54% | 67.21% | 55.00% | 1900 |

### Confusion matrix

Rows are true labels; columns are predicted labels. Label order is `World`, `Sports`, `Business`, `Science/Technology`.

```text
              World  Sports  Business  Science/Technology
World           956     139       312                 493
Sports          193    1127       208                 372
Business        250     140       908                 602
Science/Tech    170     118       335                1277
```

The most useful result is the 90.18% top-3 accuracy: the frozen representation usually places the correct topic among its top candidates, but the four-way decision boundary is substantially weaker than the binary prompt-injection task. Business and Science/Technology are the most confused classes.
