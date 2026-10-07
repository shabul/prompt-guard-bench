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

Results will be filled after the local Gemma 1B evaluation completes.
