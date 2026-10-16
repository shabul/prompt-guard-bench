# SMS pilot extension protocol (fixed before the new model runs)

Date: 2026-10-16. These are exploratory comparisons because the existing 400-row SMS test was already viewed. None of these results will become a confirmatory paper endpoint.

Use the same UCI archive SHA-256, normalized deduplication, seed-42 split IDs, 128 fit labels (64 per class), and 200 validation labels (160 ham, 40 spam) as `pilot_sms.py`. The 400 test rows have 320 ham and 80 spam. Each method therefore consumes 328 unique development labels. No test labels enter fitting or threshold choice.

Compare two further methods against the existing TF-IDF, length/format, and Gemma rows:

1. Frozen [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (Apache 2.0 per model card), default sentence embedding without normalization, `StandardScaler` and `LogisticRegression(C=1.0, max_iter=1000, random_state=42)`. Do not tune embedding or head hyperparameters on test.
2. SetFit 1.2.0 using the same pinned MiniLM checkpoint, fitting its encoder and classification head on the same 128 training examples. Use its documented `Trainer` API with `TrainingArguments(batch_size=16, num_epochs=1, num_iterations=20, max_steps=80, seed=42, show_progress_bar=False, save_strategy="no")`, no evaluation set during training, and default loss/head settings. If library compatibility requires a change, record it before looking at test results. Use the same validation-only threshold rule on class-1 scores. SetFit training time includes encoder fine-tuning and head fitting.

For both, select the score threshold that maximizes validation F1; break ties toward the higher threshold. Report test confusion counts, precision, recall, F1, false-positive rate, average precision, and warm one-message inference latency on the same Apple M5 MPS machine. Report feature extraction or fine-tuning time separately and note any batching. Store dataset/code/model revisions and package versions. Do not switch model checkpoint or tune settings after seeing test numbers; any subsequent variation is a new exploratory run.

Interpretation rule: treat a gain smaller than three F1 points or a cost comparison without matched request definitions as inconclusive. The decision about a paper methods claim remains open until independent tasks, stronger source controls, and a fresh locked final test exist.
