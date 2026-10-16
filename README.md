# Gemma 1B prompt-injection benchmark

Frozen Gemma 3 1B IT hidden-state classifier evaluated on prompt-injection data. This is a historical, exploratory benchmark. The ridge output is passed through sigmoid and is an uncalibrated score; the 0.6 threshold was selected after inspecting evaluation results.

For production handoff, prerequisites, deployment guidance, API design, threshold policy, and validation requirements, see [SHIPMENT_GUIDE.md](SHIPMENT_GUIDE.md).

For the model-to-decision design, see [APPROACH.md](APPROACH.md).

## Run

Use Python 3.12 and install the tested macOS arm64 environment with `python -m pip install -r requirements.lock.txt`, or install the broad requirements with `python -m pip install -r requirements.txt`. The commands below can use a local Gemma weights directory or the `google/gemma-3-1b-it` model ID if you have access. They save an inspectable legacy head artifact and result JSON in the working directory.

```bash
python evaluate_gemma1b.py --model /path/to/gemma-3-1b-it
```

Evaluate the external, positive-only Giskard attack set with the same saved head:

```bash
python evaluate_giskard.py --model /path/to/gemma-3-1b-it
```

The prepared deepset `oob` split is a second holdout from the same source, not an external distribution test. The Giskard set has no benign examples, so it cannot establish precision or false-positive rate. Inference timings in the historical reports cover the model forward pass only. See [PLAN.md](PLAN.md) for the new validation and benchmark protocol.

The prepared dataset is tracked in this repository. `prepare_gemma1b_dataset.py` rebuilds it from the current upstream dataset and can change tracked data; run it only when intentionally refreshing the snapshot and audit the resulting diff.

The saved artifact is JSON and contains the fitted mean, scale, coefficients, model identifier, and model-config/chat-template fingerprints. Load it only with the same model weights and tokenizer revision. The fingerprints do not verify weight bytes or pin an upstream revision; do not use this artifact for a confirmatory or production claim.

## First non-security pilot

Run binary SMS spam triage with the same Gemma feature encoder, a separate logistic head, and TF-IDF and message-format baselines:

```bash
python pilot_sms.py --model /path/to/gemma-3-1b-it
```

`python pilot_sms.py --baseline-only` checks the dataset, deterministic split, and cheap baselines without model weights. The script downloads the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), verifies its archive hash, and keeps raw messages out of Git. UCI lists the dataset under CC BY 4.0; cite Almeida and Hidalgo (2011), DOI: [10.24432/C5CC84](https://doi.org/10.24432/C5CC84). See [pilot report](research/pilot_report.md) for the exploratory setup and findings. The pilot does not replace a locked external test or support a broad methods claim.

To compare the frozen MiniLM embedding baseline and SetFit on the same pilot split, create a separate Python 3.12 environment using `requirements-research.lock.txt`, then run:

```bash
python pilot_sms_embedding.py
python pilot_sms_setfit.py
```

These commands download the pinned [MiniLM model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). The SetFit run saves its trained model under the ignored `artifacts/` directory. The [extension protocol](research/pilot_extension_protocol.md) records the fixed settings. On this one exploratory split, frozen MiniLM had the best F1; see the pilot report for metrics, cost, and limits.

## Recipe-review transfer pilot

The second non-security pilot uses [UCI Recipe Reviews and User Feedback](https://archive.ics.uci.edu/dataset/911/recipe+reviews+and+user+feedback+dataset), which UCI lists under CC BY 4.0 (DOI: [10.24432/C5FG95](https://doi.org/10.24432/C5FG95)). It holds out entire recipes and compares five methods using the same 128 fitting and 200 validation labels. Use the separate environment from `requirements-research.lock.txt`:

```bash
python pilot_recipe.py --method majority
python pilot_recipe.py --method tfidf
python pilot_recipe.py --method shape
python pilot_recipe.py --method minilm
python pilot_recipe.py --method gemma --model /path/to/gemma-3-1b-it
python pilot_recipe_setfit.py
python research/paired_recipe.py
python -m research.audit_recipe_reviewers
```

The source archive is SHA-256-checked, and run artifacts contain hashed review IDs and scores, not raw review text. The reviewer audit exports only aggregate overlap counts. SetFit was added after the recipe test was first viewed, under a [separate extension protocol](research/recipe_setfit_extension_protocol.md). The [original protocol](research/recipe_pilot_protocol.md) and [exploratory result](research/recipe_pilot_report.md) explain the group split and limits. Neither pilot is an untouched paper test.

## Experimental task-neutral primitives

`judgment_bench/` now contains a small, weight-free binary task and logistic-head scoring contract. It loads the pilot JSON heads without pickle, checks dimensions and finite values, requires encoder ID/revision compatibility, and returns an explicitly uncalibrated score. If an artifact lacks a model revision (the current Gemma pilot heads do), scoring refuses it by default; `allow_unpinned_model=True` is an explicit pilot-only override. A truncated input never receives a binary decision. This is **not** yet a stable fit/predict package or production inference API: callers must provide a trusted embedding and encoder metadata, and a model revision alone does not prove weight-byte identity.
