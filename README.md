<div align="center">

# Prompt Guard Bench

### Build small, auditable text judgments. Benchmark them before you trust them.

An experimental, local-first research workspace for turning labeled text into task-specific scores—and for finding out when a simpler or faster model works better.

[![Status: research preview](https://img.shields.io/badge/status-research%20preview-6f42c1)](PLAN.md)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab)](requirements.lock.txt)
[![Two non-security pilots](https://img.shields.io/badge/pilots-SMS%20%2B%20recipes-2ea44f)](research/pilot_decision.md)

[Why this exists](#why-this-exists) · [Quickstart](#quickstart-no-model-weights) · [Results](#what-the-pilots-show) · [Research plan](PLAN.md) · [Limitations](#research-preview-not-a-release)

</div>

## Why this exists

Many applications need a narrow judgment—*is this message spam? is this review favorable? does this text contain an instruction attack?*—rather than a generated answer. The reusable workflow is to define the label, fit a task-specific model, retain the data and model lineage, choose an operating threshold on validation data, and make the application's policy explicit.

This repository began as a frozen Gemma 3 1B prompt-injection experiment. It now uses that case study alongside two non-security pilots to ask a broader question: **which text-judgment approach is useful for a given task, label budget, and latency budget?** Gemma is one candidate, not a predetermined winner or public default.

```text
Labeled examples + task definition
               │
               ▼
      Source and split audit  ──►  Fit / validation / test IDs
               │
               ▼
  TF-IDF │ frozen MiniLM │ frozen Gemma │ SetFit
               │
               ▼
     Task-specific scorer  ──►  Validation-selected threshold
               │
               ▼
   Score + provenance + metrics + explicit limitations
```

The code currently provides benchmark scripts and experimental scoring primitives, **not** a complete installable toolkit or production defense.

## Quickstart: no model weights

The fastest way to inspect the workflow is the SMS baseline-only pilot. It downloads the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), verifies the archive SHA-256, deduplicates normalized messages, and evaluates TF-IDF and message-format baselines on a deterministic split. No Gemma or MiniLM weights are needed.

```bash
git clone https://github.com/shabul/prompt-guard-bench.git
cd prompt-guard-bench
git switch gemma-1b-pi
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python pilot_sms.py --baseline-only
```

This writes an ignored `sms_pilot_results.json` in the working directory. The broad `requirements.txt` is an entry point; the exact environment tested on Python 3.12/macOS arm64 is recorded in [`requirements.lock.txt`](requirements.lock.txt). A clean install on another platform has not yet been independently reproduced.

For a second task without model weights, run the recipe-review baselines:

```bash
.venv/bin/python pilot_recipe.py --method majority
.venv/bin/python pilot_recipe.py --method tfidf
.venv/bin/python pilot_recipe.py --method shape
```

The recipe source is also downloaded from UCI and hash-checked. Review text stays out of the committed result artifacts.

## What the pilots show

These are **exploratory, already-viewed test splits**, not locked paper results. Each task used 128 fit labels and 200 validation labels; thresholds were chosen on validation, not on the pilot test rows. The two tasks use different primary metrics because recipe sentiment has an 80%-favorable sampled test set.

| Task | Metric | Frozen Gemma | Frozen MiniLM | TF-IDF | SetFit |
| --- | --- | ---: | ---: | ---: | ---: |
| SMS spam triage | Spam F1 | 0.876 | **0.930** | 0.883 | 0.889 |
| Recipe sentiment, unseen recipes | Macro F1 | **0.896** | 0.681 | 0.792 | 0.848 |

The ranking reverses across tasks. Gemma's recipe macro-F1 advantage over SetFit is +0.048 on this split, but a paired recipe-cluster bootstrap interval includes zero; SetFit also had much lower warm request latency here. That is evidence *against* declaring a universally best backend from these pilots. See the [SMS report](research/pilot_report.md), [recipe report](research/recipe_pilot_report.md), and [pilot decision](research/pilot_decision.md) for confusion counts, timing definitions, uncertainty, and caveats. The [related-work matrix](research/related_work.md) explains how this work relates to existing classification and evaluation tools.

The original prompt-injection result is a separate historical case study: a frozen Gemma hidden-state vector, a ridge head, and a sigmoid-transformed **uncalibrated score**. Its 0.6 threshold was selected after looking at evaluation results. The deepset `oob` partition is another same-source holdout; the Giskard attack set contains only positives, so it cannot measure false-positive rate. Read the [Gemma report](GEMMA_1B_REPORT.md) and [Giskard report](GISKARD_REPORT.md) before citing those numbers.

## Run the full experiments

### Historical prompt-injection reproduction

You need access to `google/gemma-3-1b-it` or a local copy of its weights. The scripts use the prepared dataset already tracked in this repository.

```bash
.venv/bin/python evaluate_gemma1b.py --model /path/to/gemma-3-1b-it
.venv/bin/python evaluate_giskard.py --model /path/to/gemma-3-1b-it
```

The first command saves `gemma1b_legacy_head.json`; the second loads that same head. Generated JSON is ignored by Git. Model-config and chat-template fingerprints are checked, but the artifact does **not** pin or hash model weight bytes. `prepare_gemma1b_dataset.py` can rebuild and change the tracked snapshot; only run it when intentionally refreshing and auditing that data.

### Cross-task comparisons

The extra embedding and SetFit methods use a separate Python 3.12 environment pinned in [`requirements-research.lock.txt`](requirements-research.lock.txt). The lock records the tested macOS arm64 environment, not a cross-platform guarantee. These methods download the pinned [MiniLM checkpoint](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

```bash
python3.12 -m venv .venv-setfit
.venv-setfit/bin/python -m pip install -r requirements-research.lock.txt
source .venv-setfit/bin/activate
```

```bash
# SMS: same 128 fit / 200 validation / 400 exploratory test rows
python pilot_sms.py --model /path/to/gemma-3-1b-it
python pilot_sms_embedding.py
python pilot_sms_setfit.py

# Recipes: entire recipe groups held out; 320 exploratory test rows
python pilot_recipe.py --method majority
python pilot_recipe.py --method tfidf
python pilot_recipe.py --method shape
python pilot_recipe.py --method minilm
python pilot_recipe.py --method gemma --model /path/to/gemma-3-1b-it
python pilot_recipe_setfit.py
python research/paired_recipe.py
python -m research.audit_recipe_reviewers
```

Run these commands from the repository root inside the research environment. SetFit saves its model under ignored `artifacts/` directories and refuses to overwrite an existing artifact directory; supply a new `--artifact-dir` for a rerun. The recipe SetFit comparison was added *after* the initial recipe test was viewed and has its own [extension protocol](research/recipe_setfit_extension_protocol.md). The reviewer audit exports aggregate counts only; it found five test reviews from reviewers seen during development despite recipe-disjoint groups.

## Experimental API surface

[`judgment_bench/`](judgment_bench/) contains a small, weight-free binary task contract and a pickle-free loader/scorer for the pilot logistic-head JSON format. It validates head dimensions and finite values, checks encoder ID/revision, and labels scores as uncalibrated. A truncated input receives **no automatic binary decision**. Gemma pilot heads lack a pinned model revision, so using one requires the explicit `allow_unpinned_model=True` pilot-only override.

This is an early primitive, not a stable `fit/evaluate/predict` package. Callers must supply a trusted embedding and encoder metadata; a revision string alone does not prove that model weight bytes are identical. The proposed complete API and release gates are in [`PLAN.md`](PLAN.md).

## Research preview, not a release

| In place | Still required before an OSS alpha or paper claim |
| --- | --- |
| Shared historical Gemma evaluator, two task pilots, source-archive checks, deterministic split IDs, saved pilot heads, exploratory reports, lightweight tests | Installable backend-neutral fit/predict API, independent clean-machine reproduction, model-weight identity checks, dataset redistribution audit, untouched final tests, longer-document task, repeated seeds and label budgets |

The source-code license has **not** been selected or added yet. Until that is resolved, treat this as a publicly visible research repository—not a licensed OSS release; [GitHub explains why a public repo still needs an explicit license](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository). The UCI pilot sources are listed as CC BY 4.0 by UCI; cite the [SMS dataset](https://doi.org/10.24432/C5CC84) and [recipe-review dataset](https://doi.org/10.24432/C5FG95) when using them. Check model and dataset terms independently before redistributing assets.

Prompt-injection scoring is not authorization, sandboxing, tool permissioning, or a production safety guarantee. Applications remain responsible for their policies and defense-in-depth controls. For the technical design and deployment cautions, see [`APPROACH.md`](APPROACH.md) and [`SHIPMENT_GUIDE.md`](SHIPMENT_GUIDE.md).

## Reproduce or contribute

Run the lightweight tests with `python -m unittest discover -s tests`. If you find a split leak, source-license issue, artifact mismatch, or reproducibility failure, please open a GitHub issue with the command, environment, and relevant run manifest (without private text or credentials). The next work is tracked in the [plan](PLAN.md); benchmark protocols and reports live in [`research/`](research/).
