# Plan: from prompt-injection experiment to a general-purpose judgment toolkit

Status: proposal, expanded 2026-10-16. This document defines the work to validate the current experiment, build a research-quality benchmark, test whether the method works beyond prompt injection, publish a paper if the evidence supports one, and release useful open-source software. The original reported metrics came from the repository's Markdown reports. A local rerun with the saved head now reproduces all historical confusion counts; generated result JSON files remain ignored by Git.

Implementation started 2026-10-16: the two evaluators now share `gemma_benchmark.py`; the main evaluator saves a legacy ridge-head JSON artifact and the Giskard evaluator loads it with model-config/chat-template fingerprint checks. Model paths are configurable, the dataset digest includes split/label/text, percentile calculation is corrected, and the reports label historical claims as exploratory. Helper tests and full local model reruns pass. This does not complete Phase 0 or Phase 1: the injection dataset/license audit, model weight revision pinning, independent-machine reproduction, and a new validation protocol are still pending.

Two non-security pilots added 2026-10-16. `pilot_sms.py` tests binary SMS triage; `pilot_recipe.py` tests review sentiment with entire recipe groups held out. Both use UCI CC BY 4.0 sources pinned by archive SHA-256, deterministic deduplicated splits, validation-selected thresholds, and TF-IDF, length/format, frozen Gemma, frozen MiniLM, and bounded SetFit comparisons. Recipe SetFit was added after the initial test was viewed and is explicitly a post-test extension. `requirements.lock.txt` records the original successful Python 3.12/macOS arm64 benchmark environment; `requirements-research.lock.txt` records the separate pilot environment. The protocols, results, [related-work matrix](research/related_work.md), and [provisional pilot decision](research/pilot_decision.md) are in `research/`. Both test partitions are exploratory and already viewed. More seeds/budgets, source auditing, an untouched final test, and prospective usefulness criteria remain necessary before a methods claim is chosen.

## 1. Product thesis

The reusable idea is **turning a frozen language model's representations into small, task-specific judgments that application code can consume**. A user supplies labeled examples and a task definition; the toolkit extracts features, fits a lightweight head, evaluates it, packages the exact model and preprocessing choices, and returns a versioned score. Application code owns the final policy.

Prompt-injection detection is the first reference task. The public project should lead with the broader workflow and use injection detection as a complete, well-evaluated example. We should not suggest that the existing injection-trained head already solves other tasks. Every new judgment needs its own labels, evaluation, and operating policy.

Potential public description: “Train small, auditable classifiers on frozen language-model features. Start with labeled text, get a versioned score and an evaluation report, then plug the score into your application.” The exact project name can be chosen after the API and first demo exist.

### Who it serves

- Developers who need local semantic signals inside an application without generation or per-request fine-tuning.
- Researchers comparing small classifier heads, embedding strategies, and model backbones on labeled text tasks.
- Security teams who want to reproduce and extend the prompt-injection case study.

### Initial use cases to demonstrate, not promise broadly

| Use case | Input and label | Application decision | Main evaluation concern |
| --- | --- | --- | --- |
| Prompt-injection screening | Text plus safe/injection label | Flag for review or restrict a risky action | False positives on ordinary quoted or technical text; attack recall |
| Support-ticket routing (later) | Ticket plus destination label | Select a queue | Multiclass confusion and abstention; outside the first binary release |
| Binary SMS triage (first non-security pilot) | Message plus ham/spam label | Send to review or inbox | Source/format shortcuts, class imbalance, and false blocks |
| Binary document triage (later breadth test) | Document plus a defined include/exclude criterion | Send to review or archive | Cross-source generalization, long text, and false exclusions |
| Document relevance | Query and passage plus relevant/irrelevant label | Rank or filter retrieval candidates | Performance by query family and retrieval source |
| Policy or quality checks | Content plus a precisely defined criterion label | Escalate, annotate, or accept | Label consistency and drift as policy changes |

Start with binary text classification in the first public release. Multiclass, pairwise relevance, and multiple independent judgments can follow only when the training and reporting contracts support them. A single output should be called a `score` until calibration is demonstrated; it should be called a `probability` only if that claim has been tested for that task and population.

## 2. Current baseline

The branch `gemma-1b-pi` contains three executable scripts: `prepare_gemma1b_dataset.py`, `evaluate_gemma1b.py`, and `evaluate_giskard.py`. The first script selects English rows from `deepset/prompt-injections` and makes 150 calibration, 100 test, and 101 `oob` rows. The evaluators now use a shared `gemma_benchmark.py` module: the first fits a legacy ridge head on frozen Gemma features and saves it; the Giskard script loads that head and measures recall on 35 attack examples.

The historical reports claim 93.33% precision and 95.45% recall on the 100-row test split at threshold 0.6, plus 35/35 detection on the Giskard set at that threshold. These numbers are a useful starting baseline, with important limits: the two deepset evaluation splits come from the same source, Giskard contains only positives, and the threshold appears to have been chosen after examining reported results. The scripts can now write and read an artifact, and the historical counts have been locally reproduced. No production inference API or independent-machine reproduction exists yet.

### Findings to address, in priority order

| Priority | Finding and evidence | Consequence | Proposed resolution |
| --- | --- | --- | --- |
| P0 | The historical evaluator fits ridge regression to numeric 0/1 labels, then applies sigmoid to its output. | A ridge prediction is not a logit. The resulting number lacks a justified probability interpretation; threshold 0.6 is tied to this arbitrary transform. | The current shared module preserves this as an explicitly uncalibrated legacy score. Next compare raw ridge scores with logistic regression or a calibrated head on held-out calibration data. |
| P0 | The original scripts fitted the head on every run and saved no artifact. | The claimed deployment flow could not be reproduced or served as written. | A versioned legacy JSON artifact is now written and reused by Giskard. Still separate fit/evaluate/predict into a stable API, pin model/tokenizer revisions, and verify load/save parity on a full run. |
| P0 | `prepare_gemma1b_dataset.py:8-25` concatenates upstream splits, shuffles within class, and slices into three subsets. The `oob` name appears in reports as if it were external data. | Source splits may be lost; similar or duplicate texts could span subsets. The OOB subset measures another within-source holdout, not distribution shift. | Audit upstream provenance and exact/near duplicates; group duplicates before splitting. Call this a second holdout or use one final locked test. Reserve “external/OOD” for separately sourced data. |
| P0 | `GEMMA_1B_REPORT.md:26` recommends 0.6 based on the displayed test/OOB results. | Threshold selection on evaluation data can make the final headline optimistic. | Introduce separate train, validation/threshold, and locked test roles. Choose the operating point on validation and publish the locked test once. |
| P1 | The original evaluators hard-coded a personal model path, enabled `trust_remote_code`, and silently truncated at 2048 tokens. | Installation was machine-specific and long-input behavior was hidden. | Model path and remote-code policy are now options and truncation is counted. Still pin model/tokenizer revisions, expose a length policy, and test long inputs. |
| P1 | Historical latency measured only the forward pass, and Giskard P95 used the wrong nearest-rank index. | Published latency is incomplete; the old P95 cannot be trusted. | The percentile calculation is fixed and embedding/forward times are recorded. Still measure full scoring request, load time, concurrent throughput, and a newly reproduced P95. |
| P1 | The original evaluators duplicated embedding and ridge-fit logic. | Fixes could diverge, and runs did not reuse one artifact. | Both now share a module and the saved head. Next move this into a stable package and test full-run artifact parity. |
| P1 | The original README had dead commands; requirements repeated `datasets` and lacked a lock. | Fresh users could not follow the quickstart reliably. | Quickstart and duplicate dependency are corrected; a lock records the tested macOS arm64 environment. Still test a clean install on another machine and define supported platforms. |
| P1 | `data/` is listed in `.gitignore` yet `git ls-files data` shows 30 tracked files, including the prepared datasets and Giskard CSV. | Ignore rules do not affect tracked files. Publication needs an explicit data and provenance decision. | Audit source licenses, redistribution rights, data cards, attribution, hashes, and whether to retain snapshots or use download scripts. Do not remove tracked data without that decision. |
| P2 | The original digest hashed concatenated text without delimiters or labels. | It did not identify the complete evaluation dataset. | The new digest includes canonical split, row index, text, and label; still record immutable upstream revisions and source IDs in the full manifest. |
| P2 | The original evaluators ran at import and relied on fixed row-count assertions. | They were hard to import, test, and extend. | Both now have entry points and basic schema checks. Still improve validation and separate model code from CLI I/O. |

The current benchmark has no verified false-positive rate on representative application traffic. A positive-only attack set cannot answer that question. A small same-source split also cannot establish generalization to other domains or languages.

## 3. Design principles

1. **Task definition first.** A judgment has an ID, input schema, label definitions, positive/negative examples, exclusions, and intended use. “Unsafe” or “good” without criteria is not a portable target.
2. **Model score and policy are separate.** Return raw/normalized score, calibration status, and model version. The caller chooses thresholds and actions for its setting. Support abstain/review outcomes.
3. **Evidence travels with the artifact.** Save model revision, tokenizer/chat template, representation method, fit data manifest, preprocessing settings, head parameters, metrics, and code version.
4. **No universal performance claim.** Publish per-task, per-dataset results with base rates, confidence intervals, and failure examples. Highlight where a model has not been evaluated.
5. **Local-first, replaceable components.** The first backend can be local Gemma. Define a narrow encoder interface so future supported backbones can be compared without changing task or reporting contracts.
6. **Default privacy.** Avoid raw user text in logs and artifacts unless the caller explicitly opts in. Keep examples and model weights out of Python package distributions by default.

### Proposed contracts

```text
TaskSpec:
  id, version, input_fields, labels, label_definitions, source_context

TrainingExample:
  id, input, label, group_id?, source?, metadata?

JudgmentArtifact:
  task_spec, encoder_id, encoder_revision, tokenizer_revision,
  chat_template_hash, pooling, max_tokens, truncation_policy,
  normalization, head_type, head_parameters, calibration?,
  data_manifest_hash, fit_config, evaluation_summary, code_version

Prediction:
  task_id, task_version, score, score_semantics, label?,
  calibration_status, artifact_id, input_status, latency_ms
```

`input_status` should make truncation, unsupported input, and preprocessing failure observable. The API should never quietly convert a failed or partial input into a confident action. For a general-purpose task, use `score` and `label`; keep `probability_injection` only as a compatibility field in the injection example if needed.

### Candidate package layout

```text
src/judgment_bench/
  task.py             # task and input schemas
  encoders/           # frozen encoder adapter; Gemma implementation first
  heads/              # ridge baseline, logistic head, calibration
  artifacts.py        # safe, versioned serialization and compatibility checks
  evaluation.py       # splits, metrics, uncertainty, reports
  policy.py           # optional explicit allow/review/block mappings
  cli.py              # fit, evaluate, predict, inspect
examples/
  prompt_injection/
  sms_triage/         # first independent binary public-data example
  document_triage/    # later longer-text task, after license and split review
  ticket_routing/     # later, when multiclass training/reporting is supported
tests/
docs/
```

Avoid making the CLI or HTTP wrapper a dependency of the core library. An HTTP service can be an optional example after a stable local API exists. Inspect and document artifact deserialization risks; do not load arbitrary pickle artifacts from untrusted sources.

## 4. Evaluation program

### Dataset and label protocol

- Record the original dataset repository, revision, license, retrieval date, and original split for every row. Verify what label `0` and `1` mean from source metadata.
- Normalize IDs for exact-duplicate detection; inspect near duplicates and template families. Split related examples by group before fitting, so variants do not leak across train/validation/test.
- Keep at least three roles: head training, threshold/calibration validation, and locked test. When data is too small, use grouped cross-validation for development and a separately acquired final test; state uncertainty honestly.
- Treat the current deepset `oob` subset as another same-source holdout unless a provenance check proves otherwise. Use Giskard as an external attack-only stress set.
- Add a task-specific benign set representative of real traffic, plus difficult negatives: quoted attacks, security analysis, coding requests, commands discussed as data, and long retrieved documents.
- For broader tasks, document annotation instructions, disagreements, adjudication, class balance, and intended population. Avoid claiming a general-purpose model from a single narrow benchmark.

### Methods and comparisons

- Preserve the existing ridge+sigmoid path as `legacy_v0` so past reports remain explainable.
- Compare raw ridge ranking, a regularized logistic head, and at least one simple baseline such as TF-IDF plus logistic regression. Report both quality and operating cost. A strong frozen encoder is useful only if it beats simpler methods where needed.
- Compare representation choices where feasible: final token, pooled hidden states, selected layer, and a dedicated embedding model. Ensure no test-driven architecture selection.
- Assess calibration with reliability plots, Brier score or log loss, and error by score bucket. If calibrating, fit calibration on validation data only.
- Evaluate at fixed thresholds chosen before final testing. Report confusion matrix, precision, recall, F1, false-positive rate, PR-AUC, and confidence intervals; show metrics by source, length, and attack/benign family.
- Measure end-to-end p50/p95/p99 latency, memory, throughput, model-load time, and behavior under concurrent requests on named hardware. Separate cached feature experiments from real inference.
- Add targeted stress tests for truncation, multilingual text, obfuscation, distribution shift, adversarial edits, empty/malformed input, and model failure. Treat generated stress cases as supplementary evidence, not a replacement for independent data.

### Reproduction record

One command should create a machine-readable run manifest and an HTML/Markdown report containing config, commit, dataset fingerprints, split IDs, artifact ID, confusion matrices, threshold table, latency definition, and seed. Public results should link to the exact run manifest. Prefer a small checked-in result fixture for smoke tests, while keeping large or restricted datasets downloadable by script where licensing permits.

## 5. Research and benchmarking track

The current result is an experiment, not yet a paper claim. A paper needs a specific question, a fair comparison, enough independent evidence to answer it, and clear boundaries on what was tested. The research track is independent of packaging: useful OSS can be released even if a new method does not outperform existing baselines.

### 5.1 Select the paper contribution before large experiments

Choose **one primary contribution** and write its falsifiable claim before final test runs:

| Candidate paper | Claim to test | Evidence required | If the claim fails |
| --- | --- | --- | --- |
| Methods paper | A frozen small LM plus a lightweight head gives a useful quality/cost or label-efficiency tradeoff across diverse text judgments. | Multiple task families; strong task-specific and generic baselines; learning curves; compute/latency; ablations. | Publish an honest negative result, workshop report, or software paper rather than claiming methodological superiority. |
| Benchmark paper | A documented evaluation suite exposes important gaps in prompt-injection detectors or reusable text judges. | Original, well-annotated test design; contamination audit; diverse systems; reliability and difficulty analysis; legal redistribution path. | Release evaluation code and a smaller dataset card without calling it a new benchmark contribution. |
| Systems/software paper | A reproducible artifact and evaluation workflow materially improves the ability to build and audit local text judgments. | Working package; two or more independent tasks; usability and reproducibility study; comparison to existing workflows. | Release as OSS and document the engineering contribution without overstating research novelty. |

Proposed first hypothesis: **the frozen Gemma representation with a small supervised head is competitive with simpler baselines at low label counts on several binary text judgments, with a measurable quality/latency tradeoff**. “Competitive” must be operationalized before testing, such as a noninferiority margin and a maximum latency or memory budget chosen for the intended deployment. Do not change the primary claim after viewing locked test results.

Before committing to a methods claim, make a related-work matrix for TF-IDF classifiers, dedicated embeddings plus a linear head, [SetFit](https://huggingface.co/docs/setfit/index), specialist injection detectors, and existing reproducible classification toolkits. Record what each supports: task scope, trainable components, total labels needed, training and inference cost, artifact portability, calibration, and benchmark protocol. SetFit fine-tunes a sentence encoder, so compare it as a strong alternative rather than describing it as another frozen-feature method. Identify the missing capability or measured tradeoff the proposed work addresses; if the matrix reveals no distinct methods claim, pursue the benchmark or software framing. Deliverable: `research/related_work.md` with cited, versioned entries and a one-paragraph differentiator that can be falsified by the pilot.

### 5.2 Benchmark suite and dataset selection

Use a **task matrix**, not one aggregate leaderboard number. Before choosing datasets, define the target population and unit of evaluation for each task. Start with the following candidate matrix; dataset inclusion depends on verified labels, license, access, and leakage review.

| Track | Candidate sources | What it tests | Required negatives or context | Main caution |
| --- | --- | --- | --- | --- |
| Within-source injection | Current deepset set, rebuilt with audited splits | Reproduction and label efficiency | Original safe rows plus hard benign examples | Small dataset; existing test has influenced threshold selection |
| External injection text | Current Giskard set; candidate [BIPIA](https://github.com/microsoft/BIPIA) and [InjecAgent](https://arxiv.org/abs/2403.02691) subsets | Transfer across attack source and surface | Matched benign content from the same surface | Attack-only data can measure recall, not precision |
| Agent workflow | Candidate [AgentDojo](https://github.com/ethz-spylab/agentdojo) | Whether screening changes actual attack success | Full task, tool-output context, and normal-task completion | A text detector score does not itself prove an agent defense |
| General text classification | Candidate classification tasks in [MTEB](https://github.com/embeddings-benchmark/mteb) | Task transfer of the *workflow* and encoder | Each dataset's own negatives and official splits | Use only tasks compatible with the defined input/output API |
| Intent / abstention | Candidate [CLINC150 original dataset](https://github.com/clinc/oos-eval) | Routing and out-of-scope rejection | Out-of-scope utterances | This is multiclass; run only after multiclass support exists, or define a justified binary subtask |

For a broad methods claim, target at least **three independent tasks across at least two non-security domains**, in addition to prompt injection. This is a proposed minimum for planning, not a statistical guarantee. Include a binary classification task with clean splits, a task with changing domain/source, and a task where abstention or false positives matter. If we only complete the injection track, frame the paper narrowly as a prompt-injection study. If we add a benchmark with new annotations, write annotation and sampling protocols before data collection.

Pilot public binary non-security tasks in Phase 1, before the full package API is fixed. UCI SMS ham/spam triage and UCI recipe-review sentiment now exercise separate task heads with verifiable source archives. SMS is short text and lacks row-level source IDs; recipe reviews have recipe-group IDs, but they are still short-to-medium reviews rather than a document inclusion/exclusion task, and recipe grouping does not imply reviewer-disjointness. Add a longer document task with source groups before making a broad transfer claim. Defer ticket routing until multiclass support exists. Pairwise relevance also remains a later extension unless the input contract explicitly supports query-passage pairs.

Build a dataset register with: source URL and immutable revision, license, permitted redistribution, row count, label definitions, train/validation/test roles, collection time, language, source type, privacy status, duplicate-group rule, and known contamination risks. Never silently merge official train and test splits. Run exact and near-duplicate checks within and across datasets; document likely benchmark exposure in pretrained backbones as a limitation even where it cannot be measured.

Add a shortcut audit to each dataset card: obtain positive and negative examples from the same source and collection surface where feasible; balance or stratify source, length, and format; report results by source; and run a source-held-out evaluation. Fit a cheap length/format/source-metadata-only classifier on the training partition. If it performs surprisingly well, inspect confounding before crediting semantic understanding to the text model. Keep dataset-source IDs separate from labels in exported predictions so the audit is repeatable.

### 5.3 Fair comparison protocol

At minimum, compare these families on the **same data partitions and selection budget**:

1. Majority-class and simple lexical/rule baseline.
2. TF-IDF with a tuned linear classifier.
3. A dedicated text-embedding model plus the same head and tuning procedure.
4. Frozen Gemma features with ridge, logistic, and calibrated variants.
5. [SetFit](https://huggingface.co/docs/setfit/index) on compatible non-security classification tasks when testing few-label claims, using the same examples and development-label budget. Report its encoder training cost separately because it fine-tunes the encoder.
6. A length/format/source-metadata-only classifier as a shortcut diagnostic where those fields are available.
7. For injection only, at least one published specialist detector, such as [Meta Prompt Guard 2](https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/prompt-guard), subject to its license and input scope.
8. Optional generative zero-shot/few-shot judge if a cost-matched comparison is part of the claim. Freeze its prompt and scoring rule before testing.

Document each baseline's model revision, preprocessing, threshold selection, hyperparameter search space, access restrictions, and inference hardware. Tune all trainable systems on the same validation information. Report both the **validation-selected operating point under a common rule** and each system's published/default threshold where relevant. Do not compare a tuned proposed model to untuned baselines, or a 1B model's isolated forward pass to a baseline's full request time.

For learning curves, report two budgets: examples used to fit the model and **all unique labeled examples consumed during development**, including hyperparameter selection, calibration, threshold selection, and any task-specific prompt examples. Sample the same IDs for all compatible methods at each budget, fix the allocation before running, and repeat sampling with declared seeds. Never call a run “16-label” if it also uses a larger labeled validation set. Report annotation time or cost when labels are newly acquired.

### 5.4 Predeclared experimental protocol

- Write a one-page protocol before the locked test: primary claim, primary metric per task, minimum useful effect/noninferiority margin, datasets, exclusions, split method, baselines, hyperparameter budget, seeds, and statistical analysis.
- Record the total labeled-development budget and its fit/selection/calibration/threshold allocation for every learning-curve point. State what happens if a small budget cannot support calibration or threshold estimation: omit that claim or use a predeclared development-only resampling rule with honest uncertainty.
- Keep training, head/model selection, calibration, threshold choice, and final test roles distinct. If data are scarce, use nested/grouped cross-validation for development and a separate locked external test where feasible.
- Use grouped splits for paraphrases, attack templates, documents, users, or source conversations. Record group IDs and inspect leakage before fitting.
- Treat the current 100-row test and 101-row same-source holdout as **exploratory historical data** for a new paper because results and threshold choices are already visible. Acquire or reserve a genuinely untouched final set for confirmatory results.
- Run at least five predeclared split/seed replicates for analyses whose randomness affects results, then check whether more are needed based on observed variance and compute budget. Report per-seed results, mean/median, and spread; do not pretend multiple seeds make correlated examples independent.
- Compute a sample-size or precision plan for each primary endpoint before collection. For rare false positives, plan around the desired upper confidence bound rather than picking a round row count. For example, with zero false positives in `n` independent benign samples, the rough 95% upper bound is `3/n`; reaching an upper bound near 1% takes about 300 such examples. This approximation is a planning aid; clustered data need cluster-aware intervals.
- Store the locked split IDs and run plan under version control. Test-set access should be limited to final runs; every post-test change must be logged and evaluated on a new holdout or clearly marked exploratory.
- Specify the threshold algorithm on validation data before test access: candidate grid or score quantiles, target FPR and confidence rule, tie breaking, and an explicit `no_eligible_threshold` result if none qualifies. Freeze the selected threshold and artifact; on the locked test report achieved FPR and recall with intervals at that fixed threshold. A threshold chosen from the test ROC/PR curve is exploratory and cannot support the primary operating-point claim.

### 5.5 Metrics, uncertainty, and statistical comparisons

| Question | Required reporting |
| --- | --- |
| Does it rank examples? | PR-AUC with class prevalence; ROC-AUC as secondary; score distributions by class. |
| Does it work at the chosen threshold? | TP/FP/FN/TN, precision, recall, specificity/FPR, F1, balanced accuracy, and exact threshold-selection rule. |
| Is it usable at low false-positive rates? | Select thresholds using the predeclared rule on validation; report locked-test recall and achieved FPR with intervals at those frozen thresholds. Report `no_eligible_threshold` or “not estimable” when validation negatives are insufficient. |
| Is its score calibrated? | Brier score, log loss, reliability diagram, expected calibration error with stated bins, and calibration-in-the-large. Apply probability language only where supported. |
| Can it abstain? | Coverage versus error/risk curve, review rate, and cost of unresolved cases. |
| Does it help an agent? | Attack success rate **and** legitimate-task completion with the detector inserted at a specified interception point; report operational false blocks. |
| Is it practical? | Feature extraction cost, head-fit time, total training/development cost, uncached end-to-end p50/p95/p99 inference, throughput, memory, model-load time, artifact size, energy/cost where measurable, and tokens processed. |

For proportions, report Wilson or exact binomial intervals; for grouped or paired metrics, bootstrap over the independent unit (source document, attack family, conversation, or user) rather than individual derived rows. For model comparisons, use paired differences on identical test IDs with confidence intervals and a suitable paired test when needed. Predeclare any multiplicity adjustment for several primary hypotheses. Report effect sizes and intervals even when a p-value is included. Include a per-dataset table and a macro summary; do not let a large easy dataset dominate the headline.

The efficiency report must identify which numbers use cached representations. Record encoder/tokenization time for all labeled development examples, model/head fitting time, selection/calibration time, and full uncached inference on the same hardware and input lengths. As a separate experiment, test one encoder pass with several task heads on identical input encoding; count the shared pass once, include every head's scoring time and memory, and compare against independent full passes. This experiment only supports a multi-task efficiency claim if the task prompts, templates, pooling, and preprocessing are actually shareable.

### 5.6 Ablations and sensitivity checks

Run a compact, predeclared grid on development data. Include a few interactions that could change the conclusion: pooling × layer, head × calibration method, and input formatting × task. Bound the number of cells and seeds in the protocol, select configurations on validation, and confirm only the chosen comparisons on the locked test. Use targeted single-factor checks for the remaining choices:

- Representation: last token vs mean pooling; final vs intermediate layer; chat template vs plain text.
- Head: raw ridge vs logistic vs calibrated variants; regularization strength; normalization on/off.
- Data: total labeled-development budgets (for example 16/32/64/128/all, where feasible), class balance, grouped vs random splits, and with/without hard negatives. Report the fit/selection allocation at every point.
- Input handling: maximum length, truncation side, document chunking, quoted-instruction treatment, and language subsets.
- Model efficiency: dtype/device and at least one smaller embedding baseline; include feature extraction, head-fit, full request time, and memory. Separately test shared encoding across task heads.
- Policy: threshold sensitivity, review band width, and failure behavior when input is truncated or inference fails.

Produce learning curves, threshold tradeoff plots, reliability plots, cost-quality Pareto plots, and family/source error breakdowns. State when a sensitivity check is underpowered. Avoid searching many variants on the locked test.

### 5.7 Failure analysis and threat validity

For prompt injection, specify the attacker control (user prompt, retrieved document, web page, tool result), defender context, intended unsafe deviation, and whether the detector sees the same text as the protected application. Distinguish jailbreak requests from indirect prompt injection, and benign quotation from executable instruction. Evaluate both standalone text classification and, where claimed, real downstream outcome. For agent tests, compare matched runs with and without the detector and record the legitimate task cost of blocking.

Create an error taxonomy for false positives, false negatives, language, length, source, attack family, and truncation. Review a sampled set of errors with written labeling rules; record annotator agreement and unresolved cases. Test simple adaptive attacks against the frozen detector as a stress study, but keep those results separate from the locked benchmark. Document where input-only detection cannot observe context needed to judge authorization or actual tool misuse.

### 5.8 Paper package and reporting standard

Prepare the paper as a set of traceable claims, each backed by a figure/table and a run artifact. A possible outline is: problem and scope; related work; task and method; dataset/label protocol; experimental setup; main cross-task results; prompt-injection case study; ablations; efficiency; limitations; reproducibility and ethics. The appendix should include prompts/templates, hyperparameter grids, split algorithms, model/data revisions, hardware, full per-dataset results, confidence interval methods, and negative findings.

Use the [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist) as a venue-neutral preparation checklist for reproducibility, statistical reporting, data/code access, and limitations; adapt the final package to the actual venue. Use [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) and [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) as documentation references. Do not assume that passing a checklist establishes novelty or acceptance.

Publication bundle: paper PDF/source, exact experiment configs, environment lock and hardware notes, dataset manifest and split IDs, training/evaluation code, artifact format and compatible weights where rights permit, raw per-example predictions or privacy-preserving substitutes, aggregate tables/figures generated from those predictions, and a `reproduce.md` command sequence. Archive a release tag and checksums so paper numbers cannot drift with `main`.

### 5.9 Benchmark harness and artifact layout

The benchmark should be an executable protocol rather than a collection of handwritten report numbers. Proposed commands and files:

```text
judge-bench data audit --suite configs/suites/paper-v1.yaml
judge-bench run --suite configs/suites/paper-v1.yaml --method configs/methods/gemma-ridge.yaml
judge-bench summarize --runs results/paper-v1/ --output reports/paper-v1/

configs/suites/paper-v1.yaml       # task IDs, dataset revisions, split/group rules, primary metrics
configs/methods/*.yaml             # frozen model revision, representation, head, search budget
research/related_work.md           # cited capabilities and tradeoff matrix
research/pilot_decision.md         # fixed pilot budget, criteria, observed evidence, chosen framing
results/<run_id>/manifest.json     # commit, environment, device, seeds, hashes, timings
results/<run_id>/predictions.*     # row ID, source/group, split, true label, score, label, status
results/<run_id>/metrics.json      # aggregate and group-level metrics with intervals
reports/paper-v1/                 # generated tables, figures, and data/limitations cards
```

The harness should reject split overlap and duplicate IDs, warn on likely near-duplicates, validate label mappings, and refuse to aggregate runs with incompatible task versions or score semantics. It should distinguish a model error from a skipped/unsupported row and report coverage. Store fit and total development-label counts, threshold origin and eligibility status, cached/uncached timing definitions, and source-shortcut audit outputs in the manifest. Provide a `--smoke` fixture that runs in CI without gated weights; run the full suite in a documented research environment. Generate all paper tables and figures from manifests and predictions, never by copying rounded values into Markdown.

### 5.10 Paper-readiness gate

Call the work paper-ready only when all of the following are true:

- The main claim and primary endpoints were fixed before final testing, and the locked test has not been used for iteration.
- At least the task breadth required by the chosen paper framing is complete; all systems use comparable splits, selection budgets, and timing definitions.
- A related-work matrix explains the distinct claim; the pilot decision records why the chosen framing survived the baselines and shortcut checks.
- Learning curves count total development labels, operating thresholds were frozen on validation, and test FPR is reported at those exact thresholds.
- Main tables include strong baselines, uncertainty, per-task results, ablations, compute/latency, and meaningful failure analysis.
- Independent reproduction from a clean machine is documented, or access restrictions and a viable alternative reproduction route are explicit.
- Dataset, model, and code rights are checked; privacy and harmful-content handling are documented.
- Every abstract claim can be traced to a numbered result and exact run manifest. Negative or inconclusive outcomes are described as such.

## 6. Delivery roadmap

### Phase 0 — Correct the benchmark story

Deliverables: fix the README's dead commands; rename the `oob` claim; mark existing scores as uncalibrated; clarify how 0.6 was selected; correct the Giskard p95 computation; resolve the tracked-data policy; add a `legacy_v0` result manifest where the underlying run can be reproduced.

Done when a fresh contributor can understand exactly what was measured and no report calls a same-source split “external” or an unvalidated score a calibrated probability.

### Phase 1 — Make the experiment reproducible

Deliverables: installable package skeleton, supported Python/platform matrix, pinned model and dataset revisions, configurable paths and devices, one shared encoder/scoring implementation, deterministic split audit, and a one-command benchmark. Add unit tests for fit/scoring/serialization and one small end-to-end smoke test that does not require downloading model weights. The UCI SMS pilot supplies the first licensed public binary task with its own head; next move it behind a stable task-spec API, document any task-specific changes, build the related-work matrix, and expand source-shortcut diagnostics beyond length/format.

Done when a second machine can recreate the reported counts and result manifest, predictions from a saved artifact match the original fit within a documented tolerance, and the second task runs end to end without editing injection-specific scripts.

### Pilot decision gate — choose the research claim and default backend

Before large experiments, compare the frozen Gemma head with TF-IDF, a dedicated embedding model plus head, and SetFit where the task fits, on the injection and non-security pilots. Use matched split IDs, fit and total labeled-development budgets, and an explicit compute cap. Include a source/length/format shortcut baseline, full uncached inference cost, and at least one small-label learning-curve point. Reserve the locked confirmatory data; pilot results are exploratory.

Write `research/pilot_decision.md` with the exact configs, run IDs, budget, quality/latency/label criteria, source audit, and outcome. Continue a methods paper only if a predeclared useful region of the quality-cost-label tradeoff survives strong baselines and shortcut checks. If it does not, choose the benchmark or software question and revise the primary claim before acquiring final test results. Select the default encoder from the pilot's deployment tradeoff; Gemma remains a supported reference backend even if another backend is a better default. This gate decides framing and scope, not whether the OSS package can exist.

### Phase 2 — Establish a trustworthy first task

Deliverables: improved data protocol, validation-selected threshold, locked test, representative benign set, external attack stress set, specialist-detector comparisons, calibration check, error analysis, and corrected latency report. Freeze a written experimental protocol before final test runs. Keep all findings, including regressions.

Done when the prompt-injection example has a reproducible scorecard with uncertainty and a stated operating setting. Do not gate this phase on matching the original headline number.

### Phase 3 — General-purpose local API

Deliverables: documented `TaskSpec`, labeled-example format, fit/evaluate/predict API, versioned artifacts, CLI, input-status contract, and policy examples. Turn the SMS pilot into a documented public walkthrough, keeping its head separate from the injection head. Begin the full cross-task suite only after the pilot decision gate.

Done when a contributor can define a new binary text judgment and run it without editing the Gemma-specific benchmark scripts.

### Phase 4 — Research package and paper decision

Deliverables: complete the predeclared cross-task suite, baselines, ablations, statistical analysis, failure audit, and archived reproduction bundle. Decide whether the evidence supports a methods, benchmark, or software paper; narrow the title and claims if it does not support a broad one. Draft the paper and appendix from generated tables and figures, then run an independent reproduction pass.

Done when the section 5.10 gate is met for the selected framing. Paper submission is a separate decision based on the intended venue and its current rules.

### Parallel milestone — OSS alpha

An alpha can ship once Phase 1 and the minimal fit/evaluate/predict API from Phase 3 are working; it does not wait for Phase 2's confirmatory injection study or the full research package. Exit criteria: installable library and CLI; loadable, versioned artifacts; injection and binary non-security examples using separate heads; a small reproducible benchmark report; documented score semantics, model/data rights, and known limits; and one independent clean-machine quickstart. Tag alpha as experimental and label its benchmark numbers exploratory. The full paper suite, multiclass routing, and agent-workflow claims are outside the alpha gate.

### Phase 5 — Public OSS release

Deliverables: license choice after dependency/data/model review; `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, release notes, task/data/model cards, CI, package distribution, minimal quickstart, troubleshooting, and two runnable examples. Publish a clear compatibility table for CPU, Apple Silicon, and any tested accelerator. Decide whether examples download gated weights and make that step explicit.

Done when a new user can install, run a small local example, inspect its limitations, reproduce at least one benchmark, and file a useful issue without private paths or credentials. The alpha and later stable OSS release can precede paper submission once their respective legal and usability checks are complete; they do not require a positive research result.

OSS release gate:

- `pip install` or equivalent succeeds on every advertised platform; the quickstart runs from a clean clone without editing personal paths. Publish exact tested versions and a small no-weights smoke path.
- Core API has versioned input/output schemas, documented error and abstention behavior, artifact compatibility tests, and a deprecation policy. A new task example exercises the API without changing library internals.
- CI runs unit and integration smoke tests, lint/type checks where configured, license checks, and a reproducibility check that regenerated small-result tables match committed fixtures. Full model benchmarks run on a scheduled or manually triggered runner with pinned dependencies.
- Model and dataset licenses, attribution, generated examples, privacy, and redistribution have been reviewed. Provide download instructions for gated or restricted assets; never package private weights or credentials.
- Publish a security policy, issue/PR templates, contribution and governance guidance, release signing/checksums if practical, and a maintainer response policy. State which benchmark claims are historical, exploratory, and confirmatory.
- A clean-room contributor follows the quickstart and one benchmark reproduction guide; record and fix the blockers found before `v1.0`.

### Phase 6 — Expand based on evidence

Possible extensions: multiclass routing, multiple independent binary judgments on one cached representation, ranking pairs, batch and streaming inference, optional service adapters, additional encoders, and label/active-learning utilities. Each extension needs a real user task, an API example, and a benchmark before it becomes a public promise.

## 7. Public narrative and launch materials

The homepage/README should answer, in order: what developers can build; what a judgment is; a short installation and prediction example; how to train on their labels; what the first benchmark actually shows; and where the limitations live. Lead with the SMS triage example while it is the only independent binary task; replace or supplement it with validated document triage when long-text evidence exists. Show prompt-injection detection as a deeper case study. Introduce ticket routing only after multiclass support and its own evaluation exist.

Suggested demo story: define a label with examples → fit a head over a frozen encoder → inspect validation errors → select a policy threshold → save the artifact → predict locally with a versioned result. Show a few realistic failures as well as successes. Explain that “general purpose” describes the toolkit and workflow, not a universal pretrained safety classifier.

Positioning to test with early users: “local semantic judgments for application code,” “small classifiers on frozen LLM features,” and “reproducible text decision models.” Avoid “guardrail for everything” or an accuracy claim without task and dataset context. Invite community contributions of task adapters, datasets with verified redistribution rights, and independent benchmark runs.

## 8. Decisions needed before implementation

These are project choices, with proposed defaults so work can proceed:

| Decision | Proposed default | Revisit when |
| --- | --- | --- |
| First public abstraction | Binary text judgment with explicit label definitions | A second task reveals a missing input or output concept |
| First reference encoder | Local Gemma 3 1B IT behind an adapter | Pilot chooses the public default on measured quality, full cost, and accessibility |
| First head | Regularized logistic regression, retaining ridge as baseline | Calibration and task results support a different method |
| Result field | `score` plus documented semantics | Held-out calibration supports probability terminology |
| Distribution | Python library and CLI first | Stable API and demand justify an HTTP server |
| Benchmark data | Publish manifests; redistribute only after license review | Data rights and reproducibility tradeoffs are resolved |
| Naming | Keep repository name for now; use task-neutral package/API names | The two-task demo and public positioning are ready |

### Experiment budget and stopping rules

Before the pilot, put a concrete resource envelope in `research/pilot_decision.md`: available hardware, wall-clock and compute limits, maximum manual labels, candidate methods, allowed hyperparameter trials, seeds, and dataset sizes. Before the full study, revise those limits in the predeclared protocol. Estimate the run count as tasks × methods × budgets × seeds × selected ablation cells; time a small sample to project total extraction and training cost. Cache features only for experiments whose method permits it, and always report uncached serving cost.

Stop expanding a methods study if the pilot misses its predeclared usefulness margin against strong baselines, source shortcuts dominate the result, or the planned final test cannot measure the primary FPR/quality target with adequate precision under the budget. Narrow to a defensible benchmark/software contribution or collect better data before further model sweeps. Stop adding tasks or ablation cells when they do not test a stated claim. Record the decision and all exploratory runs; a negative pilot does not stop the OSS alpha.

## 9. Immediate next actions

1. Reproduce the current numbers with a captured environment and save a manifest. If that is not possible, label the reports as historical and keep them separate from new results.
2. Inspect source labels, data licenses, tracked snapshots, and exact/near-duplicate overlap. Correct the session memo and documentation accordingly.
3. Draft `research/related_work.md`; the licensed SMS pilot is selected, and its lack of row-level source groups is documented. Choose a second dataset with source groups for that audit.
4. Extract the shared encoder/head code and artifact format while preserving `legacy_v0`; run injection and non-security pilots through the same API.
5. Fix the pilot budget and compare TF-IDF, embedding plus head, Gemma plus head, and SetFit where applicable. Count all development labels and full uncached inference; document the research framing and default-backend decision.
6. Draft the predeclared full-study protocol: primary metric, useful-effect margin, baseline roster, group/source split rule, final-test source, fixed validation-threshold rule, uncertainty and sample-size target, and run budget.
7. Build a representative benign set and locked evaluation path before revising threshold claims; then choose additional non-security tasks only where they test the selected claim.

This order protects credibility: first make the existing case study reproducible, then test the workflow on distinct tasks under a declared protocol, then release the toolkit and any paper with claims matched to evidence.
