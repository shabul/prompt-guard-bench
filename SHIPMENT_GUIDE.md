# Shipping the Gemma 3 1B prompt-injection detector

This repository contains a local prompt-injection screening component built from `google/gemma-3-1b-it` and a frozen hidden-state ridge classifier. Gemma is used as an encoder; it does not generate the application response. The classifier returns an injection probability that your application converts into an allow, review, or block decision.

## What is included

- `evaluate_gemma1b.py`: evaluates the detector on the prepared deepset dataset.
- `prepare_gemma1b_dataset.py`: downloads the source dataset, keeps English rows, and creates deterministic calibration, test, and OOB splits.
- `evaluate_giskard.py`: evaluates recall and latency on Giskard's positive-only attack set.
- `GEMMA_1B_REPORT.md` and `GISKARD_REPORT.md`: recorded benchmark results.

The current classifier is a research benchmark and a strong starting point for a service. It is not, by itself, a complete security boundary.

## Prerequisites

Required:

- macOS with Apple Silicon recommended; CPU also works but is slower.
- Python 3.11 or newer compatible with the pinned environment.
- At least 8 GB available memory; 16 GB or more is recommended.
- Local Gemma 3 1B IT weights at:

  `/Users/Shabul/model-weights/Google/gemma-3-1b-it`

- Access to the Gemma model under its Hugging Face license. Authenticate with Hugging Face and accept the model terms before downloading it.

Create the environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

For a different machine, either change `MODEL` in the scripts or expose the model path through a small configuration change. Do not download weights on every request.

## Reproduce the benchmark

Prepare the deterministic English dataset and disjoint splits:

```bash
.venv/bin/python prepare_gemma1b_dataset.py
```

This creates 150 calibration rows, 100 test rows, and 101 out-of-bag rows using seed 42. Fit the classifier on calibration and evaluate it:

```bash
.venv/bin/python evaluate_gemma1b.py
```

Evaluate the external Giskard attack set:

```bash
.venv/bin/python evaluate_giskard.py
```

The Giskard file contains attacks only. It is valid for measuring attack recall, but not precision, accuracy, or F1. Those metrics require benign examples.

## Recommended production design

Run one long-lived model process per worker. Load the tokenizer, Gemma weights, calibration statistics, and ridge weights once at startup. For every request:

1. Normalize and length-limit the text.
2. Apply cheap deterministic checks.
3. Embed the text using the exact chat template used during calibration.
4. Standardize the final hidden state with the saved calibration mean and scale.
5. Apply the saved ridge weights and sigmoid to obtain `probability_injection`.
6. Pass the score to a policy layer.
7. Log the model version, score, action, latency, and input source.

Do not refit the head at request time. Do not change the chat template, truncation length, tokenizer, model revision, or calibration data without producing a new model version and rerunning evaluation.

A practical initial policy is:

```text
score < 0.40       allow
0.40 <= score < 0.60  allow with monitoring or secondary checks
0.60 <= score < 0.75  review, sanitize, or isolate
score >= 0.75       block or require explicit review
```

These are starting points, not universal thresholds. The current experiment showed excellent attack recall around 0.6–0.7, but the Giskard set has no benign rows and therefore cannot establish the false-positive rate.

## Where to apply the detector

Scan more than the initial user message. Depending on the application, inspect:

- user messages;
- retrieved documents before they enter the model context;
- uploaded files and web content;
- tool outputs;
- tool arguments and planned actions;
- conversation history when it is reused across turns.

For an agent, a high-risk score should prevent tool execution, not merely change the assistant's wording. Keep authorization checks separate: prompt-injection detection does not prove that a requested action is permitted.

## API contract for an application wrapper

Expose the detector behind an internal endpoint such as:

```http
POST /v1/prompt-injection/check
Content-Type: application/json
```

```json
{
  "text": "content to inspect",
  "source": "user",
  "request_id": "optional-id"
}
```

Return a versioned, explainable decision:

```json
{
  "decision": "review",
  "probability_injection": 0.68,
  "threshold": 0.60,
  "model_version": "gemma-3-1b-it-ridge-v1",
  "source": "user",
  "latency_ms": 342
}
```

The repository currently provides the inference/evaluation logic, not a production HTTP server. The wrapper should own authentication, request limits, tracing, redaction, retries, health checks, and policy enforcement.

## Validation before enabling blocking

Build a representative labeled validation set with both classes. Include ordinary questions, coding requests, long documents, quoted attacks, multilingual inputs, RAG content, obfuscated attacks, and tool-use requests. Keep calibration, tuning, test, and OOD sets disjoint.

Report at minimum:

- precision, recall, F1, accuracy, and confusion matrix;
- false-positive rate on benign traffic;
- recall by attack family;
- p50, p95, and p99 latency;
- timeout and model-load failure rates;
- results by input source and input length.

Deploy in shadow mode first. Review false positives and missed attacks, select a threshold against an explicit business cost, then enable blocking only for high-confidence cases. Keep a human-review path for borderline scores.

## Operational safeguards

- Pin the model revision, tokenizer, calibration artifact, and code commit.
- Store weights outside the repository and never commit them.
- Do not log raw sensitive prompts by default; log hashes or redacted samples.
- Add maximum input length and request timeouts.
- Fail closed for high-risk tool actions if the detector is unavailable; fail open only where the product owner has explicitly accepted that risk.
- Monitor score distribution, latency, false positives, and newly observed attack patterns.
- Re-run the full evaluation after changing the model, prompt format, tools, retrieval system, or threshold.
- Treat detector output as one signal in a defense-in-depth system, alongside authorization, sandboxing, output validation, and least-privilege tool access.

## Current evidence and limitations

On the 35-row positive-only Giskard set, the current Gemma 1B head detected 35/35 attacks at threshold 0.6 and 34/35 at 0.7. Mean latency was approximately 339 ms per prompt on the development Mac. These results demonstrate recall and local feasibility; they do not establish production precision, generalization, or safety guarantees.
