# Pilot decision gate: provisional software-first framing

Date: 2026-10-16. This decision is based on exploratory tests that have been viewed during development. It is **not** a preregistered methods-study decision or a final paper claim. No useful-effect margin, latency ceiling, or total compute budget was fixed before all pilot results; those cannot be retroactively supplied.

## Evidence available

| Task | Fit / validation / test labels | Best observed thresholded result | Cost and counterevidence |
| --- | --- | --- | --- |
| UCI SMS ham/spam, same-source deduplicated split | 128 / 200 / 400 | Frozen MiniLM F1 0.930; Gemma F1 0.876; TF-IDF F1 0.883; SetFit F1 0.889 | Gemma warm mean request ~74 ms versus MiniLM ~3.2 ms and TF-IDF ~0.13 ms; no source-group holdout |
| UCI recipe-review sentiment, recipe-group-held-out split | 128 / 200 / 320 | Gemma macro F1 0.896; SetFit 0.848; TF-IDF 0.792; frozen MiniLM 0.681 | Gemma warm mean request ~85 ms versus SetFit ~4.1 ms and TF-IDF ~0.13 ms; Gemma-minus-SetFit paired recipe interval for macro F1 is [-0.009, 0.099] |

The pilots use the same 328 total labeled-development budget per task, with 128 fitting and 200 validation labels. Source ZIP hashes, split IDs and digest, code hashes, model revision where available, and run environments are in ignored JSON manifests. See [SMS report](pilot_report.md), [recipe report](recipe_pilot_report.md), and their protocols for the exact settings. The two tasks do not satisfy the plan's proposed minimum of three independent tasks across at least two non-security domains; both are short/medium text classification, and the recipe result uses a noisy star-rating proxy. An aggregate audit found 5 test reviews from reviewers seen in development despite recipe-disjoint groups. SetFit recipe was added only after the five-method test had been viewed. The original prompt-injection scores also come from historical, test-influenced thresholding and do not repair this limitation.

## Decision now

Do not claim that frozen Gemma plus a linear head is a generally superior method, and do not choose Gemma as the OSS default on these results. The ranking reverses across tasks and the strongest recipe alternative, SetFit, is much cheaper at inference while not clearly separated by this one group bootstrap. A general-purpose, backend-neutral **software and evaluation workflow** is the most defensible provisional product framing: define a task, fit a small head or supported alternative, retain provenance and score semantics, evaluate on appropriate groups, and let applications choose policy. Prompt injection remains a reference example, not the project's sole identity. This is a product/research direction, not a publication acceptance claim.

## Gate before a methods paper or public-default decision

1. Write a new prospective protocol with a primary task matrix, explicit useful-effect/noninferiority margin and latency/compute budget, fixed seeds and hyperparameter search, primary endpoint, and multiplicity rule. Do this before accessing new test sets.
2. Add an independently sourced, longer-document binary task with verified license and genuine source groups. Keep original official split roles if present. Audit reviewer or author overlap, near duplicates, and pretrained-model exposure where possible.
3. On development data only, run repeated grouped splits and at least one smaller labeled-development budget. Compare TF-IDF, frozen MiniLM, SetFit, Gemma, and shortcut baselines with the same row IDs and selection budget. Treat training cost, memory, warm/cold request latency, and model size as part of the comparison.
4. Reserve untouched final tests. Select model, threshold, and operating target on development data, then open each final test once. Report group-aware uncertainty, class prevalence, calibration status, and all post-test deviations.
5. Independently test a clean install and reproduce the data audit and one benchmark run. Resolve model weight pinning, dataset redistribution rights, and artifact compatibility before OSS alpha.

An OSS alpha may proceed before the paper gate if it clearly labels these pilots exploratory and offers a useful, tested, backend-neutral API. A negative methods result does not invalidate that software goal.
