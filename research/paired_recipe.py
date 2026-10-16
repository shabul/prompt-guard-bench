"""Exploratory paired, recipe-cluster bootstrap for the recipe pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


METHODS = ("majority", "tfidf", "shape", "minilm", "gemma", "setfit")


def load_aligned_reports(directory):
    reports = {name: json.loads((directory / f"recipe_{name}_results.json").read_text()) for name in METHODS}
    reference = reports[METHODS[0]]
    keys = ("data_sha256", "split_ids", "split_counts", "recipe_groups", "task_id")
    for name, report in reports.items():
        if any(report[key] != reference[key] for key in keys):
            raise ValueError(f"{name} differs in data, groups, or split")
        for split in ("validation", "test"):
            left = [(row["id"], row["label"], row["recipe_code"]) for row in reference["predictions"][split]]
            right = [(row["id"], row["label"], row["recipe_code"]) for row in report["predictions"][split]]
            if left != right:
                raise ValueError(f"{name} differs in {split} prediction rows")
    return reports


def metrics(rows):
    truth = [row["label"] for row in rows]
    predicted = [row["predicted"] for row in rows]
    return {
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
    }


def paired_bootstrap(reports, challenger, reference, repetitions=2000, seed=20261016):
    first = reports[challenger]["predictions"]["test"]
    second = reports[reference]["predictions"]["test"]
    groups = sorted({row["recipe_code"] for row in first})
    if len(groups) != 20:
        raise ValueError(f"expected 20 independent recipe groups, found {len(groups)}")
    by_group = {
        name: {group: [row for row in rows if row["recipe_code"] == group] for group in groups}
        for name, rows in ((challenger, first), (reference, second))
    }
    observed = {metric: metrics(first)[metric] - metrics(second)[metric] for metric in ("macro_f1", "balanced_accuracy")}
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in observed}
    excluded = 0
    for _ in range(repetitions):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sample = {name: [row for group in selected for row in by_group[name][group]] for name in (challenger, reference)}
        if len({row["label"] for row in sample[challenger]}) != 2:
            excluded += 1
            continue
        left, right = metrics(sample[challenger]), metrics(sample[reference])
        for metric in draws:
            draws[metric].append(left[metric] - right[metric])
    return {
        "comparison": f"{challenger} minus {reference}",
        "observed": observed,
        "percentile_95_interval": {metric: [float(x) for x in np.quantile(values, [0.025, 0.975])] for metric, values in draws.items()},
        "unit": "held-out recipe code",
        "groups": len(groups),
        "repetitions": repetitions,
        "excluded_one_class_replicates": excluded,
        "seed": seed,
        "interpretation": "exploratory descriptive uncertainty on one selected pilot split; not a confirmatory interval",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("recipe_paired_results.json"))
    args = parser.parse_args()
    reports = load_aligned_reports(args.directory)
    result = {
        "data_sha256": reports["gemma"]["data_sha256"],
        "comparisons": [paired_bootstrap(reports, "gemma", method) for method in ("tfidf", "minilm", "setfit")],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
