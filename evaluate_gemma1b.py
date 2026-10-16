"""Reproduce the historical Gemma ridge benchmark and save its fitted head."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from gemma_benchmark import GemmaEncoder, binary_metrics, dataset_digest, fit_legacy_ridge, legacy_score, load_artifact, save_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-1b-it", help="Hugging Face model ID or local weights path")
    parser.add_argument("--data", default="data/gemma1b-prompt-injections")
    parser.add_argument("--artifact", default="gemma1b_legacy_head.json")
    parser.add_argument("--output", default="gemma1b_results.json")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from datasets import load_from_disk

    data = load_from_disk(args.data)
    required = ("calibration", "test", "oob")
    if not all(split in data and len(data[split]) for split in required):
        raise ValueError(f"dataset must contain nonempty {required} splits")
    encoder = GemmaEncoder(args.model, args.device, args.trust_remote_code)

    def collect(split):
        rows = []
        for source in data[split]:
            vector, timing = encoder.embed(source["text"])
            rows.append({"vector": vector, "label": int(source["label"]), **timing})
        return rows

    training = collect("calibration")
    head = fit_legacy_ridge([row["vector"] for row in training], [row["label"] for row in training])
    data_hash = dataset_digest({split: data[split] for split in required})
    artifact = save_artifact(args.artifact, encoder, head, data_hash)
    loaded_head = load_artifact(args.artifact)["head"]
    if abs(legacy_score(training[0]["vector"], head) - legacy_score(training[0]["vector"], loaded_head)) > 1e-7:
        raise ValueError("saved head does not reproduce the fitted score")

    scored_splits = {}
    for split in ("test", "oob"):
        rows = collect(split)
        scored_splits[split] = [
            {key: value for key, value in row.items() if key != "vector"} | {"score": legacy_score(row["vector"], head)}
            for row in rows
        ]
    thresholds = {
        str(value): {split: binary_metrics(rows, value) for split, rows in scored_splits.items()}
        for value in (0.5, 0.6, 0.7, 0.8, 0.9)
    }
    summary = {
        "model": args.model,
        "method": "legacy frozen final hidden state + ridge + sigmoid score",
        "score_semantics": artifact["score_semantics"],
        "device": encoder.device,
        "dtype": "float32",
        "calibration_rows": len(training),
        "test_rows": len(data["test"]),
        "oob_rows": len(data["oob"]),
        "split_provenance": "deepset same-source test and second holdout; exploratory",
        "threshold_origin": "historical displayed thresholds; 0.6 was selected after evaluation",
        "thresholds": thresholds,
        "dataset_sha256": data_hash,
        "artifact": args.artifact,
        "model_fingerprints": encoder.fingerprints(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
        },
        "truncated_rows": {split: sum(row["truncated"] for row in rows) for split, rows in scored_splits.items()},
    }
    Path(args.output).write_text(json.dumps({"summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
