"""Evaluate a saved legacy Gemma head on Giskard's positive-only CSV."""

import argparse
import csv
import json
from pathlib import Path

from gemma_benchmark import GemmaEncoder, legacy_score, load_artifact, nearest_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="gemma1b_legacy_head.json")
    parser.add_argument("--csv", default="data/giskard_prompt_injections.csv")
    parser.add_argument("--output", default="giskard_results.json")
    parser.add_argument("--model", help="override model location; must match saved model ID or be its local copy")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    with Path(args.csv).open(encoding="utf-8", newline="") as file:
        source_rows = list(csv.DictReader(file))
    if not source_rows or not all(row.get("group") in {"Jailbreak", "Hijacking attacks"} for row in source_rows):
        raise ValueError("expected a nonempty Giskard attack-only CSV")
    encoder = GemmaEncoder(args.model or artifact["model_id"], args.device, args.trust_remote_code, artifact["max_length"])
    if encoder.tokenizer.truncation_side != artifact["tokenizer_truncation_side"]:
        raise ValueError("tokenizer truncation side differs from fitted artifact")
    for field, actual in encoder.fingerprints().items():
        if actual != artifact.get(field):
            raise ValueError(f"{field} differs from fitted artifact")

    rows = []
    for source in source_rows:
        vector, timing = encoder.embed(source["prompt"])
        rows.append({
            "index": source["index"],
            "name": source["name"],
            "group": source["group"],
            "score": legacy_score(vector, artifact["head"]),
            **timing,
        })
    thresholds = {}
    for value in (0.5, 0.6, 0.7, 0.8, 0.9):
        detected = sum(row["score"] >= value for row in rows)
        thresholds[str(value)] = {"detected": detected, "recall": detected / len(rows)}
    latencies = [row["forward_latency_ms"] for row in rows]
    summary = {
        "model": encoder.model_id,
        "artifact": args.artifact,
        "score_semantics": artifact["score_semantics"],
        "dataset": "Giskard-AI/prompt-injections prompt_injections.csv",
        "n": len(rows),
        "groups": {group: sum(row["group"] == group for row in rows) for group in sorted({row["group"] for row in rows})},
        "positive_only": True,
        "thresholds": thresholds,
        "mean_forward_latency_ms": sum(latencies) / len(latencies),
        "p95_forward_latency_ms": nearest_rank(latencies, 0.95),
        "quantile_method": "nearest_rank",
        "truncated_rows": sum(row["truncated"] for row in rows),
    }
    Path(args.output).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
