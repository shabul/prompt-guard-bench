"""Frozen MiniLM embedding baseline on the existing exploratory SMS split."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from pilot_sms import SOURCE_SHA256, fit_linear, load_sms, make_splits, summary_for
from gemma_benchmark import dataset_digest


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EXPECTED_SPLIT_SHA256 = "dc74e3dd1dfcbed40c774449a03a2fc5aea2986ad8a3e7f2da683c3c4b063f81"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--output", default="sms_embedding_results.json")
    parser.add_argument("--artifact", default="sms_embedding_head.json")
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("this pilot extension compares only the fixed seed-42 split")

    source, source_info = load_sms()
    splits = make_splits(source, args.seed)
    data_hash = dataset_digest(splits)
    if data_hash != EXPECTED_SPLIT_SHA256 or source_info["archive_sha256"] != SOURCE_SHA256:
        raise ValueError("SMS split differs from the original pilot")

    started = time.perf_counter()
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device=args.device)
    load_seconds = time.perf_counter() - started
    texts = {name: [row["text"] for row in rows] for name, rows in splits.items()}
    labels = {name: [row["label"] for row in rows] for name, rows in splits.items()}

    vectors = {}
    feature_seconds = {}
    for name in ("train", "validation"):
        started = time.perf_counter()
        vectors[name] = model.encode(texts[name], batch_size=32, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        feature_seconds[name] = time.perf_counter() - started
    test_vectors = []
    test_embedding_ms = []
    for text in texts["test"]:
        started = time.perf_counter()
        test_vectors.append(model.encode(text, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False))
        test_embedding_ms.append((time.perf_counter() - started) * 1000)
    vectors["test"] = np.stack(test_vectors)
    feature_seconds["test"] = sum(test_embedding_ms) / 1000

    started = time.perf_counter()
    scaler, head = fit_linear(vectors["train"], labels["train"], args.seed)
    head_fit_seconds = time.perf_counter() - started
    validation_scores = head.decision_function(scaler.transform(vectors["validation"]))
    test_scores = []
    test_head_ms = []
    for vector in vectors["test"]:
        started = time.perf_counter()
        test_scores.append(float(head.decision_function(scaler.transform([vector]))[0]))
        test_head_ms.append((time.perf_counter() - started) * 1000)
    result = summary_for({"validation": validation_scores, "test": np.asarray(test_scores)}, splits)

    artifact = {
        "version": "binary_logistic_v1",
        "task_id": "sms-spam-v1",
        "score_semantics": "uncalibrated_logistic_decision_function",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pooling": "sentence_transformers_default",
        "max_seq_length": model.max_seq_length,
        "train_data_sha256": dataset_digest({"train": splits["train"]}),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": head.coef_[0].tolist(),
        "intercept": float(head.intercept_[0]),
        "validation_threshold": result["selection"],
    }
    Path(args.artifact).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    restored = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    sample = vectors["test"][0]
    restored_score = float(np.dot((sample - restored["mean"]) / restored["scale"], restored["coef"]) + restored["intercept"])
    if not np.isclose(restored_score, test_scores[0], atol=1e-6):
        raise ValueError("saved embedding head does not reproduce score")

    report = {
        "status": "exploratory; test viewed during original pilot",
        "method": "frozen MiniLM sentence embeddings + standardized logistic head",
        "source_archive_sha256": SOURCE_SHA256,
        "data_sha256": data_hash,
        "split_counts": {name: {str(label): labels[name].count(label) for label in (0, 1)} for name in splits},
        "development_labels": 328,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "artifact": args.artifact,
        "result": result,
        "timing": {
            "model_load_seconds": load_seconds,
            "feature_seconds_by_split": feature_seconds,
            "head_fit_seconds": head_fit_seconds,
            "mean_test_request_ms": float(np.mean(np.asarray(test_embedding_ms) + np.asarray(test_head_ms))),
            "definition": "warm one-message encode plus head score; train and validation encoded in batches of 32",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "transformers": importlib.metadata.version("transformers"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ("pilot_sms_embedding.py", "pilot_sms.py", "gemma_benchmark.py")},
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("MiniLM:", result["test"], "AP:", round(result["test_average_precision"], 4))
    print("mean warm request ms:", round(report["timing"]["mean_test_request_ms"], 3))
    print("report:", args.output)


if __name__ == "__main__":
    main()
