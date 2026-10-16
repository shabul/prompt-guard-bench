"""Exploratory binary SMS triage pilot: Gemma features versus simple baselines."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import random
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

from gemma_benchmark import GemmaEncoder, classification_metrics, dataset_digest


SOURCE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SOURCE_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
TASK_ID = "sms-spam-v1"
SPLIT_SIZES = {"train": {0: 64, 1: 64}, "validation": {0: 160, 1: 40}, "test": {0: 320, 1: 80}}


def load_sms(source_url=SOURCE_URL):
    with urllib.request.urlopen(source_url, timeout=30) as response:
        archive = response.read()
    actual_hash = hashlib.sha256(archive).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"source archive hash differs: {actual_hash}")
    with zipfile.ZipFile(io.BytesIO(archive)) as contents:
        lines = contents.read("SMSSpamCollection").decode("latin-1").splitlines()
    unique = {}
    for index, line in enumerate(lines):
        kind, text = line.split("\t", 1)
        if kind not in {"ham", "spam"} or not text.strip():
            raise ValueError(f"bad source row {index}")
        key = text.strip().casefold()
        label = int(kind == "spam")
        if key in unique and unique[key]["label"] != label:
            raise ValueError(f"conflicting labels for source row {index}")
        unique.setdefault(key, {"id": hashlib.sha256(key.encode("utf-8")).hexdigest(), "text": text, "label": label})
    return list(unique.values()), {"archive_sha256": actual_hash, "source_rows": len(lines), "unique_normalized_rows": len(unique)}


def make_splits(rows, seed=42):
    rng = random.Random(seed)
    by_label = {label: [row for row in rows if row["label"] == label] for label in (0, 1)}
    for group in by_label.values():
        rng.shuffle(group)
    positions = {0: 0, 1: 0}
    splits = {}
    for name, counts in SPLIT_SIZES.items():
        selected = []
        for label, count in counts.items():
            start = positions[label]
            selected.extend(by_label[label][start:start + count])
            positions[label] += count
        if len(selected) != sum(counts.values()):
            raise ValueError(f"not enough rows for {name}")
        rng.shuffle(selected)
        splits[name] = selected
    ids = [row["id"] for split in splits.values() for row in split]
    if len(ids) != len(set(ids)):
        raise ValueError("normalized duplicate crossed split boundary")
    return splits


def choose_f1_threshold(scores, labels):
    """Select on validation only; highest threshold wins an F1 tie."""
    candidates = sorted(set(float(score) for score in scores), reverse=True)
    candidates.insert(0, float(np.nextafter(max(candidates), np.inf)))
    best = None
    for threshold in candidates:
        value = f1_score(labels, np.asarray(scores) >= threshold, zero_division=0)
        if best is None or value > best[0]:
            best = (float(value), threshold)
    return {"threshold": best[1], "validation_f1": best[0], "rule": "maximum validation F1, highest threshold on tie"}


def summary_for(scores_by_split, splits):
    validation_labels = [row["label"] for row in splits["validation"]]
    choice = choose_f1_threshold(scores_by_split["validation"], validation_labels)
    test_labels = [row["label"] for row in splits["test"]]
    test_rows = [{"score": float(score), "label": label} for score, label in zip(scores_by_split["test"], test_labels)]
    return {
        "selection": choice,
        "test": classification_metrics(test_rows, choice["threshold"]),
        "test_average_precision": float(average_precision_score(test_labels, scores_by_split["test"])),
    }


def shape_features(text):
    length = max(len(text), 1)
    return [
        len(text),
        len(text.split()),
        sum(char.isdigit() for char in text) / length,
        sum(char.isupper() for char in text) / length,
        text.count("!"),
        int(bool(re.search(r"https?://|www\.", text, re.IGNORECASE))),
    ]


def fit_linear(features, labels, seed):
    scaler = StandardScaler().fit(features)
    head = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(scaler.transform(features), labels)
    return scaler, head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-only", action="store_true", help="run split and cheap baselines without loading Gemma")
    parser.add_argument("--output", default="sms_pilot_results.json")
    parser.add_argument("--artifact", default="sms_gemma_head.json")
    args = parser.parse_args()

    source_rows, source_info = load_sms()
    splits = make_splits(source_rows, args.seed)
    labels = {name: [row["label"] for row in rows] for name, rows in splits.items()}
    texts = {name: [row["text"] for row in rows] for name, rows in splits.items()}
    report = {
        "task_id": TASK_ID,
        "status": "exploratory pilot; test examined during development",
        "source": {"url": SOURCE_URL, "license": "CC BY 4.0 per UCI repository", **source_info},
        "split_rule": "deduplicate by stripped casefolded text; stratified seeded sampling from single source pool",
        "seed": args.seed,
        "split_counts": {name: {str(label): count for label, count in sizes.items()} for name, sizes in SPLIT_SIZES.items()},
        "split_ids": {name: [row["id"] for row in rows] for name, rows in splits.items()},
        "data_sha256": dataset_digest(splits),
        "timing_definition": "warm single-text test requests; Gemma mean is separately measured encoding plus head scoring",
        "code_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("pilot_sms.py", "gemma_benchmark.py", "requirements.lock.txt")
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "methods": {},
    }

    started = time.perf_counter()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    train_tfidf = vectorizer.fit_transform(texts["train"])
    tfidf_head = LogisticRegression(C=1.0, max_iter=1000, random_state=args.seed).fit(train_tfidf, labels["train"])
    tfidf_fit = time.perf_counter() - started
    tfidf_scores = {"validation": tfidf_head.decision_function(vectorizer.transform(texts["validation"]))}
    tfidf_test_times = []
    tfidf_test_scores = []
    for text in texts["test"]:
        started = time.perf_counter()
        tfidf_test_scores.append(float(tfidf_head.decision_function(vectorizer.transform([text]))[0]))
        tfidf_test_times.append((time.perf_counter() - started) * 1000)
    tfidf_scores["test"] = np.asarray(tfidf_test_scores)
    report["methods"]["tfidf_logistic"] = {
        **summary_for(tfidf_scores, splits),
        "fit_seconds": tfidf_fit,
        "mean_test_request_ms": float(np.mean(tfidf_test_times)),
    }

    started = time.perf_counter()
    shape = {name: np.asarray([shape_features(text) for text in group], dtype=float) for name, group in texts.items()}
    shape_scaler, shape_head = fit_linear(shape["train"], labels["train"], args.seed)
    shape_fit = time.perf_counter() - started
    shape_scores = {"validation": shape_head.decision_function(shape_scaler.transform(shape["validation"]))}
    shape_test_times = []
    shape_test_scores = []
    for text in texts["test"]:
        started = time.perf_counter()
        shape_test_scores.append(float(shape_head.decision_function(shape_scaler.transform([shape_features(text)]))[0]))
        shape_test_times.append((time.perf_counter() - started) * 1000)
    shape_scores["test"] = np.asarray(shape_test_scores)
    report["methods"]["length_format_logistic"] = {
        **summary_for(shape_scores, splits),
        "fit_seconds": shape_fit,
        "mean_test_request_ms": float(np.mean(shape_test_times)),
    }

    if not args.baseline_only:
        encoder = GemmaEncoder(args.model, args.device)
        vectors = {}
        embed_timings = {}
        for name, rows in splits.items():
            embedded = [encoder.embed(row["text"]) for row in rows]
            vectors[name] = np.stack([vector.numpy() for vector, _ in embedded])
            embed_timings[name] = [timing for _, timing in embedded]
        started = time.perf_counter()
        scaler, head = fit_linear(vectors["train"], labels["train"], args.seed)
        fit_seconds = time.perf_counter() - started
        gemma_scores = {"validation": head.decision_function(scaler.transform(vectors["validation"]))}
        head_test_times = []
        gemma_test_scores = []
        for vector in vectors["test"]:
            started = time.perf_counter()
            gemma_test_scores.append(float(head.decision_function(scaler.transform([vector]))[0]))
            head_test_times.append((time.perf_counter() - started) * 1000)
        gemma_scores["test"] = np.asarray(gemma_test_scores)
        gemma_result = summary_for(gemma_scores, splits)
        mean_embedding_ms = sum(t["embedding_latency_ms"] for t in embed_timings["test"]) / len(embed_timings["test"])
        gemma_result.update({
            "model_fingerprints": encoder.fingerprints(),
            "head_fit_seconds": fit_seconds,
            "embedding_seconds_by_split": {name: sum(t["embedding_latency_ms"] for t in group) / 1000 for name, group in embed_timings.items()},
            "mean_test_embedding_ms": mean_embedding_ms,
            "mean_test_head_ms": float(np.mean(head_test_times)),
            "mean_test_request_ms": mean_embedding_ms + float(np.mean(head_test_times)),
            "truncated_rows": {name: sum(t["truncated"] for t in group) for name, group in embed_timings.items()},
        })
        artifact = {
            "version": "binary_logistic_v1",
            "task_id": TASK_ID,
            "labels": {"0": "ham", "1": "spam"},
            "score_semantics": "uncalibrated_logistic_decision_function",
            "model_id": encoder.model_id,
            "model_fingerprints": encoder.fingerprints(),
            "pooling": "final_hidden_state_last_token",
            "max_length": encoder.max_length,
            "train_data_sha256": dataset_digest({"train": splits["train"]}),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": head.coef_[0].tolist(),
            "intercept": float(head.intercept_[0]),
            "validation_threshold": gemma_result["selection"],
        }
        Path(args.artifact).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        restored = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
        first = vectors["test"][0]
        restored_score = float(np.dot((first - restored["mean"]) / restored["scale"], restored["coef"]) + restored["intercept"])
        if not np.isclose(restored_score, gemma_scores["test"][0], atol=1e-6):
            raise ValueError("saved Gemma head does not reproduce fitted score")
        report["methods"]["gemma_logistic"] = gemma_result
        report["artifact"] = args.artifact

    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, result in report["methods"].items():
        print(name, "test:", result["test"], "AP:", round(result["test_average_precision"], 4))
    print("report:", args.output)


if __name__ == "__main__":
    main()
