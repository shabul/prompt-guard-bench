"""Recipe-review binary pilot with entire recipe groups held out."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

from gemma_benchmark import GemmaEncoder, dataset_digest
from pilot_sms import fit_linear, shape_features, summary_for
from pilot_sms_embedding import MODEL_ID as EMBEDDING_MODEL_ID, MODEL_REVISION as EMBEDDING_REVISION


SOURCE_URL = "https://archive.ics.uci.edu/static/public/911/recipe+reviews+and+user+feedback+dataset.zip"
SOURCE_SHA256 = "aff630962b32fb8e8fc43ba8eddf79c45e2ae70321e81dda383c95f04887b77f"
SOURCE_FILENAME = "Recipe Reviews and User Feedback Dataset.csv"
SPLIT_COUNTS = {"train": {0: 64, 1: 64}, "validation": {0: 40, 1: 160}, "test": {0: 64, 1: 256}}


def load_recipe_rows():
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        archive = response.read()
    digest = hashlib.sha256(archive).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"recipe archive hash differs: {digest}")
    with zipfile.ZipFile(io.BytesIO(archive)) as contents:
        frame = pd.read_csv(io.BytesIO(contents.read(SOURCE_FILENAME)))
    required = {"text", "stars", "recipe_code"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing source columns: {required - set(frame.columns)}")
    candidates = {}
    for _, source in frame.iterrows():
        if source["stars"] not in (1, 2, 4, 5) or not isinstance(source["text"], str):
            continue
        text = source["text"].strip()
        if not text:
            continue
        key = text.casefold()
        row = {
            "id": hashlib.sha256(key.encode("utf-8")).hexdigest(),
            "text": text,
            "label": int(source["stars"] in (4, 5)),
            "recipe_code": int(source["recipe_code"]),
        }
        candidates.setdefault(key, []).append(row)
    conflicting = sum(len({row["label"] for row in group}) > 1 for group in candidates.values())
    rows = [group[0] for group in candidates.values() if len({row["label"] for row in group}) == 1]
    return rows, {"archive_sha256": digest, "source_rows": len(frame), "unique_usable_rows": len(rows), "conflicting_normalized_text_groups_removed": conflicting}


def make_recipe_splits(rows, seed=42):
    rng = random.Random(seed)
    groups = sorted({row["recipe_code"] for row in rows})
    if len(groups) != 100:
        raise ValueError(f"expected 100 recipe groups, found {len(groups)}")
    rng.shuffle(groups)
    group_sets = {"train": set(groups[:60]), "validation": set(groups[60:80]), "test": set(groups[80:])}
    result = {}
    for name, needed in SPLIT_COUNTS.items():
        selected = []
        for label, count in needed.items():
            choices = [row for row in rows if row["recipe_code"] in group_sets[name] and row["label"] == label]
            rng.shuffle(choices)
            if len(choices) < count:
                raise ValueError(f"{name}: only {len(choices)} examples for class {label}, need {count}")
            selected.extend(choices[:count])
        rng.shuffle(selected)
        result[name] = selected
    ids = [row["id"] for group in result.values() for row in group]
    if len(ids) != len(set(ids)):
        raise ValueError("normalized duplicate crossed split boundary")
    assert all(not (group_sets[a] & group_sets[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    return result, {name: sorted(values) for name, values in group_sets.items()}


def run_method(method, splits, args):
    texts = {name: [row["text"] for row in rows] for name, rows in splits.items()}
    labels = {name: [row["label"] for row in rows] for name, rows in splits.items()}
    timing = {}
    artifact = None

    if method == "majority":
        scores = {name: np.ones(len(rows)) for name, rows in splits.items() if name != "train"}
        timing["definition"] = "constant prediction; no model request"
    elif method == "tfidf":
        started = time.perf_counter()
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        train_features = vectorizer.fit_transform(texts["train"])
        head = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(train_features, labels["train"])
        timing["fit_seconds"] = time.perf_counter() - started
        scores = {"validation": head.decision_function(vectorizer.transform(texts["validation"]))}
        test_scores, latencies = [], []
        for text in texts["test"]:
            started = time.perf_counter()
            test_scores.append(float(head.decision_function(vectorizer.transform([text]))[0]))
            latencies.append((time.perf_counter() - started) * 1000)
        scores["test"] = np.asarray(test_scores)
        timing["mean_test_request_ms"] = float(np.mean(latencies))
    elif method == "shape":
        features = {name: np.asarray([shape_features(text) for text in group], dtype=float) for name, group in texts.items()}
        started = time.perf_counter()
        scaler, head = fit_linear(features["train"], labels["train"], 42)
        timing["fit_seconds"] = time.perf_counter() - started
        scores = {"validation": head.decision_function(scaler.transform(features["validation"]))}
        test_scores, latencies = [], []
        for text in texts["test"]:
            started = time.perf_counter()
            test_scores.append(float(head.decision_function(scaler.transform([shape_features(text)]))[0]))
            latencies.append((time.perf_counter() - started) * 1000)
        scores["test"] = np.asarray(test_scores)
        timing["mean_test_request_ms"] = float(np.mean(latencies))
    else:
        if method == "gemma":
            encoder = GemmaEncoder(args.model, args.device)
            model_info = {"model_id": args.model, "fingerprints": encoder.fingerprints(), "pooling": "final_hidden_state_last_token"}
            def embed_one(text):
                vector, details = encoder.embed(text)
                return vector.numpy(), details["embedding_latency_ms"]
        else:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(EMBEDDING_MODEL_ID, revision=EMBEDDING_REVISION, device=args.device)
            model_info = {"model_id": EMBEDDING_MODEL_ID, "revision": EMBEDDING_REVISION, "pooling": "sentence_transformers_default"}
            def embed_one(text):
                started = time.perf_counter()
                vector = encoder.encode(text, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
                return vector, (time.perf_counter() - started) * 1000
        vectors, embed_times = {}, {}
        for name, group in texts.items():
            embedded = [embed_one(text) for text in group]
            vectors[name] = np.stack([vector for vector, _ in embedded])
            embed_times[name] = [latency for _, latency in embedded]
        started = time.perf_counter()
        scaler, head = fit_linear(vectors["train"], labels["train"], 42)
        timing["head_fit_seconds"] = time.perf_counter() - started
        scores = {"validation": head.decision_function(scaler.transform(vectors["validation"]))}
        test_scores, head_times = [], []
        for vector in vectors["test"]:
            started = time.perf_counter()
            test_scores.append(float(head.decision_function(scaler.transform([vector]))[0]))
            head_times.append((time.perf_counter() - started) * 1000)
        scores["test"] = np.asarray(test_scores)
        timing["embedding_seconds_by_split"] = {name: sum(values) / 1000 for name, values in embed_times.items()}
        timing["mean_test_request_ms"] = float(np.mean(np.asarray(embed_times["test"]) + np.asarray(head_times)))
        timing["definition"] = "warm one-message embedding plus separately measured head scoring"
        artifact = {
            "version": "binary_logistic_v1", "task_id": "recipe-sentiment-v1",
            "score_semantics": "uncalibrated_logistic_decision_function",
            **model_info, "train_data_sha256": dataset_digest({"train": splits["train"]}),
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": head.coef_[0].tolist(), "intercept": float(head.intercept_[0]),
        }

    result = summary_for(scores, splits)
    threshold = result["selection"]["threshold"]
    predicted = np.asarray(scores["test"]) >= threshold
    result["test"]["macro_f1"] = float(f1_score(labels["test"], predicted, average="macro", zero_division=0))
    result["test"]["balanced_accuracy"] = float(balanced_accuracy_score(labels["test"], predicted))
    predictions = {
        name: [
            {
                "id": row["id"],
                "recipe_code": row["recipe_code"],
                "label": row["label"],
                "score": float(score),
                "predicted": int(score >= threshold),
            }
            for row, score in zip(splits[name], scores[name])
        ]
        for name in ("validation", "test")
    }
    if artifact is not None:
        artifact["validation_threshold"] = result["selection"]
        artifact_path = Path(f"recipe_{method}_head.json")
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        sample = vectors["test"][0]
        restored = json.loads(artifact_path.read_text(encoding="utf-8"))
        check = float(np.dot((sample - restored["mean"]) / restored["scale"], restored["coef"]) + restored["intercept"])
        if not np.isclose(check, scores["test"][0], atol=1e-6):
            raise ValueError("saved head does not reproduce score")
        result["artifact"] = str(artifact_path)
    return result, timing, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("majority", "tfidf", "shape", "gemma", "minilm"), required=True)
    parser.add_argument("--model", default="google/gemma-3-1b-it", help="local Gemma path for --method gemma")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--output", help="defaults to recipe_<method>_results.json")
    args = parser.parse_args()
    rows, source_info = load_recipe_rows()
    splits, groups = make_recipe_splits(rows)
    result, timing, predictions = run_method(args.method, splits, args)
    report = {
        "status": "exploratory group-held-out recipe pilot",
        "task_id": "recipe-sentiment-v1",
        "labels": {"0": "1 or 2 stars", "1": "4 or 5 stars"},
        "method": args.method,
        "source": {"url": SOURCE_URL, "license": "CC BY 4.0 per UCI repository", **source_info},
        "split_rule": "seed 42; 60/20/20 recipes; no exact normalized duplicate text",
        "split_counts": {name: {str(label): sum(row["label"] == label for row in group) for label in (0, 1)} for name, group in splits.items()},
        "recipe_groups": groups,
        "split_ids": {name: [row["id"] for row in group] for name, group in splits.items()},
        "data_sha256": dataset_digest(splits),
        "development_labels": 328,
        "result": result,
        "predictions": predictions,
        "timing": timing,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": importlib.metadata.version("torch"), "scikit_learn": importlib.metadata.version("scikit-learn")},
        "code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ("pilot_recipe.py", "pilot_sms.py", "gemma_benchmark.py")},
    }
    destination = args.output or f"recipe_{args.method}_results.json"
    Path(destination).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.method, result["test"], "AP:", round(result["test_average_precision"], 4), "report:", destination)


if __name__ == "__main__":
    main()
