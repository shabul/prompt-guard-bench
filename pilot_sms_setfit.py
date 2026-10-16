"""SetFit baseline on the fixed exploratory SMS pilot split."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from gemma_benchmark import dataset_digest
from pilot_sms import load_sms, make_splits, summary_for
from pilot_sms_embedding import MODEL_ID, MODEL_REVISION, EXPECTED_SPLIT_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--output", default="sms_setfit_results.json")
    parser.add_argument("--artifact-dir", default="artifacts/sms_setfit")
    args = parser.parse_args()
    artifact_path = Path(args.artifact_dir)
    if artifact_path.exists():
        raise FileExistsError(f"artifact directory already exists: {artifact_path}; choose a new --artifact-dir")

    source, source_info = load_sms()
    splits = make_splits(source, seed=42)
    data_hash = dataset_digest(splits)
    if data_hash != EXPECTED_SPLIT_SHA256:
        raise ValueError("SMS split differs from original pilot")
    training = Dataset.from_dict({
        "text": [row["text"] for row in splits["train"]],
        "label": [row["label"] for row in splits["train"]],
    })

    started = time.perf_counter()
    model = SetFitModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.model_body.to(args.device)
    load_seconds = time.perf_counter() - started
    train_args = TrainingArguments(
        output_dir="artifacts/setfit_checkpoints",
        batch_size=16,
        num_epochs=1,
        num_iterations=20,
        max_steps=80,
        seed=42,
        show_progress_bar=False,
        save_strategy="no",
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=training)
    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started
    classes = list(model.model_head.classes_)
    if classes != [0, 1]:
        raise ValueError(f"unexpected SetFit class order: {classes}")

    validation_texts = [row["text"] for row in splits["validation"]]
    validation_scores = model.predict_proba(validation_texts, batch_size=32, as_numpy=True, show_progress_bar=False)[:, 1]
    test_scores = []
    test_latencies = []
    for row in splits["test"]:
        started = time.perf_counter()
        probability = model.predict_proba([row["text"]], batch_size=1, as_numpy=True, show_progress_bar=False)[0, 1]
        test_scores.append(float(probability))
        test_latencies.append((time.perf_counter() - started) * 1000)
    result = summary_for({"validation": validation_scores, "test": np.asarray(test_scores)}, splits)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact_path)
    reloaded = SetFitModel.from_pretrained(str(artifact_path))
    reloaded.model_body.to(args.device)
    check = reloaded.predict_proba([splits["test"][0]["text"]], batch_size=1, as_numpy=True, show_progress_bar=False)[0, 1]
    if not np.isclose(check, test_scores[0], atol=1e-5):
        raise ValueError("saved SetFit model does not reproduce score")

    report = {
        "status": "exploratory; test viewed during original pilot",
        "method": "SetFit fine-tuned MiniLM encoder + default classification head",
        "source_archive_sha256": source_info["archive_sha256"],
        "data_sha256": data_hash,
        "development_labels": 328,
        "fit_labels": 128,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "training_config": {
            "batch_size": 16, "num_epochs": 1, "num_iterations": 20,
            "max_steps": 80, "seed": 42, "loss_and_head": "SetFit defaults",
        },
        "result": result,
        "artifact_dir": str(artifact_path),
        "timing": {
            "model_load_seconds": load_seconds,
            "train_seconds": train_seconds,
            "mean_test_request_ms": float(np.mean(test_latencies)),
            "definition": "warm one-message SetFit predict_proba on Apple M5 MPS",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "setfit": importlib.metadata.version("setfit"),
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "transformers": importlib.metadata.version("transformers"),
        },
        "code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ("pilot_sms_setfit.py", "pilot_sms_embedding.py", "pilot_sms.py", "gemma_benchmark.py")},
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SetFit:", result["test"], "AP:", round(result["test_average_precision"], 4))
    print("mean warm request ms:", round(report["timing"]["mean_test_request_ms"], 3))
    print("train seconds:", round(train_seconds, 2))
    print("report:", args.output)


if __name__ == "__main__":
    main()
