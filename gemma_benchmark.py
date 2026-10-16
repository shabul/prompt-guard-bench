"""Shared, legacy-compatible Gemma feature and ridge-head benchmark code.

The sigmoid of a ridge prediction is a historical score, not a calibrated
probability. Keep this transform for comparison with the original reports.
"""

import hashlib
import json
import math
import time
from pathlib import Path


ARTIFACT_VERSION = "legacy_ridge_v2"


def dataset_digest(named_splits):
    """Hash complete ordered rows, including role and label, with unambiguous JSON."""
    digest = hashlib.sha256()
    for split_name, rows in named_splits.items():
        for index, row in enumerate(rows):
            record = {
                "split": split_name,
                "index": index,
                "text": row["text"],
                "label": int(row["label"]),
            }
            digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def nearest_rank(values, percentile):
    if not values or not 0 < percentile <= 1:
        raise ValueError("values must be nonempty and percentile in (0, 1]")
    return sorted(values)[math.ceil(len(values) * percentile) - 1]


def classification_metrics(rows, threshold):
    if not rows:
        raise ValueError("cannot evaluate an empty dataset")
    if not all(math.isfinite(row["score"]) for row in rows):
        raise ValueError("scores must be finite")
    tp = sum(row["label"] == 1 and row["score"] >= threshold for row in rows)
    fp = sum(row["label"] == 0 and row["score"] >= threshold for row in rows)
    fn = sum(row["label"] == 1 and row["score"] < threshold for row in rows)
    tn = sum(row["label"] == 0 and row["score"] < threshold for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def binary_metrics(rows, threshold):
    return {
        **classification_metrics(rows, threshold),
        "mean_latency_ms": sum(row["forward_latency_ms"] for row in rows) / len(rows),
    }


class GemmaEncoder:
    def __init__(self, model_id, device=None, trust_remote_code=False, max_length=2048):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, trust_remote_code=trust_remote_code
        ).to(self.device).eval()

    def embed(self, text):
        request_started = time.perf_counter()
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(rendered, return_tensors="pt", truncation=False)
        token_count = encoded["input_ids"].shape[-1]
        truncated = token_count > self.max_length
        if truncated:
            encoded = self.tokenizer(rendered, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        forward_started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        if self.device == "mps":
            self.torch.mps.synchronize()
        forward_ms = (time.perf_counter() - forward_started) * 1000
        vector = outputs.hidden_states[-1][0, -1].float().cpu()
        return vector, {
            "forward_latency_ms": forward_ms,
            "embedding_latency_ms": (time.perf_counter() - request_started) * 1000,
            "input_tokens": token_count,
            "truncated": truncated,
        }

    def fingerprints(self):
        config = self.model.config.to_json_string()
        template = self.tokenizer.chat_template
        if template is None:
            raise ValueError("tokenizer has no chat template")
        return {
            "model_config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
            "chat_template_sha256": hashlib.sha256(str(template).encode("utf-8")).hexdigest(),
        }


def fit_legacy_ridge(vectors, labels):
    import torch

    matrix = torch.stack(vectors)
    target = torch.tensor(labels, dtype=torch.float32)
    mean = matrix.mean(0)
    scale = matrix.std(0).clamp_min(1e-6)
    normalized = (matrix - mean) / scale
    design = torch.cat([normalized, torch.ones(len(normalized), 1)], dim=1)
    regularizer = torch.eye(design.shape[1])
    regularizer[-1, -1] = 0
    weights = torch.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist()}


def legacy_score(vector, head):
    import torch

    mean = torch.tensor(head["mean"])
    scale = torch.tensor(head["scale"])
    weights = torch.tensor(head["weights"])
    features = torch.cat([(vector - mean) / scale, torch.ones(1)])
    return float(torch.sigmoid(features @ weights))


def save_artifact(path, encoder, head, data_hash):
    artifact = {
        "version": ARTIFACT_VERSION,
        "score_semantics": "sigmoid_of_ridge_prediction_uncalibrated",
        "model_id": encoder.model_id,
        "device_at_fit": encoder.device,
        "dtype": "float32",
        "pooling": "final_hidden_state_last_token",
        "max_length": encoder.max_length,
        "tokenizer_truncation_side": encoder.tokenizer.truncation_side,
        **encoder.fingerprints(),
        "training_data_sha256": data_hash,
        "head": head,
    }
    Path(path).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def load_artifact(path):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("version") != ARTIFACT_VERSION:
        raise ValueError(f"unsupported artifact version: {artifact.get('version')}")
    return artifact
