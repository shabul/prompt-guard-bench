"""Validated, pickle-free scoring for exploratory binary logistic heads."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np

from .contracts import Prediction


@dataclass(frozen=True)
class LogisticHeadArtifact:
    task_id: str
    model_id: str
    model_revision: str | None
    score_semantics: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    validation_threshold: float | None
    train_data_sha256: str

    @classmethod
    def from_mapping(cls, record):
        if record.get("version") != "binary_logistic_v1":
            raise ValueError("unsupported logistic artifact version")
        if record.get("score_semantics") != "uncalibrated_logistic_decision_function":
            raise ValueError("unsupported score semantics")
        selection = record.get("validation_threshold")
        threshold = selection.get("threshold") if isinstance(selection, dict) else None
        artifact = cls(
            task_id=record["task_id"], model_id=record["model_id"],
            model_revision=record.get("model_revision", record.get("revision")),
            score_semantics=record["score_semantics"],
            mean=tuple(float(x) for x in record["mean"]),
            scale=tuple(float(x) for x in record["scale"]),
            coef=tuple(float(x) for x in record["coef"]),
            intercept=float(record["intercept"]),
            validation_threshold=None if threshold is None else float(threshold),
            train_data_sha256=record["train_data_sha256"],
        )
        if not artifact.task_id or not artifact.model_id or not re.fullmatch(r"[0-9a-f]{64}", artifact.train_data_sha256):
            raise ValueError("artifact identity or training digest invalid")
        if not artifact.mean or len({len(artifact.mean), len(artifact.scale), len(artifact.coef)}) != 1:
            raise ValueError("linear head dimensions differ")
        if any(not math.isfinite(x) for x in (*artifact.mean, *artifact.scale, *artifact.coef, artifact.intercept)):
            raise ValueError("linear head parameters must be finite")
        if any(x <= 0 for x in artifact.scale):
            raise ValueError("linear head scale must be positive")
        if artifact.validation_threshold is not None and not math.isfinite(artifact.validation_threshold):
            raise ValueError("validation threshold must be finite")
        return artifact

    @classmethod
    def from_json(cls, path):
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def score_embedding(embedding, artifact, *, model_id, model_revision=None, truncated=False, apply_threshold=True, allow_unpinned_model=False):
    """Score an already-computed feature vector; caller supplies trusted encoder metadata."""
    if model_id != artifact.model_id:
        raise ValueError("encoder model ID differs from artifact")
    if artifact.model_revision is None and not allow_unpinned_model:
        raise ValueError("artifact has no model revision; explicit pilot-only override required")
    if artifact.model_revision is not None and model_revision != artifact.model_revision:
        raise ValueError("encoder revision differs from artifact")
    vector = np.asarray(embedding, dtype=float)
    if vector.ndim != 1 or len(vector) != len(artifact.coef) or not np.all(np.isfinite(vector)):
        raise ValueError("embedding must be a finite vector matching head dimension")
    score = float(np.dot((vector - artifact.mean) / artifact.scale, artifact.coef) + artifact.intercept)
    if not math.isfinite(score):
        raise ValueError("computed score is not finite")
    threshold = artifact.validation_threshold if apply_threshold else None
    if apply_threshold and threshold is None:
        raise ValueError("artifact has no validation threshold")
    return Prediction(
        task_id=artifact.task_id,
        score=score,
        score_semantics=artifact.score_semantics,
        label=None if threshold is None or truncated else int(score >= threshold),
        threshold=threshold,
        input_status="truncated" if truncated else "ok",
    )
