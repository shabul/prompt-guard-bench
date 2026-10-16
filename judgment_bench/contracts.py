"""Minimal binary task and prediction contracts with explicit score semantics."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BinaryTaskSpec:
    task_id: str
    negative_label: str
    positive_label: str
    intended_use: str

    def __post_init__(self):
        if not all((self.task_id, self.negative_label, self.positive_label, self.intended_use)):
            raise ValueError("task ID, both label definitions, and intended use are required")
        if self.negative_label == self.positive_label:
            raise ValueError("positive and negative label definitions must differ")


@dataclass(frozen=True)
class Prediction:
    task_id: str
    score: float
    score_semantics: str
    label: int | None
    threshold: float | None
    input_status: str

    def __post_init__(self):
        if not math.isfinite(self.score):
            raise ValueError("prediction score must be finite")
        if self.label not in (None, 0, 1):
            raise ValueError("binary label must be 0, 1, or absent")
        if self.threshold is not None and not math.isfinite(self.threshold):
            raise ValueError("decision threshold must be finite")
        if self.input_status not in ("ok", "truncated"):
            raise ValueError("unsupported input status")
