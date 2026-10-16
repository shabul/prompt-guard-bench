"""Experimental task-neutral primitives; not yet a stable public API."""

from .contracts import BinaryTaskSpec, Prediction
from .linear import LogisticHeadArtifact, score_embedding

__all__ = ["BinaryTaskSpec", "Prediction", "LogisticHeadArtifact", "score_embedding"]
