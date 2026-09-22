"""Result types shared by every backend and by the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..decision.policy import DecisionResult


@dataclass(frozen=True)
class EmbeddingSummary:
    """Compact, JSON-safe description of one 128-d embedding.

    The raw vector stays in memory (``vector``) but is never printed in full;
    the summary is what ends up in the CLI output and in result JSON files.
    """

    role: str
    vector: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def dim(self) -> int:
        return int(self.vector.size) if self.vector is not None else 0

    def as_dict(self) -> Dict[str, Any]:
        if self.vector is None:
            return {"role": self.role, "dim": 0}
        flat = np.asarray(self.vector, dtype=np.float64).reshape(-1)
        return {
            "role": self.role,
            "dim": int(flat.size),
            "l2_norm": round(float(np.linalg.norm(flat)), 6),
            "mean": round(float(flat.mean()), 6),
            "min": round(float(flat.min()), 6),
            "max": round(float(flat.max()), 6),
        }


@dataclass
class AuthResult:
    """Everything one inference call produced, in a serialisable form."""

    decision: DecisionResult
    face_embedding: EmbeddingSummary
    voice_embedding: EmbeddingSummary
    backend: str
    attention: Optional[Dict[str, Any]] = None
    model_files: Dict[str, str] = field(default_factory=dict)
    inputs: Dict[str, str] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.decision.score

    @property
    def threshold(self) -> float:
        return self.decision.threshold

    @property
    def accepted(self) -> bool:
        return self.decision.accepted

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "backend": self.backend,
            "decision": self.decision.as_dict(),
            "face_embedding": self.face_embedding.as_dict(),
            "voice_embedding": self.voice_embedding.as_dict(),
            "attention_weights": self.attention,
            "model_files": self.model_files,
            "inputs": self.inputs,
            "timings_ms": self.timings_ms,
        }
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload


def summarise(role: str, vector: np.ndarray) -> EmbeddingSummary:
    return EmbeddingSummary(role=role, vector=np.asarray(vector).reshape(-1))
