"""End-to-end pipeline: (face image, voice clip) -> AuthResult.

The pipeline is backend-agnostic. Both backends consume the *same* tensors
produced by ``multimodal_auth.preprocessing`` and the *same* threshold from
``multimodal_auth.decision``, which is what makes the cross-backend
equivalence test meaningful.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..config.loader import Config, load_config
from ..decision.policy import DecisionResult, evaluate
from ..errors import InvalidAudioError, InvalidImageError
from .types import AuthResult, summarise

VALID_BACKENDS = ("onnx", "pytorch")


def create_backend(config: Config, backend: str | None = None):
    """Instantiate the requested backend (lazily, so only one runtime loads)."""
    name = (backend or config.runtime.backend).lower()
    if name == "onnx":
        from .onnx_backend import OnnxBackend

        return OnnxBackend(config)
    if name == "pytorch":
        from .torch_backend import TorchBackend

        return TorchBackend(config)
    raise ValueError("unknown backend %r (expected one of %s)" % (backend, ", ".join(VALID_BACKENDS)))


class MultimodalPipeline:
    """Preprocess -> embed -> fuse -> decide."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        backend: str | None = None,
        threshold: float | None = None,
        engine=None,
    ):
        self.config = config or load_config()
        self.backend_name = (backend or self.config.runtime.backend).lower()
        self.threshold = (
            self.config.threshold if threshold is None else float(threshold)
        )
        # validated here so a bad override fails before any file is touched
        evaluate(0.5, self.threshold)
        self._engine = engine

    # ------------------------------------------------------------------ setup
    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_backend(self.config, self.backend_name)
        return self._engine

    def describe(self) -> Dict[str, object]:
        info: Dict[str, object] = {
            "backend": self.backend_name,
            "threshold": self.threshold,
            "config": str(self.config.source),
            "model_files": dict(self.engine.model_files),
        }
        if hasattr(self.engine, "ort_version"):
            info["onnxruntime_version"] = self.engine.ort_version
        if self.backend_name == "pytorch":
            info["torch_version"] = getattr(self.engine._torch, "__version__", "unknown")
        return info

    # ------------------------------------------------------------------- run
    def run(
        self,
        face_source,
        voice_source,
        *,
        timings: bool = True,
    ) -> AuthResult:
        """Authenticate one (face, voice) pair.

        Raises :class:`InvalidImageError` / :class:`InvalidAudioError` when an
        input cannot be used; never returns a made-up score.
        """
        from ..preprocessing.audio import preprocess_voice
        from ..preprocessing.face import preprocess_face

        started = time.perf_counter()
        face_tensor = preprocess_face(face_source, self.config.face)
        voice_tensor = preprocess_voice(voice_source, self.config.voice)
        preprocess_seconds = time.perf_counter() - started

        face_embedding = np.asarray(self.engine.embed_face(face_tensor), dtype=np.float32).reshape(-1)
        voice_embedding = np.asarray(self.engine.embed_voice(voice_tensor), dtype=np.float32).reshape(-1)

        expected_dim = self.config.models["face"].embed_dim
        for role, vector in (("face", face_embedding), ("voice", voice_embedding)):
            if vector.size != expected_dim:
                raise ValueError(
                    "%s embedding has %d dims, expected %d" % (role, vector.size, expected_dim)
                )

        score, attention = self.engine.score(voice_embedding, face_embedding)
        decision: DecisionResult = evaluate(float(score), self.threshold)

        warnings = []
        if attention is None:
            warnings.append(
                "attention weights are only available with --backend pytorch "
                "(the INT8 ONNX graph exposes the final score only)"
            )

        result = AuthResult(
            decision=decision,
            face_embedding=summarise("face", face_embedding),
            voice_embedding=summarise("voice", voice_embedding),
            backend=self.backend_name,
            attention=attention,
            model_files=dict(self.engine.model_files),
            inputs={
                "face": _describe_source(face_source),
                "voice": _describe_source(voice_source),
            },
            warnings=warnings,
        )
        if timings:
            result.timings_ms = {
                "preprocess": round(preprocess_seconds * 1000.0, 3),
                "face_embed": round(getattr(self.engine, "last_face_seconds", 0.0) * 1000.0, 3),
                "voice_embed": round(getattr(self.engine, "last_voice_seconds", 0.0) * 1000.0, 3),
                "fusion": round(getattr(self.engine, "last_fusion_seconds", 0.0) * 1000.0, 3),
                "total": round((time.perf_counter() - started) * 1000.0, 3),
            }
        return result


def _describe_source(source) -> str:
    if isinstance(source, (str, Path)):
        return str(Path(source).resolve())
    if isinstance(source, np.ndarray):
        return "<ndarray shape=%s>" % (tuple(source.shape),)
    return "<%s>" % type(source).__name__


__all__ = [
    "MultimodalPipeline",
    "create_backend",
    "InvalidImageError",
    "InvalidAudioError",
]
