"""PyTorch backend - research path, and the only backend that can expose the
attention-gate weights.

It loads ``weights/pytorch/*.pth`` with ``strict=True``, so it doubles as a
structural check that the audited checkpoints still match the model
definitions byte-for-byte.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..config.loader import Config
from ..errors import ModelArtifactError


class TorchBackend:
    name = "pytorch"

    def __init__(self, config: Config):
        self.config = config
        self._torch = self._import_torch()
        self._modules: Dict[str, object] = {}
        self.load_seconds: Dict[str, float] = {}
        self.provider_used = {"device": "cpu"}

    @staticmethod
    def _import_torch():
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ModelArtifactError(
                "PyTorch is not installed; use `--backend onnx`, or install "
                "requirements/research.txt (%s)" % exc
            )
        return torch

    def _load_state_dict(self, path: Path, label: str):
        if not path.is_file():
            raise ModelArtifactError(
                "missing checkpoint for %r: %s (see docs/MODEL_ARTIFACT_MANIFEST.md)"
                % (label, path)
            )
        torch = self._torch
        try:
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as first_error:  # older pickles are not weights-only safe
            try:
                return torch.load(str(path), map_location="cpu", weights_only=False)
            except Exception as exc:
                raise ModelArtifactError("cannot load checkpoint %s: %s" % (path, exc))

    def _build(self, role: str):
        from ..models.face_extractor import FaceExtractor
        from ..models.voice_extractor import VoiceExtractor
        from ..fusion.attention_fusion import ModalAttentionFusion

        spec = self.config.models[role]
        if role == "face":
            model = FaceExtractor(embed_dim=spec.embed_dim)
        elif role == "voice":
            model = VoiceExtractor(n_mfcc=self.config.voice.n_mfcc, embed_dim=spec.embed_dim)
        else:
            model = ModalAttentionFusion(
                voice_dim=self.config.models["voice"].embed_dim,
                face_dim=self.config.models["face"].embed_dim,
                hidden_dim=spec.hidden_dim or 64,
            )
        state = self._load_state_dict(self.config.pytorch_path(role), role)
        if not isinstance(state, dict):
            raise ModelArtifactError("checkpoint %s does not contain a state_dict" % role)
        if any(key.startswith("module.") for key in state):
            state = {key[len("module."):]: value for key, value in state.items()}
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ModelArtifactError(
                "checkpoint %s does not match the model definition: %s"
                % (self.config.pytorch_path(role), exc)
            )
        model.eval()
        return model

    @property
    def modules(self) -> Dict[str, object]:
        if not self._modules:
            for role in ("face", "voice", "fusion"):
                started = time.perf_counter()
                self._modules[role] = self._build(role)
                self.load_seconds[role] = time.perf_counter() - started
        return self._modules

    @property
    def model_files(self) -> Dict[str, str]:
        return {role: str(self.config.pytorch_path(role)) for role in ("face", "voice", "fusion")}

    # -------------------------------------------------------------- inference
    def _forward(self, role: str, tensor: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            batch = torch.from_numpy(np.ascontiguousarray(tensor[None, ...], dtype=np.float32))
            output = self.modules[role](batch)
        return output.numpy().reshape(-1)

    def embed_face(self, tensor: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        embedding = self._forward("face", tensor)
        self.last_face_seconds = time.perf_counter() - start
        return embedding.astype(np.float32)

    def embed_voice(self, tensor: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        embedding = self._forward("voice", tensor)
        self.last_voice_seconds = time.perf_counter() - start
        return embedding.astype(np.float32)

    def score(
        self, voice_embedding: np.ndarray, face_embedding: np.ndarray
    ) -> Tuple[float, Optional[Dict[str, object]]]:
        from ..fusion.inspect_attention import fuse_with_attention

        torch = self._torch
        voice = torch.from_numpy(np.ascontiguousarray(voice_embedding[None, :], dtype=np.float32))
        face = torch.from_numpy(np.ascontiguousarray(face_embedding[None, :], dtype=np.float32))
        start = time.perf_counter()
        breakdown = fuse_with_attention(self.modules["fusion"], voice, face)
        self.last_fusion_seconds = time.perf_counter() - start
        attention = {
            "voice_weight": round(breakdown.voice_weight, 6),
            "face_weight": round(breakdown.face_weight, 6),
            "source": "pytorch attention_net softmax output",
        }
        return float(breakdown.score), attention
