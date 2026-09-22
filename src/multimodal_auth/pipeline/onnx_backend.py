"""ONNX Runtime backend - the canonical, dependency-light inference path.

The INT8 graphs in ``weights/onnx/`` require ``onnxruntime>=1.24.1`` because
they contain ``ConvInteger`` nodes (the original ``requirements_pi.txt`` pin of
1.16.0 cannot even load them - see docs/LIMITATIONS.md). Startup is therefore
validated explicitly and reported as a :class:`ModelArtifactError`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..config.loader import Config
from ..errors import ModelArtifactError

SUPPORTED_ORT_MIN = (1, 24, 1)


class OnnxBackend:
    """Loads the three INT8 graphs and exposes embedding/score primitives."""

    name = "onnx"

    def __init__(self, config: Config):
        self.config = config
        self._sessions: Dict[str, object] = {}
        self._providers: List[str] = [config.runtime.provider]
        self.load_seconds: Dict[str, float] = {}
        self.provider_used: Dict[str, str] = {}
        # Fails immediately - before any preprocessing - when onnxruntime is
        # missing or too old for the INT8 graphs. Sessions stay lazy.
        self.ort_version = self._check_onnxruntime()

    # ------------------------------------------------------------------ load
    @staticmethod
    def _check_onnxruntime() -> str:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ModelArtifactError(
                "onnxruntime is not installed; run "
                "`pip install -r requirements/runtime.txt` (%s)" % exc
            )
        version = tuple(int(part) for part in ort.__version__.split(".")[:3])
        if version < SUPPORTED_ORT_MIN:
            raise ModelArtifactError(
                "onnxruntime %s is too old for the INT8 graphs (ConvInteger needs >= %s)"
                % (ort.__version__, ".".join(str(v) for v in SUPPORTED_ORT_MIN))
            )
        return ort.__version__

    def _load(self, role: str, path: Path):
        import onnxruntime as ort

        if not path.is_file():
            raise ModelArtifactError(
                "missing ONNX artefact for %r: %s (see docs/MODEL_ARTIFACT_MANIFEST.md)"
                % (role, path)
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.config.runtime.intra_op_num_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        started = time.perf_counter()
        try:
            session = ort.InferenceSession(str(path), sess_options=options, providers=self._providers)
        except Exception as exc:
            raise ModelArtifactError("cannot load ONNX model %s: %s" % (path, exc))
        self.load_seconds[role] = time.perf_counter() - started
        self.provider_used[role] = session.get_providers()[0]
        expected_inputs = {
            "face": ["face_input"],
            "voice": ["voice_input"],
            "fusion": ["voice_input", "face_input"],
        }[role]
        names = [item.name for item in session.get_inputs()]
        if names != expected_inputs:
            raise ModelArtifactError(
                "unexpected inputs for %r: %s (expected %s)" % (role, names, expected_inputs)
            )
        return session

    @property
    def sessions(self) -> Dict[str, object]:
        if not self._sessions:
            self._sessions = {
                role: self._load(role, self.config.onnx_path(role))
                for role in ("face", "voice", "fusion")
            }
        return self._sessions

    @property
    def model_files(self) -> Dict[str, str]:
        return {role: str(self.config.onnx_path(role)) for role in ("face", "voice", "fusion")}

    # -------------------------------------------------------------- inference
    def embed_face(self, tensor: np.ndarray) -> np.ndarray:
        session = self.sessions["face"]
        batch = np.ascontiguousarray(tensor[None, ...], dtype=np.float32)
        start = time.perf_counter()
        outputs = session.run(["output"], {"face_input": batch})
        self.last_face_seconds = time.perf_counter() - start
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)

    def embed_voice(self, tensor: np.ndarray) -> np.ndarray:
        session = self.sessions["voice"]
        batch = np.ascontiguousarray(tensor[None, ...], dtype=np.float32)
        start = time.perf_counter()
        outputs = session.run(["output"], {"voice_input": batch})
        self.last_voice_seconds = time.perf_counter() - start
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)

    def score(self, voice_embedding: np.ndarray, face_embedding: np.ndarray):
        session = self.sessions["fusion"]
        voice = np.ascontiguousarray(voice_embedding[None, :], dtype=np.float32)
        face = np.ascontiguousarray(face_embedding[None, :], dtype=np.float32)
        start = time.perf_counter()
        outputs = session.run(["output"], {"voice_input": voice, "face_input": face})
        self.last_fusion_seconds = time.perf_counter() - start
        return float(np.asarray(outputs[0]).reshape(-1)[0]), None
