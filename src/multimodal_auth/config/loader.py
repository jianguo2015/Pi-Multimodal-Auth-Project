"""Typed, validated configuration loader.

Design rules
------------
* One YAML file (``configs/default.yaml``) is the single source of truth.
* Every value is validated on load; nothing is silently defaulted at runtime.
* No absolute paths: relative paths are resolved against the project root,
  which is discovered from this file's location, so the repository works from
  any directory (including paths with non-ASCII characters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import yaml

from ..errors import ConfigError

DEFAULT_CONFIG_RELPATH = Path("configs") / "default.yaml"
VALID_BACKENDS = ("onnx", "pytorch")


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root.

    Walks upwards from ``start`` (default: this file) until a directory
    containing ``configs/default.yaml`` is found. Never depends on the current
    working directory.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / DEFAULT_CONFIG_RELPATH).is_file():
            return candidate
    raise ConfigError(
        "could not locate the project root (no %s above %s)"
        % (DEFAULT_CONFIG_RELPATH, here)
    )


def _require(mapping: Dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError("missing required key %r in %s" % (key, where))
    return mapping[key]


def _as_int(value: Any, name: str, minimum: int = 1) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise ConfigError("%s must be an integer, got %r" % (name, value))
    if ivalue < minimum:
        raise ConfigError("%s must be >= %d, got %d" % (name, minimum, ivalue))
    return ivalue


def _as_float(value: Any, name: str, low: float, high: float) -> float:
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise ConfigError("%s must be a number, got %r" % (name, value))
    if not (low < fvalue < high):
        raise ConfigError(
            "%s must be strictly between %s and %s, got %s"
            % (name, low, high, fvalue)
        )
    return fvalue


def _as_shape(value: Any, name: str) -> Tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError("%s must be a list of positive integers" % name)
    dims = tuple(_as_int(d, "%s[%d]" % (name, i)) for i, d in enumerate(value))
    if not dims:
        raise ConfigError("%s must not be empty" % name)
    return dims


@dataclass(frozen=True)
class ModelSpec:
    """One network: its artefacts on disk plus its expected tensor contract."""

    role: str
    onnx_file: str
    pytorch_file: str
    input_shape: Tuple[int, ...]
    #: feature dimension produced by an extractor; ``None`` for the fusion head,
    #: whose only output is a scalar score.
    embed_dim: int | None = None
    hidden_dim: int | None = None


@dataclass(frozen=True)
class FacePreprocessConfig:
    """Image -> (3, S, S) float32 in roughly [-1, 1]."""

    image_size: int = 112
    mean: float = 127.5
    scale: float = 128.0
    cascade: str = "haarcascade_frontalface_default.xml"
    scale_factor: float = 1.1
    min_neighbors: int = 4
    largest_face: bool = True
    center_crop_fallback: bool = True


@dataclass(frozen=True)
class VoicePreprocessConfig:
    """Waveform -> (1, n_mfcc, max_pad_len) float32."""

    sample_rate: int = 16000
    n_mfcc: int = 40
    max_pad_len: int = 400


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str = "onnx"
    provider: str = "CPUExecutionProvider"
    intra_op_num_threads: int = 2


@dataclass(frozen=True)
class DecisionConfig:
    """Single global threshold, frozen at 0.85 by the Phase 0 audit."""

    threshold: float = 0.85


@dataclass(frozen=True)
class Config:
    root: Path
    source: Path
    project_name: str
    status: str
    onnx_dir: Path
    pytorch_dir: Path
    models: Dict[str, ModelSpec] = field(default_factory=dict)
    face: FacePreprocessConfig = field(default_factory=FacePreprocessConfig)
    voice: VoicePreprocessConfig = field(default_factory=VoicePreprocessConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    # -------------------------------------------------------------- helpers
    def onnx_path(self, role: str) -> Path:
        return self.onnx_dir / self.models[role].onnx_file

    def pytorch_path(self, role: str) -> Path:
        return self.pytorch_dir / self.models[role].pytorch_file

    @property
    def threshold(self) -> float:
        return self.decision.threshold

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot of the effective configuration."""
        return {
            "project": {"name": self.project_name, "status": self.status},
            "source": str(self.source),
            "root": str(self.root),
            "models": {
                role: {
                    "onnx": str(self.onnx_path(role)),
                    "pytorch": str(self.pytorch_path(role)),
                    "input_shape": list(spec.input_shape),
                    "embed_dim": spec.embed_dim,
                    "hidden_dim": spec.hidden_dim,
                }
                for role, spec in self.models.items()
            },
            "preprocessing": {
                "face": {
                    "image_size": self.face.image_size,
                    "mean": self.face.mean,
                    "scale": self.face.scale,
                    "cascade": self.face.cascade,
                    "scale_factor": self.face.scale_factor,
                    "min_neighbors": self.face.min_neighbors,
                    "largest_face": self.face.largest_face,
                    "center_crop_fallback": self.face.center_crop_fallback,
                },
                "voice": {
                    "sample_rate": self.voice.sample_rate,
                    "n_mfcc": self.voice.n_mfcc,
                    "max_pad_len": self.voice.max_pad_len,
                },
            },
            "runtime": {
                "backend": self.runtime.backend,
                "provider": self.runtime.provider,
                "intra_op_num_threads": self.runtime.intra_op_num_threads,
            },
            "decision": {"threshold": self.decision.threshold},
        }


def _build_models(raw: Dict[str, Any]) -> Dict[str, ModelSpec]:
    specs: Dict[str, ModelSpec] = {}
    for role in ("face", "voice", "fusion"):
        block = raw.get(role)
        if not isinstance(block, dict):
            raise ConfigError("models.%s must be a mapping" % role)
        hidden = block.get("hidden_dim")
        specs[role] = ModelSpec(
            role=role,
            onnx_file=str(_require(block, "onnx", "models.%s" % role)),
            pytorch_file=str(_require(block, "pytorch", "models.%s" % role)),
            input_shape=_as_shape(
                _require(block, "input_shape", "models.%s" % role),
                "models.%s.input_shape" % role,
            ),
            embed_dim=(
                None
                if role == "fusion"
                else _as_int(
                    _require(block, "embed_dim", "models.%s" % role),
                    "models.%s.embed_dim" % role,
                )
            ),
            hidden_dim=None
            if hidden is None
            else _as_int(hidden, "models.%s.hidden_dim" % role),
        )
    return specs

def load_config(
    path: str | Path | None = None,
    root: str | Path | None = None,
) -> Config:
    """Load and validate the YAML configuration.

    Parameters
    ----------
    path:
        Explicit config file. Defaults to ``<root>/configs/default.yaml``.
    root:
        Project root override, used by tests and by the copy-out check.
    """
    resolved_root = Path(root).resolve() if root else find_project_root()
    config_path = (
        Path(path).resolve() if path else resolved_root / DEFAULT_CONFIG_RELPATH
    )

    if not config_path.is_file():
        raise ConfigError("configuration file not found: %s" % config_path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError("invalid YAML in %s: %s" % (config_path, exc))
    except OSError as exc:  # pragma: no cover - filesystem dependent
        raise ConfigError("cannot read %s: %s" % (config_path, exc))
    if not isinstance(raw, dict):
        raise ConfigError("configuration must be a YAML mapping: %s" % config_path)

    project = raw.get("project") or {}
    paths = raw.get("paths") or {}
    models = _build_models(raw.get("models") or {})
    pre = raw.get("preprocessing") or {}
    face_raw = pre.get("face") or {}
    voice_raw = pre.get("voice") or {}
    runtime_raw = raw.get("runtime") or {}
    decision_raw = raw.get("decision") or {}

    normalisation = face_raw.get("normalization") or {}
    detector = face_raw.get("detector") or {}

    backend = str(runtime_raw.get("backend", "onnx"))
    if backend not in VALID_BACKENDS:
        raise ConfigError(
            "runtime.backend must be one of %s, got %r"
            % (", ".join(VALID_BACKENDS), backend)
        )

    face_cfg = FacePreprocessConfig(
        image_size=_as_int(face_raw.get("image_size", 112), "face.image_size"),
        mean=float(normalisation.get("mean", 127.5)),
        scale=float(normalisation.get("scale", 128.0)),
        cascade=str(detector.get("cascade", "haarcascade_frontalface_default.xml")),
        scale_factor=float(detector.get("scale_factor", 1.1)),
        min_neighbors=_as_int(
            detector.get("min_neighbors", 4), "face.detector.min_neighbors", minimum=1
        ),
        largest_face=bool(detector.get("largest_face", True)),
        center_crop_fallback=str(detector.get("fallback", "center_crop")) == "center_crop",
    )
    if face_cfg.scale == 0:
        raise ConfigError("face.normalization.scale must not be zero")

    voice_cfg = VoicePreprocessConfig(
        sample_rate=_as_int(voice_raw.get("sample_rate", 16000), "voice.sample_rate"),
        n_mfcc=_as_int(voice_raw.get("n_mfcc", 40), "voice.n_mfcc"),
        max_pad_len=_as_int(voice_raw.get("max_pad_len", 400), "voice.max_pad_len"),
    )

    if models["voice"].input_shape != (1, voice_cfg.n_mfcc, voice_cfg.max_pad_len):
        raise ConfigError(
            "models.voice.input_shape %s is inconsistent with the MFCC settings (1,%d,%d)"
            % (list(models["voice"].input_shape), voice_cfg.n_mfcc, voice_cfg.max_pad_len)
        )
    if models["face"].input_shape != (3, face_cfg.image_size, face_cfg.image_size):
        raise ConfigError(
            "models.face.input_shape %s is inconsistent with image_size %d"
            % (list(models["face"].input_shape), face_cfg.image_size)
        )
    if models["face"].embed_dim != models["voice"].embed_dim:
        raise ConfigError("face and voice embed_dim must match for the fusion head")
    if models["fusion"].input_shape != (
        models["face"].embed_dim,
        models["voice"].embed_dim,
    ):
        raise ConfigError(
            "models.fusion.input_shape %s must equal (face_embed_dim, voice_embed_dim)"
            % list(models["fusion"].input_shape)
        )

    runtime_cfg = RuntimeConfig(
        backend=backend,
        provider=str(runtime_raw.get("provider", "CPUExecutionProvider")),
        intra_op_num_threads=_as_int(
            runtime_raw.get("intra_op_num_threads", 2),
            "runtime.intra_op_num_threads",
            minimum=1,
        ),
    )
    decision_cfg = DecisionConfig(
        threshold=_as_float(decision_raw.get("threshold", 0.85), "decision.threshold", 0.0, 1.0)
    )

    return Config(
        root=resolved_root,
        source=config_path,
        project_name=str(project.get("name", "multimodal-auth")),
        status=str(project.get("status", "")),
        onnx_dir=resolved_root / str(paths.get("onnx_models", "weights/onnx")),
        pytorch_dir=resolved_root / str(paths.get("pytorch_models", "weights/pytorch")),
        models=models,
        face=face_cfg,
        voice=voice_cfg,
        runtime=runtime_cfg,
        decision=decision_cfg,
    )

