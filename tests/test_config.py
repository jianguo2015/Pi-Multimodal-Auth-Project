"""Configuration loading and validation."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from multimodal_auth.config import load_config
from multimodal_auth.config.loader import find_project_root
from multimodal_auth.errors import ConfigError

#: Values frozen by the Phase 0 audit; changing them changes published results.
FROZEN = {
    "threshold": 0.85,
    "face_image_size": 112,
    "face_mean": 127.5,
    "face_scale": 128.0,
    "face_scale_factor": 1.1,
    "face_min_neighbors": 4,
    "voice_sample_rate": 16000,
    "voice_n_mfcc": 40,
    "voice_max_pad_len": 400,
    "embed_dim": 128,
    "fusion_hidden_dim": 64,
}


def test_project_root_discovery(config, project_root):
    assert config.root == project_root
    assert config.source == project_root / "configs" / "default.yaml"
    assert find_project_root(config.source) == project_root


def test_frozen_values(config):
    assert config.threshold == FROZEN["threshold"]
    assert config.face.image_size == FROZEN["face_image_size"]
    assert config.face.mean == FROZEN["face_mean"]
    assert config.face.scale == FROZEN["face_scale"]
    assert config.face.scale_factor == FROZEN["face_scale_factor"]
    assert config.face.min_neighbors == FROZEN["face_min_neighbors"]
    assert config.voice.sample_rate == FROZEN["voice_sample_rate"]
    assert config.voice.n_mfcc == FROZEN["voice_n_mfcc"]
    assert config.voice.max_pad_len == FROZEN["voice_max_pad_len"]
    assert config.models["face"].embed_dim == FROZEN["embed_dim"]
    assert config.models["voice"].embed_dim == FROZEN["embed_dim"]
    assert config.models["fusion"].hidden_dim == FROZEN["fusion_hidden_dim"]


def test_tensor_contracts(config):
    assert config.models["face"].input_shape == (3, 112, 112)
    assert config.models["voice"].input_shape == (1, 40, 400)
    assert config.models["fusion"].input_shape == (128, 128)
    assert config.models["fusion"].embed_dim is None  # a score, not an embedding


def test_paths_resolve_under_the_project_root(config, project_root):
    assert config.onnx_path("face") == project_root / "weights" / "onnx" / "face_extractor_quant.onnx"
    assert config.pytorch_path("voice") == project_root / "weights" / "pytorch" / "voice_extractor_weights.pth"
    for role in ("face", "voice", "fusion"):
        assert config.onnx_path(role).is_relative_to(project_root)
        assert config.pytorch_path(role).is_relative_to(project_root)


def test_default_yaml_has_no_absolute_paths(config):
    text = config.source.read_text(encoding="utf-8")
    assert ":\\\\" not in text and ":/" not in text.replace("http://", "").replace("https://", "")


def test_as_dict_is_json_serialisable(config):
    payload = config.as_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert payload["decision"]["threshold"] == FROZEN["threshold"]
    assert payload["root"] == str(config.root)
    assert payload["models"]["face"]["input_shape"] == [3, 112, 112]
    assert json.loads(text)["decision"]["threshold"] == FROZEN["threshold"]


def test_relocatable_copy(tmp_path, config):
    """A copy of the repo tree on another path loads with identical values."""
    copy_root = tmp_path / "elsewhere"
    (copy_root / "configs").mkdir(parents=True)
    (copy_root / "configs" / "default.yaml").write_text(
        config.source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    reloaded = load_config(root=copy_root)
    assert reloaded.root == copy_root.resolve()
    assert reloaded.threshold == config.threshold
    assert reloaded.onnx_dir == copy_root.resolve() / "weights" / "onnx"


def test_config_is_immutable(config):
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.decision.threshold = 0.5


# ---------------------------------------------------------------- error cases
def _write(tmp_path: Path, payload) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def _baseline(config) -> dict:
    return yaml.safe_load(config.source.read_text(encoding="utf-8"))


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(path=tmp_path / "nope.yaml")


def test_malformed_yaml(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("models: [oops\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=path)


def test_non_mapping_config(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path=path)


def test_missing_model_key(tmp_path, config):
    payload = _baseline(config)
    del payload["models"]["fusion"]["onnx"]
    with pytest.raises(ConfigError, match="models.fusion"):
        load_config(path=_write(tmp_path, payload))


def test_threshold_out_of_range(tmp_path, config):
    for bad in (0.0, 1.0, 1.5, -0.2):
        payload = _baseline(config)
        payload["decision"]["threshold"] = bad
        with pytest.raises(ConfigError, match="threshold"):
            load_config(path=_write(tmp_path, payload))


def test_non_numeric_threshold(tmp_path, config):
    payload = _baseline(config)
    payload["decision"]["threshold"] = "high"
    with pytest.raises(ConfigError, match="threshold"):
        load_config(path=_write(tmp_path, payload))


def test_unknown_backend(tmp_path, config):
    payload = _baseline(config)
    payload["runtime"]["backend"] = "tensorflow"
    with pytest.raises(ConfigError, match="backend"):
        load_config(path=_write(tmp_path, payload))


def test_voice_shape_must_match_mfcc_settings(tmp_path, config):
    payload = _baseline(config)
    payload["models"]["voice"]["input_shape"] = [1, 20, 400]
    with pytest.raises(ConfigError, match="inconsistent with the MFCC"):
        load_config(path=_write(tmp_path, payload))


def test_face_shape_must_match_image_size(tmp_path, config):
    payload = _baseline(config)
    payload["preprocessing"]["face"]["image_size"] = 96
    with pytest.raises(ConfigError, match="inconsistent with image_size"):
        load_config(path=_write(tmp_path, payload))


def test_fusion_input_must_match_embedding_dims(tmp_path, config):
    payload = _baseline(config)
    payload["models"]["fusion"]["input_shape"] = [64, 64]
    with pytest.raises(ConfigError, match="fusion.input_shape"):
        load_config(path=_write(tmp_path, payload))


def test_embed_dims_must_match(tmp_path, config):
    payload = _baseline(config)
    payload["models"]["voice"]["embed_dim"] = 64
    with pytest.raises(ConfigError, match="embed_dim must match"):
        load_config(path=_write(tmp_path, payload))


def test_zero_normalisation_scale_rejected(tmp_path, config):
    payload = _baseline(config)
    payload["preprocessing"]["face"]["normalization"]["scale"] = 0.0
    with pytest.raises(ConfigError, match="scale must not be zero"):
        load_config(path=_write(tmp_path, payload))


def test_non_positive_threads_rejected(tmp_path, config):
    payload = _baseline(config)
    payload["runtime"]["intra_op_num_threads"] = 0
    with pytest.raises(ConfigError, match="intra_op_num_threads"):
        load_config(path=_write(tmp_path, payload))
