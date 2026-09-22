"""End-to-end pipeline behaviour on inputs that are always available."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from multimodal_auth.decision import Decision, evaluate
from multimodal_auth.errors import InvalidAudioError, InvalidImageError, ModelArtifactError
from multimodal_auth.pipeline import MultimodalPipeline


@pytest.fixture(scope="module")
def onnx_pipeline():
    return MultimodalPipeline(backend="onnx")


def test_sessions_and_models_are_loaded_lazily():
    """Constructing a pipeline must not touch the disk; the engine loads on use."""
    fresh = MultimodalPipeline(backend="onnx")
    assert fresh._engine is None
    fresh.run
    assert fresh._engine is None  # still nothing loaded


def test_describe(onnx_pipeline, sample_face_image, sample_voice_clip):
    onnx_pipeline.run(sample_face_image, sample_voice_clip)  # force the engine to load
    info = onnx_pipeline.describe()
    assert info["backend"] == "onnx"
    assert info["threshold"] == 0.85
    assert set(info["model_files"]) == {"face", "voice", "fusion"}
    assert info["onnxruntime_version"]


def test_run_produces_a_complete_result(onnx_pipeline, sample_face_image, sample_voice_clip):
    result = onnx_pipeline.run(sample_face_image, sample_voice_clip)
    assert 0.0 <= result.score <= 1.0
    assert result.threshold == 0.85
    assert result.decision.decision in (Decision.ACCEPT, Decision.REJECT)
    assert result.face_embedding.dim == 128
    assert result.voice_embedding.dim == 128
    assert set(result.timings_ms) == {"preprocess", "face_embed", "voice_embed", "fusion", "total"}
    assert all(value >= 0 for value in result.timings_ms.values())


def test_decision_matches_the_policy(onnx_pipeline, sample_face_image, sample_voice_clip):
    result = onnx_pipeline.run(sample_face_image, sample_voice_clip)
    expected = evaluate(result.score, result.threshold)
    assert result.accepted is expected.accepted
    assert result.decision.margin == pytest.approx(expected.margin)


def test_result_is_json_serialisable(onnx_pipeline, sample_face_image, sample_voice_clip):
    payload = onnx_pipeline.run(sample_face_image, sample_voice_clip).as_dict()
    text = json.dumps(payload, ensure_ascii=False)
    assert "decision" in text and "attention_weights" in text
    assert payload["face_embedding"]["dim"] == 128
    # the raw 128-d vectors are never dumped into reports
    assert "vector" not in payload["face_embedding"]


def test_inference_is_deterministic(onnx_pipeline, sample_face_image, sample_voice_clip):
    first = onnx_pipeline.run(sample_face_image, sample_voice_clip).score
    second = onnx_pipeline.run(sample_face_image, sample_voice_clip).score
    assert first == second


def test_onnx_backend_reports_no_attention(onnx_pipeline, sample_face_image, sample_voice_clip):
    result = onnx_pipeline.run(sample_face_image, sample_voice_clip)
    assert result.attention is None
    assert any("attention" in warning for warning in result.warnings)


def test_threshold_override(onnx_pipeline, sample_face_image, sample_voice_clip):
    strict = MultimodalPipeline(backend="onnx", threshold=0.999)
    result = strict.run(sample_face_image, sample_voice_clip)
    assert result.threshold == 0.999
    assert result.decision.decision is Decision.REJECT


def test_invalid_threshold_rejected_at_construction():
    from multimodal_auth.errors import ConfigError

    with pytest.raises(ConfigError):
        MultimodalPipeline(backend="onnx", threshold=1.5)


def test_missing_face_raises(onnx_pipeline, tmp_path, sample_voice_clip):
    with pytest.raises(InvalidImageError):
        onnx_pipeline.run(tmp_path / "nope.jpg", sample_voice_clip)


def test_missing_audio_raises(onnx_pipeline, sample_face_image, tmp_path):
    with pytest.raises(InvalidAudioError):
        onnx_pipeline.run(sample_face_image, tmp_path / "nope.wav")


def test_missing_onnx_artifact_is_reported(config, tmp_path, sample_face_image, sample_voice_clip):
    broken = dataclasses.replace(
        config, onnx_dir=tmp_path / "no-such-directory"
    )
    pipeline = MultimodalPipeline(broken, backend="onnx")
    with pytest.raises(ModelArtifactError, match="missing ONNX artefact"):
        pipeline.run(sample_face_image, sample_voice_clip)


def test_unknown_backend():
    from multimodal_auth.pipeline.pipeline import create_backend

    from multimodal_auth.config import load_config

    with pytest.raises(ValueError, match="unknown backend"):
        create_backend(load_config(), "magic")


def test_torch_backend_agrees_with_onnx(sample_face_image, sample_voice_clip):
    torch = pytest.importorskip("torch")

    onnx_result = MultimodalPipeline(backend="onnx").run(sample_face_image, sample_voice_clip)
    torch_result = MultimodalPipeline(backend="pytorch").run(sample_face_image, sample_voice_clip)

    assert torch_result.backend == "pytorch"
    assert abs(torch_result.score - onnx_result.score) < 1e-2
    assert torch_result.attention is not None
    total = torch_result.attention["voice_weight"] + torch_result.attention["face_weight"]
    assert total == pytest.approx(1.0, abs=1e-6)


def test_torch_backend_embeddings_track_onnx(config, sample_face_image, sample_voice_clip):
    """INT8 vs PyTorch embeddings must stay highly aligned, not bit-identical."""
    pytest.importorskip("torch")

    from multimodal_auth.preprocessing import preprocess_face, preprocess_voice

    face_tensor = preprocess_face(sample_face_image, config.face)
    voice_tensor = preprocess_voice(sample_voice_clip, config.voice)

    from multimodal_auth.pipeline.onnx_backend import OnnxBackend
    from multimodal_auth.pipeline.torch_backend import TorchBackend

    onnx_backend = OnnxBackend(config)
    torch_backend = TorchBackend(config)

    def cosine(left, right):
        left = np.asarray(left, dtype=np.float64).reshape(-1)
        right = np.asarray(right, dtype=np.float64).reshape(-1)
        return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))

    face_similarity = cosine(
        onnx_backend.embed_face(face_tensor), torch_backend.embed_face(face_tensor)
    )
    voice_similarity = cosine(
        onnx_backend.embed_voice(voice_tensor), torch_backend.embed_voice(voice_tensor)
    )
    assert face_similarity > 0.99, face_similarity
    assert voice_similarity > 0.99, voice_similarity
