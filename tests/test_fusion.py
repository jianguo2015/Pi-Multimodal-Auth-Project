"""Fusion head: attention gating must stay a valid convex combination."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from multimodal_auth.fusion import ModalAttentionFusion, fuse_with_attention  # noqa: E402


@pytest.fixture(scope="module")
def fusion_model():
    from multimodal_auth.config import load_config

    config = load_config()
    model = ModalAttentionFusion(
        voice_dim=config.models["voice"].embed_dim,
        face_dim=config.models["face"].embed_dim,
        hidden_dim=config.models["fusion"].hidden_dim or 64,
    )
    state = torch.load(str(config.pytorch_path("fusion")), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)  # the audited checkpoint must still fit
    model.eval()
    return model


def _embeddings(dim=128, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, dim, generator=generator),
        torch.randn(1, dim, generator=generator),
    )


def test_checkpoint_matches_definition(fusion_model):
    keys = set(fusion_model.state_dict())
    assert "v_proj.weight" in keys and "f_proj.weight" in keys
    assert "attention_net.0.weight" in keys and "classifier.4.weight" in keys


def test_attention_weights_sum_to_one(fusion_model):
    for seed in range(8):
        voice, face = _embeddings(seed=seed)
        breakdown = fuse_with_attention(fusion_model, voice, face)
        assert breakdown.voice_weight + breakdown.face_weight == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= breakdown.voice_weight <= 1.0
        assert 0.0 <= breakdown.face_weight <= 1.0


def test_attention_breakdown_matches_forward(fusion_model):
    """The introspection helper must not drift from the model's own forward."""
    for seed in range(8):
        voice, face = _embeddings(seed=seed)
        expected = float(fusion_model(voice, face).reshape(-1)[0])
        breakdown = fuse_with_attention(fusion_model, voice, face)
        assert breakdown.score == pytest.approx(expected, abs=1e-7)


def test_fused_feature_is_the_weighted_sum(fusion_model):
    voice, face = _embeddings(seed=3)
    breakdown = fuse_with_attention(fusion_model, voice, face)
    manual = breakdown.voice_feature * breakdown.voice_weight + breakdown.face_feature * breakdown.face_weight
    assert torch.allclose(manual, breakdown.fused_feature, atol=1e-6)


def test_score_is_a_probability(fusion_model):
    for seed in range(5):
        voice, face = _embeddings(seed=seed)
        breakdown = fuse_with_attention(fusion_model, voice, face)
        assert 0.0 <= breakdown.score <= 1.0


def test_inference_is_deterministic_in_eval_mode(fusion_model):
    voice, face = _embeddings(seed=11)
    first = fuse_with_attention(fusion_model, voice, face).score
    second = fuse_with_attention(fusion_model, voice, face).score
    assert first == second


def test_breakdown_keeps_training_flag(fusion_model):
    fusion_model.train()
    voice, face = _embeddings(seed=5)
    fuse_with_attention(fusion_model, voice, face)
    assert fusion_model.training is True
    fusion_model.eval()


def test_random_weights_still_sum_to_one():
    model = ModalAttentionFusion()
    model.eval()
    voice, face = _embeddings(seed=42)
    breakdown = fuse_with_attention(model, voice, face)
    assert breakdown.voice_weight + breakdown.face_weight == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(breakdown.score)
