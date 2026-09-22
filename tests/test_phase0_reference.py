"""Reproduce the Phase 0 reference numbers on the local biometric data.

These tests are marked ``private_data`` and are skipped automatically when
``data/processed`` is absent, so a fresh clone still gets a green suite. They
are the strongest check in the repository: they reproduce the exact figures
published in ``docs/REPRODUCTION_STATUS.md``

    genuine  mean 0.9182  min 0.8943  max 0.9293  accept@0.85 = 1.00
    impostor max 0.0511 over the expanded 625-pair grid   accept@0.85 = 0.00

They also need the research extras (``requirements/research.txt``, i.e. torch).
That dependency must surface as *skipped tests*, never as a collection error,
because the documented public command ``pytest -m "not private_data"`` still
collects every module before the marker filter runs. The model imports are
therefore deferred until torch is known to be importable.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from multimodal_auth.decision import evaluate
from multimodal_auth.preprocessing import preprocess_face, preprocess_voice

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from multimodal_auth.fusion import ModalAttentionFusion
    from multimodal_auth.models import FaceExtractor, VoiceExtractor

pytestmark = [
    pytest.mark.private_data,
    pytest.mark.skipif(
        not TORCH_AVAILABLE,
        reason="the reference tier needs requirements/research.txt (torch)",
    ),
]

PHASE0_GENUINE = {"mean": 0.9182, "min": 0.8943, "max": 0.9293}


@pytest.fixture(scope="module")
def processed_root(project_root):
    return project_root / "data" / "processed"


@pytest.fixture(scope="module")
def nets(config):
    voice = VoiceExtractor(n_mfcc=config.voice.n_mfcc, embed_dim=128)
    face = FaceExtractor(embed_dim=128)
    fusion = ModalAttentionFusion(voice_dim=128, face_dim=128, hidden_dim=64)
    voice.load_state_dict(torch.load(str(config.pytorch_path("voice")), map_location="cpu"))
    face.load_state_dict(torch.load(str(config.pytorch_path("face")), map_location="cpu"))
    fusion.load_state_dict(torch.load(str(config.pytorch_path("fusion")), map_location="cpu"))
    for model in (voice, face, fusion):
        model.eval()
    return voice, face, fusion


def _embeddings(net, paths):
    with torch.no_grad():
        return torch.cat(
            [net(torch.tensor(np.load(p), dtype=torch.float32).unsqueeze(0)) for p in sorted(paths)]
        )


def _grid(fusion, left, right):
    with torch.no_grad():
        return np.array(
            [
                [float(fusion(left[i: i + 1], right[j: j + 1])[0]) for j in range(right.size(0))]
                for i in range(left.size(0))
            ]
        )


def test_genuine_grid_matches_phase0(processed_root, nets):
    voice_net, face_net, fusion = nets
    me_voices = _embeddings(voice_net, (processed_root / "voice_features" / "User_Me").glob("*.npy"))
    me_faces = _embeddings(face_net, (processed_root / "face_features" / "User_Me").glob("*.npy"))

    assert me_voices.shape == (10, 128)
    assert me_faces.shape == (5, 128)

    scores = _grid(fusion, me_voices, me_faces).reshape(-1)
    assert scores.size == 50
    assert scores.mean() == pytest.approx(PHASE0_GENUINE["mean"], abs=1e-3)
    assert scores.min() == pytest.approx(PHASE0_GENUINE["min"], abs=1e-3)
    assert scores.max() == pytest.approx(PHASE0_GENUINE["max"], abs=1e-3)
    assert all(evaluate(float(s), 0.85).accepted for s in scores)


def test_impostors_are_rejected(processed_root, nets):
    voice_net, face_net, fusion = nets
    me_voices = _embeddings(voice_net, (processed_root / "voice_features" / "User_Me").glob("*.npy"))
    me_faces = _embeddings(face_net, (processed_root / "face_features" / "User_Me").glob("*.npy"))

    scores = []
    for stranger in sorted(p for p in (processed_root / "voice_features").iterdir() if p.is_dir()):
        if stranger.name == "User_Me":
            continue
        stranger_voices = _embeddings(voice_net, stranger.glob("*.npy"))
        scores.append(_grid(fusion, stranger_voices, me_faces).reshape(-1))
    for stranger in sorted(p for p in (processed_root / "face_features").iterdir() if p.is_dir()):
        if stranger.name == "User_Me":
            continue
        stranger_faces = _embeddings(face_net, stranger.glob("*.npy"))
        scores.append(_grid(fusion, me_voices, stranger_faces).reshape(-1))

    impostors = np.concatenate(scores)
    assert impostors.size >= 500
    assert impostors.max() < 0.10
    assert not any(evaluate(float(s), 0.85).accepted for s in impostors)


def test_preprocessing_is_bit_exact_on_raw_inputs(project_root, config):
    """Fresh preprocessing must reproduce the stored Phase 0 features exactly."""
    raw_face = project_root / "data" / "raw" / "face" / "User_Me"
    raw_voice = project_root / "data" / "raw" / "voice" / "User_Me"
    if not raw_face.is_dir() or not raw_voice.is_dir():
        pytest.skip("raw inputs are not present on this machine")

    stored_face = project_root / "data" / "processed" / "face_features" / "User_Me"
    stored_voice = project_root / "data" / "processed" / "voice_features" / "User_Me"

    for path in sorted(raw_face.glob("*.jpg"))[:3]:
        fresh = preprocess_face(path, config.face)
        np.testing.assert_array_equal(fresh, np.load(stored_face / (path.stem + ".npy")))

    for path in sorted(raw_voice.glob("*.wav"))[:3]:
        fresh = preprocess_voice(path, config.voice)
        np.testing.assert_array_equal(fresh, np.load(stored_voice / (path.stem + ".npy")))


def test_onnx_pipeline_agrees_with_pytorch_on_real_features(processed_root, config):
    """INT8 vs PyTorch agreement measured in Phase 0 was well under 1e-2."""
    from multimodal_auth.pipeline.onnx_backend import OnnxBackend
    from multimodal_auth.pipeline.torch_backend import TorchBackend

    onnx_backend = OnnxBackend(config)
    torch_backend = TorchBackend(config)

    differences = []
    for voice_path in sorted((processed_root / "voice_features" / "User_Me").glob("*.npy"))[:3]:
        voice_feature = np.load(voice_path).astype(np.float32)
        for face_path in sorted((processed_root / "face_features" / "User_Me").glob("*.npy"))[:2]:
            face_feature = np.load(face_path).astype(np.float32)
            onnx_score = onnx_backend.score(
                onnx_backend.embed_voice(voice_feature), onnx_backend.embed_face(face_feature)
            )[0]
            torch_score = torch_backend.score(
                torch_backend.embed_voice(voice_feature), torch_backend.embed_face(face_feature)
            )[0]
            differences.append(abs(onnx_score - torch_score))

    assert max(differences) < 1e-2
