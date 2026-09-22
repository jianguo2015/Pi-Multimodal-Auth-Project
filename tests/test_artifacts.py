"""The shipped artifacts must match the published manifest, byte for byte.

``docs/MODEL_ARTIFACT_MANIFEST.md`` carries a machine-readable table; these
tests are what make it a contract rather than documentation. If any training
or export script ever writes into ``weights/`` again, this suite fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MANIFEST = Path(__file__).resolve().parents[1] / "docs" / "MODEL_ARTIFACT_MANIFEST.md"
ROW = re.compile(
    r"^\|\s*`(?P<path>weights/[^`]+)`\s*\|\s*(?P<size>\d+)\s*\|\s*`(?P<sha>[0-9a-f]{64})`",
    re.MULTILINE,
)

ONNX_INPUTS = {
    "face_extractor": ["face_input"],
    "voice_extractor": ["voice_input"],
    "attention_fusion": ["voice_input", "face_input"],
}


def manifest_rows() -> dict:
    text = MANIFEST.read_text(encoding="utf-8")
    rows = {match["path"]: match for match in ROW.finditer(text)}
    assert rows, "manifest table not found or not in the expected format"
    return rows


def test_manifest_lists_every_shipped_file(project_root):
    rows = manifest_rows()
    on_disk = {
        path.relative_to(project_root).as_posix()
        for path in (project_root / "weights").rglob("*")
        if path.is_file()
    }
    assert set(rows) == on_disk, "weights/ and the manifest disagree"


@pytest.mark.parametrize("path", sorted(manifest_rows()))
def test_artifacts_are_unmodified(project_root, path):
    from multimodal_auth.safety import file_digest

    row = manifest_rows()[path]
    target = project_root / path
    assert target.stat().st_size == int(row["size"]), "size drift in %s" % path
    assert file_digest(target) == row["sha"], "hash drift in %s" % path


def test_onnxruntime_can_load_the_int8_graphs(project_root, config):
    """Guards the deployment blocker found in Phase 0: ORT must be >= 1.24.1."""
    ort = pytest.importorskip("onnxruntime")
    version = tuple(int(part) for part in ort.__version__.split(".")[:3])
    assert version >= (1, 24, 1), "onnxruntime %s cannot load ConvInteger graphs" % ort.__version__

    for role in ("face", "voice", "fusion"):
        session = ort.InferenceSession(
            str(config.onnx_path(role)), providers=["CPUExecutionProvider"]
        )
        stem = session.get_modelmeta().producer_name  # touch the model
        assert stem is not None


@pytest.mark.parametrize("role", ["face", "voice", "fusion"])
def test_onnx_input_contract(project_root, config, role):
    ort = pytest.importorskip("onnxruntime")

    session = ort.InferenceSession(str(config.onnx_path(role)), providers=["CPUExecutionProvider"])
    names = [item.name for item in session.get_inputs()]
    stem = config.models[role].onnx_file.replace("_quant.onnx", "").replace("_fp32.onnx", "")
    assert names == ONNX_INPUTS[stem], "fusion feeds voice first, face second"
    assert [item.name for item in session.get_outputs()] == ["output"]


@pytest.mark.parametrize("role", ["face", "voice", "fusion"])
def test_onnx_dynamic_batch_axis(project_root, config, role):
    ort = pytest.importorskip("onnxruntime")

    session = ort.InferenceSession(str(config.onnx_path(role)), providers=["CPUExecutionProvider"])
    batch = session.get_inputs()[0].shape[0]
    assert batch in ("batch_size", 1, None)


def test_pytorch_checkpoints_load_strictly(config):
    torch = pytest.importorskip("torch")

    from multimodal_auth.fusion import ModalAttentionFusion
    from multimodal_auth.models import FaceExtractor, VoiceExtractor

    models = {
        "face": FaceExtractor(embed_dim=config.models["face"].embed_dim),
        "voice": VoiceExtractor(
            n_mfcc=config.voice.n_mfcc, embed_dim=config.models["voice"].embed_dim
        ),
        "fusion": ModalAttentionFusion(
            voice_dim=config.models["voice"].embed_dim,
            face_dim=config.models["face"].embed_dim,
            hidden_dim=config.models["fusion"].hidden_dim or 64,
        ),
    }
    for role, model in models.items():
        state = torch.load(str(config.pytorch_path(role)), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)


def test_no_stray_files_under_weights(project_root):
    """No checkpoints from a training run may ever end up in weights/."""
    suspicious = [
        path.name
        for pattern in ("*.pth", "*.onnx", "*.npy", "*.json", "*.log", "*.tmp")
        for path in (project_root / "weights").rglob(pattern)
        if path.name not in {"attention_fusion_weights.pth", "face_extractor_weights.pth",
                            "voice_extractor_weights.pth"}
        and not path.name.endswith(("_quant.onnx", "_fp32.onnx"))
    ]
    assert suspicious == []
