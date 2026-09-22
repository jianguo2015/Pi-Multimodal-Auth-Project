"""CLI contract: exit codes, output formats, and refusing unsafe writes."""

from __future__ import annotations

import json

import pytest

from multimodal_auth.cli import EXIT_INPUT, EXIT_OK, EXIT_REJECT, run_infer


def test_missing_face_is_an_input_error(tmp_path, sample_voice_clip, capsys):
    code = run_infer(
        ["--face", str(tmp_path / "nope.jpg"), "--voice", str(sample_voice_clip)]
    )
    assert code == EXIT_INPUT
    assert "input error" in capsys.readouterr().err


def test_missing_voice_is_an_input_error(tmp_path, sample_face_image, capsys):
    code = run_infer(
        ["--face", str(sample_face_image), "--voice", str(tmp_path / "nope.wav")]
    )
    assert code == EXIT_INPUT
    assert "input error" in capsys.readouterr().err


def test_human_report(sample_face_image, sample_voice_clip, capsys):
    code = run_infer(["--face", str(sample_face_image), "--voice", str(sample_voice_clip)])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    for expected in (
        "Multimodal Authentication Result",
        "Fusion score",
        "Threshold",
        "Decision",
        "LIMITATION",
        "Research prototype",
    ):
        assert expected in out
    assert "0.85" in out


def test_json_output(sample_face_image, sample_voice_clip, capsys):
    code = run_infer(
        ["--face", str(sample_face_image), "--voice", str(sample_voice_clip), "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["backend"] == "onnx"
    assert payload["decision"]["rule"] == "score >= threshold"
    assert payload["face_embedding"]["dim"] == 128


def test_show_config(sample_face_image, capsys):
    code = run_infer(["--show-config", "--face", str(sample_face_image), "--voice", "x"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["decision"]["threshold"] == 0.85


def test_fail_on_reject_exit_code(sample_face_image, sample_voice_clip, capsys):
    """Synthetic inputs are not the enrolled identity, so the decision is REJECT."""
    code = run_infer(
        [
            "--face", str(sample_face_image),
            "--voice", str(sample_voice_clip),
            "--fail-on-reject",
        ]
    )
    capsys.readouterr()
    assert code == EXIT_REJECT


def test_invalid_threshold_is_a_config_error(sample_face_image, sample_voice_clip, capsys):
    from multimodal_auth.cli import EXIT_CONFIG

    code = run_infer(
        [
            "--face", str(sample_face_image),
            "--voice", str(sample_voice_clip),
            "--threshold", "1.5",
        ]
    )
    assert code == EXIT_CONFIG
    assert "configuration error" in capsys.readouterr().err


def test_result_can_be_written_to_a_file(sample_face_image, sample_voice_clip, tmp_path, capsys):
    target = tmp_path / "result.json"
    code = run_infer(
        [
            "--face", str(sample_face_image),
            "--voice", str(sample_voice_clip),
            "--json",
            "--output", str(target),
        ]
    )
    capsys.readouterr()
    assert code == EXIT_OK
    assert json.loads(target.read_text(encoding="utf-8"))["backend"] == "onnx"


def test_result_cannot_be_written_into_weights(project_root, sample_face_image, sample_voice_clip, capsys):
    from multimodal_auth.cli import EXIT_ARTIFACT

    target = project_root / "weights" / "result.json"
    code = run_infer(
        [
            "--face", str(sample_face_image),
            "--voice", str(sample_voice_clip),
            "--output", str(target),
        ]
    )
    capsys.readouterr()
    assert code == EXIT_ARTIFACT
    assert not target.exists()


def test_pytorch_backend_selection(sample_face_image, sample_voice_clip, capsys):
    pytest.importorskip("torch")
    code = run_infer(
        [
            "--face", str(sample_face_image),
            "--voice", str(sample_voice_clip),
            "--backend", "pytorch",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["backend"] == "pytorch"
    assert payload["attention_weights"]["voice_weight"] > 0
