"""Preprocessing: face images and audio clips."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from multimodal_auth.errors import InvalidAudioError, InvalidImageError
from multimodal_auth.preprocessing import (
    load_audio,
    mfcc_features,
    preprocess_face,
    preprocess_voice,
    read_image_bgr,
    write_image_bgr,
)


# ------------------------------------------------------------------ face
def test_face_shape_dtype_and_range(sample_face_image, config):
    tensor = preprocess_face(sample_face_image, config.face)
    assert tensor.shape == (3, config.face.image_size, config.face.image_size)
    assert tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0


def test_face_is_deterministic(sample_face_image, config):
    first = preprocess_face(sample_face_image, config.face)
    second = preprocess_face(sample_face_image, config.face)
    assert np.array_equal(first, second)


def test_face_accepts_a_preloaded_array(sample_face_image, config):
    image = read_image_bgr(sample_face_image)
    from_array = preprocess_face(image, config.face)
    from_path = preprocess_face(sample_face_image, config.face)
    assert np.array_equal(from_array, from_path)


def test_face_missing_file(config, tmp_path):
    with pytest.raises(InvalidImageError, match="not found"):
        preprocess_face(tmp_path / "missing.jpg", config.face)


def test_face_undecodable_file(config, tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"this is not an image")
    with pytest.raises(InvalidImageError):
        preprocess_face(broken, config.face)


def test_face_empty_file(config, tmp_path):
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    with pytest.raises(InvalidImageError):
        preprocess_face(empty, config.face)


def test_face_wrong_channel_count(config):
    """A 2-D array passed directly (no file) must be rejected explicitly."""
    gray = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(InvalidImageError, match="3-channel"):
        preprocess_face(gray, config.face)


def test_face_center_crop_fallback_on_flat_image(config, tmp_path):
    """No detectable face -> deterministic centre crop, not an exception."""
    from multimodal_auth.preprocessing import write_image_bgr

    flat = np.full((200, 120, 3), 128, dtype=np.uint8)
    path = tmp_path / "flat.jpg"
    write_image_bgr(path, flat)
    tensor = preprocess_face(path, config.face)
    assert tensor.shape == (3, config.face.image_size, config.face.image_size)
    # centre crop of a uniform image normalises to (128 - 127.5) / 128
    expected = (128.0 - config.face.mean) / config.face.scale
    assert np.allclose(tensor, expected, atol=2e-2)


def test_read_and_write_roundtrip_unicode_path(tmp_path):
    """Non-ASCII paths are the default in this project; they must work."""
    from multimodal_auth.preprocessing import write_image_bgr

    directory = tmp_path / "图像-voice-测试"
    target = directory / "sample-image.jpg"
    original = np.zeros((32, 32, 3), dtype=np.uint8)
    original[:, :, 0] = 200
    write_image_bgr(target, original)
    loaded = read_image_bgr(target)
    assert loaded.shape == original.shape
    assert np.abs(loaded.astype(int) - original.astype(int)).max() <= 8  # JPEG lossy


# ----------------------------------------------------------------- voice
def test_voice_shape_dtype_and_padding(sample_voice_clip, config):
    tensor = preprocess_voice(sample_voice_clip, config.voice)
    assert tensor.shape == (1, config.voice.n_mfcc, config.voice.max_pad_len)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_voice_tail_is_zero_padded_for_short_clips(config, tmp_path):
    short = tmp_path / "short.wav"
    sf.write(str(short), np.zeros(8000, dtype=np.float32), 16000)
    tensor = preprocess_voice(short, config.voice)
    assert tensor.shape == (1, 40, 400)
    # a 0.5 s clip yields far fewer than 400 frames, so the tail must be zeros
    assert np.all(tensor[:, :, -1] == 0.0)


def test_voice_is_cropped_for_long_clips(config, tmp_path):
    """>400 MFCC frames (about 12.8 s at 16 kHz) must be cropped, not padded."""
    rng = np.random.default_rng(7)
    long_clip = tmp_path / "long.wav"
    sf.write(str(long_clip), (0.1 * rng.standard_normal(16 * 16000)).astype(np.float32), 16000)
    tensor = preprocess_voice(long_clip, config.voice)
    assert tensor.shape == (1, 40, 400)
    assert tensor[:, :, -1].any()  # genuinely cropped, not zero-padded


def test_voice_missing_file(config, tmp_path):
    with pytest.raises(InvalidAudioError, match="not found"):
        preprocess_voice(tmp_path / "missing.wav", config.voice)


def test_voice_empty_file(config, tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(InvalidAudioError, match="empty"):
        preprocess_voice(empty, config.voice)


def test_voice_undecodable_file(config, tmp_path):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"not audio at all" * 100)
    with pytest.raises(InvalidAudioError):
        preprocess_voice(broken, config.voice)


def test_voice_too_short_is_rejected(config, tmp_path):
    tiny = tmp_path / "tiny.wav"
    sf.write(str(tiny), np.zeros(400, dtype=np.float32), 16000)
    with pytest.raises(InvalidAudioError, match="too short"):
        preprocess_voice(tiny, config.voice)


def test_mfcc_rejects_empty_waveform(config):
    with pytest.raises(InvalidAudioError):
        mfcc_features(np.zeros(0, dtype=np.float32), config.voice)


def test_load_audio_resamples(config, tmp_path):
    path = tmp_path / "8k.wav"
    sf.write(str(path), np.zeros(16000, dtype=np.float32), 8000)
    waveform = load_audio(path, sample_rate=16000)
    assert waveform.size == 32000