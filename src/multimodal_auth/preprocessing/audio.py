"""Voice preprocessing: audio file -> (1, n_mfcc, max_pad_len) float32 tensor.

Numerical pipeline copied verbatim from the audited original
``utils/audio_ops.py::extract_mfcc`` (``librosa.load(sr=16000)``, 40 MFCCs,
crop-or-zero-pad the time axis to 400, prepend a channel axis). No
normalisation is applied - the original did not apply any, and the frozen
embedding comparison depends on that.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import numpy as np

from ..config.loader import VoicePreprocessConfig
from ..errors import InvalidAudioError

#: 0.1 s at 16 kHz. Anything shorter cannot carry a usable utterance and is
#: rejected instead of being silently zero-padded into a meaningless tensor.
MIN_AUDIO_SAMPLES = 1600


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode a mono waveform at ``sample_rate`` using librosa."""
    file_path = Path(path)
    if not file_path.is_file():
        raise InvalidAudioError("audio file not found: %s" % file_path)
    try:
        if file_path.stat().st_size == 0:
            raise InvalidAudioError("audio file is empty: %s" % file_path)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        raise InvalidAudioError("cannot stat audio file %s: %s" % (file_path, exc))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, _ = librosa.load(str(file_path), sr=sample_rate)
    except Exception as exc:
        raise InvalidAudioError("cannot decode audio %s: %s" % (file_path, exc))

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size < MIN_AUDIO_SAMPLES:
        raise InvalidAudioError(
            "audio too short: %d samples (< %d required) in %s"
            % (waveform.size, MIN_AUDIO_SAMPLES, file_path)
        )
    return waveform


def mfcc_features(
    waveform: np.ndarray,
    config: VoicePreprocessConfig | None = None,
    sample_rate: int | None = None,
) -> np.ndarray:
    """Crop-or-pad MFCC time axis and add the channel dimension."""
    config = config or VoicePreprocessConfig()
    sr = sample_rate or config.sample_rate

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size < MIN_AUDIO_SAMPLES:
        # librosa happily returns a single padded frame for a tiny or empty
        # input, which would silently turn into an all-zero tensor. Refuse it.
        raise InvalidAudioError(
            "waveform too short for MFCC extraction: %d samples (< %d required)"
            % (waveform.size, MIN_AUDIO_SAMPLES)
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=config.n_mfcc)

    if mfcc.ndim != 2 or mfcc.shape[0] != config.n_mfcc or mfcc.shape[1] == 0:
        raise InvalidAudioError("MFCC extraction produced an unusable shape %s" % (mfcc.shape,))

    if mfcc.shape[1] > config.max_pad_len:
        mfcc = mfcc[:, : config.max_pad_len]
    else:
        pad_width = config.max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")

    return np.ascontiguousarray(np.expand_dims(mfcc, axis=0), dtype=np.float32)


def preprocess_voice(
    source: str | Path | np.ndarray,
    config: VoicePreprocessConfig | None = None,
) -> np.ndarray:
    """Convert an audio file (or waveform) into the ``VoiceExtractor`` input."""
    config = config or VoicePreprocessConfig()
    if isinstance(source, (str, Path)):
        waveform = load_audio(source, sample_rate=config.sample_rate)
    else:
        waveform = np.asarray(source, dtype=np.float32).reshape(-1)
        if waveform.size < MIN_AUDIO_SAMPLES:
            raise InvalidAudioError(
                "waveform too short: %d samples (< %d required)"
                % (waveform.size, MIN_AUDIO_SAMPLES)
            )
    return mfcc_features(waveform, config, sample_rate=config.sample_rate)
