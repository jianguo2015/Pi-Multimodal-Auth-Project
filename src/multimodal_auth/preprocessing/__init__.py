"""Preprocessing package: raw files -> model-ready tensors.

Both entry points are pure functions of (input, config) and are deterministic,
which is what makes the regression tests against Phase 0 features possible.
"""

from .audio import load_audio, mfcc_features, preprocess_voice  # noqa: F401
from .face import (  # noqa: F401
    build_cascade,
    detect_faces,
    preprocess_face,
    read_image_bgr,
    write_image_bgr,
)

__all__ = [
    "load_audio",
    "mfcc_features",
    "preprocess_voice",
    "build_cascade",
    "detect_faces",
    "preprocess_face",
    "read_image_bgr",
    "write_image_bgr",
]
