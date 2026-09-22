"""Shared pytest fixtures.

Design rules:

* tests never read the private biometric data in ``data/`` unless the
  ``private_data`` marker is used **and** the data exists (see
  ``test_phase0_reference.py``);
* tests never write outside ``tmp_path`` or ``<root>/outputs``;
* the synthetic sample inputs come from ``examples/make_examples.py`` so that
  the tests exercise exactly what a user gets after cloning.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodal_auth.config import load_config  # noqa: E402
from multimodal_auth.preprocessing import write_image_bgr  # noqa: E402


def _load_example_generator():
    path = ROOT / "examples" / "make_examples.py"
    spec = importlib.util.spec_from_file_location("_example_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXAMPLES = _load_example_generator()


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def config():
    return load_config(root=ROOT)


@pytest.fixture(scope="session")
def sample_face_image(tmp_path_factory) -> Path:
    """A deterministic 512x512 synthetic image."""
    path = tmp_path_factory.mktemp("inputs") / "sample_face.jpg"
    write_image_bgr(path, EXAMPLES.make_face())
    return path


@pytest.fixture(scope="session")
def sample_voice_clip(tmp_path_factory) -> Path:
    """A deterministic 2 s harmonic tone at 16 kHz."""
    import soundfile as sf

    path = tmp_path_factory.mktemp("inputs") / "sample_voice.wav"
    sf.write(str(path), EXAMPLES.make_voice(), 16000)
    return path


@pytest.fixture(scope="session")
def private_data_root() -> Path:
    return ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def has_private_data(private_data_root: Path) -> bool:
    return (
        (private_data_root / "voice_features" / "User_Me").is_dir()
        and (private_data_root / "face_features" / "User_Me").is_dir()
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``private_data`` tests when the local biometric data is absent."""
    data_root = ROOT / "data" / "processed"
    available = (data_root / "voice_features" / "User_Me").is_dir()
    if available:
        return
    skip = pytest.mark.skip(reason="private biometric data not present on this machine")
    for item in items:
        if "private_data" in item.keywords:
            item.add_marker(skip)
