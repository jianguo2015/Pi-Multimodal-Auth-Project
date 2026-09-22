"""Feature-extractor networks (verbatim copies of the audited originals).

The modules require PyTorch, which is only needed for the research/export
workflow, so imports are resolved lazily. ``import multimodal_auth`` therefore
never pulls in torch and the ONNX-only path stays lightweight.
"""

from typing import TYPE_CHECKING

_LAZY = {
    "FaceExtractor": "face_extractor",
    "VoiceExtractor": "voice_extractor",
}

__all__ = ["FaceExtractor", "VoiceExtractor"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .face_extractor import FaceExtractor
    from .voice_extractor import VoiceExtractor


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    from importlib import import_module

    module = import_module(".%s" % module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
