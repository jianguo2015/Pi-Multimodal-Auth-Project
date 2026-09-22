"""Attention-gated fusion of the two modality embeddings.

PyTorch is imported lazily, so this package can be imported (and its safety
documentation read) without torch installed.
"""

from typing import TYPE_CHECKING

__all__ = ["ModalAttentionFusion", "FusionBreakdown", "fuse_with_attention"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .attention_fusion import ModalAttentionFusion
    from .inspect_attention import FusionBreakdown, fuse_with_attention


def __getattr__(name: str):
    from importlib import import_module

    if name == "ModalAttentionFusion":
        module = import_module(".attention_fusion", __name__)
    elif name in {"FusionBreakdown", "fuse_with_attention"}:
        module = import_module(".inspect_attention", __name__)
    else:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    value = getattr(module, name)
    globals()[name] = value
    return value
