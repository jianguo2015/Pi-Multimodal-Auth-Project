"""Inference pipeline package."""

from .pipeline import MultimodalPipeline, create_backend  # noqa: F401
from .types import AuthResult, EmbeddingSummary, summarise  # noqa: F401

__all__ = [
    "MultimodalPipeline",
    "create_backend",
    "AuthResult",
    "EmbeddingSummary",
    "summarise",
]
