"""Read-only introspection of the fusion model.

Why this module exists
----------------------
``ModalAttentionFusion.forward`` returns only the final score. The attention
gate values are the most interesting part of the architecture, but exposing
them must not change the model definition, because the audited ``.pth``
artefacts have to keep loading with ``strict=True`` into a byte-identical
class.

The helper below therefore re-runs the *same* submodules in the *same* order
and returns the intermediate tensors. ``tests/test_fusion.py`` asserts that the
score produced here is bit-identical to ``model.forward``, so the duplication
cannot silently drift.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class FusionBreakdown(NamedTuple):
    """Intermediate values of one fusion forward pass (batch size 1)."""

    score: float
    voice_weight: float
    face_weight: float
    voice_feature: torch.Tensor
    face_feature: torch.Tensor
    fused_feature: torch.Tensor


@torch.no_grad()
def fuse_with_attention(model, voice_embedding, face_embedding) -> FusionBreakdown:
    """Return the score *and* the attention breakdown of a fusion pass.

    Parameters
    ----------
    model:
        A :class:`ModalAttentionFusion` in ``eval()`` mode.
    voice_embedding, face_embedding:
        Tensors of shape ``(1, embed_dim)``.

    Mirrors ``ModalAttentionFusion.forward`` step by step:
    ``relu(proj)`` -> ``concat`` -> ``softmax(attention)`` ->
    ``w_v * v_feat + w_f * f_feat`` -> ``classifier`` -> ``sigmoid``.
    BatchNorm1d runs in evaluation mode, exactly like the ONNX export.
    """
    was_training = model.training
    model.eval()
    try:
        voice_feature = model.relu(model.v_proj(voice_embedding))
        face_feature = model.relu(model.f_proj(face_embedding))

        concat_feature = torch.cat((voice_feature, face_feature), dim=1)
        weights = model.attention_net(concat_feature)  # [1, 2]

        voice_weight = weights[:, 0].unsqueeze(1)
        face_weight = weights[:, 1].unsqueeze(1)

        fused_feature = (voice_feature * voice_weight) + (face_feature * face_weight)
        score = model.classifier(fused_feature)
    finally:
        if was_training:
            model.train()

    return FusionBreakdown(
        score=float(score.reshape(-1)[0]),
        voice_weight=float(voice_weight.reshape(-1)[0]),
        face_weight=float(face_weight.reshape(-1)[0]),
        voice_feature=voice_feature,
        face_feature=face_feature,
        fused_feature=fused_feature,
    )
