"""Human-readable rendering of an :class:`AuthResult`.

Kept separate from ``cli.py`` so that tests can assert on the exact text
without going through argparse, and so the CLI stays small.
"""

from __future__ import annotations

from typing import TextIO

from .pipeline.types import AuthResult

DISCLAIMER = (
    "Research prototype. Trained on two toy identities and synthetic distractor "
    "features, evaluated on those same small sets; not a production biometric "
    "system and not suitable for real access-control decisions. "
    "See docs/LIMITATIONS.md."
)


def _embedding_line(label: str, summary) -> str:
    stats = summary.as_dict()
    return "%-15s : %d-d  (l2=%.4f, mean=%+.4f, range [%+.3f, %+.3f])" % (
        label,
        stats["dim"],
        stats["l2_norm"],
        stats["mean"],
        stats["min"],
        stats["max"],
    )


def attention_text(result: AuthResult) -> str:
    """Attention weights when the backend can provide them, else the reason."""
    if result.attention:
        return "voice=%.4f  face=%.4f  (%s)" % (
            result.attention["voice_weight"],
            result.attention["face_weight"],
            result.attention["source"],
        )
    return "unavailable with the %s backend (the INT8 graph exposes the score only)" % result.backend


def render(result: AuthResult, config_path: str | None = None) -> str:
    """Render the full report as a single string (used verbatim by the CLI)."""
    lines = [
        "=== Multimodal Authentication Result ===",
        "%-15s : %s" % ("Face input", result.inputs["face"]),
        "%-15s : %s" % ("Voice input", result.inputs["voice"]),
        "%-15s : %s" % ("Backend", result.backend),
        _embedding_line("Face embedding", result.face_embedding),
        _embedding_line("Voice embedding", result.voice_embedding),
        "%-15s : %s" % ("Attention", attention_text(result)),
        "%-15s : %.6f" % ("Fusion score", result.score),
        "%-15s : %.2f" % ("Threshold", result.threshold),
        "%-15s : %s  (%s, margin %+.6f)"
        % ("Decision", result.decision.decision.value, result.decision.rule, result.decision.margin),
    ]
    if result.timings_ms:
        t = result.timings_ms
        lines.append(
            "%-15s : preprocess=%.1f ms  face=%.1f ms  voice=%.1f ms  fusion=%.2f ms  total=%.1f ms"
            % (
                "Timing",
                t.get("preprocess", 0.0),
                t.get("face_embed", 0.0),
                t.get("voice_embed", 0.0),
                t.get("fusion", 0.0),
                t.get("total", 0.0),
            )
        )
    if config_path:
        lines.append("%-15s : %s" % ("Config", config_path))
    lines.append("-" * 60)
    lines.append("LIMITATION: %s" % DISCLAIMER)
    return "\n".join(lines)


def print_report(result: AuthResult, config_path: str | None = None, stream: TextIO | None = None) -> None:
    import sys

    print(render(result, config_path), file=stream or sys.stdout)


__all__ = ["DISCLAIMER", "attention_text", "render", "print_report"]
