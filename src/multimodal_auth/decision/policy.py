"""Accept / reject decision policy.

The Phase 0 audit found exactly one threshold in the original project
(``config/settings.yaml -> inference.fusion_threshold: 0.85``) and no other
decision rule anywhere. Phase 1 preserves that single rule and makes it
explicit, testable and configurable - nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import ConfigError

DEFAULT_THRESHOLD = 0.85


class Decision(str, Enum):
    """Outcome of comparing a fusion score against the threshold."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class DecisionResult:
    score: float
    threshold: float
    decision: Decision
    margin: float          # score - threshold; negative means rejected
    rule: str = "score >= threshold"

    @property
    def accepted(self) -> bool:
        return self.decision is Decision.ACCEPT

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "threshold": self.threshold,
            "decision": self.decision.value,
            "margin": self.margin,
            "rule": self.rule,
        }


def validate_threshold(threshold: float) -> float:
    """Threshold must be a probability strictly inside (0, 1)."""
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        raise ConfigError("threshold must be a number, got %r" % (threshold,))
    if not 0.0 < value < 1.0:
        raise ConfigError("threshold must be strictly between 0 and 1, got %s" % value)
    return value


def validate_score(score: float) -> float:
    """Fusion scores come out of a sigmoid, so they must be in [0, 1]."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        raise ConfigError("score must be a number, got %r" % (score,))
    if not 0.0 <= value <= 1.0:
        raise ConfigError("score must be within [0, 1], got %s" % value)
    return value


def decide(score: float, threshold: float = DEFAULT_THRESHOLD) -> Decision:
    """Apply the frozen rule: accept when ``score >= threshold``."""
    return evaluate(score, threshold).decision


def evaluate(score: float, threshold: float = DEFAULT_THRESHOLD) -> DecisionResult:
    """Full decision record, including the margin for reporting."""
    valid_score = validate_score(score)
    valid_threshold = validate_threshold(threshold)
    decision = Decision.ACCEPT if valid_score >= valid_threshold else Decision.REJECT
    return DecisionResult(
        score=valid_score,
        threshold=valid_threshold,
        decision=decision,
        margin=valid_score - valid_threshold,
    )
