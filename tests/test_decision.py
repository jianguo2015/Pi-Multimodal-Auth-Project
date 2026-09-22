"""Decision policy: the single frozen threshold rule."""

from __future__ import annotations

import pytest

from multimodal_auth.decision import (
    DEFAULT_THRESHOLD,
    Decision,
    decide,
    evaluate,
    validate_score,
    validate_threshold,
)
from multimodal_auth.errors import ConfigError


def test_default_threshold_matches_phase0():
    assert DEFAULT_THRESHOLD == 0.85


def test_boundary_is_inclusive():
    """score == threshold must ACCEPT (the original used ``score >= threshold``)."""
    assert decide(0.85, 0.85) is Decision.ACCEPT
    assert decide(0.8500001, 0.85) is Decision.ACCEPT


def test_just_below_threshold_is_rejected():
    assert decide(0.8499999, 0.85) is Decision.REJECT


def test_margin_direction():
    accepted = evaluate(0.9, 0.85)
    rejected = evaluate(0.1, 0.85)
    assert accepted.margin == pytest.approx(0.05)
    assert rejected.margin == pytest.approx(-0.75)
    assert accepted.accepted is True
    assert rejected.accepted is False


def test_phase0_operating_points():
    """The scores documented in Phase 0 must land on the documented side."""
    assert evaluate(0.9182, 0.85).decision is Decision.ACCEPT       # genuine mean
    assert evaluate(0.8943, 0.85).decision is Decision.ACCEPT       # genuine min
    assert evaluate(0.0483, 0.85).decision is Decision.REJECT       # impostor max
    assert evaluate(0.0511, 0.85).decision is Decision.REJECT       # wider-grid impostor max


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -5.0])
def test_invalid_scores_rejected(bad):
    with pytest.raises(ConfigError, match=r"within \[0, 1\]"):
        validate_score(bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.2, -0.5])
def test_invalid_thresholds_rejected(bad):
    with pytest.raises(ConfigError, match="strictly between"):
        validate_threshold(bad)


def test_non_numeric_inputs_rejected():
    with pytest.raises(ConfigError):
        validate_score("high")
    with pytest.raises(ConfigError):
        validate_threshold(None)


def test_as_dict_shape():
    payload = evaluate(0.9182).as_dict()
    assert set(payload) == {"score", "threshold", "decision", "margin", "rule"}
    assert payload["rule"] == "score >= threshold"
    assert payload["decision"] == "ACCEPT"


def test_decision_enum_values():
    assert Decision.ACCEPT.value == "ACCEPT"
    assert Decision.REJECT.value == "REJECT"
