"""Exception hierarchy for the package.

Every failure mode the pipeline can hit is expressed as one of these types so
that callers (CLI, tests) never have to inspect third-party error strings.
"""


class MultimodalAuthError(Exception):
    """Base class for all package errors."""


class ConfigError(MultimodalAuthError):
    """Configuration file is missing, malformed or internally inconsistent."""


class InvalidInputError(MultimodalAuthError):
    """User supplied input could not be used (missing / unreadable / empty)."""


class InvalidImageError(InvalidInputError):
    """Image could not be decoded or contains no usable region."""


class InvalidAudioError(InvalidInputError):
    """Audio could not be decoded or is too short to analyse."""


class ModelArtifactError(MultimodalAuthError):
    """A required model artefact is missing or cannot be loaded."""


class SafetyError(MultimodalAuthError):
    """An operation was refused because it would modify protected artefacts."""
