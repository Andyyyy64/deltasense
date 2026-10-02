"""Failures are never encoded as successful empty observations."""


class DeltaSenseError(Exception):
    """Base for actionable DeltaSense failures."""


class InputError(DeltaSenseError, ValueError):
    """An image path, image pair, checkpoint, or setting is invalid."""


class UnsupportedTaskError(InputError):
    """Only detection and instance segmentation are supported in v0.1."""


class InferenceError(DeltaSenseError, RuntimeError):
    """Loading or running the selected model failed; no fallback is performed."""


class ObservationError(DeltaSenseError, ValueError):
    """The model returned malformed or inconsistent prediction data."""
