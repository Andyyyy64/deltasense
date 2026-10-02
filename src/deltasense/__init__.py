"""DeltaSense: compare model observations without claiming verified physical change."""

from .engine import DeltaSense
from .errors import (
    DeltaSenseError,
    InferenceError,
    InputError,
    ObservationError,
    UnsupportedTaskError,
)
from .result import ComparisonResult

__version__ = "0.1.0"
__all__ = [
    "DeltaSense",
    "ComparisonResult",
    "DeltaSenseError",
    "InputError",
    "InferenceError",
    "ObservationError",
    "UnsupportedTaskError",
]
