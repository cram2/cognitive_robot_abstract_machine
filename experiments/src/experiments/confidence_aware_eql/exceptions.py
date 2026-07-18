from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Optional


@dataclass
class UnfamiliarSampleWarning(Warning):
    """Raised when a queried instance is unlikely under the learned distribution.

    The warning carries enough context to trace which rule node rejected the
    instance and why, so a caller can surface an explainable reason instead of a
    silent low-confidence result.

    .. note::
        This subclasses :class:`Warning` so it can be issued through
        :func:`warnings.warn` or collected and raised, while still being a plain
        dataclass with documented fields.
    """

    node_name: str
    """Name of the rule-tree node that evaluated the instance."""

    reason: str
    """Human-readable explanation of why the instance was flagged."""

    log_likelihood: Optional[float] = None
    """Log-likelihood of the instance under the model, or ``None`` when the
    instance could not be scored (for example a missing feature)."""

    def __post_init__(self) -> None:
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"UnfamiliarSampleWarning at '{self.node_name}': {self.reason}"


@dataclass
class UnknownFeatureValueError(Exception):
    """Raised when a categorical feature receives a value outside its domain."""

    feature_name: str
    """Name of the feature whose value was not recognised."""

    value: object
    """The unrecognised value that was supplied."""

    def __post_init__(self) -> None:
        super().__init__(
            f"unknown value {self.value!r} for categorical feature '{self.feature_name}'"
        )
