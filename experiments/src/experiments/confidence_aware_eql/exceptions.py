from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from typing_extensions import Any


@dataclass
class UnfamiliarSampleException(DataclassException):
    """Raised when a rule fires for an instance unlikely under the learned distribution.

    The exception names the rule node that scored the instance, so a caller
    evaluating the rule tree can intercept it and handle the unfamiliar case
    instead of acting on a potentially overconfident deterministic result.
    """

    node_name: str
    """Name of the rule node whose conclusions fired for the instance."""

    log_likelihood: float
    """Log-likelihood of the instance under the model."""

    threshold: float
    """The cutoff the log-likelihood was compared against."""

    def error_message(self) -> str:
        """Describe which rule node flagged the instance and by how much."""
        return (
            f"At node '{self.node_name}' the instance has log-likelihood "
            f"{self.log_likelihood:.2f}, which is below the familiarity threshold "
            f"{self.threshold:.2f}."
        )

    def suggest_correction(self) -> str:
        """Advise how an unfamiliar instance should be handled."""
        return (
            "Handle the instance as out-of-distribution instead of applying the "
            "rule's deterministic conclusion, or extend the training data so that "
            "instances like it become familiar."
        )


@dataclass
class UnknownFeatureValueError(DataclassException):
    """Raised when a feature receives a value outside the domain it was learned on."""

    feature_name: str
    """Name of the feature whose value was not recognised."""

    value: Any
    """The unrecognised value that was supplied."""

    def error_message(self) -> str:
        """Describe which feature received an unusable value."""
        return (
            f"The value {self.value!r} of feature '{self.feature_name}' is not part "
            f"of the domain the model was learned on."
        )

    def suggest_correction(self) -> str:
        """Advise how to make the value usable."""
        return (
            f"Give '{self.feature_name}' a value that occurs in the training data, "
            f"or extend the training data so that the value is represented."
        )