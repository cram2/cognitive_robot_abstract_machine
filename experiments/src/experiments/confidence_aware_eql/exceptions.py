from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from typing_extensions import Any


@dataclass
class UnfamiliarSampleWarning(Warning):
    """
    Reports that an instance is unlikely under the learned distribution.

    The warning names the rule node that scored the instance, so the
    deterministic result of the rule tree can be accompanied by an explicit
    statement of doubt.

    .. note::
        This subclasses :class:`Warning` rather than
        :class:`krrood.exceptions.DataclassException` because it reports a doubt
        about a result rather than an error: the evaluation continues and the
        deterministic result is still produced.
    """

    node_name: str
    """Name of the rule-tree node that evaluated the instance."""

    log_likelihood: float
    """
    Log-likelihood of the instance under the model.
    """

    threshold: float
    """The cutoff the log-likelihood was compared against."""

    def __post_init__(self) -> None:
        super().__init__(str(self))

    def __str__(self) -> str:
        return (
            f"At node '{self.node_name}' the instance has log-likelihood "
            f"{self.log_likelihood:.2f}, which is below the familiarity threshold "
            f"{self.threshold:.2f}."
        )


@dataclass
class UnknownFeatureValueError(DataclassException):
    """
    Raised when a feature receives a value outside the domain it was learned on.
    """

    feature_name: str
    """Name of the feature whose value was not recognised."""

    value: Any
    """
    The unrecognised value that was supplied.
    """

    def error_message(self) -> str:
        """
        Describe which feature received an unusable value.
        """
        return (
            f"The value {self.value!r} of feature '{self.feature_name}' is not part "
            f"of the domain the model was learned on."
        )

    def suggest_correction(self) -> str:
        """
        Advise how to make the value usable.
        """
        return (
            f"Give '{self.feature_name}' a value that occurs in the training data, "
            f"or extend the training data so that the value is represented."
        )
