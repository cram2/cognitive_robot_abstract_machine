from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Type, Any

from krrood.exceptions import DataclassException, InputError


@dataclass
class NoInstancesProvidedError(InputError):
    """
    Raised when a :class:`~krrood.parametrization.feature_extraction.feature_extractor.F
    eatureExtractor` is constructed from an empty instance list.
    """

    def error_message(self) -> str:
        return "At least one instance must be provided to build a FeatureExtractor."

    def suggest_correction(self) -> str:
        return (
            "Pass a non-empty list of DAO instances to FeatureExtractor.from_instances."
        )


@dataclass
class MissingFieldNameError(InputError):
    """
    Raised when :attr:`~krrood.parametrization.feature_extraction.aggregations.Aggregati
    onStatistic.field_name` is ``None`` but :attr:`~krrood.parametrization.feature_extra
    ction.aggregations.AggregationStatistic.aggregation_features` or a method that
    depends on it is accessed.
    """

    def error_message(self) -> str:
        return "field_name must be set to access aggregation_features."

    def suggest_correction(self) -> str:
        return "Construct AggregationStatistic with @aggregation_statistic(field_name='field name')."


@dataclass
class OutOfDomainValueError(DataclassException):
    """
    Raised when a computed aggregation statistic falls outside the training domain of
    its latent variable.
    """

    feature_name: str
    """
    The name of the feature whose value was out of domain.
    """

    value: Any
    """
    The computed value that violated the domain constraint.
    """

    def error_message(self) -> str:
        return f"Value {self.value!r} for feature '{self.feature_name}' is outside its training domain."

    def suggest_correction(self) -> str:
        return (
            "Ensure the observed value falls within the domain of the corresponding latent variable, "
            "or extend the model's training domain to include this value."
        )


@dataclass
class UnsupportedFeatureTypeError(DataclassException):
    """
    Raised when a feature column has a type that cannot be preprocessed for a
    :class:`~probabilistic_model.learning.jpt.jpt.JointProbabilityTree`.
    """

    feature_type: Type
    """
    The unsupported type encountered during preprocessing.
    """

    column_name: str
    """
    The name of the dataframe column that triggered the error.
    """

    def error_message(self) -> str:
        return (
            f"Unsupported type {self.feature_type!r} for column '{self.column_name}'."
        )

    def suggest_correction(self) -> str:
        return (
            "Only bool, enum, and random_events compatible types are supported. "
            "Ensure the feature column maps to one of those types."
        )
