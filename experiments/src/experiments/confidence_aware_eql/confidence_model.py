"""Learn a confidence model over familiar objects and score new ones.

The confidence model is a relational probabilistic circuit fitted per class on the
familiar instances of that class. It answers one question about a new object: how
likely is it under the distribution of familiar instances of its own class. An
object whose likelihood falls below its class's calibrated threshold does not
resemble anything the model was trained on and is judged unfamiliar.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from krrood.entity_query_language.factories import a
from krrood.entity_query_language.query.match import Match
from krrood.exceptions import DataclassException
from krrood.ormatic.data_access_objects.dao import get_dao_schema, to_dao
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from typing_extensions import Any, List

# %% exceptions


@dataclass
class UnmodeledClassError(DataclassException):
    """Raised when scoring an instance whose class no per-class model was fitted for."""

    instance_class: type
    """The class no confidence model was fitted for."""

    def error_message(self) -> str:
        return f"No confidence model was fitted for {self.instance_class!r}."

    def suggest_correction(self) -> str:
        return (
            "Include at least one instance of this class in the instances passed to "
            "fit_from_instances."
        )


@dataclass
class AmbiguousThresholdError(DataclassException):
    """Raised when reading the single-class threshold of a model fitted on several classes."""

    fitted_classes: List[type]
    """The classes this confidence model was actually fitted on."""

    def error_message(self) -> str:
        return (
            f"threshold is ambiguous: this confidence model was fitted on "
            f"{len(self.fitted_classes)} classes {self.fitted_classes!r}."
        )

    def suggest_correction(self) -> str:
        return "Call threshold_for(instance) to get the threshold for a specific instance's class."


# %% grounding query construction


def _build_grounding_query(domain_class: type, instance: Any) -> Match:
    """Build a query mirroring an instance's collection structure for RSPN grounding.

    Every constructor field of ``domain_class`` without a default is filled with an
    ellipsis placeholder, except for the class's collection relationships - found
    through the same DAO schema :class:`RelationalProbabilisticCircuit` uses to
    discover its exchangeable parts - which are expanded into one recursively built
    sub-query per actual child, so the grounded circuit's structure matches the
    instance being scored.

    :param domain_class: The domain class the query is built for.
    :param instance: The domain instance whose collection structure the query mirrors.
    :return: A resolved query ready to pass to :meth:`RelationalProbabilisticCircuit.ground`.
    """
    schema = get_dao_schema(type(to_dao(instance)))
    collection_relationships = {
        relationship.key: relationship for relationship in schema.collection_relationships
    }
    kwargs = {}
    for field in dataclasses.fields(domain_class):
        if field.name in collection_relationships:
            child_domain_type = collection_relationships[field.name].domain_type
            kwargs[field.name] = [
                _build_grounding_query(child_domain_type, child)
                for child in getattr(instance, field.name)
            ]
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            kwargs[field.name] = ...
    query = a(domain_class)(**kwargs)
    query.resolve()
    return query


# %% per-class model


@dataclass
class PerClassConfidenceModel:
    """A relational probabilistic circuit and familiarity threshold fitted on one class."""

    circuit: RelationalProbabilisticCircuit
    """The relational probabilistic circuit fitted on this class's training instances."""

    threshold: float
    """The log-likelihood below which an instance of this class is judged unfamiliar."""

    @classmethod
    def fit(cls, domain_class: type, instances: List[Any]) -> PerClassConfidenceModel:
        """Fit a relational probabilistic circuit and calibrate its threshold.

        The threshold is calibrated as the first percentile of the training
        instances' own log-likelihoods, so an instance less likely than almost every
        familiar one is judged unfamiliar.

        :param domain_class: The class every instance belongs to.
        :param instances: The familiar instances of ``domain_class`` to learn from.
        :return: A fitted per-class confidence model.
        """
        circuit = RelationalProbabilisticCircuit(domain_class)
        circuit.fit([to_dao(instance) for instance in instances])
        model = cls(circuit, threshold=-np.inf)
        training_log_likelihoods = [
            model.log_likelihood_of(instance) for instance in instances
        ]
        model.threshold = float(np.percentile(training_log_likelihoods, 1.0))
        return model

    def log_likelihood_of(self, instance: Any) -> float:
        """Score a single instance of this model's class under the fitted circuit.

        The instance is grounded into a concrete circuit matching its own collection
        structure; any of its features not modeled at the top level (for instance, a
        collection child's own attributes) are left unobserved and marginalized out
        rather than raising.

        :param instance: The instance to score; must belong to this model's class.
        :return: The instance's log-likelihood under the fitted circuit.
        """
        grounded = self.circuit.ground(_build_grounding_query(self.circuit.class_, instance))
        dataframe = self.circuit.feature_extractor.create_dataframe([to_dao(instance)])
        dataframe = self.circuit.feature_extractor.preprocess_dataframe(dataframe)
        variable_names = [variable.name for variable in grounded.variables]
        event = np.full((1, len(variable_names)), np.nan)
        for index, name in enumerate(variable_names):
            if name in dataframe.columns:
                event[0, index] = dataframe[name].iloc[0]
        return float(grounded.log_likelihood(event)[0])


# %% confidence model


@dataclass
class ConfidenceModel:
    """Fitted per-class relational probabilistic circuits scoring object familiarity.

    One :class:`PerClassConfidenceModel` is fitted per distinct class found among the
    training instances, each with its own calibrated threshold. Scoring an instance
    dispatches to its own class's model, so instances of different classes are never
    pooled into a shared distribution.
    """

    models_by_class: dict[type, PerClassConfidenceModel]
    """Mapping from each observed class to its fitted per-class confidence model."""

    @classmethod
    def fit_from_instances(cls, instances: List[Any]) -> ConfidenceModel:
        """Fit one confidence model per distinct class found among the instances.

        :param instances: The familiar instances the model is learned from, of any
            number of distinct classes.
        :return: A fitted confidence model ready to score new instances.
        """
        instances_by_class: dict[type, List[Any]] = {}
        for instance in instances:
            instances_by_class.setdefault(type(instance), []).append(instance)
        models_by_class = {
            domain_class: PerClassConfidenceModel.fit(domain_class, class_instances)
            for domain_class, class_instances in instances_by_class.items()
        }
        return cls(models_by_class)

    def _model_for(self, instance: Any) -> PerClassConfidenceModel:
        """Look up the per-class model for an instance's class.

        :param instance: The instance whose class model is looked up.
        :return: The fitted model for ``type(instance)``.
        :raises UnmodeledClassError: If no model was fitted for the instance's class.
        """
        domain_class = type(instance)
        if domain_class not in self.models_by_class:
            raise UnmodeledClassError(domain_class)
        return self.models_by_class[domain_class]

    @property
    def threshold(self) -> float:
        """The familiarity threshold, when exactly one class was fitted.

        :raises AmbiguousThresholdError: If more than one class was fitted, since the
            threshold then depends on which class an instance belongs to; use
            :meth:`threshold_for` instead.
        """
        if len(self.models_by_class) != 1:
            raise AmbiguousThresholdError(list(self.models_by_class))
        return next(iter(self.models_by_class.values())).threshold

    def threshold_for(self, instance: Any) -> float:
        """Return the familiarity threshold that applies to an instance's class.

        :param instance: The instance whose class's threshold is looked up.
        :return: The familiarity threshold of the instance's class.
        :raises UnmodeledClassError: If no model was fitted for the instance's class.
        """
        return self._model_for(instance).threshold

    def log_likelihood_of(self, instance: Any) -> float:
        """Return the log-likelihood of a single object under its class's model.

        :param instance: The object whose familiarity is scored.
        :return: The instance's log-likelihood under its class's fitted circuit.
        :raises UnmodeledClassError: If no model was fitted for the instance's class.
        """
        return self._model_for(instance).log_likelihood_of(instance)

    def is_familiar(self, instance: Any) -> bool:
        """Whether an object is familiar under its class's model.

        :param instance: The object to judge.
        :return: ``True`` when the object's log-likelihood is at or above its class's
            familiarity threshold, ``False`` otherwise.
        :raises UnmodeledClassError: If no model was fitted for the instance's class.
        """
        return self.log_likelihood_of(instance) >= self.threshold_for(instance)
