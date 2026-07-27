from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.parametrization.model_registries import ModelRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.learning.jpt.variables import AnnotatedVariable
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_model import ProbabilisticModel
from typing_extensions import Any, Dict, List

from experiments.confidence_aware_eql.engine.schema import FeatureSchema


@dataclass
class MomentRegistry(ModelRegistry):
    """
    A registry returning a fully factorised model with prescribed moments.

    The generative backend of the query language asks this registry for a model over the
    variables of a match statement. The registry answers with a fully factorised model
    whose continuous variables carry the mean and variance of the corresponding
    :class:`AnnotatedVariable`, so a cluster of familiar instances is described by its
    moments rather than by hand-written sampling.
    """

    annotated_variables: List[AnnotatedVariable]
    """The moments of the continuous variables of the cluster."""

    def get_model(self, parameters: UnderspecifiedParameters) -> ProbabilisticModel:
        """
        Return the fully factorised model for the requested variables.

        :param parameters: The variables the backend needs a model for. Their names are
            class-qualified, for example ``KitchenObject.weight``.
        :return: A fully factorised model over those variables.
        """
        moment_by_name = {
            annotated.variable.name: annotated for annotated in self.annotated_variables
        }
        variables = list(parameters.variables.values())
        means: Dict[Any, float] = {}
        variances: Dict[Any, float] = {}
        for variable in variables:
            annotated = moment_by_name.get(self._bare_name(variable))
            if annotated is None:
                continue
            means[variable] = annotated.mean
            variances[variable] = annotated.standard_deviation**2
        return fully_factorized(variables, means, variances)

    @staticmethod
    def _bare_name(variable: Any) -> str:
        """
        Return the feature name of a possibly class-qualified variable.

        :param variable: A variable of the match statement, whose name may carry a class
            prefix such as ``KitchenObject.weight``.
        :return: The bare feature name, for example ``weight``.
        """
        return variable.name.split(".")[-1]


@dataclass
class FamiliarCluster:
    """
    One cluster of familiar instances, generated through the query language.
    """

    match_statement: Any
    """
    The match statement describing the class and the underspecified fields.
    """

    annotated_variables: List[AnnotatedVariable]
    """
    The moments of the continuous variables of the cluster.
    """

    def sample(self, number_of_instances: int) -> List[Any]:
        """
        Generate familiar instances of this cluster with the generative backend.

        :param number_of_instances: How many instances to draw.
        :return: The generated instances, as objects of the queried class.
        """
        backend = ProbabilisticBackend(
            model_registry=MomentRegistry(self.annotated_variables),
            number_of_samples=number_of_instances,
        )
        return list(backend.evaluate(self.match_statement))


@dataclass
class TrainingDataGenerator:
    """
    Generates a familiar training matrix from a set of clusters.
    """

    schema: FeatureSchema
    """
    The schema whose variables define the columns of the training matrix.
    """

    clusters: List[FamiliarCluster]
    """
    The clusters the familiar instances are generated from.
    """

    def sample(self, instances_per_cluster: int) -> np.ndarray:
        """
        Generate and encode the familiar instances of every cluster.

        :param instances_per_cluster: How many instances to draw per cluster.
        :return: The training matrix whose columns follow the schema variables.
        """
        rows = [
            self.schema.encode(instance)
            for cluster in self.clusters
            for instance in cluster.sample(instances_per_cluster)
        ]
        return np.vstack(rows)
