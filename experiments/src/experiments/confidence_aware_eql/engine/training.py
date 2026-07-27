from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from probabilistic_model.learning.jpt.variables import AnnotatedVariable
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from random_events.variable import Variable
from typing_extensions import List


@dataclass
class ClusterPrototype:
    """
    One cluster of familiar instances, described per variable by a mean and a spread.

    A prototype is turned into a fully factorised circuit, which is then sampled to
    obtain familiar training instances for that cluster. Using the existing fully
    factorised backend avoids hand-written sampling code.
    """

    annotated_variables: List[AnnotatedVariable]
    """The variables of the cluster together with their mean and standard deviation."""

    @property
    def circuit(self) -> ProbabilisticCircuit:
        """
        The fully factorised distribution described by this prototype.
        """
        means = {
            annotated.variable: annotated.mean for annotated in self.annotated_variables
        }
        variances = {
            annotated.variable: annotated.standard_deviation**2
            for annotated in self.annotated_variables
        }
        return fully_factorized(
            [annotated.variable for annotated in self.annotated_variables],
            means,
            variances,
        )

    def sample(self, number_of_instances: int, variables: List[Variable]) -> np.ndarray:
        """
        Draw familiar instances from this cluster.

        :param number_of_instances: How many instances to draw.
        :param variables: The variable order the returned columns must follow.
        :return: A matrix of samples whose columns follow ``variables``.
        """
        circuit = self.circuit
        samples = np.asarray(circuit.sample(number_of_instances), dtype=float)
        circuit_order = [variable.name for variable in circuit.variables]
        permutation = [circuit_order.index(variable.name) for variable in variables]
        return samples[:, permutation]


@dataclass
class TrainingDataGenerator:
    """
    Generates a familiar training set from a set of cluster prototypes.
    """

    prototypes: List[ClusterPrototype]
    """
    The clusters of familiar instances to sample from.
    """

    def sample(
        self,
        instances_per_prototype: int,
        variables: List[Variable],
        random_seed: int = 0,
    ) -> np.ndarray:
        """
        Draw an equal number of instances from every prototype.

        :param instances_per_prototype: How many instances to draw per cluster.
        :param variables: The variable order the returned columns must follow.
        :param random_seed: The seed making the generated training set reproducible.
        :return: The stacked training matrix whose columns follow ``variables``.
        """
        np.random.seed(random_seed)
        return np.vstack(
            [
                prototype.sample(instances_per_prototype, variables)
                for prototype in self.prototypes
            ]
        )
