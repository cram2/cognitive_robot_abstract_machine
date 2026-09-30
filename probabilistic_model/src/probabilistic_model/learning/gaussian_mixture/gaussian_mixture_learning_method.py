"""
The circuit shared by the Gaussian mixture learning methods.
"""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass

import numpy.typing as npt
import pandas as pd
from random_events.variable import Continuous, Variable
from typing_extensions import Iterable, List, Optional, Sequence

from probabilistic_model.distributions.distributions import DiscreteDistribution
from probabilistic_model.distributions.multivariate_gaussian import (
    Covariance,
    MultivariateGaussianDistribution,
)
from probabilistic_model.learning.gaussian_mixture.covariance_type import (
    CovarianceType,
)
from probabilistic_model.learning.jpt.variables import (
    AnnotatedVariable,
    infer_variables_from_dataframe,
)
from probabilistic_model.learning.learning_method import LearningMethod
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)


@dataclass
class GaussianMixtureLearningMethod(LearningMethod, ABC):
    """
    A learning method whose circuit is a sum unit over one product per component: a
    multivariate Gaussian leaf times one leaf per discrete variable.
    """

    @staticmethod
    def _variables(
        data: pd.DataFrame, variables: Optional[Iterable[AnnotatedVariable]]
    ) -> List[Variable]:
        """
        :return: The given variables, or those inferred from the data.
        """
        if variables is None:
            variables = infer_variables_from_dataframe(data)
        return [annotated_variable.variable for annotated_variable in variables]

    @staticmethod
    def _continuous_values(
        data: pd.DataFrame, continuous: Sequence[Continuous]
    ) -> npt.NDArray:
        """
        :return: The columns of the continuous variables, in their order, as floats.
        """
        return data[[variable.name for variable in continuous]].to_numpy(dtype=float)

    @staticmethod
    def _circuit(
        covariance_type: CovarianceType,
        continuous: Sequence[Continuous],
        weights: npt.NDArray,
        means: Optional[npt.NDArray],
        covariances: Optional[npt.NDArray],
        discrete_distributions: Sequence[Sequence[DiscreteDistribution]],
    ) -> ProbabilisticCircuit:
        """
        :param covariance_type: The layout of the covariances.
        :param continuous: The continuous variables, in the order of the means.
        :param weights: The weight of each component.
        :param means: The mean of each component, or ``None`` without continuous
            variables.
        :param covariances: The covariances, or ``None`` without continuous variables.
        :param discrete_distributions: For each component, the distributions of the
            discrete variables.
        :return: The mixture as a simplified circuit.
        """
        continuous = tuple(continuous)
        if continuous:
            covariances = covariance_type.full_covariances(
                covariances, len(weights), len(continuous)
            )
        circuit = ProbabilisticCircuit()
        root = SumUnit(probabilistic_circuit=circuit)
        for component, (weight, discrete) in enumerate(
            zip(weights, discrete_distributions)
        ):
            product = ProductUnit(probabilistic_circuit=circuit)
            if continuous:
                product.add_subcircuit(
                    leaf(
                        MultivariateGaussianDistribution(
                            variables=continuous,
                            mean=means[component],
                            covariance=Covariance.from_matrix(covariances[component]),
                        ),
                        circuit,
                    )
                )
            for distribution in discrete:
                product.add_subcircuit(leaf(distribution, circuit))
            root.add_subcircuit(product, math.log(weight))
        circuit.simplify()
        return circuit
