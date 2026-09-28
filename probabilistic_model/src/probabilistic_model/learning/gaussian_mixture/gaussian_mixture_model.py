"""
Fitting continuous variables with scikit-learn's Gaussian mixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from random_events.variable import Continuous
from sklearn.mixture import GaussianMixture
from typing_extensions import Iterable, Optional, Sequence

from probabilistic_model.exceptions import NonContinuousVariableError
from probabilistic_model.learning.gaussian_mixture.covariance_type import (
    CovarianceType,
)
from probabilistic_model.learning.gaussian_mixture.gaussian_mixture_learning_method import (
    GaussianMixtureLearningMethod,
)
from probabilistic_model.learning.jpt.variables import AnnotatedVariable
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)


@dataclass
class GaussianMixtureModel(GaussianMixtureLearningMethod):
    """
    Fits continuous variables with a scikit-learn :class:`GaussianMixture`.
    """

    model: GaussianMixture = field(default_factory=GaussianMixture)
    """
    The mixture to fit.
    """

    def fit(
        self,
        data: pd.DataFrame,
        variables: Optional[Iterable[AnnotatedVariable]] = None,
    ) -> ProbabilisticCircuit:
        """
        :raises NonContinuousVariableError: If a variable is not continuous.
        """
        variables = self._variables(data, variables)
        non_continuous = [
            variable for variable in variables if not isinstance(variable, Continuous)
        ]
        if non_continuous:
            raise NonContinuousVariableError(non_continuous)
        self.model.fit(self._continuous_values(data, variables))
        return self.to_probabilistic_circuit(variables)

    def to_probabilistic_circuit(
        self, variables: Sequence[Continuous]
    ) -> ProbabilisticCircuit:
        """
        Convert the fitted mixture, also one fitted outside of :meth:`fit`.

        :param variables: The variables, in the order of the columns it was fitted on.
        :return: The mixture as a circuit.
        """
        return self._circuit(
            CovarianceType(self.model.covariance_type),
            variables,
            self.model.weights_,
            self.model.means_,
            self.model.covariances_,
            [[] for _ in self.model.weights_],
        )
