"""
Fitting continuous, symbolic and integer variables with StepMix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from random_events.variable import Continuous, Integer, Symbolic
from sklearn.base import clone
from stepmix.stepmix import StepMix
from typing_extensions import Any, Iterable, List, Optional, Union

from probabilistic_model.distributions.distributions import (
    DiscreteDistribution,
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.learning.gaussian_mixture.covariance_type import (
    CovarianceType,
)
from probabilistic_model.learning.gaussian_mixture.gaussian_mixture_learning_method import (
    GaussianMixtureLearningMethod,
)
from probabilistic_model.learning.gaussian_mixture.initialization_method import (
    InitializationMethod,
)
from probabilistic_model.learning.gaussian_mixture.measurement import (
    CategoricalBlock,
    GaussianBlock,
    Measurement,
)
from probabilistic_model.learning.jpt.variables import AnnotatedVariable
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.utils import MissingDict


@dataclass
class StepMixModel(GaussianMixtureLearningMethod):
    """
    Fits continuous, symbolic and integer variables with a :class:`StepMix` mixture.

    A component is a Gaussian over the continuous variables times one categorical
    distribution per discrete variable.
    """

    model: StepMix = field(
        default_factory=lambda: StepMix(
            init_params=InitializationMethod.K_MEANS,
            n_init=10,
            verbose=0,
            progress_bar=0,
        )
    )
    """
    The mixture to fit, by default started from the best of ten k-means runs, since a
    single random start often ends in a poor local optimum. Its ``init_params`` is one
    of :class:`InitializationMethod`. Each :meth:`fit` replaces it by a copy fitted
    on the data.
    """

    covariance_type: CovarianceType = CovarianceType.FULL
    """
    The shape of the Gaussians' covariances.
    """

    covariance_regularization: float = 1e-6
    """
    The regularization added to the diagonal of the covariances.
    """

    clipped_probability: float = 1e-12
    """
    StepMix clips categorical probabilities to at least ``1e-15``; probabilities up
    to this are read as zero.
    """

    def fit(
        self,
        data: pd.DataFrame,
        variables: Optional[Iterable[AnnotatedVariable]] = None,
    ) -> ProbabilisticCircuit:
        variables = self._variables(data, variables)
        continuous = [
            variable for variable in variables if isinstance(variable, Continuous)
        ]
        discrete = [
            variable for variable in variables if not isinstance(variable, Continuous)
        ]
        columns = [self._continuous_values(data, continuous)] if continuous else []
        outcomes = []
        for variable in discrete:
            codes, keys = self._outcome_codes(variable, data[variable.name])
            columns.append(codes.reshape(-1, 1))
            outcomes.append(keys)

        measurement = Measurement(
            (
                GaussianBlock(
                    continuous, self.covariance_type, self.covariance_regularization
                )
                if continuous
                else None
            ),
            [CategoricalBlock(variable) for variable in discrete],
        )

        # StepMix keeps the outcomes it saw in a previous fit, so refit a fresh copy
        self.model = clone(self.model).set_params(measurement=measurement.to_stepmix())
        self.model.fit(np.column_stack(columns))

        parameters = self.model.get_parameters()
        fitted = parameters["measurement"]
        discrete_distributions = [
            [
                self._distribution(
                    block.variable, fitted[block.name]["pis"][component], keys
                )
                for block, keys in zip(measurement.categorical, outcomes)
            ]
            for component in range(self.model.n_components)
        ]
        gaussian = fitted[measurement.gaussian.name] if continuous else {}
        return self._circuit(
            self.covariance_type,
            continuous,
            parameters["weights"],
            gaussian.get("means"),
            gaussian.get("covariances"),
            discrete_distributions,
        )

    @staticmethod
    def _outcome_codes(
        variable: Union[Symbolic, Integer], values: pd.Series
    ) -> tuple[npt.NDArray, List[Any]]:
        """
        :return: The code ``0, 1, ...`` of every row's outcome, which StepMix reads,
            and the key of each code in a distribution of the variable.
        """
        if isinstance(variable, Symbolic):
            elements = list(variable.domain.all_elements)
            index_of = {element: index for index, element in enumerate(elements)}
            codes = np.array([index_of[value] for value in values])
            keys = [hash(element) for element in variable.domain.simple_sets]
            return codes, keys
        unique, codes = np.unique(values.to_numpy(), return_inverse=True)
        return codes, [hash(value) for value in unique]

    def _distribution(
        self,
        variable: Union[Symbolic, Integer],
        probabilities: npt.NDArray,
        keys: List[Any],
    ) -> DiscreteDistribution:
        """
        :param probabilities: One component's probability of each outcome code.
        :param keys: The key of each outcome code.
        :return: The distribution, without the outcomes StepMix only kept by clipping.
        """
        kept = probabilities > self.clipped_probability
        total = probabilities[kept].sum()
        result = MissingDict(float)
        for key, probability in zip(
            np.asarray(keys)[: len(probabilities)][kept], probabilities[kept]
        ):
            result[key] = float(probability / total)
        distribution_class = (
            SymbolicDistribution
            if isinstance(variable, Symbolic)
            else IntegerDistribution
        )
        return distribution_class(variable=variable, probabilities=result)
