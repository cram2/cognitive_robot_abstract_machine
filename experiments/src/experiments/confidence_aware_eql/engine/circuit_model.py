from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from random_events.variable import Variable
from sklearn.mixture import GaussianMixture
from random_events.variable import Variable
from typing_extensions import List


@dataclass
class GaussianMixtureCircuit:
    """
    A tractable probabilistic circuit compiled from a Gaussian mixture.

    A :class:`sklearn.mixture.GaussianMixture` is fitted from data and then compiled
    into a sum-product circuit of Gaussian leaves, which supports exact log-likelihood,
    marginal and conditional queries.

    .. warning::
        The mixture is fitted with a diagonal covariance, and only the diagonal is read
        when the circuit is compiled. Correlations between variables inside a component
        are therefore not represented: within one component the variables are treated as
        independent, and dependencies are only captured through the mixture weights of
        the components. A full covariance would need leaves over several variables, or a
        model such as a joint probability tree.
    """

    gaussian_mixture: GaussianMixture
    """
    The fitted mixture that defines the component means, variances and weights.
    """

    variables: List[Variable]
    """
    The variables of the model, in the order of the columns of the training data.
    """

    circuit: ProbabilisticCircuit = field(init=False)
    """
    The compiled sum-product circuit used for the queries.
    """

    _training_to_circuit_permutation: List[int] = field(init=False)
    """
    Column permutation from the training order to the variable order of the circuit.
    """

    def __post_init__(self) -> None:
        self.circuit = self._compile_circuit()
        training_order = [variable.name for variable in self.variables]
        self._training_to_circuit_permutation = [
            training_order.index(variable.name) for variable in self.circuit.variables
        ]

    @property
    def variable_names(self) -> List[str]:
        """
        The variable names in training-column order.
        """
        return [variable.name for variable in self.variables]

    @property
    def number_of_components(self) -> int:
        """
        :return: The number of components of the underlying Gaussian mixture.
        """
        return self.gaussian_mixture.n_components

    def log_likelihood(self, rows: np.ndarray) -> np.ndarray:
        """
        Compute the log-likelihood of complete instances.

        :param rows: A matrix of instances whose columns follow :attr:`variables`.
        :return: The log-likelihood of every row.
        """
        rows = np.asarray(rows, dtype=float)
        if rows.ndim == 1:
            rows = rows[np.newaxis, :]
        return self.circuit.log_likelihood(
            rows[:, self._training_to_circuit_permutation]
        )

    def marginal(self, variables: List[Variable]) -> ProbabilisticCircuit:
        """
        Marginalise the circuit onto a subset of its variables.

        :param variables: The variables to keep.
        :return: The circuit over the given variables only.
        """
        kept_names = {variable.name for variable in variables}
        kept = [
            variable
            for variable in self.circuit.variables
            if variable.name in kept_names
        ]
        return self.circuit.marginal(kept)

    def _compile_circuit(self) -> ProbabilisticCircuit:
        """
        Build the sum-product circuit from the parameters of the fitted mixture.

        :return: A circuit whose root is a sum over one product per mixture component.
        """
        circuit = ProbabilisticCircuit()
        mixture_node = SumUnit(probabilistic_circuit=circuit)
        for component_index in range(self.number_of_components):
            product_node = ProductUnit(probabilistic_circuit=circuit)
            for variable_index, variable in enumerate(self.variables):
                mean = float(
                    self.gaussian_mixture.means_[component_index, variable_index]
                )
                standard_deviation = float(
                    np.sqrt(
                        self.gaussian_mixture.covariances_[
                            component_index, variable_index
                        ]
                    )
                )
                gaussian = GaussianDistribution(
                    location=mean, scale=standard_deviation, variable=variable
                )
                product_node.add_subcircuit(leaf(gaussian, circuit))
            component_weight = float(self.gaussian_mixture.weights_[component_index])
            mixture_node.add_subcircuit(product_node, np.log(component_weight))
        return circuit


def fit_gaussian_mixture_circuit(
    data: np.ndarray,
    variables: List[Variable],
    number_of_components: int = 0,
    maximum_components: int = 8,
    random_seed: int = 0,
    covariance_regularisation: float = 1e-4,
) -> GaussianMixtureCircuit:
    """
    Fit a Gaussian mixture to data and compile it into a tractable circuit.

    :param data: The training matrix whose columns follow ``variables``.
    :param variables: The variables of the model, in the order of the columns.
    :param number_of_components: The number of mixture components, or ``0`` to select
        the number automatically by the Bayesian information criterion.
    :param maximum_components: The largest number of components tried during automatic
        selection.
    :param random_seed: The seed used for the mixture fitting.
    :param covariance_regularisation: The value added to the diagonal of every
        covariance to keep nearly constant variables well conditioned.
    :return: The compiled circuit together with the mixture it was compiled from.
    """
    data = np.asarray(data, dtype=float)
    if number_of_components == 0:
        number_of_components = _select_component_count(
            data, maximum_components, random_seed, covariance_regularisation
        )
    gaussian_mixture = GaussianMixture(
        n_components=number_of_components,
        covariance_type="diag",
        random_state=random_seed,
        reg_covar=covariance_regularisation,
    ).fit(data)
    return GaussianMixtureCircuit(gaussian_mixture, list(variables))


def _select_component_count(
    data: np.ndarray,
    maximum_components: int,
    random_seed: int,
    covariance_regularisation: float,
) -> int:
    """
    Choose the number of components that minimises the Bayesian information criterion.

    :param data: The training matrix the mixtures are fitted to.
    :param maximum_components: The largest number of components tried.
    :param random_seed: The seed used for every fit.
    :param covariance_regularisation: The value added to the diagonal of every
        covariance.
    :return: The number of components with the lowest criterion value.
    """
    best_component_count = 1
    best_criterion = np.inf
    for candidate in range(1, min(maximum_components, len(data)) + 1):
        mixture = GaussianMixture(
            n_components=candidate,
            covariance_type="diag",
            random_state=random_seed,
            reg_covar=covariance_regularisation,
        ).fit(data)
        criterion = mixture.bic(data)
        if criterion < best_criterion:
            best_criterion = criterion
            best_component_count = candidate
    return best_component_count
