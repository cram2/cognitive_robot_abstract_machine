from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.mixture import GaussianMixture
from typing_extensions import List

from random_events.variable import Continuous
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    ProductUnit,
    leaf,
)


@dataclass
class GaussianMixtureCircuit:
    """A tractable probabilistic circuit compiled from a Gaussian mixture.

    A real :class:`sklearn.mixture.GaussianMixture` is fitted from data and then
    compiled into a sum-product circuit of Gaussian leaves. The circuit supports
    exact and efficient log-likelihood and marginal queries, which is what the
    confidence check relies on.

    .. note::
        The circuit orders its variables alphabetically, so queries permute
        incoming rows from domain order into circuit order before evaluation.
    """

    gaussian_mixture: GaussianMixture
    """The fitted mixture that defines the component means, variances, and weights."""

    feature_names: List[str]
    """Feature names in domain order, matching the columns of the training data."""

    circuit: ProbabilisticCircuit = field(init=False)
    """The compiled sum-product circuit used for likelihood queries."""

    _domain_to_circuit_permutation: List[int] = field(init=False)
    """Column permutation mapping domain order to circuit variable order."""

    def __post_init__(self) -> None:
        self.circuit = self._compile_circuit()
        circuit_order = [variable.name for variable in self.circuit.variables]
        self._domain_to_circuit_permutation = [
            self.feature_names.index(name) for name in circuit_order
        ]

    @property
    def number_of_components(self) -> int:
        """The number of mixture components in the underlying Gaussian mixture."""
        return self.gaussian_mixture.n_components

    def log_likelihood(self, rows: np.ndarray) -> np.ndarray:
        """Return the log-likelihood of each row under the circuit.

        :param rows: A matrix of instances in domain feature order.
        """
        rows = np.asarray(rows, dtype=float)
        if rows.ndim == 1:
            rows = rows[np.newaxis, :]
        return self.circuit.log_likelihood(rows[:, self._domain_to_circuit_permutation])

    def marginal(self, feature_names: List[str]) -> ProbabilisticCircuit:
        """Return the circuit marginalised onto the given features."""
        variables = [
            variable
            for variable in self.circuit.variables
            if variable.name in feature_names
        ]
        return self.circuit.marginal(variables)

    def _compile_circuit(self) -> ProbabilisticCircuit:
        """Build the sum-product circuit from the fitted mixture parameters."""
        variables = [Continuous(name) for name in self.feature_names]
        circuit = ProbabilisticCircuit()
        mixture_node = SumUnit(probabilistic_circuit=circuit)
        for component_index in range(self.number_of_components):
            product_node = ProductUnit(probabilistic_circuit=circuit)
            for feature_index, variable in enumerate(variables):
                mean = float(self.gaussian_mixture.means_[component_index, feature_index])
                standard_deviation = float(
                    np.sqrt(self.gaussian_mixture.covariances_[component_index, feature_index])
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
    feature_names: List[str],
    number_of_components: int = 0,
    maximum_components: int = 8,
    random_seed: int = 0,
    covariance_regularisation: float = 1e-4,
) -> GaussianMixtureCircuit:
    """Fit a Gaussian mixture to data and compile it into a tractable circuit.

    :param data: Training matrix in domain feature order.
    :param feature_names: Feature names in domain order.
    :param number_of_components: Number of mixture components, or ``0`` to select
        the count automatically by the Bayesian information criterion.
    :param maximum_components: Upper bound tried during automatic selection.
    :param random_seed: Seed for reproducible mixture fitting.
    :param covariance_regularisation: Value added to the diagonal of each
        covariance to keep near-constant features well conditioned.
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
    return GaussianMixtureCircuit(gaussian_mixture, list(feature_names))


def _select_component_count(
    data: np.ndarray,
    maximum_components: int,
    random_seed: int,
    covariance_regularisation: float,
) -> int:
    """Choose the component count minimising the Bayesian information criterion."""
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
