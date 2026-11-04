"""
Expectation maximization that learns the parameters of units of an existing circuit, and
the learning method that fits a circuit on a dataframe with it.
"""

from __future__ import annotations

# %% imports
import copy
import itertools
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from random_events.variable import Variable
from typing_extensions import TYPE_CHECKING

from probabilistic_model.distributions.distributions import DiscreteDistribution
from probabilistic_model.distributions.gaussian import (
    GaussianDistribution,
    TruncatedGaussianDistribution,
)
from probabilistic_model.learning.exceptions import (
    StartingCircuitVariablesMismatchError,
    UnsupportedLeafDistributionError,
)
from probabilistic_model.learning.learning_method import LearningMethod
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    InnerUnit,
    LeafUnit,
    ProbabilisticCircuit,
    SumUnit,
    Unit,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    SampleArray,
    SampleColumn,
    SampleNodeValues,
    SampleValues,
)
from probabilistic_model.utils import logsumexp

if TYPE_CHECKING:
    from probabilistic_model.learning.jpt.variables import AnnotatedVariable
    from probabilistic_model.probabilistic_model import ProbabilisticModel


# %% learning data structures
@dataclass(frozen=True)
class LearnableUnits:
    """
    The units whose parameters expectation maximization updates.
    """

    sum_units: frozenset[SumUnit]
    """
    Sum units whose weights are updated.
    """

    leaf_units: frozenset[LeafUnit]
    """
    Leaf units whose distributions are updated.
    """

    @classmethod
    def from_units(cls, units: Collection[Unit]) -> LearnableUnits:
        """
        Select the sum and leaf units among some units; product units have no
        parameters.

        :param units: The units to learn.
        :return: The sum and leaf units among them.
        :raises UnsupportedLeafDistributionError: If a leaf has a distribution that
            cannot be learned.
        """
        leaf_units = frozenset(unit for unit in units if isinstance(unit, LeafUnit))
        for leaf_unit in leaf_units:
            if not cls._can_fit(leaf_unit.distribution):
                raise UnsupportedLeafDistributionError(leaf_unit.distribution)
        sum_units = frozenset(unit for unit in units if isinstance(unit, SumUnit))
        return cls(sum_units=sum_units, leaf_units=leaf_units)

    @staticmethod
    def _can_fit(distribution: ProbabilisticModel) -> bool:
        """
        Check whether the maximization step can fit a leaf distribution to weighted
        rows.

        A truncated Gaussian cannot be fitted: its location and scale describe the
        Gaussian before truncation, not the mean and spread of its rows.

        :param distribution: The distribution of a leaf.
        :return: Whether the distribution is a Gaussian or a discrete distribution.
        """
        if isinstance(distribution, TruncatedGaussianDistribution):
            return False
        return isinstance(distribution, (GaussianDistribution, DiscreteDistribution))


@dataclass
class ExpectationStepResult:
    """
    The posterior statistics of one expectation step.
    """

    average_log_likelihood: float
    """
    Average log-likelihood of the rows under the whole circuit.
    """

    log_responsibilities: SampleNodeValues
    """
    Log-responsibility of every unit for every row, one column per unit index; ``-inf``
    for units the rows do not reach.
    """


# %% expectation maximization
@dataclass
class ExpectationMaximization:
    """
    Expectation maximization for the parameters of some units of an existing circuit.

    The expectation step evaluates the whole circuit; the maximization step updates the
    weights of the learned sum units and the distributions of the learned leaves, while
    every other unit keeps its parameters.
    """

    smoothing: float = 1e-8
    """
    Added to the expected count of every edge below a sum unit.
    """

    minimum_mixture_proportion: float = 1e-12
    """
    Lower bound for a learned sum unit weight before taking its logarithm.
    """

    minimum_gaussian_variance: float = 1e-6
    """
    Lower bound for the variance of a learned Gaussian leaf.
    """

    def learn(
        self,
        circuit: ProbabilisticCircuit,
        data: SampleArray,
        learnable_units: LearnableUnits,
        iterations: int = 1,
    ) -> list[float]:
        """
        Alternate expectation and maximization steps on a circuit, in place.

        :param circuit: The circuit whose parameters are learned.
        :param data: One row per sample, columns ordered like the circuit's variables.
        :param learnable_units: The units of the circuit to update.
        :param iterations: Number of expectation maximization iterations.
        :return: The average log-likelihood of the rows under the circuit before every
            iteration; empty if there are no rows.
        """
        data = np.asarray(data)
        if len(data) == 0:
            return []
        history = []
        for _ in range(iterations):
            expectation = self._expectation_step(circuit, data)
            history.append(expectation.average_log_likelihood)
            self._maximization_step(circuit, learnable_units, expectation, data)
        return history

    def _expectation_step(
        self, circuit: ProbabilisticCircuit, data: SampleArray
    ) -> ExpectationStepResult:
        """
        Evaluate the whole circuit and pass the responsibilities from the root down to
        every reached unit.

        :param circuit: The circuit to evaluate.
        :param data: One row per sample, columns ordered like the circuit's variables.
        :return: The average log-likelihood and the responsibilities of every unit.
        """
        average_log_likelihood = float(np.mean(circuit.log_likelihood(data)))
        log_responsibilities = np.full(
            (len(data), max(circuit.graph.node_indices()) + 1), -np.inf
        )
        log_responsibilities[:, circuit.root.index] = 0.0

        for unit in itertools.chain.from_iterable(circuit.layers):
            self._add_responsibility_to_children(unit, log_responsibilities)

        return ExpectationStepResult(average_log_likelihood, log_responsibilities)

    def _add_responsibility_to_children(
        self, unit: Unit, log_responsibilities: SampleNodeValues
    ) -> None:
        """
        Add the share of a unit's responsibility for every row to each of its children.

        A sum unit divides its responsibility among its children by how well each
        explains the row; a product unit gives every child its whole responsibility.

        :param unit: The unit whose responsibility is complete.
        :param log_responsibilities: Log-responsibility of every unit for every row;
            updated in place.
        """
        if not isinstance(unit, InnerUnit) or np.all(
            np.isneginf(log_responsibilities[:, unit.index])
        ):
            return
        if isinstance(unit, SumUnit):
            for log_weight, child in unit.log_weighted_subcircuits:
                self._accumulate(
                    log_responsibilities,
                    child,
                    self._edge_log_responsibility(
                        log_responsibilities, unit, log_weight, child
                    ),
                )
            return
        for child in unit.subcircuits:
            self._accumulate(
                log_responsibilities, child, log_responsibilities[:, unit.index]
            )

    @staticmethod
    def _edge_log_responsibility(
        log_responsibilities: SampleNodeValues,
        sum_unit: SumUnit,
        log_weight: float,
        child: Unit,
    ) -> SampleValues:
        """
        Compute how much of a sum unit's responsibility for every row passes through one
        of its edges.

        Reads the likelihoods the last evaluation of the circuit left on the units.

        :param log_responsibilities: Log-responsibility of every unit for every row.
        :param sum_unit: The parent of the edge.
        :param log_weight: Log-weight of the edge.
        :param child: The child of the edge.
        :return: Log-responsibility of the child through the edge for every row;
            ``-inf`` for rows that are impossible under the sum unit.
        """
        log_responsibility = (
            log_responsibilities[:, sum_unit.index]
            + log_weight
            + np.asarray(child.result_of_current_query)
            - np.asarray(sum_unit.result_of_current_query)
        )
        return np.where(np.isnan(log_responsibility), -np.inf, log_responsibility)

    @staticmethod
    def _accumulate(
        log_responsibilities: SampleNodeValues,
        unit: Unit,
        log_responsibility: SampleValues,
    ) -> None:
        """
        Add a per-row log-responsibility to those already collected for a unit.

        :param log_responsibilities: Log-responsibility of every unit for every row;
            updated in place.
        :param unit: The unit that receives the responsibility.
        :param log_responsibility: Log-responsibility to add for every row.
        """
        log_responsibilities[:, unit.index] = np.logaddexp(
            log_responsibilities[:, unit.index], log_responsibility
        )

    def _maximization_step(
        self,
        circuit: ProbabilisticCircuit,
        learnable_units: LearnableUnits,
        expectation: ExpectationStepResult,
        data: SampleArray,
    ) -> None:
        """
        Update the weights and leaf distributions of the learnable units.

        :param circuit: The circuit the learnable units belong to.
        :param learnable_units: The units to update.
        :param expectation: The result of the preceding expectation step.
        :param data: The rows the expectation step evaluated.
        """
        for sum_unit in learnable_units.sum_units:
            log_weighted_children = sum_unit.log_weighted_subcircuits
            counts = np.array(
                [
                    np.exp(
                        logsumexp(
                            self._edge_log_responsibility(
                                expectation.log_responsibilities,
                                sum_unit,
                                log_weight,
                                child,
                            )
                        )
                    )
                    + self.smoothing
                    for log_weight, child in log_weighted_children
                ]
            )
            if float(np.sum(counts)) <= 0.0:
                continue
            for (_, child), log_weight in zip(
                log_weighted_children, self._normalized_log_weights(counts)
            ):
                circuit.add_edge(sum_unit, child, log_weight=float(log_weight))
            sum_unit.normalize()

        variable_to_index_map = circuit.variable_to_index_map
        for leaf_unit in learnable_units.leaf_units:
            weights = np.exp(expectation.log_responsibilities[:, leaf_unit.index])
            if float(np.sum(weights)) <= 0.0:
                continue
            self._update_leaf_distribution(
                leaf_unit, data, weights, variable_to_index_map
            )

    def _normalized_log_weights(self, counts: np.ndarray) -> np.ndarray:
        """
        Turn the expected counts of a sum unit's children into log-weights.

        :param counts: Smoothed expected count of every child.
        :return: The log of every count's share, floored at
            :attr:`minimum_mixture_proportion`.
        """
        proportions = counts / np.sum(counts)
        return np.log(np.maximum(proportions, self.minimum_mixture_proportion))

    def _update_leaf_distribution(
        self,
        leaf_unit: LeafUnit,
        data: SampleArray,
        weights: SampleValues,
        variable_to_index_map: dict[Variable, int],
    ) -> None:
        """
        Fit the distribution of a leaf to the weighted rows, in place.

        :param leaf_unit: The leaf whose distribution is updated.
        :param data: One row per sample, columns ordered like the circuit's variables.
        :param weights: Non-negative responsibility of every row, with a positive total.
        :param variable_to_index_map: Column of every variable in ``data``.
        """
        distribution = leaf_unit.distribution
        values = data[:, variable_to_index_map[distribution.variable]]

        if isinstance(distribution, GaussianDistribution):
            self._fit_gaussian(distribution, values, weights)
            return
        self._fit_discrete(distribution, values, weights)

    def _fit_gaussian(
        self,
        distribution: GaussianDistribution,
        values: SampleColumn,
        weights: SampleValues,
    ) -> None:
        """
        Set a Gaussian to the weighted mean and standard deviation of the values, its
        variance floored at :attr:`minimum_gaussian_variance`.

        :param distribution: The Gaussian to update in place.
        :param values: The value of its variable in every row.
        :param weights: Non-negative weight of every row, with a positive total.
        """
        total_weight = np.sum(weights)
        mean = np.sum(weights * values) / total_weight
        variance = np.sum(weights * (values - mean) ** 2) / total_weight
        distribution.location = float(mean)
        distribution.scale = float(
            np.sqrt(max(variance, self.minimum_gaussian_variance))
        )

    @staticmethod
    def _fit_discrete(
        distribution: DiscreteDistribution,
        values: SampleColumn,
        weights: SampleValues,
    ) -> None:
        """
        Set every probability of a discrete distribution to the weighted share of the
        rows with that value.

        :param distribution: The distribution to update in place.
        :param values: The value of its variable in every row.
        :param weights: Non-negative weight of every row, with a positive total.
        """
        value_hashes = np.array([hash(value) for value in values])
        probabilities = distribution.probabilities
        for key in list(probabilities.keys()):
            probabilities[key] = float(np.sum(weights[value_hashes == key]))
        distribution.normalize()


# %% learning method
@dataclass
class ExpectationMaximizationLearning(LearningMethod):
    """
    Fits a copy of a starting circuit on a dataframe by learning all of its parameters
    with expectation maximization; the structure stays that of the starting circuit.
    """

    starting_circuit: ProbabilisticCircuit
    """
    The circuit whose structure and initial parameters every fit starts from; it stays
    unchanged.
    """

    iterations: int
    """
    Number of expectation maximization iterations of every fit.
    """

    expectation_maximization: ExpectationMaximization = field(
        default_factory=ExpectationMaximization
    )
    """
    The expectation maximization the parameters are learned with.
    """

    def fit(
        self,
        data: pd.DataFrame,
        variables: Iterable[AnnotatedVariable] | None = None,
    ) -> ProbabilisticCircuit:
        """
        Like :meth:`LearningMethod.fit`, but given variables must be those of the
        starting circuit.

        :raises StartingCircuitVariablesMismatchError: If the variables differ from
            those of the starting circuit.
        :raises UnsupportedLeafDistributionError: If the starting circuit has a leaf
            that cannot be learned.
        """
        circuit = copy.deepcopy(self.starting_circuit)
        if variables is not None:
            given_variables = {
                annotated_variable.variable for annotated_variable in variables
            }
            if given_variables != set(circuit.variables):
                raise StartingCircuitVariablesMismatchError(
                    circuit_variables=frozenset(circuit.variables),
                    given_variables=frozenset(given_variables),
                )
        self.expectation_maximization.learn(
            circuit,
            data[[variable.name for variable in circuit.variables]].to_numpy(),
            LearnableUnits.from_units(circuit.nodes()),
            self.iterations,
        )
        return circuit
