"""
Expectation maximization that learns one column of a progressive probabilistic circuit
at a time.
"""

from __future__ import annotations

# %% imports
import itertools
from dataclasses import dataclass

import numpy as np
from random_events.variable import Variable

from probabilistic_model.distributions.distributions import DiscreteDistribution
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.learning.progressive.exceptions import (
    UnsupportedLeafDistributionError,
)
from probabilistic_model.learning.progressive.progressive_circuit import (
    CircuitColumn,
    ProgressiveProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    InnerUnit,
    LeafUnit,
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


# %% learning data structures
@dataclass(frozen=True)
class LearnableUnits:
    """
    The units whose parameters learning a column updates.
    """

    sum_units: frozenset[SumUnit]
    """
    Sum units whose weights are updated.
    """

    leaf_units: frozenset[LeafUnit]
    """
    Leaf units whose distributions are updated.
    """


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
class ProgressiveExpectationMaximization:
    """
    Expectation maximization for one column of a progressive probabilistic circuit.

    While the column is learned, the root gives it the whole weight, so it is trained as
    the only model of its task; it still reaches earlier columns through its edges to
    them. Only the units of the column are updated. Afterwards the root weights every
    column by its share of all rows the columns were learned from.

    .. warning::

        Later columns read from earlier ones, so learning an earlier column again also
        changes the later columns.
    """

    progressive_circuit: ProgressiveProbabilisticCircuit
    """
    The circuit whose columns are learned.
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
        self, data: SampleArray, column: CircuitColumn, iterations: int = 1
    ) -> list[float]:
        """
        Learn the parameters of a column.

        :param data: One row per sample, columns ordered like the circuit's variables.
        :param column: The column to learn.
        :param iterations: Number of expectation maximization iterations.
        :return: The average log-likelihood of the rows under the column before every
            iteration.
        :raises UnregisteredColumnError: If the column belongs to another progressive
            circuit.
        :raises UnsupportedLeafDistributionError: If the column has a leaf that cannot
            be learned.
        """
        self.progressive_circuit.validate_column(column)
        data = np.asarray(data)
        if len(data) == 0:
            return []

        learnable_units = self.learnable_units(column)
        self.progressive_circuit.restrict_root_to(column)
        history = []
        for _ in range(iterations):
            expectation = self._expectation_step(data)
            history.append(expectation.average_log_likelihood)
            self._maximization_step(learnable_units, expectation, data)
        column.sample_count = len(data)
        self.progressive_circuit.weight_root_by_sample_count()
        return history

    def learnable_units(self, column: CircuitColumn) -> LearnableUnits:
        """
        Collect the units whose parameters learning a column updates.

        :param column: The column to learn.
        :return: The sum and leaf units of the column.
        :raises UnsupportedLeafDistributionError: If the column has a leaf that cannot
            be learned.
        """
        units = self.progressive_circuit.units_of(column)
        leaf_units = frozenset(unit for unit in units if isinstance(unit, LeafUnit))
        for leaf_unit in leaf_units:
            if not isinstance(
                leaf_unit.distribution, (GaussianDistribution, DiscreteDistribution)
            ):
                raise UnsupportedLeafDistributionError(leaf_unit.distribution)
        sum_units = frozenset(unit for unit in units if isinstance(unit, SumUnit))
        return LearnableUnits(sum_units=sum_units, leaf_units=leaf_units)

    def _expectation_step(self, data: SampleArray) -> ExpectationStepResult:
        """
        Evaluate the whole circuit and pass the responsibilities from the root down to
        every reached unit.

        :param data: One row per sample, columns ordered like the circuit's variables.
        :return: The average log-likelihood and the responsibilities of every unit.
        """
        circuit = self.progressive_circuit.circuit
        average_log_likelihood = float(np.mean(circuit.log_likelihood(data)))
        log_responsibilities = np.full(
            (len(data), max(circuit.graph.node_indices()) + 1), -np.inf
        )
        log_responsibilities[:, self.progressive_circuit.root.index] = 0.0

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
        learnable_units: LearnableUnits,
        expectation: ExpectationStepResult,
        data: SampleArray,
    ) -> None:
        """
        Update the weights and leaf distributions of the learnable units.

        :param learnable_units: The units to update.
        :param expectation: The result of the preceding expectation step.
        :param data: The rows the expectation step evaluated.
        """
        circuit = self.progressive_circuit.circuit
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
