"""
The tensorized input layer whose leaves are binomial distributions.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from random_events.interval import Interval
from random_events.product_algebra import SimpleEvent
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Variable
from scipy.special import gammaln
from scipy.stats import binom
from sortedcontainers import SortedSet
from typing_extensions import Any, Self

from probabilistic_model.distributions.binomial import BinomialDistribution
from probabilistic_model.exceptions import (
    ShapeMismatchError,
    SuccessProbabilityOutsideUnitIntervalError,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
    NodeIndices,
    NodeIntegers,
    NodeMask,
    NodeStateValues,
    NodeValues,
    SampleArray,
    SampleColumn,
    SampleNodeValues,
    VariableValues,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.base import (
    UnivariateInputLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.discrete_layer import (
    IntegerLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
)
from probabilistic_model.probabilistic_circuit.tensorized.query_cache import (
    QueryCache,
    memoized,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
)

# %% binomial layer


@dataclass(eq=False, repr=False, kw_only=True)
class BinomialLayer(UnivariateInputLayer):
    """
    A layer of binomial distributions over one integer variable, see
    :class:`~probabilistic_model.distributions.binomial.BinomialDistribution`.

    The success probabilities are its parameters; the numbers of trials and the offsets
    are structure. Truncated or conditioned, it becomes an :class:`IntegerLayer`.
    """

    number_of_trials: NodeIntegers
    """
    How many trials every node counts.
    """

    success_probability: NodeValues
    """
    The probability of each trial of every node to succeed.
    """

    offset: NodeIntegers
    """
    The value of no success of every node.
    """

    @property
    def number_of_nodes(self) -> int:
        return len(self.number_of_trials)

    @property
    def number_of_own_parameters(self) -> int:
        return self.number_of_nodes

    @property
    def parameters(self) -> list[BackendArray]:
        return [self.success_probability]

    def copy_with_parameters(self, parameters: list[BackendArray]) -> Self:
        (success_probability,) = parameters
        return dataclasses.replace(self, success_probability=success_probability)

    def projected_parameters(
        self, parameters: list[BackendArray]
    ) -> list[BackendArray]:
        (success_probability,) = parameters
        return [self.namespace.clip(success_probability, 0.0, 1.0)]

    def validate_own(self):
        """
        :raises ShapeMismatchError: If the nodes do not each have a number of trials, a
            success probability and an offset.
        :raises SuccessProbabilityOutsideUnitIntervalError: If a success probability is
            not within [0, 1].
        """
        for other in (self.success_probability, self.offset):
            if self.number_of_trials.shape != other.shape:
                raise ShapeMismatchError(self.number_of_trials.shape, other.shape)
        success_probability = self.host_copy(self.success_probability)
        # the negated comparison also catches a success probability that is nan
        outside = ~((success_probability >= 0) & (success_probability <= 1))
        if outside.any():
            raise SuccessProbabilityOutsideUnitIntervalError(
                success_probabilities=success_probability[outside].tolist()
            )

    # %% probabilities of the numbers of successes

    @property
    def successes(self) -> NodeIntegers:
        """
        :return: Every number of successes up to the largest number of trials of any
            node, shape (#numbers,).
        """
        return np.arange(int(self.number_of_trials.max(initial=0)) + 1)

    def log_coefficients_of_successes(self) -> NodeStateValues:
        """
        :return: The logarithm of the binomial coefficient of every number of
            :attr:`successes` for every node, ``-inf`` above its number of trials.
        """
        trials = self.number_of_trials[:, None]
        successes = self.successes[None, :]
        possible = successes <= trials
        return self.backend_array(
            values=np.where(
                possible,
                gammaln(trials + 1)
                - gammaln(successes + 1)
                - gammaln(np.where(possible, trials - successes, 0) + 1),
                -np.inf,
            )
        )

    def log_probabilities_of_successes(self) -> NodeStateValues:
        """
        :return: The log-probability of every number of :attr:`successes` under every
            node, ``-inf`` above its number of trials, shape (#nodes, #numbers).
        """
        successes = self.successes[None, :]
        return self.log_coefficients_of_successes() + self.log_success_terms(
            successes=self.backend_array(values=successes),
            failures=self.backend_array(
                values=np.maximum(self.number_of_trials[:, None] - successes, 0)
            ),
            success_probability=self.success_probability[:, None],
        )

    def log_success_terms(
        self,
        *,
        successes: BackendArray,
        failures: BackendArray,
        success_probability: BackendArray,
    ) -> BackendArray:
        """
        :param successes: Numbers of successes.
        :param failures: The numbers of failures that go with them.
        :param success_probability: The success probabilities of the nodes, shaped to
            broadcast against the numbers.
        :return: The log-probability of these successes and failures in one order,
            zero for none of either even where its probability is zero.
        """
        namespace = self.namespace
        probability = success_probability
        succeeding = probability > 0
        failing = probability < 1
        log_success = namespace.log(namespace.where(succeeding, probability, 1.0))
        log_failure = namespace.log(namespace.where(failing, 1 - probability, 1.0))
        return namespace.where(
            successes > 0,
            namespace.where(succeeding, successes * log_success, -namespace.inf),
            0.0,
        ) + namespace.where(
            failures > 0,
            namespace.where(failing, failures * log_failure, -namespace.inf),
            0.0,
        )

    def backend_array(self, *, values: np.ndarray) -> BackendArray:
        """
        :param values: Numbers on the host.
        :return: The numbers as floats in an array of the backend of the parameters.
        """
        return self.namespace.asarray(
            values, dtype=self.floating_point_type, device=self.device
        )

    def successes_of_column(self, values: SampleColumn) -> SampleNodeValues:
        """
        :param values: Values of the variable of this layer, shape (#events,).
        :return: The number of successes every node needs for every value, shape
            (#events, #nodes).
        """
        return self.float_column(values=values) - self.backend_array(values=self.offset)

    def entries_of_nodes(
        self, *, table: NodeStateValues, successes: SampleNodeValues
    ) -> SampleNodeValues:
        """
        :param table: One value per node and number of :attr:`successes`.
        :param successes: Whole numbers of successes per event and node, clipped to
            the :attr:`successes`.
        :return: The value of the table at the node and the number of successes of
            every entry, shape (#events, #nodes).
        """
        namespace = self.namespace
        number_of_columns = table.shape[1]
        flat_indices = (
            namespace.astype(
                successes,
                namespace.__array_namespace_info__().default_dtypes(device=self.device)[
                    "indexing"
                ],
            )
            + namespace.arange(self.number_of_nodes, device=self.device)[None, :]
            * number_of_columns
        )
        return namespace.reshape(
            namespace.take(
                namespace.reshape(table, (-1,)), namespace.reshape(flat_indices, (-1,))
            ),
            successes.shape,
        )

    # %% numeric queries

    @memoized
    def log_likelihood_of_nodes(
        self, events: SampleArray, cache: QueryCache | None = None
    ) -> SampleNodeValues:
        namespace = self.namespace
        successes = self.successes_of_column(self.column_of(events))
        trials = self.backend_array(values=self.number_of_trials)[None, :]
        possible = (
            (successes >= 0)
            & (successes <= trials)
            & (successes == namespace.floor(successes))
        )
        safe_successes = namespace.where(possible, successes, 0.0)
        log_coefficients = self.entries_of_nodes(
            table=self.log_coefficients_of_successes(),
            successes=safe_successes,
        )
        log_likelihoods = log_coefficients + self.log_success_terms(
            successes=safe_successes,
            failures=trials - safe_successes,
            success_probability=self.success_probability[None, :],
        )
        return namespace.where(possible, log_likelihoods, -namespace.inf)

    @memoized
    def cumulative_distribution_of_nodes(
        self, events: SampleArray, cache: QueryCache | None = None
    ) -> SampleNodeValues:
        namespace = self.namespace
        successes = namespace.floor(self.successes_of_column(self.column_of(events)))
        cumulative = namespace.cumulative_sum(
            namespace.exp(self.log_probabilities_of_successes()), axis=1
        )
        reached = self.entries_of_nodes(
            table=cumulative,
            successes=namespace.clip(successes, 0.0, float(len(self.successes) - 1)),
        )
        return namespace.where(successes < 0, 0.0, reached)

    @memoized
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: QueryCache | None = None,
    ) -> NodeValues:
        interval: Interval = event[variables[self.variable]]
        values = self.offset[:, None] + self.successes[None, :]
        contained = [value for value in np.unique(values).tolist() if value in interval]
        namespace = self.namespace
        return namespace.sum(
            namespace.where(
                namespace.asarray(np.isin(values, contained), device=self.device),
                namespace.exp(self.log_probabilities_of_successes()),
                0.0,
            ),
            axis=1,
        )

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> NodeValues:
        """
        Like :meth:`~probabilistic_model.probabilistic_circuit.tensorized.input_layer.ba
        se.UnivariateInputLayer.moment_of_nodes_own`, as the expectation of the centered
        values under the probabilities of the successes of every node.
        """
        values = self.backend_array(
            values=self.offset[:, None] + self.successes[None, :]
        )
        return self.namespace.sum(
            self.namespace.exp(self.log_probabilities_of_successes())
            * (values - center) ** order,
            axis=1,
        )

    def sample_of_nodes(self, nodes: NodeIndices, variables: SortedSet) -> SampleColumn:
        return (
            self.offset[nodes]
            + np.random.binomial(
                self.number_of_trials[nodes],
                self.host_copy(self.success_probability)[nodes],
            )
        ).astype(float)

    # %% structural queries

    def to_integer_layer(self) -> IntegerLayer:
        """
        :return: An integer layer with the probabilities of the nodes of this one, over
            every value any of them can take.
        """
        states = np.arange(
            int(self.offset.min(initial=0)),
            int((self.offset + self.number_of_trials).max(initial=-1)) + 1,
        )
        probabilities = binom.pmf(
            states[None, :] - self.offset[:, None],
            self.number_of_trials[:, None],
            self.host_copy(self.success_probability)[:, None],
        )
        rows, columns = np.nonzero(probabilities)
        return IntegerLayer(
            self.variable,
            states,
            DenseProbabilityTable.from_entries(
                rows, columns, probabilities[rows, columns], probabilities.shape
            ),
        )

    def type_of_truncated_layer(
        self, assignment: AbstractCompositeSet, singleton_allowed: bool
    ) -> type[Layer]:
        return IntegerLayer

    def log_truncated_of_assignment(
        self, assignment: AbstractCompositeSet, singleton_allowed: bool
    ) -> LayerWithLogProbabilities:
        return self.to_integer_layer().log_truncated_of_assignment(
            assignment, singleton_allowed
        )

    def log_conditional_of_value(self, value: Any) -> LayerWithLogProbabilities:
        return self.to_integer_layer().log_conditional_of_value(value)

    def apply_translation_own(self, translation: VariableValues):
        self.offset = self.offset + int(translation[self.variable])

    # %% nodes

    def node_distribution(self, index: int, variable: Variable) -> BinomialDistribution:
        return BinomialDistribution(
            variable=variable,
            number_of_trials=int(self.number_of_trials[index]),
            success_probability=float(self.host_copy(self.success_probability)[index]),
            offset=int(self.offset[index]),
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: list[BinomialDistribution]
    ) -> Self:
        return cls(
            variable=variable_index,
            number_of_trials=np.array(
                [distribution.number_of_trials for distribution in distributions],
                dtype=np.int64,
            ),
            success_probability=np.array(
                [distribution.success_probability for distribution in distributions],
                dtype=float,
            ),
            offset=np.array(
                [distribution.offset for distribution in distributions], dtype=np.int64
            ),
        )

    def select_nodes(self, mask: NodeMask) -> Self:
        return self.__class__(
            variable=self.variable,
            number_of_trials=self.number_of_trials[mask],
            success_probability=self.success_probability[mask],
            offset=self.offset[mask],
        )

    @classmethod
    def concatenate(cls, layers: list[Self]) -> Self:
        return cls(
            variable=layers[0].variable,
            number_of_trials=np.concatenate(
                [layer.number_of_trials for layer in layers]
            ),
            success_probability=np.concatenate(
                [layer.success_probability for layer in layers]
            ),
            offset=np.concatenate([layer.offset for layer in layers]),
        )

    def has_equal_parameters(self, other: Self) -> bool:
        return (
            np.array_equal(self.number_of_trials, other.number_of_trials)
            and np.array_equal(self.offset, other.offset)
            and np.array_equal(
                self.host_copy(self.success_probability),
                self.host_copy(other.success_probability),
            )
        )
