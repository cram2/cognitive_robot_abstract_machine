"""
The tensorized input layers whose leaves are binomial and truncated binomial
distributions.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from random_events.interval import Interval, singleton
from random_events.product_algebra import SimpleEvent
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Variable
from scipy.special import gammaln
from sortedcontainers import SortedSet
from typing_extensions import Any, Self

from probabilistic_model.distributions.binomial import (
    BinomialDistribution,
    TruncatedBinomialDistribution,
)
from probabilistic_model.exceptions import (
    ShapeMismatchError,
    SuccessProbabilityOutsideUnitIntervalError,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
    NodeIndices,
    NodeIntegers,
    NodeMask,
    NodeStateMask,
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

    The success probabilities are its parameters; everything else is structure.
    Truncated or conditioned, it becomes a :class:`TruncatedBinomialLayer`.
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

    def copy_with_parameters(self, *, parameters: list[BackendArray]) -> Self:
        (success_probability,) = parameters
        return dataclasses.replace(self, success_probability=success_probability)

    def projected_parameters(
        self, *, parameters: list[BackendArray]
    ) -> list[BackendArray]:
        """
        Like :meth:`Layer.projected_parameters`, bringing every success probability back
        into [0, 1].
        """
        (success_probability,) = parameters
        return [self.namespace.clip(success_probability, 0.0, 1.0)]

    def validate_own(self) -> None:
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

    @property
    def values_of_successes(self) -> NodeStateValues:
        """
        :return: The value every node takes at every number of :attr:`successes`, shape
            (#nodes, #numbers).
        """
        return self.offset[:, None] + self.successes[None, :]

    def possible_successes(self) -> NodeStateMask:
        """
        :return: The numbers of :attr:`successes` every node can have, shape (#nodes,
            #numbers).
        """
        return self.successes[None, :] <= self.number_of_trials[:, None]

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
            node, ``-inf`` where it is not possible, shape (#nodes, #numbers).
        """
        successes = self.successes[None, :]
        return self.log_coefficients_of_successes() + self.log_success_terms(
            successes=self.backend_array(values=successes),
            failures=self.backend_array(
                values=np.maximum(self.number_of_trials[:, None] - successes, 0)
            ),
        )

    def log_success_terms(
        self, *, successes: BackendArray, failures: BackendArray
    ) -> BackendArray:
        """
        :param successes: Numbers of successes, one row per node or broadcast to it.
        :param failures: The numbers of failures that go with them.
        :return: The log-probability of these successes and failures in one order
            under every node, zero for none of either even where its probability is.
        """
        namespace = self.namespace
        probability = self.success_probability[:, None]
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

    def selected_successes(self, *, assignment: AbstractCompositeSet) -> NodeStateMask:
        """
        :param assignment: The assignment of the variable of this layer.
        :return: The numbers of :attr:`successes` at which every node takes a value of
            the assignment.
        """
        values = self.values_of_successes
        contained = [
            value for value in np.unique(values).tolist() if value in assignment
        ]
        return np.isin(values, contained)

    def successes_of_column(self, *, values: SampleColumn) -> SampleNodeValues:
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
        :param successes: Whole numbers of successes per event and node, within the
            :attr:`successes`.
        :return: The value of the table at the node and the number of successes of
            every entry, shape (#events, #nodes).
        """
        namespace = self.namespace
        index_type = namespace.__array_namespace_info__().default_dtypes(
            device=self.device
        )["indexing"]
        flat_indices = (
            namespace.astype(successes, index_type)
            + namespace.arange(self.number_of_nodes, device=self.device)[None, :]
            * table.shape[1]
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
        self, *, events: SampleArray, cache: QueryCache | None = None
    ) -> SampleNodeValues:
        namespace = self.namespace
        successes = self.successes_of_column(values=self.column_of(events))
        possible = (
            (successes >= 0)
            & (successes <= len(self.successes) - 1)
            & (successes == namespace.floor(successes))
        )
        log_likelihoods = self.entries_of_nodes(
            table=self.log_probabilities_of_successes(),
            successes=namespace.where(possible, successes, 0.0),
        )
        return namespace.where(possible, log_likelihoods, -namespace.inf)

    @memoized
    def cumulative_distribution_of_nodes(
        self, *, events: SampleArray, cache: QueryCache | None = None
    ) -> SampleNodeValues:
        namespace = self.namespace
        successes = namespace.floor(
            self.successes_of_column(values=self.column_of(events))
        )
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
        *,
        event: SimpleEvent,
        variables: SortedSet,
        cache: QueryCache | None = None,
    ) -> NodeValues:
        namespace = self.namespace
        selected = self.selected_successes(assignment=event[variables[self.variable]])
        return namespace.sum(
            namespace.where(
                namespace.asarray(selected, device=self.device),
                namespace.exp(self.log_probabilities_of_successes()),
                0.0,
            ),
            axis=1,
        )

    def moment_of_nodes_own(
        self, *, order: int, center: float, variable: Variable
    ) -> NodeValues:
        """
        Like :meth:`UnivariateInputLayer.moment_of_nodes_own`, as the expectation of the
        centered values under the probabilities of the successes of every node.
        """
        values = self.backend_array(values=self.values_of_successes)
        return self.namespace.sum(
            self.namespace.exp(self.log_probabilities_of_successes())
            * (values - center) ** order,
            axis=1,
        )

    def sample_of_nodes(
        self, *, nodes: NodeIndices, variables: SortedSet
    ) -> SampleColumn:
        """
        Like :meth:`UnivariateInputLayer.sample_of_nodes`, drawing the number of
        successes of every entry from the probabilities of its node, all nodes at once.
        """
        cumulative = np.cumsum(
            np.exp(self.host_copy(self.log_probabilities_of_successes()))[nodes], axis=1
        )
        drawn = np.random.random((len(nodes), 1)) * cumulative[:, -1:]
        successes = np.minimum(
            (cumulative < drawn).sum(axis=1), len(self.successes) - 1
        )
        return (self.offset[nodes] + successes).astype(float)

    # %% structural queries

    def type_of_truncated_layer(
        self, *, assignment: AbstractCompositeSet, singleton_allowed: bool
    ) -> type[Layer]:
        return TruncatedBinomialLayer

    def log_truncated_of_assignment(
        self, *, assignment: AbstractCompositeSet, singleton_allowed: bool
    ) -> LayerWithLogProbabilities:
        selected = self.selected_successes(assignment=assignment)
        probabilities = np.exp(self.host_copy(self.log_probabilities_of_successes()))
        with np.errstate(divide="ignore"):
            log_probabilities = np.log(
                np.where(selected, probabilities, 0.0).sum(axis=1)
            )
        return LayerWithLogProbabilities(
            TruncatedBinomialLayer.from_binomial_layer(
                layer=self, allowed_successes=self.possible_successes() & selected
            ),
            log_probabilities,
        )

    def log_conditional_of_value(self, *, value: Any) -> LayerWithLogProbabilities:
        return self.log_truncated_of_assignment(
            assignment=singleton(value), singleton_allowed=False
        )

    def apply_translation_own(self, *, translation: VariableValues) -> None:
        """
        Moves the :attr:`offset` of every node by the translation of its variable.
        """
        self.offset = self.offset + int(translation[self.variable])

    # %% nodes

    def node_distribution(
        self, *, index: int, variable: Variable
    ) -> BinomialDistribution:
        return BinomialDistribution(
            variable=variable,
            number_of_trials=int(self.number_of_trials[index]),
            success_probability=float(self.host_copy(self.success_probability)[index]),
            offset=int(self.offset[index]),
        )

    @classmethod
    def from_distributions(
        cls, *, variable_index: int, distributions: list[BinomialDistribution]
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

    def select_nodes(self, *, mask: NodeMask) -> Self:
        return dataclasses.replace(
            self,
            number_of_trials=self.number_of_trials[mask],
            success_probability=self.success_probability[mask],
            offset=self.offset[mask],
        )

    @classmethod
    def concatenate(cls, *, layers: list[Self]) -> Self:
        return dataclasses.replace(
            layers[0],
            number_of_trials=np.concatenate(
                [layer.number_of_trials for layer in layers]
            ),
            success_probability=np.concatenate(
                [layer.success_probability for layer in layers]
            ),
            offset=np.concatenate([layer.offset for layer in layers]),
        )

    def has_equal_parameters(self, *, other: Self) -> bool:
        return (
            np.array_equal(self.number_of_trials, other.number_of_trials)
            and np.array_equal(self.offset, other.offset)
            and np.array_equal(
                self.host_copy(self.success_probability),
                self.host_copy(other.success_probability),
            )
        )


# %% truncated binomial layer


@dataclass(eq=False, repr=False, kw_only=True)
class TruncatedBinomialLayer(BinomialLayer):
    """
    A layer of binomial distributions over one integer variable, each restricted to its
    :attr:`allowed_successes` and renormalized, see
    :class:`~probabilistic_model.distributions.binomial.TruncatedBinomialDistribution`.
    """

    allowed_successes: NodeStateMask
    """
    The numbers of :attr:`successes` every node is restricted to.
    """

    @classmethod
    def from_binomial_layer(
        cls, *, layer: BinomialLayer, allowed_successes: NodeStateMask
    ) -> Self:
        """
        :param layer: The layer whose nodes to truncate.
        :param allowed_successes: The numbers of successes every node is restricted to.
        :return: The nodes of the layer restricted to these numbers of successes, with
            copies of its arrays on the host.
        """
        return cls(
            variable=layer.variable,
            number_of_trials=layer.number_of_trials.copy(),
            success_probability=layer.host_copy(layer.success_probability).copy(),
            offset=layer.offset.copy(),
            allowed_successes=allowed_successes,
        )

    def validate_own(self) -> None:
        """
        Like :meth:`BinomialLayer.validate_own`, and also:

        :raises ShapeMismatchError: If the nodes do not each have a flag per number of
            successes.
        """
        super().validate_own()
        expected_shape = (self.number_of_nodes, len(self.successes))
        if self.allowed_successes.shape != expected_shape:
            raise ShapeMismatchError(self.allowed_successes.shape, expected_shape)

    def possible_successes(self) -> NodeStateMask:
        """
        Like :meth:`BinomialLayer.possible_successes`, only the
        :attr:`allowed_successes`.
        """
        return super().possible_successes() & self.allowed_successes

    def log_probabilities_of_successes(self) -> NodeStateValues:
        """
        Like :meth:`BinomialLayer.log_probabilities_of_successes`, restricted to the
        :attr:`allowed_successes` and renormalized.
        """
        namespace = self.namespace
        restricted = namespace.where(
            namespace.asarray(self.allowed_successes, device=self.device),
            super().log_probabilities_of_successes(),
            -namespace.inf,
        )
        largest = namespace.max(restricted, axis=1, keepdims=True)
        possible = largest > -namespace.inf
        shift = namespace.where(possible, largest, 0.0)
        mass = namespace.sum(namespace.exp(restricted - shift), axis=1, keepdims=True)
        log_mass = shift + namespace.log(namespace.where(possible, mass, 1.0))
        return namespace.where(possible, restricted - log_mass, -namespace.inf)

    def node_distribution(
        self, *, index: int, variable: Variable
    ) -> TruncatedBinomialDistribution:
        """
        Like :meth:`BinomialLayer.node_distribution`, restricted to the values the node
        allows.
        """
        untruncated = super().node_distribution(index=index, variable=variable)
        truncation_interval = Interval()
        values = self.values_of_successes[index][self.possible_successes()[index]]
        for value in values.tolist():
            truncation_interval |= singleton(value)
        return TruncatedBinomialDistribution(
            variable=variable,
            number_of_trials=untruncated.number_of_trials,
            success_probability=untruncated.success_probability,
            offset=untruncated.offset,
            truncation_interval=truncation_interval,
        )

    @classmethod
    def from_distributions(
        cls, *, variable_index: int, distributions: list[TruncatedBinomialDistribution]
    ) -> Self:
        """
        Like :meth:`BinomialLayer.from_distributions`, restricting every node to the
        allowed values of its distribution.
        """
        untruncated = BinomialLayer.from_distributions(
            variable_index=variable_index, distributions=distributions
        )
        values = untruncated.values_of_successes
        return cls.from_binomial_layer(
            layer=untruncated,
            allowed_successes=np.array(
                [
                    [value in distribution.allowed_values for value in row]
                    for distribution, row in zip(distributions, values.tolist())
                ],
                dtype=bool,
            ).reshape(values.shape),
        )

    def select_nodes(self, *, mask: NodeMask) -> Self:
        """
        Like :meth:`BinomialLayer.select_nodes`, keeping the allowed successes of the
        selected nodes.
        """
        selected = super().select_nodes(mask=mask)
        return dataclasses.replace(
            selected,
            allowed_successes=self.allowed_successes[mask][
                :, : len(selected.successes)
            ],
        )

    @classmethod
    def concatenate(cls, *, layers: list[Self]) -> Self:
        """
        Like :meth:`BinomialLayer.concatenate`, joining the allowed successes of the
        layers too.
        """
        concatenated = super().concatenate(layers=layers)
        return dataclasses.replace(
            concatenated,
            allowed_successes=np.concatenate(
                [
                    np.pad(
                        layer.allowed_successes,
                        (
                            (0, 0),
                            (0, len(concatenated.successes) - len(layer.successes)),
                        ),
                    )
                    for layer in layers
                ]
            ),
        )

    def has_equal_parameters(self, *, other: Self) -> bool:
        """
        Like :meth:`BinomialLayer.has_equal_parameters`, comparing the allowed successes
        too.
        """
        return super().has_equal_parameters(other=other) and np.array_equal(
            self.allowed_successes, other.allowed_successes
        )
