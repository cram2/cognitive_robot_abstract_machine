from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from random_events.interval import Interval
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Any, Dict, List, Optional, Self, Tuple, Type

from probabilistic_model.distributions.distributions import (
    DiscreteDistribution,
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
    NodeMask,
    NodeValues,
    SampleArray,
    SampleColumn,
    SampleNodeValues,
    StateIndices,
    StateMask,
    States,
)
from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    UndefinedCumulativeDistributionError,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.base import (
    InputLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
    ProbabilityTable,
)
from probabilistic_model.probabilistic_circuit.tensorized.query_cache import (
    QueryCache,
    memoized,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
)
from probabilistic_model.probabilistic_circuit.tensorized.symbolic_encoding import (
    SymbolicEncoding,
)
from probabilistic_model.utils import MissingDict


@dataclass(eq=False, repr=False)
class DiscreteLayer(InputLayer, ABC):
    """
    Abstract base class for the input layers of discrete univariate distributions.

    The probability of every state for every node is held by a
    :class:`~probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table.ProbabilityTable`,
    whose type decides how the probabilities are stored. Choose it with the
    ``table_type`` of :meth:`from_distributions` or with :meth:`stored_as`.
    """

    states: States
    """
    The states of the variable, sorted ascending: the columns of the table.
    """

    table: ProbabilityTable
    """
    The probability of every state for every node.
    """

    @property
    def number_of_nodes(self) -> int:
        return self.table.number_of_nodes

    @property
    def number_of_states(self) -> int:
        """
        :return: The number of states of the variable.
        """
        return len(self.states)

    @property
    def number_of_own_parameters(self) -> int:
        return self.table.number_of_stored_entries

    def validate_own(self):
        if self.table.number_of_states != self.number_of_states:
            raise ShapeMismatchError(
                (self.table.number_of_nodes, self.table.number_of_states),
                (self.number_of_nodes, self.number_of_states),
            )

    @abstractmethod
    def selected_states(self, assignment: AbstractCompositeSet) -> StateMask:
        """
        :param assignment: The assignment of the variable of this layer.
        :return: The states the assignment contains.
        """
        raise NotImplementedError

    def state_indices_of(self, values: SampleColumn) -> StateIndices:
        """
        Look up the index of every value in :attr:`states`.

        :param values: The values of the variable as the layers read them: the value
            itself for an integer variable and the position of the domain element for a
            symbolic variable, see :class:`SymbolicEncoding`.
        :return: The index of every value, or ``-1`` for values that are not a state.
        """
        values = np.asarray(values, dtype=float).reshape(-1)
        states = self.states.astype(float)
        positions = np.searchsorted(states, values)
        positions = np.clip(positions, 0, max(self.number_of_states - 1, 0))
        found = states[positions] == values
        return np.where(found, positions, -1)

    @memoized
    def log_likelihood_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        return self.table.log_probabilities_of_states(
            self.state_indices_of(self.column_of(events))
        )

    @memoized
    def cumulative_distribution_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        raise UndefinedCumulativeDistributionError(self.__class__)

    @memoized
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> NodeValues:
        return self.table.probability_of_states(
            self.selected_states(event[variables[self.variable]])
        )

    def type_of_truncated_layer(
        self, assignment: AbstractCompositeSet, singleton_allowed: bool
    ) -> Type[Layer]:
        return self.__class__

    def log_truncated_of_assignment(
        self, assignment: AbstractCompositeSet, singleton_allowed: bool
    ) -> LayerWithLogProbabilities:
        # a single state left is handled like any other set of states
        return self.renormalized_to(self.selected_states(assignment))

    def log_conditional_of_value(self, value: Any) -> LayerWithLogProbabilities:
        # the value is read like the values of events, see state_indices_of
        selected = np.zeros(self.number_of_states, dtype=bool)
        [index] = self.state_indices_of(np.array([value], dtype=float))
        if index >= 0:
            selected[index] = True
        return self.renormalized_to(selected)

    def renormalized_to(self, selected: StateMask) -> LayerWithLogProbabilities:
        """
        :param selected: The states to keep.
        :return: The layer with the probability of every other state set to zero and
            renormalized, and the log-probability of the kept states under every node.
            A node without probability for the kept states keeps none and is removed by
            the prune pass.
        """
        with np.errstate(divide="ignore"):
            node_log_probabilities = np.log(self.table.probability_of_states(selected))
        return LayerWithLogProbabilities(
            self.with_table(self.table.restricted_to(selected).normalized()),
            node_log_probabilities,
        )

    def normalize_own(self):
        self.table = self.table.normalized()

    def probabilities_of_node(self, node: int) -> MissingDict:
        """
        :param node: The index of a node.
        :return: The probability of every state with a non-zero probability.
        """
        indices, probabilities = self.table.probabilities_of_node(node)
        return MissingDict(
            float,
            {
                int(state): float(probability)
                for state, probability in zip(self.states[indices], probabilities)
            },
        )

    def with_table(self, table: ProbabilityTable) -> Self:
        """
        :param table: The probability table of the new layer, over the states of this
            one.
        :return: A layer over the same variable and states as this one with this table.
        """
        return self.__class__(self.variable, self.states.copy(), table)

    def stored_as(self, table_type: Type[ProbabilityTable]) -> Self:
        """
        :param table_type: The type of table to store the probabilities in.
        :return: This layer with its probabilities stored in a table of that type.
        """
        return self.with_table(table_type.of(self.table))

    @classmethod
    def from_distributions(
        cls,
        variable_index: int,
        distributions: List[DiscreteDistribution],
        table_type: Type[ProbabilityTable] = DenseProbabilityTable,
    ) -> Self:
        """
        :param table_type: The type of table to store the probabilities in.
        """
        return cls(variable_index, *cls.parameters_of(distributions, table_type))

    @classmethod
    def parameters_of(
        cls,
        distributions: List[DiscreteDistribution],
        table_type: Type[ProbabilityTable],
    ) -> Tuple[States, ProbabilityTable]:
        """
        :param distributions: Distributions over the variable of this layer.
        :param table_type: The type of table to store the probabilities in.
        :return: The states that any of the distributions has, and the table of the
            probability of every state under every distribution.
        """
        probabilities_by_state = [
            cls.probabilities_by_state_of(distribution)
            for distribution in distributions
        ]
        states = np.array(
            sorted(
                {
                    state
                    for probabilities in probabilities_by_state
                    for state in probabilities
                }
            ),
            dtype=np.int64,
        )
        rows = np.concatenate(
            [
                np.full(len(probabilities), row, dtype=np.int64)
                for row, probabilities in enumerate(probabilities_by_state)
            ]
        )
        columns = np.searchsorted(
            states,
            np.array(
                [
                    state
                    for probabilities in probabilities_by_state
                    for state in probabilities
                ],
                dtype=np.int64,
            ),
        )
        values = np.array(
            [
                probability
                for probabilities in probabilities_by_state
                for probability in probabilities.values()
            ],
            dtype=float,
        )
        return states, table_type.from_entries(
            rows, columns, values, (len(distributions), len(states))
        )

    @classmethod
    def probabilities_by_state_of(
        cls, distribution: DiscreteDistribution
    ) -> Dict[int, float]:
        """
        :param distribution: A distribution over the variable of this layer.
        :return: The probability of every state of the distribution, keyed by the state
            as this layer stores it.
        """
        return dict(distribution.probabilities)

    def select_nodes(self, mask: NodeMask) -> Self:
        return self.with_table(self.table.select_nodes(mask))

    @classmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        # truncating a discrete layer never changes its states, so the tables line up;
        # the result is stored like the table of the first layer
        table_type = type(layers[0].table)
        return layers[0].with_table(
            table_type.concatenate([table_type.of(layer.table) for layer in layers])
        )

    def sample_of_nodes(self, nodes: NodeIndices, variables: SortedSet) -> SampleColumn:
        indices = self.table.sample_state_indices(nodes)
        return np.where(
            indices >= 0, self.states[np.maximum(indices, 0)].astype(float), np.nan
        )

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> Self:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        result = self.with_table(self.table.copy())
        memo[id(self)] = result
        return result


@dataclass(eq=False, repr=False)
class SymbolicLayer(DiscreteLayer):
    """
    A layer of categorical distributions over one symbolic variable.

    The states are the positions of the domain elements, see :class:`SymbolicEncoding`.
    """

    domain_hashes: SampleColumn
    """
    The hash of every domain element of the variable, at the position of the element.
    """

    def with_table(self, table: ProbabilityTable) -> Self:
        return self.__class__(
            self.variable, self.states.copy(), table, self.domain_hashes
        )

    @classmethod
    def from_distributions(
        cls,
        variable_index: int,
        distributions: List[SymbolicDistribution],
        table_type: Type[ProbabilityTable] = DenseProbabilityTable,
    ) -> Self:
        return cls(
            variable_index,
            *cls.parameters_of(distributions, table_type),
            SymbolicEncoding(distributions[0].variable).hashes,
        )

    @classmethod
    def probabilities_by_state_of(
        cls, distribution: SymbolicDistribution
    ) -> Dict[int, float]:
        encoding = SymbolicEncoding(distribution.variable)
        return {
            encoding.index_of_element(hash_value): probability
            for hash_value, probability in distribution.probabilities.items()
        }

    def node_distribution(self, index: int, variable: Variable) -> SymbolicDistribution:
        return SymbolicDistribution(
            variable=variable,
            probabilities=MissingDict(
                float,
                {
                    int(self.domain_hashes[state]): probability
                    for state, probability in self.probabilities_of_node(index).items()
                },
            ),
        )

    def selected_states(self, assignment: Set) -> StateMask:
        hashes = np.array(
            [hash(simple_set) for simple_set in assignment.simple_sets], dtype=float
        )
        return np.isin(self.domain_hashes[self.states], hashes)


@dataclass(eq=False, repr=False)
class IntegerLayer(DiscreteLayer):
    """
    A layer of distributions over one integer variable.
    """

    def node_distribution(self, index: int, variable: Variable) -> IntegerDistribution:
        return IntegerDistribution(
            variable=variable, probabilities=self.probabilities_of_node(index)
        )

    def selected_states(self, assignment: Interval) -> StateMask:
        return np.array(
            [state in assignment for state in self.states.tolist()], dtype=bool
        )

    @memoized
    def cumulative_distribution_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        column = np.asarray(self.column_of(events), dtype=float).reshape(-1, 1)
        reached = column >= self.states.reshape(1, -1)
        return self.table.dot(reached.astype(float).T).T

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> NodeValues:
        deviations = (self.states.astype(float) - center) ** order
        return self.table.dot(deviations)
