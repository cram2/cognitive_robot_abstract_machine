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
from typing_extensions import Any, List, Optional, Self, Type

from probabilistic_model.distributions.distributions import (
    DiscreteDistribution,
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeMask,
    NodeStateValues,
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
from probabilistic_model.probabilistic_circuit.tensorized.query_cache import (
    QueryCache,
    memoized,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
)
from probabilistic_model.probabilistic_circuit.tensorized.utils import (
    embedded_logsumexp,
)
from probabilistic_model.utils import MissingDict


@dataclass(eq=False, repr=False)
class DiscreteLayer(InputLayer, ABC):
    """
    Abstract base class for the input layers of discrete univariate distributions.

    The probability of every state of the variable is stored for every node, so that a
    likelihood is a single gather from a (#nodes, #states) block.
    """

    states: States
    """
    The states of the variable, sorted ascending.
    """

    log_probabilities: NodeStateValues
    """
    The logarithmic probability of every state for every node.
    """

    @property
    def number_of_nodes(self) -> int:
        return self.log_probabilities.shape[0]

    @property
    def number_of_states(self) -> int:
        """
        :return: The number of states of the variable.
        """
        return len(self.states)

    @property
    def number_of_own_parameters(self) -> int:
        return int(self.log_probabilities.size)

    @property
    def probabilities(self) -> NodeStateValues:
        """
        :return: The probabilities of every state for every node in linear space.
        """
        return np.exp(self.log_probabilities)

    def validate_own(self):
        if self.log_probabilities.shape[1] != self.number_of_states:
            raise ShapeMismatchError(
                (self.number_of_nodes, self.number_of_states),
                self.log_probabilities.shape,
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

        :param values: The values, as they appear in a sample array.
        :return: The index of every value, or ``-1`` for values that are not a state.
        """
        hashes = np.asarray(
            [hash(value) for value in np.asarray(values).reshape(-1)], dtype=np.int64
        )
        positions = np.searchsorted(self.states, hashes)
        positions = np.clip(positions, 0, max(self.number_of_states - 1, 0))
        found = self.states[positions] == hashes
        return np.where(found, positions, -1)

    @memoized
    def log_likelihood_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        indices = self.state_indices_of(self.column_of(events))
        result = np.full((len(indices), self.number_of_nodes), -np.inf)
        known = indices >= 0
        if known.any():
            result[known] = self.log_probabilities[:, indices[known]].T
        return result

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
        selected = self.selected_states(event[variables[self.variable]])
        return self.probabilities[:, selected].sum(axis=1)

    def type_of_truncated_layer(
        self, assignment: AbstractCompositeSet, singleton_allowed: bool
    ) -> Type[Layer]:
        return self.__class__

    def log_truncated_of_assignment(
        self, assignment: AbstractCompositeSet, singleton_allowed: bool
    ) -> LayerWithLogProbabilities:
        """
        Truncating a discrete distribution keeps the probabilities of the states the
        assignment contains and renormalizes, which is one masked row-sum for the whole
        layer.
        """
        return self.renormalized_to(self.selected_states(assignment))

    def log_conditional_of_value(self, value: Any) -> LayerWithLogProbabilities:
        """
        Conditioning on a value is truncating to the state of that value.
        """
        selected = np.zeros(self.number_of_states, dtype=bool)
        [index] = self.state_indices_of(np.array([value], dtype=object))
        if index >= 0:
            selected[index] = True
        return self.renormalized_to(selected)

    def renormalized_to(self, selected: StateMask) -> LayerWithLogProbabilities:
        """
        :param selected: The states to keep.
        :return: The layer with the probability of every other state set to zero and
            renormalized, and the log-probability of the kept states under every node.
            A node without probability for the kept states keeps its parameters and is
            removed by the prune pass.
        """
        probabilities = np.where(selected, self.probabilities, 0.0)
        total = probabilities.sum(axis=1)
        alive = total > 0

        with np.errstate(divide="ignore", invalid="ignore"):
            log_probabilities = np.log(
                probabilities / np.where(alive, total, 1.0)[:, None]
            )
            node_log_probabilities = np.where(
                alive, np.log(np.where(alive, total, 1.0)), -np.inf
            )

        log_probabilities = np.where(
            alive[:, None], log_probabilities, self.log_probabilities
        )
        return LayerWithLogProbabilities(
            self.__class__(self.variable, self.states.copy(), log_probabilities),
            node_log_probabilities,
        )

    def normalize_own(self):
        self.log_probabilities = self.log_probabilities - embedded_logsumexp(
            self.log_probabilities, axis=1
        ).reshape(-1, 1)

    def probabilities_of_node(self, node: int) -> MissingDict:
        """
        :param node: The index of a node.
        :return: The probability of every state with a non-zero probability.
        """
        return MissingDict(
            float,
            {
                int(state): float(probability)
                for state, probability in zip(self.states, self.probabilities[node])
                if probability > 0
            },
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[DiscreteDistribution]
    ) -> Self:
        states = sorted(
            {
                state
                for distribution in distributions
                for state in distribution.probabilities
            }
        )
        state_to_column = {state: index for index, state in enumerate(states)}

        probabilities = np.zeros((len(distributions), len(states)))
        for row, distribution in enumerate(distributions):
            for state, probability in distribution.probabilities.items():
                probabilities[row, state_to_column[state]] = probability

        with np.errstate(divide="ignore"):
            log_probabilities = np.log(probabilities)
        return cls(variable_index, np.array(states, dtype=np.int64), log_probabilities)

    def select_nodes(self, mask: NodeMask) -> Self:
        return self.__class__(
            self.variable, self.states.copy(), self.log_probabilities[mask]
        )

    @classmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        """
        Truncating a discrete layer never changes its states, so the probability blocks
        of the layers line up.
        """
        return cls(
            layers[0].variable,
            layers[0].states.copy(),
            np.concatenate([layer.log_probabilities for layer in layers]),
        )

    def sample_of_node(
        self, node: int, amount: int, variables: SortedSet
    ) -> SampleColumn:
        probabilities = self.probabilities[node]
        total = probabilities.sum()
        if total <= 0:
            return np.full(amount, np.nan)
        return np.random.choice(self.states, size=amount, p=probabilities / total)

    def __deepcopy__(self, memo=None) -> Self:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        result = self.__class__(
            self.variable, self.states.copy(), self.log_probabilities.copy()
        )
        memo[id(self)] = result
        return result


@dataclass(eq=False, repr=False)
class SymbolicLayer(DiscreteLayer):
    """
    A layer of categorical distributions over one symbolic variable.

    The states are the hashes of the domain elements, which is the representation that
    the events of this package use.
    """

    def node_distribution(self, index: int, variable: Variable) -> SymbolicDistribution:
        return SymbolicDistribution(
            variable=variable, probabilities=self.probabilities_of_node(index)
        )

    def selected_states(self, assignment: Set) -> StateMask:
        hashes = np.array(
            [hash(element) for element in assignment.simple_sets], dtype=np.int64
        )
        return np.isin(self.states, hashes)


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
        return reached.astype(float) @ self.probabilities.T

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> NodeValues:
        deviations = (self.states.astype(float) - center) ** order
        return self.probabilities @ deviations
