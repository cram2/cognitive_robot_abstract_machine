from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
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
from probabilistic_model.probabilistic_circuit.np.inner_layer import memoized
from probabilistic_model.probabilistic_circuit.np.input_layer import InputLayer
from probabilistic_model.utils import MissingDict


@dataclass(eq=False, repr=False)
class DiscreteLayer(InputLayer, ABC):
    """
    Abstract base class for the input layers of discrete univariate distributions.

    The probability of every state of the variable is stored for every node, so that a
    likelihood is a single gather from a (#nodes, #states) block.
    """

    states: npt.NDArray
    """
    The states of the variable, sorted ascending.

    A state is identified by the hash of the domain element for symbolic variables and by
    the value itself for integer variables, which is the representation that the events
    of this package use.
    """

    log_probabilities: npt.NDArray
    """
    The logarithmic probability of every state for every node, shape (#nodes, #states).
    """

    def __post_init__(self):
        super().__post_init__()
        states = np.asarray(self.states).reshape(-1)
        log_probabilities = np.asarray(self.log_probabilities, dtype=float).reshape(
            -1, len(states)
        )
        order = np.argsort(states, kind="stable")
        self.states = states[order]
        self.log_probabilities = log_probabilities[:, order]

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
    def probabilities(self) -> npt.NDArray:
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
    def selected_states(self, assignment: AbstractCompositeSet) -> npt.NDArray:
        """
        :param assignment: The assignment of the variable of this layer.
        :return: A boolean mask of the states the assignment contains.
        """
        raise NotImplementedError

    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[Dict] = None,
    ) -> npt.NDArray:
        if cache is None:
            cache = {}
        key = ("probability_of_simple_event", id(self))
        if key in cache:
            return cache[key]

        selected = self.selected_states(event[variables[self.variable]])
        result = np.exp(self.log_probabilities[:, selected]).sum(axis=1)

        cache[key] = result
        return result

    def log_truncated_of_assignment(
        self, assignment: AbstractCompositeSet, singleton_allowed: bool
    ) -> Tuple["DiscreteLayer", npt.NDArray]:
        """
        Truncating a discrete distribution keeps the probabilities of the states the
        assignment contains and renormalizes, which is one masked row-sum for the whole
        layer.
        """
        selected = self.selected_states(assignment)
        probabilities = np.where(selected, np.exp(self.log_probabilities), 0.0)
        total = probabilities.sum(axis=1)
        alive = total > 0

        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = probabilities / np.where(alive, total, 1.0)[:, None]
            log_probabilities = np.log(normalized)
            node_log_probabilities = np.where(
                alive, np.log(np.where(alive, total, 1.0)), -np.inf
            )

        # impossible nodes keep their parameters and are dropped by the prune pass
        log_probabilities = np.where(
            alive[:, None], log_probabilities, self.log_probabilities
        )
        return (
            self.__class__(self.variable, self.states.copy(), log_probabilities),
            node_log_probabilities,
        )

    def state_indices_of(self, values: npt.NDArray) -> npt.NDArray:
        """
        Look up the index of every value in :attr:`states`.

        :param values: The values, as they appear in an event array.
        :return: The index of every value, or ``-1`` for values that are not a state.
        """
        values = np.asarray(
            [hash(value) for value in np.asarray(values).reshape(-1)], dtype=np.int64
        )
        positions = np.searchsorted(self.states, values)
        positions = np.clip(positions, 0, max(self.number_of_states - 1, 0))
        found = self.states[positions] == values
        return np.where(found, positions, -1)

    @memoized("log_likelihood")
    def log_likelihood_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
    ) -> npt.NDArray:
        indices = self.state_indices_of(self.column_of(x))
        result = np.full((len(indices), self.number_of_nodes), -np.inf)
        known = indices >= 0
        if known.any():
            result[known] = self.log_probabilities[:, indices[known]].T
        return result

    @memoized("cumulative_distribution")
    def cumulative_distribution_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
    ) -> npt.NDArray:
        raise NotImplementedError

    def normalize_own(self):
        from probabilistic_model.probabilistic_circuit.np.utils import (
            embedded_logsumexp,
        )

        self.log_probabilities = self.log_probabilities - embedded_logsumexp(
            self.log_probabilities, axis=1
        ).reshape(-1, 1)

    def probabilities_of_node(self, node: int) -> MissingDict:
        """
        :param node: The index of a node.
        :return: The probability of every state with a non-zero probability.
        """
        probabilities = np.exp(self.log_probabilities[node])
        return MissingDict(
            float,
            {
                int(state): float(probability)
                for state, probability in zip(self.states, probabilities)
                if probability > 0
            },
        )

    @classmethod
    def states_and_log_probabilities_of(
        cls, distributions: List[DiscreteDistribution]
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Collect the states and the probability block of a list of discrete distributions.

        :param distributions: The distributions.
        :return: The sorted states and the logarithmic probabilities.
        """
        states = sorted(
            {state for distribution in distributions for state in distribution.probabilities}
        )
        state_to_column = {state: index for index, state in enumerate(states)}

        probabilities = np.zeros((len(distributions), len(states)))
        for row, distribution in enumerate(distributions):
            for state, probability in distribution.probabilities.items():
                probabilities[row, state_to_column[state]] = probability

        with np.errstate(divide="ignore"):
            return np.array(states, dtype=np.int64), np.log(probabilities)

    def select_nodes(self, mask: npt.NDArray) -> Self:
        return self.__class__(
            self.variable, self.states.copy(), self.log_probabilities[mask]
        )

    @classmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        # truncating a discrete layer never changes its states, so the blocks line up
        return cls(
            layers[0].variable,
            layers[0].states.copy(),
            np.concatenate([layer.log_probabilities for layer in layers]),
        )

    def sample_of_node(
        self, node: int, amount: int, variables: SortedSet
    ) -> npt.NDArray:
        probabilities = np.exp(self.log_probabilities[node])
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

    def to_json(self, **kwargs) -> Dict[str, Any]:
        result = super().to_json(**kwargs)
        result["states"] = self.states.tolist()
        result["log_probabilities"] = self.log_probabilities.tolist()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            data["variable"],
            np.array(data["states"], dtype=np.int64),
            np.array(data["log_probabilities"], dtype=float),
        )


@dataclass(eq=False, repr=False)
class SymbolicLayer(DiscreteLayer):
    """
    A layer of categorical distributions over one symbolic variable.
    """

    @classmethod
    def rustworkx_classes(cls) -> Tuple[Type, ...]:
        return (SymbolicDistribution,)

    def node_distribution(
        self, index: int, variable: Variable
    ) -> SymbolicDistribution:
        return SymbolicDistribution(
            variable=variable, probabilities=self.probabilities_of_node(index)
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[SymbolicDistribution]
    ) -> Self:
        states, log_probabilities = cls.states_and_log_probabilities_of(distributions)
        return cls(variable_index, states, log_probabilities)

    def selected_states(self, assignment: Set) -> npt.NDArray:
        hashes = np.array(
            [hash(element) for element in assignment.simple_sets], dtype=np.int64
        )
        return np.isin(self.states, hashes)


@dataclass(eq=False, repr=False)
class IntegerLayer(DiscreteLayer):
    """
    A layer of distributions over one integer variable.
    """

    @classmethod
    def rustworkx_classes(cls) -> Tuple[Type, ...]:
        return (IntegerDistribution,)

    def node_distribution(self, index: int, variable: Variable) -> IntegerDistribution:
        return IntegerDistribution(
            variable=variable, probabilities=self.probabilities_of_node(index)
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[IntegerDistribution]
    ) -> Self:
        states, log_probabilities = cls.states_and_log_probabilities_of(distributions)
        return cls(variable_index, states, log_probabilities)

    def selected_states(self, assignment: Interval) -> npt.NDArray:
        return np.array(
            [state in assignment for state in self.states.tolist()], dtype=bool
        )

    @memoized("cumulative_distribution")
    def cumulative_distribution_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
    ) -> npt.NDArray:
        column = np.asarray(self.column_of(x), dtype=float).reshape(-1, 1)
        reached = column >= self.states.reshape(1, -1)
        return reached.astype(float) @ np.exp(self.log_probabilities).T

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> npt.NDArray:
        deviations = (self.states.astype(float) - center) ** order
        return np.exp(self.log_probabilities) @ deviations
