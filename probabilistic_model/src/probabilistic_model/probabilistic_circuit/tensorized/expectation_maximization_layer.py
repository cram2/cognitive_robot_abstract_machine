"""
The layers whose parameters tensorized expectation maximization learns, and the
sufficient statistics their maximization step reads.
"""

from __future__ import annotations

# %% imports
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Self

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
    NodeValues,
)

if TYPE_CHECKING:
    from probabilistic_model.learning.expectation_maximization import (
        MaximizationRegularization,
    )


# %% sufficient statistics


@dataclass(kw_only=True)
class SufficientStatistics(ABC):
    """
    What the maximization step of a layer needs to know about the rows of the batches
    since the last update, accumulated by adding the statistics of every batch.
    """

    @property
    @abstractmethod
    def arrays(self) -> list[BackendArray]:
        """
        :return: Every array of the statistics, in the order :meth:`from_arrays` takes
            them.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_arrays(cls, *, arrays: list[BackendArray]) -> Self:
        """
        :param arrays: Arrays in the order and with the shapes of :attr:`arrays`.
        :return: The statistics made of these arrays.
        """
        raise NotImplementedError

    def added(self, *, other: Self) -> Self:
        """
        :param other: Statistics of the same layer on other rows.
        :return: The statistics of the rows of both.
        """
        return self.from_arrays(
            arrays=[own + others for own, others in zip(self.arrays, other.arrays)]
        )


# %% layers


class ExpectationMaximizationLayer(ABC):
    """
    Interface of a layer whose parameters expectation maximization learns from the
    gradient of the total log-likelihood of a batch.

    The node count of a node is the derivative of the total log-likelihood by the log-
    value of the node, the expected number of rows the node explains.
    """

    @abstractmethod
    def expected_statistics(
        self, *, parameter_gradients: list[BackendArray], node_counts: NodeValues
    ) -> SufficientStatistics:
        """
        :param parameter_gradients: The derivative of the total log-likelihood of a
            batch by every parameter array, in the order of the layer's parameters.
        :param node_counts: The node count of every node of the layer on the batch.
        :return: The sufficient statistics of the batch, zero for zero gradients and
            counts.
        """
        raise NotImplementedError

    @abstractmethod
    def maximized_parameters(
        self,
        *,
        statistics: SufficientStatistics,
        regularization: MaximizationRegularization,
        step_size: float,
    ) -> list[BackendArray]:
        """
        :param statistics: The sufficient statistics accumulated since the last update,
            computed with the layer's parameters.
        :param regularization: How the learned weights and variances stay positive.
        :param step_size: The share of the way from the parameters to those that
            maximize the expected log-likelihood, one for a full update.
        :return: The updated parameter arrays; nodes no row reached keep theirs.
        """
        raise NotImplementedError
