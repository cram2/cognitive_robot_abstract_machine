from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from random_events.product_algebra import Event
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Generic, Iterable, Self, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

from probabilistic_model.exceptions import (
    EmptyCompositionError,
    MixtureOverDifferentVariablesError,
    NumberOfWeightsMismatchError,
    ProductOverSharedVariablesError,
)
from probabilistic_model.probabilistic_model import ProbabilisticModel

NodeType = TypeVar("NodeType")
"""
How a representation refers to one node of a circuit it holds.
"""

RepresentationType = TypeVar(
    "RepresentationType", bound="ProbabilisticCircuitRepresentation"
)
"""
The representation a circuit is read from and answered in.
"""

# %% what the mixtures of a circuit look like


@dataclass
class UnnormalizedMixture(Generic[NodeType], SubClassSafeGeneric):
    """
    A mixture of a circuit whose component weights do not sum to one.
    """

    mixture: NodeType
    """
    The mixture, as the representation it was read from refers to its nodes.
    """

    log_weight_sum: float
    """
    What the logarithmic weights of its components sum to, instead of zero.
    """


@dataclass
class MixtureOverVariable(Generic[NodeType], SubClassSafeGeneric):
    """
    One mixture of a circuit, seen through one variable it models.
    """

    mixture: NodeType
    """
    The mixture, as the representation it was read from refers to its nodes.
    """

    variable: Variable
    """
    The variable the supports are restricted to.
    """

    component_supports: list[Event]
    """
    What every component of the mixture models of the variable.
    """


# %% the representation


@dataclass(eq=False)
class ProbabilisticCircuitRepresentation(
    ProbabilisticModel, Generic[NodeType], SubClassSafeGeneric, ABC
):
    """
    A way of representing a probabilistic circuit, which composes circuits of its own
    kind.

    A circuit can be held as a graph of units or as layers of arrays. Both answer the
    same queries and both can be mixed and multiplied, so what they have in common is
    declared here and a composition answers in the representation it was given.
    """

    @staticmethod
    def validate_mixture(
        components: list[ProbabilisticCircuitRepresentation],
        log_weights: list[float],
    ) -> None:
        """
        Check that circuits and weights make up a smooth mixture.

        :param components: The components to mix.
        :param log_weights: The logarithmic weight of every component.
        :raises EmptyCompositionError: If there is no component.
        :raises NumberOfWeightsMismatchError: If there is not one weight per component.
        :raises MixtureOverDifferentVariablesError: If the components are not over the
            same variables.
        """
        if not components:
            raise EmptyCompositionError(ProbabilisticCircuitRepresentation.mixture_of)
        if len(log_weights) != len(components):
            raise NumberOfWeightsMismatchError(len(log_weights), len(components))
        variables_per_component = [
            list(component.variables) for component in components
        ]
        if any(
            variables != variables_per_component[0]
            for variables in variables_per_component[1:]
        ):
            raise MixtureOverDifferentVariablesError(variables_per_component)

    @staticmethod
    def validate_product(
        factors: list[ProbabilisticCircuitRepresentation],
    ) -> None:
        """
        Check that circuits make up a decomposable product.

        :param factors: The factors to multiply.
        :raises EmptyCompositionError: If there is no factor.
        :raises ProductOverSharedVariablesError: If several factors are over one
            variable.
        """
        if not factors:
            raise EmptyCompositionError(ProbabilisticCircuitRepresentation.product_of)
        seen = SortedSet()
        shared = SortedSet()
        for factor in factors:
            shared |= seen & SortedSet(factor.variables)
            seen |= SortedSet(factor.variables)
        if shared:
            raise ProductOverSharedVariablesError(list(shared))

    @classmethod
    @abstractmethod
    def mixture_of(cls, components: list[Self], log_weights: Iterable[float]) -> Self:
        """
        Mix copies of circuits over the same variables, normalized.

        The components stay as they are.

        :param components: The components to mix, at least one.
        :param log_weights: The logarithmic weight of every component.
        :return: The mixture.
        :raises EmptyCompositionError: If there is no component.
        :raises NumberOfWeightsMismatchError: If there is not one weight per component.
        :raises MixtureOverDifferentVariablesError: If the components are not over the
            same variables.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def product_of(cls, factors: list[Self]) -> Self:
        """
        Multiply copies of circuits over disjoint variables.

        The factors stay as they are.

        :param factors: The factors to multiply, at least one.
        :return: The product.
        :raises EmptyCompositionError: If there is no factor.
        :raises ProductOverSharedVariablesError: If several factors are over one
            variable.
        """
        raise NotImplementedError

    # %% reading the mixtures

    @abstractmethod
    def unnormalized_mixtures(
        self, tolerance: float = 1e-6
    ) -> list[UnnormalizedMixture[NodeType]]:
        """
        Find the mixtures whose component weights do not sum to one.

        A mixture of no component counts as normalized.

        :param tolerance: How far the logarithmic weights of a mixture may sum away from
            zero.
        :return: Every mixture of the circuit that is not normalized.
        """
        raise NotImplementedError

    @abstractmethod
    def mixtures_over(
        self, variables: SortedSet[Variable]
    ) -> list[MixtureOverVariable[NodeType]]:
        """
        Read what every component of every mixture models of some variables.

        :param variables: The variables to read the mixtures through.
        :return: Every mixture of the circuit that mixes at least two components, once
            per variable it models.
        """
        raise NotImplementedError
