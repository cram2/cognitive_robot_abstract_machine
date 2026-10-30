from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from random_events.product_algebra import VariableMap
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Any, Generic, Iterable, Self, TypeVar

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    DenseEdgeValues,
)
from probabilistic_model.probabilistic_circuit.probabilistic_circuit_representation import (
    NodeType,
    ProbabilisticCircuitRepresentation,
)
from probabilistic_model.probabilistic_model import PartialPointType, ProbabilisticModel
from probabilistic_model.query_results import LogMode

"""
The type of the nodes of a circuit type.
"""


@dataclass
class ConditionedCopies:
    """
    Copies of a circuit, one per point of a :class:`ProductsOfCopies`, each conditioned
    on its point and restricted to some variables of the circuit, which are renamed.
    """

    circuit: GroundableCircuit
    """
    The circuit to copy. Copies of the same circuit have the same structure.
    """

    variables: SortedSet[Variable]
    """
    The variables of the circuit the copies keep.
    """

    renaming: dict[Variable, Variable]
    """
    The new name of every kept variable.
    """


@dataclass
class ProductsOfCopies:
    """
    Products whose ``i``-th one multiplies the ``i``-th copy of every factor.
    """

    points: list[PartialPointType]
    """
    The point every product conditions its conditioned copies on, one per product.
    """

    conditioned_copies: list[ConditionedCopies]
    """
    The conditioned copies every product multiplies.
    """

    point_masses: dict[Variable, list[Any]] = field(default_factory=dict)
    """
    Variables every product multiplies a point mass of, with the value of the point
    mass in every product.
    """

    components: list[GroundableCircuit] = field(default_factory=list)
    """
    A circuit every product multiplies, one per product, or nothing.
    """


@dataclass(eq=False)
class GroundableCircuit(ProbabilisticCircuitRepresentation[NodeType], ABC):
    """
    A circuit type that relational probabilistic circuits are fitted and grounded in.
    """

    @property
    @abstractmethod
    def variable_to_index_map(self) -> dict[Variable, int]:
        """
        :return: The index of every variable in the variables of the circuit.
        """
        raise NotImplementedError

    @abstractmethod
    def log_mode(self, check_determinism: bool = True) -> LogMode:
        """
        :param check_determinism: Whether to check that the circuit is deterministic.
        :return: The mode of the circuit and its log-likelihood.
        """
        raise NotImplementedError

    @abstractmethod
    def marginal_in_place(self, variables: Iterable[Variable]) -> Self | None:
        """
        Marginalize the circuit to the variables in place.

        :param variables: The variables to keep.
        :return: The marginalized circuit, or ``None`` if it keeps no variable.
        """
        raise NotImplementedError

    @abstractmethod
    def update_variables(self, new_variables: VariableMap):
        """
        Replace variables of the circuit in place.

        :param new_variables: The new variable of every variable to replace.
        """
        raise NotImplementedError

    # %% operations on nodes

    @abstractmethod
    def condition_on_point_in_place(self, point: PartialPointType) -> bool:
        """
        Condition the circuit on a point in place, keeping its structure. Variables of
        the point that the circuit does not model are ignored.

        :param point: The point.
        :return: Whether the point is possible. An impossible point leaves the circuit
            unchanged.
        """
        raise NotImplementedError

    @abstractmethod
    def lowest_product_nodes_over(
        self, variables: SortedSet[Variable]
    ) -> list[NodeType]:
        """
        :param variables: The variables every returned node must model.
        :return: The product nodes that model all of the variables and have no such
            node below them, so none of them is an ancestor of another.
        """
        raise NotImplementedError

    @abstractmethod
    def subcircuit_at(self, node: NodeType) -> Self:
        """
        :param node: A node of the circuit.
        :return: The circuit rooted at the node.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_variables_in_place(
        self, variables: SortedSet[Variable], nodes: list[NodeType]
    ) -> list[NodeType]:
        """
        Marginalize variables out of the circuit in place, keeping its structure.

        :param variables: The variables to remove.
        :param nodes: Nodes of the circuit to keep track of.
        :return: The nodes in the marginalized circuit.
        """
        raise NotImplementedError

    @abstractmethod
    def root_mixture_components(self) -> list[Self]:
        """
        :return: The circuits that the root of the circuit mixes, or nothing if the root
            does not mix.
        """
        raise NotImplementedError

    @abstractmethod
    def multiply_nodes_by_mixtures(
        self,
        nodes: list[NodeType],
        products: ProductsOfCopies,
        log_weights: DenseEdgeValues | None,
    ):
        """
        Make product nodes of the circuit multiply mixtures of products, in place.

        :param nodes: The product nodes.
        :param products: The products.
        :param log_weights: The weight of every product in the mixture of every node,
            shape (#nodes, #products), ``-inf`` where a node does not use a product.
            ``None`` if there is one product, which every node multiplies.
        """
        raise NotImplementedError
