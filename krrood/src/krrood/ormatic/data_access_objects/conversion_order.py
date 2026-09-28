from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Type, TYPE_CHECKING

import rustworkx

from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.exceptions import ConversionOrderCycle

if TYPE_CHECKING:
    from krrood.entity_query_language.core.mapped_variable import MappedVariable


# %% what asks for an order


@dataclass(frozen=True)
class ConversionOrderConstraint(ABC):
    """
    One reason why an alternative mapping type is converted before another.
    """

    earlier: Type[AlternativeMapping]
    """
    The mapping type that is converted first.
    """

    later: Type[AlternativeMapping]
    """
    The mapping type that is converted after it.
    """

    @abstractmethod
    def description(self) -> str:
        """
        :return: What this constraint orders and where it comes from.
        """

    @abstractmethod
    def suggest_correction(self) -> str:
        """
        :return: Advice on how to drop this constraint.
        """


@dataclass(frozen=True)
class DeclaredOrder(ConversionOrderConstraint):
    """
    An order the later mapping asks for in its
    :func:`AlternativeMapping.required_pre_build_classes`.
    """

    def description(self) -> str:
        return (
            f"{self.earlier.__name__} before {self.later.__name__}, declared by "
            f"{self.later.__name__}."
            f"{AlternativeMapping.required_pre_build_classes.__name__}"
        )

    def suggest_correction(self) -> str:
        return (
            f"drop {self.earlier.original_class()} from {self.later.__name__}."
            f"{AlternativeMapping.required_pre_build_classes.__name__}"
        )


@dataclass(frozen=True)
class HoldingOrder(ConversionOrderConstraint):
    """
    An order that holding one mapping inside another asks for.

    A mapping builds its domain object out of what it holds, so a mapping it still holds
    at that point ends up inside that domain object instead of the domain object of the
    held mapping.
    """

    def description(self) -> str:
        return f"{self.earlier.__name__} before {self.later.__name__}, which holds it"

    def suggest_correction(self) -> str:
        return (
            f"declare {self.later.original_class()} in {self.earlier.__name__}."
            f"{AlternativeMapping.required_pre_build_classes.__name__} to convert "
            f"{self.later.__name__} first instead"
        )


# %% the order of one conversion


@dataclass
class ConversionOrder:
    """
    The order the alternative mappings of a single conversion are converted in.
    """

    declared_dependencies: rustworkx.PyDiGraph
    """
    The mapping types of the conversion and the orders declared in their
    :func:`AlternativeMapping.required_pre_build_classes`.
    """

    references: Dict[AlternativeMapping, List[Tuple[Any, MappedVariable]]]
    """
    Every occurrence of an alternative mapping instance inside another object, keyed by
    the mapping instance being referenced.
    """

    graph: rustworkx.PyDiGraph = field(init=False)
    """
    The declared orders together with the orders holding asks for, where an edge
    ``(source, target)`` carries the constraint putting ``source`` before ``target``.
    """

    def __post_init__(self):
        self.graph = self.declared_dependencies.copy()
        index_of_type = {
            self.graph[index]: index for index in self.graph.node_indices()
        }
        for held, holder in self._held_and_holder_types():
            self._order_held_before_holder(held, holder, index_of_type)

    def _held_and_holder_types(self) -> Set[Tuple[Type, Type]]:
        """
        The distinct pairs of the type of a held mapping and the type holding it.

        The types are what the order is asked of, so a conversion holding the same pair
        a thousand times asks once.

        :return: Pairs of the held type and the type holding it.
        """
        held_and_holder_types = set()
        for held, references_to_held in self.references.items():
            held_type = type(held)
            for holder, _ in references_to_held:
                held_and_holder_types.add((held_type, type(holder)))
        return held_and_holder_types

    def _order_held_before_holder(
        self,
        held: Type[AlternativeMapping],
        holder: Type,
        index_of_type: Dict[Type, int],
    ):
        """
        Add the order holding asks for, unless what is declared already asks for the
        opposite.

        A declaration is what the author of the mappings decided, so it wins and the
        held mapping is left to be converted afterwards.

        :param held: The type of the mapping being held.
        :param holder: The type of whatever holds it, which is only ordered when it is
            an alternative mapping itself.
        :param index_of_type: The node of every mapping type of the conversion.
        """
        if held is holder or not issubclass(holder, AlternativeMapping):
            return
        held_index = index_of_type.get(held)
        holder_index = index_of_type.get(holder)
        if held_index is None or holder_index is None:
            return
        if rustworkx.has_path(self.declared_dependencies, holder_index, held_index):
            return
        if self.graph.has_edge(held_index, holder_index):
            return
        self.graph.add_edge(held_index, holder_index, HoldingOrder(held, holder))

    def sort_types(self) -> List[Type[AlternativeMapping]]:
        """
        Put the mapping types of the conversion in the order they are converted in.

        :raises ConversionOrderCycle: If the constraints leave no such order.
        :return: The mapping types, every one of them after the ones it waits for.
        """
        cycle = rustworkx.digraph_find_cycle(self.graph)
        if cycle:
            raise ConversionOrderCycle(
                [self.graph.get_edge_data(source, target) for source, target in cycle]
            )
        return [self.graph[index] for index in rustworkx.topological_sort(self.graph)]
