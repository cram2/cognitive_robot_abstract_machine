"""
The alternative mappings that persist a
:class:`~krrood.symbol_graph.symbol_graph.SymbolGraph` and the instances it wraps.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Self

from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.symbol_graph.symbol_graph import (
    PredicateClassRelation,
    Symbol,
    SymbolGraph,
    WrappedInstance,
)


@dataclass(kw_only=True, eq=False)
class SymbolGraphMapping(AlternativeMapping[SymbolGraph]):
    """
    Mapping specific for SymbolGraph.

    Import this class when you want to persist SymbolGraph.
    """

    instances: list[WrappedInstance]
    """
    The instances the graph wraps.
    """

    predicate_relations: list[PredicateClassRelation]
    """
    The relations between the wrapped instances.
    """

    @classmethod
    def from_domain_object(cls, *, obj: SymbolGraph) -> Self:
        return cls(
            instances=obj.instances,
            predicate_relations=obj.predicate_relations,
        )

    def to_domain_object(self) -> SymbolGraph:
        result = SymbolGraph()
        for instance in self.instances:
            result.add_instance(wrapped_instance=instance)
        for relation in self.predicate_relations:
            result.add_relation(relation=relation)
        return result


@dataclass(kw_only=True, eq=False)
class WrappedInstanceMapping(AlternativeMapping[WrappedInstance]):
    """
    Mapping specific for a :class:`~krrood.symbol_graph.symbol_graph.WrappedInstance`,
    which persists only the symbol it wraps.
    """

    instance: Symbol | None
    """
    The wrapped symbol.
    """

    @classmethod
    def from_domain_object(cls, *, obj: WrappedInstance) -> Self:
        return cls(instance=obj.instance)

    def to_domain_object(self) -> WrappedInstance:
        return WrappedInstance(instance=self.instance)
