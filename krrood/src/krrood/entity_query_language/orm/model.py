from dataclasses import dataclass

from typing_extensions import List, Optional

from ..predicate import Symbol
from ..symbol_graph import SymbolGraph, WrappedInstance, PredicateClassRelation
from ...ormatic.dao import AlternativeMapping, T


@dataclass
class SymbolGraphMapping(AlternativeMapping[SymbolGraph]):
    """
    Mapping specific for SymbolGraph.
    Import this class when you want to persist SymbolGraph.
    """

    instances: List[WrappedInstance]

    predicate_relations: List[PredicateClassRelation]

    @classmethod
    def from_domain_object(cls, obj: SymbolGraph):
        return cls(
            instances=obj.wrapped_instances,
            predicate_relations=list(obj.relations()),
        )

    def to_domain_object(self) -> T:
        result = SymbolGraph()
        for instance in self.instances:
            result.add_instance(instance)
        for relation in self.predicate_relations:
            result.add_relation(relation)
        return result


@dataclass
class WrappedInstanceMapping(AlternativeMapping[WrappedInstance]):
    instance: Optional[Symbol]

    @classmethod
    def from_domain_object(cls, obj: WrappedInstance):
        return cls(obj.instance)

    def to_domain_object(self) -> T:
        return WrappedInstance(self.instance)
