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
    def create_instance(cls, obj: SymbolGraph):
        return cls(
            instances=obj.wrapped_instances,
            predicate_relations=list(obj.relations()),
        )

    def create_from_dao(self) -> T:
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
    def create_instance(cls, obj: WrappedInstance):
        return cls(obj.instance)

    def create_from_dao(self) -> T:
        return WrappedInstance(self.instance)
