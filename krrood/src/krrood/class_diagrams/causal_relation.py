"""
Causal relations between the variables of a class, kept in the class diagram next to the
relations between classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from probabilistic_model.causal_structure.exceptions import UnknownGraphNode
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    EdgeMark,
    GraphEdge,
    GraphNode,
)
from typing_extensions import Iterable

from krrood.class_diagrams.class_diagram import (
    ClassDiagram,
    ClassRelation,
    WrappedClass,
)
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    Call,
    MappedVariable,
)

# %% what a causal relation connects


@dataclass(kw_only=True, frozen=True)
class CausalEndpoint(ABC):
    """
    One end of a causal relation: a variable of the class the relation belongs to.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: The name of the variable, as a query's parameters name it.
        """


@dataclass(kw_only=True, frozen=True)
class AttributeEndpoint(CausalEndpoint):
    """
    An attribute reached from the class through a chain of fields. Through a field that
    holds a collection, it is that attribute of every part in the collection.
    """

    fields: tuple[WrappedField, ...]
    """
    The fields from the class to the attribute, the first one declared by the class
    itself.
    """

    @classmethod
    def along(
        cls, *, class_diagram: ClassDiagram, access_path: Iterable[MappedVariable]
    ) -> AttributeEndpoint:
        """
        :param class_diagram: The diagram holding every class on the path.
        :param access_path: The steps of a symbolic attribute, from the class on.
        :return: The attribute the steps reach.
        :raises ClassIsUnMappedInClassDiagram: If a class on the path is not in the
            diagram.
        :raises FieldIsUnMappedInClassDiagram: If a step names a field its class does
            not have.
        """
        return cls(
            fields=tuple(
                class_diagram.get_wrapped_class(clazz=step._owner_class_).get_field(
                    name=step._attribute_name_
                )
                for step in access_path
                if isinstance(step, Attribute)
            )
        )

    @property
    def name(self) -> str:
        return ".".join(
            [self.fields[0].clazz.name]
            + [wrapped_field.name for wrapped_field in self.fields]
        )


@dataclass(kw_only=True, frozen=True)
class StatisticEndpoint(CausalEndpoint):
    """
    An aggregation statistic over the parts in one of the collections of the class, or
    of every part in one of its collections.
    """

    statistic: Call
    """
    The symbolic call of the statistic method.
    """

    collection: WrappedField | None = None
    """
    The field of the class holding the parts the statistic is computed for, or
    ``None`` if it is computed for the class itself.
    """

    @property
    def name(self) -> str:
        if self.collection is None:
            return self.statistic._name_
        return ".".join(
            [self.collection.clazz.name, self.collection.name, self.statistic._name_]
        )


# %% the relation


@dataclass(kw_only=True, eq=False)
class CausalRelation(ClassRelation):
    """
    A causal relation between two variables of one class: a loop on that class in the
    class diagram, which shows at each end whether that end is caused.
    """

    first: CausalEndpoint
    """
    The variable at one end.
    """

    second: CausalEndpoint
    """
    The variable at the other end.
    """

    mark_at_first: EdgeMark
    """
    What the relation shows at :attr:`first`.
    """

    mark_at_second: EdgeMark
    """
    What the relation shows at :attr:`second`.
    """

    def edge_among(self, *, node_of: dict[str, GraphNode]) -> GraphEdge:
        """
        :param node_of: The node of every variable, by the variable's name.
        :return: The relation as an edge between the nodes of its two variables.
        :raises UnknownGraphNode: If no node stands for one of them.
        """
        for endpoint in (self.first, self.second):
            if endpoint.name not in node_of:
                raise UnknownGraphNode(endpoint.name, list(node_of))
        return GraphEdge(
            node_of[self.first.name],
            node_of[self.second.name],
            self.mark_at_first,
            self.mark_at_second,
        )

    def __str__(self):
        return f"causal({self.first.name}, {self.second.name})"


# %% the causal relations of one class


@dataclass(kw_only=True)
class CausalRelationsOfClass:
    """
    The causal relations a class diagram holds for one class.
    """

    class_diagram: ClassDiagram
    """
    The diagram holding the relations.
    """

    owner: WrappedClass
    """
    The class whose variables the relations connect.
    """

    @property
    def relations(self) -> tuple[CausalRelation, ...]:
        """
        :return: The causal relations of the class.
        """
        return tuple(
            relation
            for relation in self.class_diagram.get_out_edges(cls=self.owner)
            if isinstance(relation, CausalRelation)
        )

    def replace_relations(self, *, relations: Iterable[CausalRelation]) -> None:
        """
        Replace every causal relation of the class.

        :param relations: The new relations, each a loop on :attr:`owner`.
        """
        for relation in self.relations:
            self.class_diagram.remove_relation(relation=relation)
        for relation in relations:
            self.class_diagram.add_relation(relation=relation)

    def causal_graph(self, *, nodes: list[GraphNode]) -> CausalGraph:
        """
        :param nodes: A node for each variable the relations connect, and any others.
        :return: The graph over the nodes, with one edge per relation.
        :raises UnknownGraphNode: If a relation connects a variable no node stands for.
        """
        node_of = {node.name: node for node in nodes}
        return CausalGraph.from_edges(
            nodes, [relation.edge_among(node_of=node_of) for relation in self.relations]
        )
