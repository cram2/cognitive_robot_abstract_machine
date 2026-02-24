from __future__ import annotations

import os
import weakref
from collections import defaultdict
from dataclasses import dataclass, field, InitVar

from rustworkx import PyDiGraph
from typing_extensions import (
    Any,
    Iterable,
    Optional,
    List,
    Type,
    Dict,
    DefaultDict,
    Callable,
    ClassVar
)

from .. import logger
from ..class_diagrams import ClassDiagram
from ..class_diagrams.wrapped_field import WrappedField
from ..ontomatic.property_descriptor.attribute_introspector import (
    DescriptorAwareIntrospector,
)
from ..singleton import SingletonMeta
from ..utils import recursive_subclasses


@dataclass(unsafe_hash=True)
class PredicateClassRelation:
    """
    Edge data representing a predicate-based relation between two wrapped instances.

    The relation carries a flag indicating whether it was inferred or added directly.
    """

    source: WrappedInstance
    """
    The source of the predicate
    """
    target: WrappedInstance
    """
    The target of the predicate
    """
    wrapped_field: WrappedField
    """
    The dataclass field in the source class that represents this relation with the target.
    """
    inferred: bool = False
    """
    Whether it was inferred or not.
    """

    def __post_init__(self):
        self.source = SymbolGraph().ensure_wrapped_instance(self.source)
        self.target = SymbolGraph().ensure_wrapped_instance(self.target)

    def add_to_graph(self) -> bool:
        """
        Add the relation to the graph.

        :return: True if the relation was newly added, False if it already existed.
        """
        return SymbolGraph().add_relation(self)

    def __str__(self):
        """Return the predicate type name for labeling the edge."""
        return self.__class__.__name__

    @property
    def color(self) -> str:
        return "red" if self.inferred else "black"


@dataclass
class WrappedInstance:
    """
    A node wrapper around a concrete Symbol instance used in the instance graph.
    """

    instance: InitVar[Symbol]
    """
    The instance to wrap. Only passed as initialization variable.
    """

    instance_reference: weakref.ReferenceType[Symbol] = field(init=False, default=None)
    """
    A weak reference to the symbol instance this wraps.
    """

    index: Optional[int] = field(init=False, default=None)
    """
    Index in the instance graph of the symbol graph that manages this object.
    """

    _symbol_graph_: Optional[SymbolGraph] = field(
        init=False, hash=False, default=None, repr=False
    )
    """
    The symbol graph that manages this object.
    """

    inferred: bool = False
    """
    Rather is instance was inferred or constructed.
    """

    instance_type: Type[Symbol] = field(init=False, default=None)
    """
    The type of the instance.
    This is needed to clean it up from the cache after the instance reference died.
    """

    def __post_init__(self, instance: Symbol):
        self.instance_reference = weakref.ref(instance)
        self.instance_type = type(instance)

    @property
    def instance(self) -> Optional[Symbol]:
        """
        :return: The symbol that is referenced to. Can return None if this symbol is garbage collected already.
        """
        return self.instance_reference()

    @property
    def name(self):
        """Return a unique display name composed of class name and node index."""
        return self.instance.__class__.__name__ + str(self.index)

    @property
    def color(self) -> str:
        return "red" if self.inferred else "black"

    def __eq__(self, other):
        return (
            self.instance == other.instance
            if self.instance is not None and other.instance is not None
            else False
        )

    def __hash__(self):
        return id(self.instance)


@dataclass
class SymbolGraph(metaclass=SingletonMeta):
    """
    A singleton combination of a class and instance diagram.
    This class tracks the life cycles `Symbol` instance created in the python process.
    Furthermore, relations between instances are also tracked.

    Relations are represented as edges where each edge has a relation object attached to it. The relation object
    contains also the Predicate object that represents the relation.

    The construction of this object will do nothing if a singleton instance of this already exists.
    Make sure to call `clear()` before constructing this object if you want a new one.
    """

    _class_diagram: ClassDiagram = field(default=None)
    """
    The class diagram of all registered classes.
    """

    _instance_graph: PyDiGraph[WrappedInstance, PredicateClassRelation] = field(
        default_factory=PyDiGraph, init=False
    )
    """
    A directed graph that stores all instances of `Symbol` and how they relate to each other.
    """

    _instance_index: Dict[int, WrappedInstance] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Dictionary that maps the ids of objects to wrapped instances.
    Used for faster access when only the WrappedInstance.instance is available.
    """

    _class_to_wrapped_instances: DefaultDict[Type, List[WrappedInstance]] = field(
        init=False, default_factory=lambda: defaultdict(list)
    )
    """
    A dictionary that sorts the wrapped instances by the type inside them.
    This enables quick behavior similar to selecting everything from an entire table in SQL.
    """

    _relation_index: Dict[WrappedField, set[tuple[int, int]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self):
        if self._class_diagram is None:
            # fetch all symbols and construct the graph
            self._class_diagram = ClassDiagram(
                list(recursive_subclasses(Symbol)),
                introspector=DescriptorAwareIntrospector(),
            )

    @property
    def class_diagram(self) -> ClassDiagram:
        return self._class_diagram

    def add_node(self, wrapped_instance: WrappedInstance):
        """
        Add a wrapped instance to the cache.

        :param wrapped_instance: The instance to add.
        """
        wrapped_instance.index = self._instance_graph.add_node(wrapped_instance)
        wrapped_instance._symbol_graph_ = self
        self._instance_index[id(wrapped_instance.instance)] = wrapped_instance
        self._class_to_wrapped_instances[wrapped_instance.instance_type].append(
            wrapped_instance
        )

    def remove_node(self, wrapped_instance: WrappedInstance):
        """
        Remove a wrapped instance from the cache.

        :param wrapped_instance: The instance to remove.
        """
        self._instance_index.pop(id(wrapped_instance.instance), None)
        self._class_to_wrapped_instances[wrapped_instance.instance_type].remove(
            wrapped_instance
        )
        self._instance_graph.remove_node(wrapped_instance.index)

    def remove_dead_instances(self):
        for node in self._instance_graph.nodes():
            if node.instance is None:
                self.remove_node(node)

    def get_instances_of_type(self, type_: Type[Symbol]) -> Iterable[Symbol]:
        """
        Get all wrapped instances of the given type and all its subclasses.

        :param type_: The symbol type to look for
        :return: All wrapped instances that refer to an instance of the given type.
        """
        yield from (
            instance.instance
            for cls in [type_] + recursive_subclasses(type_)
            for instance in list(self._class_to_wrapped_instances[cls])
        )

    def get_wrapped_instance(self, instance: Any) -> Optional[WrappedInstance]:
        if isinstance(instance, WrappedInstance):
            return instance
        return self._instance_index.get(id(instance), None)

    def ensure_wrapped_instance(self, instance: Any) -> WrappedInstance:
        """
        Ensures that the given instance is wrapped into a `WrappedInstance`. If the
        instance is not already wrapped, creates a new `WrappedInstance` object and
        adds it as a node. Returns the wrapped instance.

        :param instance: The object to be checked and wrapped if necessary.:
        :return: WrappedInstance: The wrapped object.
        """
        wrapped_instance = self.get_wrapped_instance(instance)
        if wrapped_instance is None:
            wrapped_instance = WrappedInstance(instance)
            self.add_node(wrapped_instance)
        return wrapped_instance

    def clear(self) -> None:
        SingletonMeta.clear_instance(type(self))

    # Adapters to align with ORM alternative mapping expectations
    def add_instance(self, wrapped_instance: WrappedInstance) -> None:
        """Add a wrapped instance to the graph.

        This is an adapter that delegates to add_node to keep API compatibility with
        SymbolGraphMapping.create_from_dao.
        """
        self.add_node(wrapped_instance)

    def add_relation(self, relation: PredicateClassRelation) -> bool:
        """Add a relation edge to the instance graph."""
        if self.relation_exists(relation):
            return False
        self._instance_graph.add_edge(
            relation.source.index, relation.target.index, relation
        )
        if relation.wrapped_field not in self._relation_index:
            self._relation_index[relation.wrapped_field] = set()
        self._relation_index[relation.wrapped_field].add(
            (relation.source.index, relation.target.index)
        )
        return True

    def relation_exists(self, relation: PredicateClassRelation) -> bool:
        return (
            relation.source.index,
            relation.target.index,
        ) in self._relation_index.get(relation.wrapped_field, set())

    def relations(self) -> Iterable[PredicateClassRelation]:
        yield from self._instance_graph.edges()

    @property
    def wrapped_instances(self) -> List[WrappedInstance]:
        return self._instance_graph.nodes()

    def get_incoming_relations_with_type(
        self,
        wrapped_instance: WrappedInstance,
        relation_type: Type[PredicateClassRelation],
    ) -> Iterable[PredicateClassRelation]:
        """
        Get all relations with the given type that are incoming to the given wrapped instance.

        :param wrapped_instance: The wrapped instance to get the relations from.
        :param relation_type: The type of the relation to filter for.
        """
        yield from self.get_incoming_relations_with_condition(
            wrapped_instance, lambda edge: isinstance(edge, relation_type)
        )

    def get_incoming_relations_with_condition(
        self,
        wrapped_instance: WrappedInstance,
        edge_condition: Callable[[PredicateClassRelation], bool],
    ) -> Iterable[PredicateClassRelation]:
        """
        Get all relations with the given condition that are incoming to the given wrapped instance.

        :param wrapped_instance: The wrapped instance to get the relations from.
        :param edge_condition: The condition to filter for.
        """
        yield from filter(edge_condition, self.get_incoming_relations(wrapped_instance))

    def get_incoming_relations(
        self,
        wrapped_instance: WrappedInstance,
    ) -> Iterable[PredicateClassRelation]:
        """
        Get all relations incoming to the given wrapped instance.

        :param wrapped_instance: The wrapped instance to get the relations from.
        """
        wrapped_instance = self.get_wrapped_instance(wrapped_instance)
        if not wrapped_instance:
            return
        yield from (
            edge for _, _, edge in self._instance_graph.in_edges(wrapped_instance.index)
        )

    def get_outgoing_relations_with_type(
        self,
        wrapped_instance: WrappedInstance,
        relation_type: Type[PredicateClassRelation],
    ) -> Iterable[PredicateClassRelation]:
        """
        Get all relations with the given type that are outgoing from the given wrapped instance.

        :param wrapped_instance: The wrapped instance to get the relations from.
        :param relation_type: The type of the relation to filter for.
        """
        yield from self.get_outgoing_relations_with_condition(
            wrapped_instance, lambda edge: isinstance(edge, relation_type)
        )

    def get_outgoing_relations_with_condition(
        self,
        wrapped_instance: WrappedInstance,
        edge_condition: Callable[[PredicateClassRelation], bool],
    ) -> Iterable[PredicateClassRelation]:
        """
        Get all relations with the given condition that are outgoing from the given wrapped instance.

        :param wrapped_instance: The wrapped instance to get the relations from.
        :param edge_condition: The condition to filter for.
        """
        yield from filter(edge_condition, self.get_outgoing_relations(wrapped_instance))

    def get_outgoing_relations(
        self,
        wrapped_instance: WrappedInstance,
    ) -> Iterable[PredicateClassRelation]:
        """
        Get all relations outgoing from the given wrapped instance.

        :param wrapped_instance: The wrapped instance to get the relations from.
        """
        wrapped_instance = self.get_wrapped_instance(wrapped_instance)
        if not wrapped_instance:
            return
        yield from (
            edge
            for _, _, edge in self._instance_graph.out_edges(wrapped_instance.index)
        )

    def to_dot(
        self,
        filepath: str,
        format_="svg",
        graph_type="instance",
        without_inherited_associations: bool = True,
    ) -> None:
        """
        Generate a dot file from the instance graph, requires graphviz and pydot libraries.

        :param filepath: The path to the dot file.
        :param format_: The format of the dot file (svg, png, ...).
        :param graph_type: The type of the graph to generate (instance, type).
        :param without_inherited_associations: Whether to include inherited associations in the graph.
        """
        import pydot

        if graph_type == "type":
            if without_inherited_associations:
                graph = self.class_diagram.to_subdiagram_without_inherited_associations(
                    True
                )._dependency_graph
            else:
                graph = self.class_diagram._dependency_graph
        else:
            graph = self._instance_graph
        if not filepath.endswith(f".{format_}"):
            filepath += f".{format_}"
        dot_str = graph.to_dot(
            lambda node: dict(
                color="black",
                fillcolor="lightblue",
                style="filled",
                label=node.name,
            ),
            lambda edge: dict(color=edge.color, style="solid", label=str(edge)),
            dict(rankdir="LR"),
        )
        dot = pydot.graph_from_dot_data(dot_str)[0]
        try:
            dot.write(filepath, format=format_)
        except FileNotFoundError:
            tmp_filepath = filepath.replace(f".{format_}", ".dot")
            dot.write(tmp_filepath, format="raw")
            try:
                os.system(f"/usr/bin/dot -T{format_} {tmp_filepath} -o {filepath}")
                os.remove(tmp_filepath)
            except Exception as e:
                logger.error(e)


@dataclass(eq=False)
class Symbol:
    """Base class for things that can be cached in the symbol graph."""

    _cache_instances_: ClassVar[bool] = True
    """
    Whether instances of this class should be cached or not in the symbol graph.
    """

    def __new__(cls, *args, **kwargs):
        """
        Updates the cache with the given instance of a symbolic type.
        """
        instance = super().__new__(cls)
        if cls._cache_instances_:
            SymbolGraph().add_node(WrappedInstance(instance))
        return instance
