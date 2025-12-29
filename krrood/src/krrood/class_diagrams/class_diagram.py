from __future__ import annotations

import logging
import os
from abc import ABC
from copy import copy
from dataclasses import dataclass
from dataclasses import field as dataclass_field, InitVar
from functools import cached_property, lru_cache

import rustworkx as rx

from .. import logger

try:
    from rustworkx_utils import RWXNode
except ImportError:
    RWXNode = None
from typing_extensions import (
    List,
    Optional,
    Dict,
    Union,
    Tuple,
    Callable,
    Iterable,
    Type,
    TYPE_CHECKING,
    Set,
)


from .attribute_introspector import (
    AttributeIntrospector,
    DataclassOnlyIntrospector,
)
from .utils import Role, get_generic_type_param
from .wrapped_field import WrappedField

from .failures import ClassIsUnMappedInClassDiagram

if TYPE_CHECKING:
    from ..entity_query_language.predicate import PropertyDescriptor


@dataclass
class ClassRelation(ABC):
    """
    Abstract base class representing a relationship between two classes in a UML class diagram.
    """

    source: WrappedClass
    """The source class in the relation."""

    target: WrappedClass
    """The target class in the relation."""

    inferred: bool = dataclass_field(default=False, init=False)
    """
    Whether this relation was inferred (e.g. associations from role takers) or explicitly defined.
    """

    def __str__(self):
        """Return the relation name for display purposes."""
        return f"{self.__class__.__name__}"

    @property
    def color(self) -> str:
        """Default edge color used when visualizing the relation."""
        if self.inferred:
            return "red"
        return "black"


@dataclass
class Inheritance(ClassRelation):
    """
    Represents an inheritance (generalization) relationship in UML.

    This is an "is-a" relationship where the source class inherits from the target class.
    In UML notation, this is represented by a solid line with a hollow triangle pointing to the parent class.
    """

    def __str__(self):
        return f"isSuperClassOf"


@dataclass(unsafe_hash=True)
class Association(ClassRelation):
    """
    Represents a general association relationship between two classes.

    This is the most general form of relationship, indicating that instances of one class
    are connected to instances of another class. In UML notation, this is shown as a solid line.
    """

    field: WrappedField
    """The field in the source class that creates this association with the target class."""

    @cached_property
    def one_to_many(self) -> bool:
        """Whether the association is one-to-many (True) or many-to-one (False)."""
        return self.field.is_one_to_many_relationship and not self.field.is_type_type

    def get_key(self, include_field_name: bool = False) -> tuple:
        """
        A tuple representing the key of the association.
        """
        if include_field_name:
            return (self.__class__, self.target.clazz, self.field.field.name)
        return (self.__class__, self.target.clazz)

    def __str__(self):
        return f"has-{self.field.public_name}"


@dataclass(eq=False)
class HasRoleTaker(Association):
    """
    This is an association between a role and a role taker where the role class contains a role taker field.
    """

    def __str__(self):
        return f"role-taker({self.field.public_name})"


@dataclass(eq=False)
class AssociationThroughRoleTaker(Association):
    """
    This is an association between a role and a role taker where the role taker class contains an association. This
    applies transitively to the role taker's role takers and so on. The path is a list of fields that are traversed to
    get to the target class.
    """

    field: WrappedField = dataclass_field(init=False)
    """
    The last field in the path that is the association to the target class.
    """
    association_path: List[Association]
    """
    The path of associations that are traversed to get to the target class.
    """
    field_path: List[WrappedField] = dataclass_field(init=False)
    """
    The path of fields that are traversed to get to the target class.
    """

    def __post_init__(self):
        self.inferred = True
        self.field_path = [a.field for a in self.association_path]
        self.field = self.field_path[-1]

    @cached_property
    def last_source(self) -> WrappedClass:
        return self.association_path[-1].source


class ParseError(TypeError):
    """
    Error that will be raised when the parser encounters something that can/should not be parsed.

    For instance, Union types
    """

    pass


@dataclass
class WrappedClass:
    """A node wrapper around a Python class used in the class diagram graph."""

    index: Optional[int] = dataclass_field(init=False, default=None)
    clazz: Type
    _class_diagram: Optional[ClassDiagram] = dataclass_field(
        init=False, hash=False, default=None, repr=False
    )
    _wrapped_field_name_map_: Dict[str, WrappedField] = dataclass_field(
        init=False, hash=False, default_factory=dict, repr=False
    )

    @cached_property
    def roles(self) -> Tuple[WrappedClass, ...]:
        """
        A tuple of roles that this class plays, represented by the HasRoleTaker instances.
         There are HasRoleTaker edges connecting the roles to this class.
        """
        return tuple(
            [
                self._class_diagram.role_association_subgraph[n]
                for n, _, _ in self._class_diagram.role_association_subgraph.in_edges(
                    self.index
                )
            ]
        )

    @cached_property
    def fields(self) -> List[WrappedField]:
        """Return wrapped fields discovered by the diagram’s attribute introspector.

        Public names from the introspector are used to index `_wrapped_field_name_map_`.
        """
        try:
            wrapped_fields: list[WrappedField] = []
            if self._class_diagram is None:
                introspector = DataclassOnlyIntrospector()
            else:
                introspector = self._class_diagram.introspector
            discovered = introspector.discover(self.clazz)
            for item in discovered:
                wf = WrappedField(
                    self,
                    item.field,
                    public_name=item.public_name,
                    property_descriptor=item.property_descriptor,
                )
                # Map under the public attribute name
                self._wrapped_field_name_map_[item.public_name] = wf
                wrapped_fields.append(wf)
            return wrapped_fields
        except TypeError as e:
            logging.error(f"Error parsing class {self.clazz}: {e}")
            raise ParseError(e) from e

    @property
    def name(self):
        """Return a unique display name composed of class name and node index."""
        return self.clazz.__name__ + str(self.index)

    def __hash__(self):
        return hash((self.index, self.clazz))


@dataclass
class ClassDiagram:
    """A graph of classes and their relations discovered via attribute introspection."""

    classes: InitVar[List[Type]]

    introspector: AttributeIntrospector = dataclass_field(
        default_factory=DataclassOnlyIntrospector, init=True, repr=False
    )

    _dependency_graph: rx.PyDiGraph[WrappedClass, ClassRelation] = dataclass_field(
        default_factory=rx.PyDiGraph, init=False
    )
    _cls_wrapped_cls_map: Dict[Type, WrappedClass] = dataclass_field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self, classes: List[Type]):
        """Initialize the diagram with the provided classes and build relations."""
        self._dependency_graph = rx.PyDiGraph()
        for clazz in classes:
            self.add_node(WrappedClass(clazz=clazz))
        self._create_all_relations()

    def get_associations_with_condition(
        self,
        clazz: Union[Type, WrappedClass],
        condition: Callable[[Association], bool],
    ) -> Iterable[Association]:
        """
        Get all associations that match the condition.

        :param clazz: The source class or wrapped class for which outgoing edges are to be retrieved.
        :param condition: The condition to filter relations by.
        """
        for relation in self.get_outgoing_relations(clazz):
            if isinstance(relation, Association) and condition(relation):
                yield relation

    def get_associations_through_role_taker_with_condition(
        self,
        clazz: Union[Type, WrappedClass],
        condition: Callable[[AssociationThroughRoleTaker], bool],
    ) -> Iterable[AssociationThroughRoleTaker]:
        """
        Get all associations through role takers that match the condition.

        :param clazz: The source class or wrapped class for which outgoing edges are to be retrieved.
        :param condition: The condition to filter relations by.
        """
        for relation in self.get_outgoing_relations(clazz):
            if isinstance(relation, AssociationThroughRoleTaker) and condition(
                relation
            ):
                yield relation

    def get_outgoing_relations(
        self,
        clazz: Union[Type, WrappedClass],
    ) -> Iterable[ClassRelation]:
        """
        Get all outgoing edge relations of the given class.

        :param clazz: The source class or wrapped class for which outgoing edges are to be retrieved.
        """
        wrapped_cls = self.get_wrapped_class(clazz)
        yield from self.get_out_edges(wrapped_cls)

    @lru_cache(maxsize=None)
    def get_common_role_taker_associations(
        self, cls1: Union[Type, WrappedClass], cls2: Union[Type, WrappedClass]
    ) -> Tuple[Optional[HasRoleTaker], Optional[HasRoleTaker]]:
        """Return pair of role-taker associations if both classes point to the same target.

        The method checks whether both classes have a HasRoleTaker association to the
        same target class and returns the matching associations, otherwise ``(None, None)``.
        """
        cls1 = self.get_wrapped_class(cls1)
        cls2 = self.get_wrapped_class(cls2)
        assoc1 = self.get_role_taker_associations_of_cls(cls1)
        if not assoc1:
            return None, None
        target_1 = assoc1.target
        for _, _, assoc2 in self._dependency_graph.in_edges(target_1.index):
            if not isinstance(assoc2, HasRoleTaker):
                continue
            if assoc2.source.clazz != cls2.clazz:
                continue
            if assoc2.field.is_role_taker:
                return assoc1, assoc2
        return None, None

    @lru_cache(maxsize=None)
    def get_role_taker_associations_of_cls(
        self, cls: Union[Type, WrappedClass]
    ) -> Optional[HasRoleTaker]:
        """Return the role-taker association of a class if present.

        A role taker is a field that is a one-to-one relationship and is not optional.
        """
        cls = self.get_wrapped_class(cls)
        for assoc in self.get_out_edges(cls):
            if isinstance(assoc, HasRoleTaker) and assoc.field.is_role_taker:
                return assoc
        return None

    @lru_cache(maxsize=None)
    def get_neighbors_with_relation_type(
        self,
        cls: Union[Type, WrappedClass],
        relation_type: Type[ClassRelation],
    ) -> Tuple[WrappedClass, ...]:
        """Return all neighbors of a class whose connecting edge matches the relation type.

        :param cls: The class or wrapped class for which neighbors are to be found.
        :param relation_type: The type of the relation to filter edges by.
        :return: A tuple containing the neighbors of the class, filtered by the specified relation type.
        """
        wrapped_cls = self.get_wrapped_class(cls)
        edge_filter_func = lambda edge: isinstance(edge, relation_type)
        filtered_neighbors = [
            self._dependency_graph.get_node_data(n)
            for n, e in self._dependency_graph.adj(wrapped_cls.index).items()
            if edge_filter_func(e)
        ]
        return tuple(filtered_neighbors)

    @lru_cache(maxsize=None)
    def get_outgoing_neighbors_with_relation_type(
        self,
        cls: Union[Type, WrappedClass],
        relation_type: Type[ClassRelation],
    ) -> Tuple[WrappedClass, ...]:
        """
        Caches and retrieves the outgoing neighbors of a given class with a specific relation type
        using the dependency graph.

        :param cls: The class or wrapped class for which outgoing neighbors are to be found.
            relation_type: The type of the relation to filter edges by.
        :return: A tuple containing the outgoing neighbors of the class, filtered by the specified relation type.
        :raises: Any exceptions raised internally by `find_successors_by_edge` or during class wrapping.
        """
        wrapped_cls = self.get_wrapped_class(cls)
        edge_filter_func = lambda edge: isinstance(edge, relation_type)
        find_successors_by_edge = self._dependency_graph.find_successors_by_edge
        return tuple(find_successors_by_edge(wrapped_cls.index, edge_filter_func))

    @lru_cache(maxsize=None)
    def get_incoming_neighbors_with_relation_type(
        self,
        cls: Union[Type, WrappedClass],
        relation_type: Type[ClassRelation],
    ) -> Tuple[WrappedClass, ...]:
        wrapped_cls = self.get_wrapped_class(cls)
        edge_filter_func = lambda edge: isinstance(edge, relation_type)
        find_predecessors_by_edge = self._dependency_graph.find_predecessors_by_edge
        return tuple(find_predecessors_by_edge(wrapped_cls.index, edge_filter_func))

    def get_out_edges(
        self, cls: Union[Type, WrappedClass]
    ) -> Tuple[ClassRelation, ...]:
        """
        Caches and retrieves the outgoing edges (relations) for the provided class in a
        dependency graph.

        :param cls: The class or wrapped class for which outgoing edges are to be retrieved.
        :return: A tuple of outgoing edges (relations) associated with the provided class.
        """
        wrapped_cls = self.get_wrapped_class(cls)
        out_edges = [
            edge for _, _, edge in self._dependency_graph.out_edges(wrapped_cls.index)
        ]
        return tuple(out_edges)

    @property
    def parent_map(self):
        """
        Build parent map from inheritance edges: child_idx -> set(parent_idx)
        """
        parent_map: dict[int, set[int]] = {}
        for u, v in self._dependency_graph.edge_list():
            rel = self._dependency_graph.get_edge_data(u, v)
            if isinstance(rel, Inheritance):
                parent_map.setdefault(v, set()).add(u)
        return parent_map

    def all_ancestors(self, node_idx: int) -> set[int]:
        """DFS to compute all ancestors for each node index"""
        parent_map = self.parent_map
        parents = parent_map.get(node_idx, set())
        if not parents:
            return set()
        stack = list(parents)
        seen: set[int] = set(parents)
        while stack:
            cur = stack.pop()
            for p in parent_map.get(cur, set()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def get_assoc_keys_by_source(
        self, include_field_name: bool = False
    ) -> dict[int, set[tuple]]:
        """
        Fetches association keys grouped by their source from the internal dependency graph.

        This method traverses the edges of the dependency graph, identifies associations,
        and groups their keys by their source nodes. Optionally includes the field name
        of associations in the resulting keys.

        :include_field_name: Optional; If True, includes the field name in the
                association keys. Defaults to False.

        :return: A dictionary where the keys are source node identifiers (int), and the
            values are sets of tuples representing association keys.
        """
        assoc_keys_by_source = {}
        for u, v in self._dependency_graph.edge_list():
            rel = self._dependency_graph.get_edge_data(u, v)
            if isinstance(rel, Association):
                assoc_keys_by_source.setdefault(u, set()).add(
                    rel.get_key(include_field_name)
                )
        return assoc_keys_by_source

    def to_subdiagram_without_inherited_associations(
        self,
        include_field_name: bool = False,
    ) -> ClassDiagram:
        """
        Return a new class diagram where association edges that are present on any
        ancestor of the source class are removed from descendants.

        Inheritance edges are preserved.
        """
        # Rebuild a fresh diagram from the same classes to avoid mutating this instance
        result = copy(self)
        # Convenience locals
        g = result._dependency_graph

        assoc_keys_by_source = result.get_assoc_keys_by_source(include_field_name)

        # Mark redundant descendant association edges for removal
        edges_to_remove: list[tuple[int, int]] = []
        for u, v in g.edge_list():
            rel = g.get_edge_data(u, v)
            if not isinstance(rel, Association):
                continue

            key = rel.get_key(include_field_name)
            # Collect all keys defined by any ancestor of u
            inherited_keys: set[tuple] = set()
            for anc in result.all_ancestors(u):
                inherited_keys |= assoc_keys_by_source.get(anc, set())

            if key in inherited_keys:
                edges_to_remove.append((u, v))

        # Remove redundant edges
        result.remove_edges(edges_to_remove)

        return result

    def remove_edges(self, edges):
        """Remove edges from the dependency graph"""
        for u, v in edges:
            try:
                self._dependency_graph.remove_edge(u, v)
            except Exception:
                pass

    @property
    def wrapped_classes(self):
        """Return all wrapped classes present in the diagram."""
        return self._dependency_graph.nodes()

    @property
    def associations(self) -> List[Association]:
        """Return all association relations present in the diagram."""
        return [
            edge
            for edge in self._dependency_graph.edges()
            if isinstance(edge, Association)
        ]

    @property
    def inheritance_relations(self) -> List[Inheritance]:
        """Return all inheritance relations present in the diagram."""
        return [
            edge
            for edge in self._dependency_graph.edges()
            if isinstance(edge, Inheritance)
        ]

    def get_wrapped_class(self, clazz: Type) -> Optional[WrappedClass]:
        """
        Gets the wrapped class corresponding to the provided class type.

        If the class type is already a WrappedClass, it will be returned as is. Otherwise, the
        method checks if the class type has an associated WrappedClass in the internal mapping
        and returns it if found.

        :param clazz : The class type to check or retrieve the associated WrappedClass.
        :return: The associated WrappedClass if it exists, None otherwise.
        """
        if isinstance(clazz, WrappedClass):
            return clazz
        try:
            return self._cls_wrapped_cls_map[clazz]
        except KeyError:
            raise ClassIsUnMappedInClassDiagram(clazz)

    def add_node(self, clazz: Union[Type, WrappedClass]):
        """
        Adds a new node to the dependency graph for the specified wrapped class.

        The method sets the position of the given wrapped class in the dependency graph,
        links it with the current class diagram, and updates the mapping of the underlying
        class to the wrapped class.

        :param clazz: The wrapped class object to be added to the dependency graph.
        """
        try:
            clazz = self.get_wrapped_class(clazz)
        except ClassIsUnMappedInClassDiagram:
            clazz = WrappedClass(clazz)
        if clazz.index is not None:
            return
        clazz.index = self._dependency_graph.add_node(clazz)
        clazz._class_diagram = self
        self._cls_wrapped_cls_map[clazz.clazz] = clazz

    def _create_all_relations(self):
        self._create_inheritance_relations()
        self._create_association_relations()
        self._create_association_relations_inferred_from_role_takers()

    def _create_inheritance_relations(self):
        """
        Creates inheritance relations between wrapped classes.

        This method identifies superclass relationships among the wrapped classes and
        establishes inheritance connections. For each class in the `wrapped_classes`
        collection, it iterates through its base classes (`__bases__`). If the base
        class exists in the wrapped classes, an inheritance relation is created and
        added to the relations list.
        """
        for clazz in self.wrapped_classes:
            for superclass in clazz.clazz.__bases__:
                try:
                    source = self.get_wrapped_class(superclass)
                except ClassIsUnMappedInClassDiagram:
                    continue
                if source:
                    relation = Inheritance(
                        source=source,
                        target=clazz,
                    )
                    self.add_relation(relation)

    def _create_association_relations(self):
        """
        Creates association relations between wrapped classes and their fields.

        This method analyzes the fields of wrapped classes and establishes relationships
        based on their target types. It determines the appropriate type of association
        (e.g., `Association` or `HasRoleTaker`) and adds the determined relations to the
        internal collection. Relations are only created when the target class is found among
        the wrapped classes.

        :raises: This method does not explicitly raise any exceptions.
        """
        for clazz in self.wrapped_classes:
            for wrapped_field in clazz.fields:
                target_type = wrapped_field.type_endpoint

                try:
                    wrapped_target_class = self.get_wrapped_class(target_type)
                except ClassIsUnMappedInClassDiagram:
                    continue

                association_type = Association
                if (
                    wrapped_field.is_role_taker
                    and issubclass(clazz.clazz, Role)
                    and target_type is clazz.clazz.get_role_taker_type()
                ):
                    role_taker_type = get_generic_type_param(clazz.clazz, Role)[0]
                    if role_taker_type is target_type:
                        association_type = HasRoleTaker

                relation = association_type(
                    field=wrapped_field,
                    source=clazz,
                    target=wrapped_target_class,
                )
                self.add_relation(relation)

    def _create_association_relations_inferred_from_role_takers(self):
        """
        Create association relations in the roles for associations inferred from role takers.
        """
        wrapped_classes = (
            self.wrapped_classes_of_role_associations_subgraph_in_topological_order
        )
        for role_taker_clazz in reversed(wrapped_classes):
            role_taker_associations = self.get_associations_with_condition(
                role_taker_clazz, lambda rel: not isinstance(rel, HasRoleTaker)
            )
            for association in role_taker_associations:
                self._infer_role_associations_for_role_taker_association(association)

    def _infer_role_associations_for_role_taker_association(
        self, role_taker_assoc: Association
    ):
        """
        Infer role associations through their role taker association.

        :param role_taker_assoc: Association of the role taker.
        """
        role_taker_clazz = role_taker_assoc.source
        for role_clazz in role_taker_clazz.roles:
            self._add_association_through_role_taker(role_clazz, role_taker_assoc)

    def _add_association_through_role_taker(
        self, role_clazz: WrappedClass, role_taker_assoc: Association
    ):
        """
        Adds an association through a role taker to the class diagram. It connects the role class with the role taker
         association target class through an AssociationThroughRoleTaker relation.

        :param role_clazz: Wrapped class of the role.
        :param role_taker_assoc: Association of the role taker.
        """
        role_taker_clazz = role_taker_assoc.source
        association_path = self.get_role_association_path(role_clazz, role_taker_clazz)
        association_path.append(role_taker_assoc)
        self.add_relation(
            AssociationThroughRoleTaker(
                association_path=association_path,
                source=role_clazz,
                target=role_taker_assoc.target,
            )
        )

    @lru_cache(maxsize=None)
    def get_role_association_path(
        self, role_clazz: WrappedClass, role_taker_clazz: WrappedClass
    ) -> List[Association]:
        """
        :param role_clazz: Wrapped class of the role.
        :param role_taker_clazz: Wrapped class of the role taker.
        :return: List of all associations between the role and role taker classes in topological order.
        """
        association_path = []
        wrapped_classes = (
            self.wrapped_classes_of_role_associations_subgraph_in_topological_order
        )
        i = wrapped_classes.index(role_clazz)
        for next_role_clazz in wrapped_classes[i:]:
            if next_role_clazz is role_taker_clazz:
                break
            association = self.role_association_subgraph.out_edges(
                next_role_clazz.index
            )[0][-1]
            association_path.append(association)
        return association_path

    @cached_property
    def wrapped_classes_of_role_associations_subgraph_in_topological_order(
        self,
    ) -> List[WrappedClass]:
        """
        :return: List of all classes in the association subgraph in topological order.
        """
        return [
            self._dependency_graph[index]
            for index in rx.topological_sort(self.role_association_subgraph)
        ]

    @cached_property
    def wrapped_classes_of_inheritance_subgraph_in_topological_order(
        self,
    ) -> List[WrappedClass]:
        """
        :return: List of all classes in the inheritance subgraph in topological order.
        """
        return [
            self._dependency_graph[index]
            for index in rx.topological_sort(self.inheritance_subgraph)
        ]

    @cached_property
    def wrapped_classes_of_association_subgraph_in_topological_order(
        self,
    ) -> List[WrappedClass]:
        """
        :return: List of all classes in the association subgraph in topological order.
        """
        return [
            self._dependency_graph[index]
            for index in rx.topological_sort(self.association_subgraph)
        ]

    @cached_property
    def inheritance_subgraph(self):
        """
        :return: The subgraph containing only inheritance relations and their incident nodes.
        """
        return self._dependency_graph.edge_subgraph(
            [(r.source.index, r.target.index) for r in self.inheritance_relations]
        )

    @cached_property
    def role_association_subgraph(self):
        """
        :return: The subgraph containing only association relations and their incident nodes.
        """
        return self._dependency_graph.edge_subgraph(
            [
                (r.source.index, r.target.index)
                for r in self.associations
                if isinstance(r, HasRoleTaker)
            ]
        )

    @cached_property
    def association_subgraph(self):
        """
        :return: The subgraph containing only association relations and their incident nodes.
        """
        return self._dependency_graph.edge_subgraph(
            [(r.source.index, r.target.index) for r in self.associations]
        )

    def add_relation(self, relation: ClassRelation):
        """
        Adds a relation to the internal dependency graph.

        The method establishes a directed edge in the graph between the source and
        target indices of the provided relation. This function is used to model
        dependencies among entities represented within the graph.

        :relation: The relation object that contains the source and target entities and
        encapsulates the relationship between them.
        """
        self._dependency_graph.add_edge(
            relation.source.index, relation.target.index, relation
        )

    def to_dot(
        self,
        filepath: str,
        format_: str = "svg",
        graph: Optional[rx.PyDiGraph] = None,
        without_inherited_associations: bool = True,
    ):
        import pydot

        if graph is None:
            if without_inherited_associations:
                graph = (
                    self.to_subdiagram_without_inherited_associations()._dependency_graph
                )
            else:
                graph = self._dependency_graph

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

    def _build_rxnode_tree(self, add_association_relations: bool = False) -> RWXNode:
        """
        Convert the class diagram graph to RWXNode tree structure for visualization.

        Creates a tree where inheritance relationships are represented as parent-child connections.
        If there are multiple root classes, they are grouped under a virtual root node.

        :param add_association_relations: If True, include association relations as parent-child connections.
        :return: Root RWXNode representing the class diagram
        """
        if not RWXNode:
            raise ImportError(
                "The rustworkx_utils package is required to visualize the class diagram."
                "Please install it with `pip install rustworkx_utils`."
            )
        # Create RWXNode for each class
        node_map = {}
        for wrapped_class in self.wrapped_classes:
            class_name = wrapped_class.clazz.__name__
            node = RWXNode(name=class_name, data=wrapped_class)
            node_map[wrapped_class.index] = node

        # Build parent-child relationships from edges
        for edge in self._dependency_graph.edge_list():
            source_idx, target_idx = edge
            relation = self._dependency_graph.get_edge_data(source_idx, target_idx)

            # For inheritance: source is parent class, target is child class
            # In RWXNode: parent class should have child class as its child
            if isinstance(relation, Inheritance):
                parent_node = node_map[source_idx]
                child_node = node_map[target_idx]
                child_node.add_parent(parent_node)
            elif isinstance(relation, Association) and add_association_relations:
                # For associations, add as parent relationship with label
                source_node = node_map[source_idx]
                target_node = node_map[target_idx]
                # Association goes from source to target
                target_node.add_parent(source_node)

        # Find root nodes (nodes without parents)
        root_nodes = [node for node in node_map.values() if not node.parents]

        # If there's only one root, return it
        if len(root_nodes) == 1:
            return root_nodes[0]

        # If there are multiple roots, create a virtual root
        virtual_root = RWXNode(name="Class Diagram")
        for root_node in root_nodes:
            root_node.add_parent(virtual_root)

        return virtual_root

    def visualize(
        self,
        filename: str = "class_diagram.pdf",
        title: str = "Class Diagram",
        figsize: tuple = (35, 30),
        node_size: int = 7000,
        font_size: int = 25,
        layout: str = "layered",
        edge_style: str = "straight",
        add_association_relations: bool = False,
        **kwargs,
    ):
        """
        Visualize the class diagram using rustworkx_utils.

        Creates a visual representation of the class diagram showing classes and their relationships.
        The diagram is saved as a PDF file.

        :param filename: Output filename for the visualization
        :param title: Title for the diagram
        :param figsize: Figure size as (width, height) tuple
        :param node_size: Size of the nodes in the visualization
        :param font_size: Font size for labels
        :param add_association_relations: Include association relationships in the visualization.
        :param kwargs: Additional keyword arguments passed to RWXNode.visualize()
        """
        root_node = self._build_rxnode_tree(
            add_association_relations=add_association_relations
        )
        root_node.visualize(
            filename=filename,
            title=title,
            figsize=figsize,
            node_size=node_size,
            font_size=font_size,
            layout=layout,
            edge_style=edge_style,
            **kwargs,
        )

    def clear(self):
        self._dependency_graph.clear()

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other
