from __future__ import annotations

import dataclasses
import logging
from abc import ABC
from copy import copy
from dataclasses import dataclass, make_dataclass, is_dataclass
from dataclasses import field, InitVar
from functools import cached_property, lru_cache
from typing import get_args, get_origin, _GenericAlias, Any

import rustworkx as rx

from ..utils import module_and_class_name

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
    TypeVar,
    get_type_hints,
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

    def __str__(self):
        """Return the relation name for display purposes."""
        return f"{self.__class__.__name__}"

    @property
    def color(self) -> str:
        """Default edge color used when visualizing the relation."""
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


class ParseError(TypeError):
    """
    Error that will be raised when the parser encounters something that can/should not be parsed.

    For instance, Union types
    """

    pass


@dataclass
class WrappedClass:
    """A node wrapper around a Python class used in the class diagram graph."""

    index: Optional[int] = field(init=False, default=None)
    clazz: Type
    _class_diagram: Optional[ClassDiagram] = field(
        init=False, hash=False, default=None, repr=False
    )
    _wrapped_field_name_map_: Dict[str, WrappedField] = field(
        init=False, hash=False, default_factory=dict, repr=False
    )

    def _get_introspector(self) -> AttributeIntrospector:
        if self._class_diagram is None:
            introspector = DataclassOnlyIntrospector()
        else:
            introspector = self._class_diagram.introspector
        return introspector

    @cached_property
    def class_to_introspect(self) -> Type:
        """
        :return: The class where the introspector should be called on.
        """
        return self.clazz

    @cached_property
    def fields(self) -> List[WrappedField]:
        """Return wrapped fields discovered by the diagram’s attribute introspector.

        Public names from the introspector are used to index `_wrapped_field_name_map_`.
        """

        wrapped_fields: list[WrappedField] = []
        introspector = self._get_introspector()
        try:
            discovered = introspector.discover(self.class_to_introspect)
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
    def name(self) -> str:
        """
        :return: The name of the class that is wrapped.
        """
        return self.clazz.__name__

    def __hash__(self):
        return hash((self.index, self.clazz))

    @property
    def name_with_entire_path(self) -> str:
        return module_and_class_name(self.clazz)


@dataclass(unsafe_hash=True)
class WrappedSpecializedGeneric(WrappedClass):
    """
    Specialization of WrappedClass for completely parameterized generic types, e.g. Generic[float].
    """

    @property
    def name_with_entire_path(self) -> str:
        return str(self.clazz)

    @cached_property
    def class_to_introspect(self):
        return make_specialized_dataclass(self.clazz)

    @property
    def name(self):
        return self.class_to_introspect.__name__


@dataclass
class ClassDiagram:
    """A graph of classes and their relations discovered via attribute introspection."""

    classes: InitVar[List[Type]]

    introspector: AttributeIntrospector = field(
        default_factory=DataclassOnlyIntrospector, init=True, repr=False
    )

    _dependency_graph: rx.PyDiGraph[WrappedClass, ClassRelation] = field(
        default_factory=rx.PyDiGraph, init=False
    )
    _cls_wrapped_cls_map: Dict[Type, WrappedClass] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self, classes: List[Type]):
        """Initialize the diagram with the provided classes and build relations."""
        self._dependency_graph = rx.PyDiGraph()
        for clazz in classes:
            self.add_node(WrappedClass(clazz=clazz))
        self._create_nodes_for_specialized_generic_type_hints()
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

    @lru_cache(maxsize=None)
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
            # Handle GenericAlias which doesn't have __bases__
            origin = get_origin(clazz.clazz)
            if origin is not None and not isinstance(clazz.clazz, type):
                bases = origin.__bases__
            else:
                try:
                    bases = clazz.clazz.__bases__
                except AttributeError:
                    continue

            for superclass in bases:
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

    def _create_all_relations(self):
        self._create_inheritance_relations()
        self._create_association_relations()

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
                # Handle GenericAlias in issubclass
                origin = get_origin(clazz.clazz)
                actual_cls = (
                    origin
                    if (origin is not None and not isinstance(clazz.clazz, type))
                    else clazz.clazz
                )

                try:
                    is_role_subclass = issubclass(actual_cls, Role)
                except TypeError:
                    is_role_subclass = False

                if wrapped_field.is_role_taker and is_role_subclass:
                    role_taker_type = get_generic_type_param(actual_cls, Role)[0]
                    if role_taker_type is target_type:
                        association_type = HasRoleTaker

                relation = association_type(
                    field=wrapped_field,
                    source=clazz,
                    target=wrapped_target_class,
                )
                self.add_relation(relation)

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
        :param kwargs: Additional keyword arguments passed to RWXNode.visualize()
        """
        root_node = self._build_rxnode_tree()
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

    def _create_nodes_for_specialized_generic_type_hints(self):
        """
        Creates nodes for specialized generic type hints utilized in the wrapped classes. This process involves
        analyzing fields for references to specialized generic types, creating corresponding nodes, and establishing
        inheritance relations as appropriate. The method ensures that all unique specialized generics referenced
        are represented as nodes. This is similar to C++ template resolving.

        :raises ClassIsUnMappedInClassDiagram: If a class referenced by a field or origin type is not mapped
            in the class diagram, it will skip further processing for that type.
        """
        # Phase 1: Collect all unique specialized generic types referenced in fields
        to_process = set()
        [
            to_process.add(wrapped_field.type_endpoint)
            for wrapped_class in self.wrapped_classes
            for wrapped_field in wrapped_class.fields
        ]

        # Phase 2: Add nodes for discovered types that do not already exists
        while to_process:
            next_type = to_process.pop()

            # skip existing nodes
            try:
                self.get_wrapped_class(next_type)
                # Already wrapped
                continue
            except ClassIsUnMappedInClassDiagram:
                pass

            # skip non dataclass generics
            if not is_dataclass(get_origin(next_type)):
                continue

            node = WrappedSpecializedGeneric(next_type)
            self.add_node(node)

            # Add explicit inheritance from the origin class
            origin = get_origin(next_type)
            if origin:
                try:
                    source_node = self.get_wrapped_class(origin)
                    self.add_relation(Inheritance(source=source_node, target=node))
                except ClassIsUnMappedInClassDiagram:
                    pass

            # Check if the new node has fields that point to other specialized generics
            for wrapped_field in node.fields:
                if wrapped_field.is_instantiation_of_generic_class:
                    try:
                        self.get_wrapped_class(wrapped_field.type_endpoint)
                    except ClassIsUnMappedInClassDiagram:
                        to_process.add(wrapped_field.type_endpoint)


def resolve_type(
    type_to_resolve: Any,
    substitution: Dict[TypeVar, Any],
    name_substitution: Dict[str, Any],
) -> Any:
    """
    Resolve type variables and forward references in a type.

    :param type_to_resolve: The type to resolve.
    :param substitution: Mapping of TypeVar to concrete types.
    :param name_substitution: Mapping of TypeVar names to concrete types.
    :return: The resolved type.
    """
    # Resolve string forward refs and TypeVar names
    if isinstance(type_to_resolve, str):
        if type_to_resolve in name_substitution:
            return name_substitution[type_to_resolve]
        return type_to_resolve

    if isinstance(type_to_resolve, TypeVar):
        return substitution.get(type_to_resolve, type_to_resolve)

    # Get arguments and recursively resolve them
    args = get_args(type_to_resolve)
    if not args:
        return type_to_resolve

    resolved_args = tuple(
        resolve_type(arg, substitution, name_substitution) for arg in args
    )

    # If the type itself can be indexed (like List[T] or Optional[T])
    params = getattr(type_to_resolve, "__parameters__", None)
    if hasattr(type_to_resolve, "__getitem__") and params:
        if len(params) < len(resolved_args):
            # Filter out NoneType if it's an Optional/Union and we have more args than parameters
            new_args = tuple(arg for arg in resolved_args if arg is not type(None))
            if len(new_args) == len(params):
                if len(params) == 1:
                    return type_to_resolve[new_args[0]]
                return type_to_resolve[new_args]

        if len(params) == 1 and len(resolved_args) == 1:
            return type_to_resolve[resolved_args[0]]
        return type_to_resolve[resolved_args]

    # Fallback: re-construct from origin (e.g. for Union/Optional or built-in generics)
    origin = get_origin(type_to_resolve)
    if origin is not None:
        # Special case for Union which might be represented as typing.Union
        # and needs to be indexed.
        if origin is Union:
            return origin[resolved_args]
        try:
            return origin[resolved_args]
        except TypeError:
            # Some origins might not be indexable directly or might need single arg
            if len(resolved_args) == 1:
                return origin[resolved_args[0]]
            raise

    return type_to_resolve


@lru_cache
def make_specialized_dataclass(alias: _GenericAlias) -> Type:
    """
    Build a concrete dataclass for a fully specialized generic alias, e.g., GenericClass[float].

    The resulting class is intended for internal use only and should never be used directly.

    :param alias: The fully specialized generic alias to build a dataclass for.
    :return: A concrete dataclass corresponding to the provided alias.
    """

    # get the template class
    template_class = get_origin(alias)
    if template_class is None:
        raise TypeError(f"{alias!r} is not a specialized generic alias")
    if not dataclasses.is_dataclass(template_class):
        raise TypeError(f"Origin {template_class!r} is not a dataclass")

    # Map TypeVar -> concrete argument
    args = get_args(alias)
    params: Tuple[TypeVar, ...] = template_class.__parameters__
    substitution = dict(zip(params, args))
    # Also map by TypeVar name to handle postponed annotations ('T')
    name_substitution = {p.__name__: a for p, a in substitution.items()}

    # Preserve dataclass parameters
    params_obj = template_class.__dataclass_params__

    # Build field specs by copying defaults/metadata and substituting types
    new_fields = []
    # Use get_type_hints to resolve any postponed annotations (strings)
    # This is important for GenericClass[T] where fields might be strings.
    try:
        resolved_hints = get_type_hints(template_class, include_extras=True)
    except Exception:
        resolved_hints = {f.name: f.type for f in dataclasses.fields(template_class)}

    for f in dataclasses.fields(template_class):
        # Use the resolved hint if available, else fallback to the raw field type
        raw_type = resolved_hints.get(f.name, f.type)
        new_type = resolve_type(raw_type, substitution, name_substitution)
        # Copy defaults and flags
        kwargs = dict(
            default=f.default,
            default_factory=f.default_factory,
            init=f.init,
            repr=f.repr,
            hash=f.hash,
            compare=f.compare,
            kw_only=getattr(f, "kw_only", False),
            metadata=(f.metadata or {}) | {"__origin_field__": f},
        )
        # Remove MISSING to satisfy make_dataclass
        if kwargs["default"] is dataclasses.MISSING:
            kwargs.pop("default")
        if kwargs["default_factory"] is dataclasses.MISSING:
            kwargs.pop("default_factory")
        new_fields.append((f.name, new_type, field(**kwargs)))

    # Name and namespace
    arg_names = [getattr(a, "__name__", repr(a)) for a in args]
    name = f"{template_class.__name__}_{'_'.join(arg_names)}"
    namespace = {
        "__origin__": template_class,
        "__args__": args,
        "__alias__": alias,
        "__module__": template_class.__module__,  # better pickling/story in repr
    }

    # create the specialized concrete class
    specialized_class = dataclasses.make_dataclass(
        name,
        fields=new_fields,
        bases=(template_class,),
        namespace=namespace,
        frozen=params_obj.frozen,
        eq=params_obj.eq,
        order=params_obj.order,
        unsafe_hash=params_obj.unsafe_hash,
        kw_only=params_obj.kw_only if hasattr(params_obj, "kw_only") else False,
        slots=getattr(template_class, "__slots__", None) is not None,
    )

    return specialized_class
