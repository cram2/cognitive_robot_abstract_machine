from __future__ import annotations

import logging
import pathlib
import uuid
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from types import ModuleType
from typing import Set

import rustworkx as rx
import sqlalchemy
import krrood.ormatic.custom_types  # type: ignore
import krrood.ormatic.data_access_objects.alternative_mappings  # type: ignore
from krrood.ormatic.helper import get_classes_of_ormatic_interface
from sortedcontainers import SortedSet
from sqlalchemy import JSON
from typing_extensions import List, Type, Dict
from typing_extensions import Optional, TextIO

from krrood.ormatic.custom_types import (
    TypeType,
    PolymorphicEnumType,
    PathType,
    JSONDataType,
)
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.dao import DataAccessObject

from krrood.ormatic.sqlalchemy_generator import SQLAlchemyGenerator
from krrood.ormatic.type_dict import TypeDict
from krrood.ormatic.utils import InheritanceStrategy, classes_of_package
from krrood.utils import module_and_class_name, recursive_subclasses
from krrood.ormatic.wrapped_table import WrappedTable, AssociationObject
from krrood.adapters.json_serializer import SubclassJSONSerializer, JSONData
from krrood.class_diagrams.class_diagram import (
    ClassDiagram,
    ClassRelation,
    WrappedClass,
)
from krrood.class_diagrams.wrapped_field import WrappedField

logger = logging.getLogger(__name__)


class AlternativelyMaps(ClassRelation):
    """
    Edge type that says that the source alternativly maps the target, e. g.
    `AlternativeMaps(source=PointMapping, target=Point)` means that PointMapping is the mapping for Point.
    """


@dataclass
class ORMatic:
    """
    ORMatic is a tool for generating SQLAlchemy ORM models from a set of dataclasses.
    """

    class_dependency_graph: ClassDiagram
    """
    The class diagram to add the orm for.
    """

    alternative_mappings: List[Type[AlternativeMapping]] = field(default_factory=list)
    """
    List of alternative mappings that should be used to map classes.
    """

    type_mappings: TypeDict = field(default_factory=TypeDict)
    """
    A dict that maps classes to custom types that should be used to save the classes.
    They keys of the type mappings must be disjoint with the classes given..
    """

    inheritance_strategy: InheritanceStrategy = InheritanceStrategy.JOINED
    """
    The inheritance strategy to use.
    """

    foreign_key_postfix = "_id"
    """
    The postfix that will be added to foreign key columns (not the relationships).
    """

    imported_modules: SortedSet[str] = field(default_factory=SortedSet, init=False)
    """
    A set of modules that need to be imported.
    """

    type_annotation_map: Dict[str, str] = field(default_factory=dict, init=False)
    """
    The string version of type mappings that is used in jinja.
    """

    inheritance_graph: rx.PyDiGraph[int] = field(default=None, init=False)
    """
    A graph that represents the inheritance structure of the classes. Extracted from the class dependency graph.
    """

    wrapped_tables: Dict[WrappedClass, WrappedTable] = field(
        default_factory=dict, init=False
    )
    """
    The wrapped tables instances for the SQLAlchemy conversion.
    """

    association_objects: List[AssociationObject] = field(
        default_factory=list, init=False
    )
    """
    List of association tables for many-to-many relationships.
    """

    def __post_init__(self):
        self.imported_modules.add(TypeDict.__module__)
        self._fill_type_mappings()
        self._create_inheritance_graph()
        self._add_alternative_mappings_to_class_diagram()
        self._create_wrapped_tables()
        self.create_type_annotations_map()

        for wrapped_table in self.wrapped_tables.values():
            self.imported_modules.add(wrapped_table.wrapped_clazz.clazz.__module__)

    def _fill_type_mappings(self):
        """
        Fill the type mappings of this with needed defaults
        """
        self.type_mappings[Type] = TypeType
        self.type_mappings[type] = TypeType
        self.type_mappings[Enum] = PolymorphicEnumType
        self.type_mappings[SubclassJSONSerializer] = JSON
        self.type_mappings[uuid.UUID] = sqlalchemy.UUID
        self.type_mappings[pathlib.Path] = PathType
        self.type_mappings[JSONData] = JSONDataType

        for key in self.type_mappings.keys():
            self.imported_modules.add(key.__module__)

    def _create_wrapped_tables(self):
        for wrapped_clazz in self.wrapped_classes_in_topological_order:

            # check if the class has an alternative mapping
            if alternative_mapping := self.get_alternative_mapping(wrapped_clazz):
                # add the alternative mapping
                self.wrapped_tables[wrapped_clazz] = WrappedTable(
                    wrapped_clazz=alternative_mapping, ormatic=self
                )
            else:
                # add the class normally
                self.wrapped_tables[wrapped_clazz] = WrappedTable(
                    wrapped_clazz=wrapped_clazz, ormatic=self
                )

    def _create_inheritance_graph(self):
        self.inheritance_graph = rx.PyDiGraph()
        self.inheritance_graph.add_nodes_from(
            [w.index for w in self.class_dependency_graph.wrapped_classes]
        )
        for edge in self.class_dependency_graph.inheritance_relations:
            self.inheritance_graph.add_edge(edge.source.index, edge.target.index, None)

    def _add_alternative_mappings_to_class_diagram(self):
        """
        Add alternative mappings to the class diagram.
        """
        for alternative_mapping in self.alternative_mappings:
            wrapped_alternative_mapping = WrappedClass(clazz=alternative_mapping)
            self.class_dependency_graph.add_node(wrapped_alternative_mapping)
            self.class_dependency_graph.add_relation(
                AlternativelyMaps(
                    source=wrapped_alternative_mapping,
                    target=self.class_dependency_graph.get_wrapped_class(
                        alternative_mapping.original_class()
                    ),
                )
            )

    @property
    def alternatively_maps_relations(self) -> List[AlternativelyMaps]:
        return [
            edge
            for edge in self.class_dependency_graph._dependency_graph.edges()
            if isinstance(edge, AlternativelyMaps)
        ]

    def get_alternative_mapping(
        self, wrapped_class: WrappedClass
    ) -> Optional[WrappedClass]:
        """
        Finds and returns an alternative mapping for the given wrapped class,
        if one exists, based on the relations specified in
        `alternatively_maps_relations`.

        :param wrapped_class: The wrapped class for which an alternative
            mapping is to be searched.
        :return: An alternate mapping of the type WrappedClass if found,
            otherwise None.
        """
        for rel in self.alternatively_maps_relations:
            if rel.target == wrapped_class:
                return rel.source
        return None

    def create_type_annotations_map(self):
        self.type_annotation_map = {}
        for clazz, custom_type in self.type_mappings.items():
            self.type_annotation_map[module_and_class_name(clazz)] = (
                module_and_class_name(custom_type)
            )
            self.imported_modules.add(clazz.__module__)
            self.imported_modules.add(custom_type.__module__)

    @property
    def wrapped_classes_in_topological_order(self) -> List[WrappedClass]:
        """
        :return: List of all tables in topological order.
        """
        return [
            self.class_dependency_graph._dependency_graph[index]
            for index in rx.topological_sort(self.inheritance_graph)
        ]

    @property
    def mapped_classes(self) -> List[Type]:
        return [key.clazz for key in self.wrapped_tables.keys()]

    def make_all_tables(self):
        for table in self.wrapped_tables.values():
            table.parse_fields()

    @classmethod
    def get_type_mappings(cls) -> TypeDict:
        """
        :return: The default type mappings that are used by ORMatic.
        """
        ormatic = cls(ClassDiagram([]))
        return ormatic.type_mappings

    def foreign_key_name(self, wrapped_field: WrappedField) -> str:
        """
        :return: A foreign key name for the given field.
        """
        return f"{wrapped_field.clazz.clazz.__name__.lower()}_{wrapped_field.field.name}{self.foreign_key_postfix}"

    def to_sqlalchemy_file(self, file: TextIO):
        """
        Generate a Python file with SQLAlchemy declarative mappings from the ORMatic models.

        :param file: The file to write to
        """
        sqlalchemy_generator = SQLAlchemyGenerator(self)
        sqlalchemy_generator.to_sqlalchemy_file(file)

    @classmethod
    def from_package(
        cls,
        packages: List[ModuleType],
        ormatic_interface_dependencies: List[ModuleType],
        ignored_classes: Set[Type],
        type_mappings: Dict[Type, Type],
        ignore_krrood_test_classes: bool = True,
    ):
        """
        Create an instance from a list of packages, dependencies, and ignored classes.


        :param packages: The packages that should be scanned for dataclasses.
        :param ormatic_interface_dependencies: The dependent ormatic_interfaces.
        :param ignored_classes: The classes that should be ignored.
        :param type_mappings: The type mappings that should be used.
        :param ignore_krrood_test_classes: Rather to ignore classes from the krrood test package.

        :return: The ORMatic instance.
        """

        all_classes, all_alternative_mappings, all_type_mappings = set(), set(), {}

        # import classes from the existing interface
        for ormatic_interface in ormatic_interface_dependencies:
            (
                interface_classes,
                interface_alternative_mappings,
                interface_type_mappings,
            ) = get_classes_of_ormatic_interface(ormatic_interface)
            all_classes |= set(interface_classes)
            all_alternative_mappings |= set(interface_alternative_mappings)
            all_type_mappings.update(interface_type_mappings)

        for package in packages:
            all_classes |= set(classes_of_package(package))

        all_classes -= ignored_classes

        all_alternative_mappings |= set(
            am
            for am in recursive_subclasses(AlternativeMapping)
            if not ignore_krrood_test_classes
            or "krrood_test" not in am.original_class().__module__
        )

        # keep only dataclasses that are not AlternativeMapping or DataAccessObject subclasses
        all_classes = {
            c
            for c in all_classes
            if is_dataclass(c)
            and not issubclass(c, (DataAccessObject, AlternativeMapping))
        }

        all_classes |= {am.original_class() for am in all_alternative_mappings}

        all_type_mappings.update(type_mappings)

        # create the new ormatic interface
        class_diagram = ClassDiagram(
            list(sorted(all_classes, key=lambda c: c.__name__, reverse=True))
        )

        # Create an ORMatic object with the classes to be mapped
        ormatic = ORMatic(
            class_diagram,
            type_mappings=TypeDict(all_type_mappings),
            alternative_mappings=list(all_alternative_mappings),
        )
        return ormatic
