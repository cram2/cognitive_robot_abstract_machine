"""
The tables ORMatic generates or reuses, and the columns and relationships they hold.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from inspect import isclass

import sqlalchemy
from sqlalchemy.ext.orderinglist import ordering_list
from typing_extensions import Any, List, TYPE_CHECKING, Optional, Set, get_origin

from krrood.ormatic.collection_shape import (
    AssociationLeafAttribute,
    CollectionShape,
    MappingLevel,
)
from krrood.ormatic.field_storage import FieldStorage, FieldToStore, ValueStorage
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.dao import (
    RelationshipInformationKey,
)
from krrood.ormatic.stored_object import StoredObjectDAO
from krrood.patterns.interface import Interface
from krrood.class_diagrams.class_diagram import (
    WrappedClass,
    Inheritance,
)
from krrood.class_diagrams.exceptions import ClassIsUnMappedInClassDiagram
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.utils import module_and_class_name
from krrood.patterns.caching import memoize

if TYPE_CHECKING:
    from krrood.ormatic.ormatic import ORMatic


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class WrappedTableNotFound(KeyError):
    type_: type
    wrapped_field: WrappedField

    def __post_init__(self):
        super().__init__(
            f"Didnt find a wrapped table for the type {self.type_}. "
            f"This happened when trying to get a wrapped table for the wrapped field {self.wrapped_field}"
        )


@dataclass(kw_only=True)
class ColumnConstructor:
    """
    Represents a column constructor that can be used to create a column in SQLAlchemy.
    """

    name: str
    """
    The name of the column.
    """

    type: str
    """
    The type of the column.

    Needs to be like "Mapped[<type>]".
    """

    constructor: str | None = None
    """
    The constructor call for sqlalchemy of the column.
    """

    def __str__(self) -> str:
        if self.constructor:
            return f"{self.name}: {self.type} = {self.constructor}"
        else:
            return f"{self.name}: {self.type}"


@dataclass(kw_only=True)
class AssociationReference:
    """
    A foreign key of an association table to the table of a mapped class, and the
    relationship that reads the object it refers to.
    """

    relationship_name: str
    """
    The name of the relationship.
    """

    foreign_key_name: str
    """
    The name of the foreign key column.
    """

    table: TableLike
    """
    The table of the class the foreign key refers to.
    """

    is_open: bool
    """
    Whether objects of other classes may be referred to, which the table of every stored
    object then holds.
    """

    nullable: bool
    """
    Whether a row may refer to no object, as a row marking an empty collection does.
    """

    @property
    def foreign_key_column(self) -> ColumnConstructor:
        """
        :return: The foreign key column.
        """
        column_type = (
            f"Mapped[{module_and_class_name(Optional)}[int]]"
            if self.nullable
            else "Mapped[int]"
        )
        return ColumnConstructor(
            name=self.foreign_key_name,
            type=column_type,
            constructor=f"mapped_column(ForeignKey('{self.table.full_primary_key_name}'), nullable={self.nullable})",
        )

    def relationship(self, *, association_name: str) -> ColumnConstructor:
        """
        :param association_name: The name of the association class.
        :return: The relationship reading the object the foreign key refers to.
        """
        if not self.is_open:
            return ColumnConstructor(
                name=self.relationship_name,
                type=f"Mapped[{self.table.tablename}]",
                constructor=f"relationship('{self.table.tablename}', foreign_keys=[{self.foreign_key_name}], lazy='selectin')",
            )
        root = module_and_class_name(StoredObjectDAO)
        return ColumnConstructor(
            name=self.relationship_name,
            type=f"Mapped[{root}]",
            constructor=(
                f"relationship({root}, "
                f"primaryjoin=lambda: {association_name}.{self.foreign_key_name} == {root}.database_id, "
                f"foreign_keys=lambda: [{association_name}.{self.foreign_key_name}], lazy='selectin', "
                f"info={{{_information_key(RelationshipInformationKey.DECLARED_TYPE)}: "
                f"{module_and_class_name(self.table.domain_class)}}})"
            ),
        )


@dataclass(kw_only=True)
class AssociationObject:
    """
    An association table with a row per value of a collection, rendered with jinja.

    A row holds its value's position and key in each level, and the value; a row without
    a value marks an empty container.
    """

    name: str
    """
    The name of the association class.
    """

    left_foreign_key: str
    """
    The name of the foreign key column to the row of the object holding the collection.
    """

    left_primary_key: str
    """
    The full primary key reference of the table of the object holding the collection
    (e.g., 'TableName.primary_key').
    """

    primary_key_name: str
    """
    The name of the primary key column.
    """

    shape: CollectionShape
    """
    The shape of the collection.
    """

    key_references: dict[int, AssociationReference]
    """
    The reference to the key's object of every level whose keys are mapped objects, by
    the depth of the level.
    """

    key_columns: dict[int, ColumnConstructor]
    """
    The column holding the key of every keyed level whose keys are not mapped objects,
    by the depth of the level.
    """

    target_reference: AssociationReference | None
    """
    The reference to the value's object, if the values are mapped objects.
    """

    value_column: ColumnConstructor | None
    """
    The column holding the value, if the values are not mapped objects.
    """

    @property
    def table_name(self):
        """
        :return: The table name encoded as a hash that is stable across different python processes.
        This needs to happen this way as it is very likely that association table names will get too long for common SQL
        dialects.
        """
        number = int(hashlib.sha256(str(self.name).encode("utf-8")).hexdigest(), 16)
        short_number = str(number)[:62]
        return f"_{short_number}"

    @property
    def primary_key(self) -> ColumnConstructor:
        """
        :return: The primary key column constructor for this association table.
        """
        return ColumnConstructor(
            name=self.primary_key_name,
            type="Mapped[int]",
            constructor="mapped_column(Integer, primary_key=True)",
        )

    @property
    def index_column_names(self) -> list[str]:
        """
        :return: The names of the columns holding the positions of a row's value, the
            outermost level first, which is the order the rows are read back in.
        """
        return [level.column_names.index for level in self.shape.levels]

    def collection_class_constructor(self, *, ormatic: ORMatic) -> str:
        """
        :param ormatic: The ORMatic instance whose generated module imports what the
            constructor needs.
        :return: The collection class of the relationship holding the rows: for a
            collection of one unkeyed level, a list that sets each appended row's
            position, so rows built without one get their place in the list.
        """
        first_level = self.shape.levels[0]
        if self.shape.is_nested or isinstance(first_level, MappingLevel):
            return module_and_class_name(list)
        ormatic.imported_modules.add(ordering_list.__module__)
        return f"{module_and_class_name(ordering_list)}('{first_level.column_names.index}')"

    @property
    def columns(self) -> list[ColumnConstructor]:
        """
        :return: The foreign key to the holding object, the level columns and the value
            columns, the value's foreign key included.
        """
        result = [
            ColumnConstructor(
                name=self.left_foreign_key,
                type="Mapped[int]",
                constructor=f"mapped_column(ForeignKey('{self.left_primary_key}'))",
            )
        ]
        for level in self.shape.levels:
            nullable = level.depth > 1
            result.append(
                ColumnConstructor(
                    name=level.column_names.index,
                    type=(
                        f"Mapped[{module_and_class_name(Optional)}[int]]"
                        if nullable
                        else "Mapped[int]"
                    ),
                    constructor=f"mapped_column(Integer, nullable={nullable})",
                )
            )
            if level.depth in self.key_columns:
                result.append(self.key_columns[level.depth])
            if level.depth in self.key_references:
                result.append(self.key_references[level.depth].foreign_key_column)
        if self.target_reference is not None:
            result.append(self.target_reference.foreign_key_column)
        if self.value_column is not None:
            result.append(self.value_column)
        return result

    @property
    def relationships(self) -> list[ColumnConstructor]:
        """
        :return: The relationships reading the objects of the keys and the value.
        """
        references = [*self.key_references.values()]
        if self.target_reference is not None:
            references.append(self.target_reference)
        return [
            reference.relationship(association_name=self.name)
            for reference in references
        ]


def _information_key(key: RelationshipInformationKey) -> str:
    """
    :param key: A key of the information a generated relationship carries.
    :return: The key as generated code spells it.
    """
    return f"{RelationshipInformationKey.__name__}.{key.name}"


class TableLike(ABC):
    """
    Common interface of :class:`WrappedTable` and :class:`ExternalTable`: something that
    names a SQLAlchemy table, whether generated in this run or already mapped by a
    dependency.
    """

    @property
    @abstractmethod
    def tablename(self) -> str: ...

    @property
    @abstractmethod
    def qualified_reference(self) -> str: ...

    @property
    @abstractmethod
    def primary_key_name(self) -> str: ...

    @property
    def full_primary_key_name(self) -> str:
        return f"{self.tablename}.{self.primary_key_name}"

    @property
    @abstractmethod
    def qualified_full_primary_key_name(self) -> str: ...

    @property
    @abstractmethod
    def domain_class(self) -> type:
        """
        :return: The class whose objects this table holds rows of.
        """

    @property
    @abstractmethod
    def own_attribute_names(self) -> set[str]:
        """
        :return: The names of the columns and relationships this table maps, including
            those it inherits where they cannot be told apart.
        """


@dataclass(kw_only=True)
class ExternalTable(TableLike):
    """
    A lookup-only stand-in for a class already mapped by an ormatic-interface
    dependency.

    It is never rendered by the generator — its columns already exist in the
    dependency's generated file. It exists purely so that local tables which use it as a
    foreign key target, relationship target, or parent class can resolve it, the same
    way they would resolve a locally generated :class:`WrappedTable`.
    """

    wrapped_clazz: WrappedClass
    """
    The wrapped (domain) class that this stands in for.
    """

    dao_class: type
    """
    The ``DataAccessObject`` subclass that already maps ``wrapped_clazz`` in the
    dependency's generated interface.
    """

    @property
    def tablename(self) -> str:
        return self.dao_class.__name__

    @property
    def qualified_reference(self) -> str:
        return module_and_class_name(self.dao_class)

    @cached_property
    def primary_key_name(self) -> str:
        return sqlalchemy.inspect(self.dao_class).primary_key[0].name

    @property
    def qualified_full_primary_key_name(self) -> str:
        return f"{self.qualified_reference}.{self.primary_key_name}"

    @cached_property
    def mapped_field_names(self) -> set[str]:
        """
        :return: Every ORM attribute name already mapped on ``dao_class``, including
            ones inherited from its own (possibly external) ancestors.
        """
        return {prop.key for prop in sqlalchemy.inspect(self.dao_class).attrs}

    @property
    def domain_class(self) -> type:
        clazz = self.wrapped_clazz.clazz
        return get_origin(clazz) or clazz

    @property
    def own_attribute_names(self) -> set[str]:
        return self.mapped_field_names

    @property
    def part_class_name(self) -> str:
        """
        :return: The name of a part class a dependent interface generates for this
            table.
        """
        return f"{self.dao_class.__name__}Part"

    @property
    def domain_class_reference(self) -> str:
        """
        :return: The qualified name of the class the table's DAO maps.
        """
        return module_and_class_name(self.dao_class.original_class())

    @property
    def part_relationships(self) -> list[ColumnConstructor]:
        """
        :return: The relationships this table's own DAO class declares, as a part class
            of the table declares them.
        """
        mapper = sqlalchemy.inspect(self.dao_class)
        dao_reference = self.qualified_reference
        result = []
        for relationship in mapper.relationships:
            if relationship.parent is not mapper:
                continue
            target = relationship.mapper.class_
            target_reference = module_and_class_name(target)
            if relationship.uselist:
                [association_column] = relationship.remote_side
                ordering_columns = ", ".join(
                    f"{target_reference}.{column.key}"
                    for column in relationship.order_by or []
                )
                ordering = (
                    f"order_by=lambda: [{ordering_columns}], "
                    if ordering_columns
                    else ""
                )
                collection_class = (
                    module_and_class_name(relationship.collection_class)
                    if isclass(relationship.collection_class)
                    else f"{module_and_class_name(ordering_list)}('{relationship.order_by[0].key}')"
                )
                arguments = (
                    f"{target_reference}, "
                    f"collection_class={collection_class}, "
                    f"cascade='all, delete-orphan', "
                    f"foreign_keys=lambda: [{target_reference}.{association_column.key}], "
                    f"{ordering}"
                    f"lazy='selectin'"
                )
            else:
                own_foreign_key = (
                    f"{dao_reference}.{next(iter(relationship.local_columns)).key}"
                )
                arguments = (
                    f"{target_reference}, uselist=False, "
                    f"foreign_keys=lambda: [{own_foreign_key}], post_update=True"
                )
                declared_type = relationship.info.get(
                    RelationshipInformationKey.DECLARED_TYPE
                )
                if declared_type is not None:
                    arguments += (
                        f", primaryjoin=lambda: {own_foreign_key} == "
                        f"{target_reference}.database_id, "
                        f"info={{{_information_key(RelationshipInformationKey.DECLARED_TYPE)}: "
                        f"{module_and_class_name(declared_type)}}}"
                    )
            result.append(
                ColumnConstructor(
                    name=relationship.key,
                    type=f"Mapped[{target_reference}]",
                    constructor=f"relationship({arguments}, overlaps='{relationship.key}')",
                )
            )
        return result


@dataclass(kw_only=True)
class WrappedTable(TableLike):
    """
    A class that wraps a dataclass and contains all the information needed to create a
    SQLAlchemy table from it.
    """

    wrapped_clazz: WrappedClass
    """
    The wrapped class that this table wraps.
    """

    ormatic: ORMatic
    """
    Reference to the ORMatic instance that created this WrappedTable.
    """

    builtin_columns: list[ColumnConstructor] = field(default_factory=list, init=False)
    """
    List of columns that can be directly mapped using builtin types.
    """

    custom_columns: list[ColumnConstructor] = field(default_factory=list, init=False)
    """
    List for custom columns that need to by fully qualified as triple of (name, type,
    constructor)
    """

    foreign_keys: list[ColumnConstructor] = field(default_factory=list, init=False)
    """
    List of columns that represent foreign keys as triple of (name, type, constructor)
    """

    relationships: list[ColumnConstructor] = field(default_factory=list, init=False)
    """
    List of relationships that should be added to the table.
    """

    part_relationships: list[ColumnConstructor] = field(
        default_factory=list, init=False
    )
    """
    The relationships of this table as the part class of an open class declares them.
    """

    mapper_args: dict[str, str] = field(default_factory=dict, init=False)
    """
    Keyword Arguments for the sqlalchemy mapper.
    """

    primary_key_name: str = "database_id"
    """
    The name of the primary key column.
    """

    polymorphic_on_name: str = "polymorphic_type"
    """
    The name of the column that will be used to identify polymorphic identities if any
    present.
    """

    skip_fields: list[WrappedField] = field(default_factory=list)
    """
    A list of fields that should be skipped when processing the dataclass.
    """

    @property
    def primary_key(self) -> ColumnConstructor:
        """
        :return: The primary key, which refers to the row of the parent table, or to the
            object's row in the table of every stored object for a root table. The
            reference is checked at commit, since a flush may insert a part first.
        """
        referenced_table = self.parent_table or self.ormatic.stored_object_table
        column_type = (
            f"ForeignKey({referenced_table.qualified_full_primary_key_name}, "
            f"deferrable=True, initially='DEFERRED')"
        )
        return ColumnConstructor(
            name=self.primary_key_name,
            type=f"Mapped[{module_and_class_name(int)}]",
            constructor=f"mapped_column({column_type}, primary_key=True, use_existing_column=True)",
        )

    def create_mapper_args(self) -> None:
        """
        Declare the table's polymorphic identity and how it joins its parent table; a
        root table also gets the type column of its hierarchy.
        """
        # every root is unconditionally polymorphic-ready: a downstream package can subclass it
        if self.parent_table is None:
            self.custom_columns.append(
                ColumnConstructor(
                    name=self.polymorphic_on_name,
                    type="Mapped[str]",
                    constructor="mapped_column(String(255), nullable=False, use_existing_column=True)",
                )
            )
            self.mapper_args.update(
                {
                    "'polymorphic_on'": f"'{self.polymorphic_on_name}'",
                    "'polymorphic_identity'": f"'{self.tablename}'",
                }
            )
            return

        self.mapper_args.update(
            {
                "'polymorphic_identity'": f"'{self.tablename}'",
                "'inherit_condition'": f"{self.primary_key_name} == {self.parent_table.qualified_full_primary_key_name}",
                # batch subclass-table loads instead of one SELECT per instance
                "'polymorphic_load'": "'selectin'",
            }
        )

    @property
    def qualified_reference(self) -> str:
        return self.tablename

    @property
    def qualified_full_primary_key_name(self) -> str:
        return self.full_primary_key_name

    @cached_property
    def tablename(self):
        result = self.wrapped_clazz.name
        result += "DAO"
        return result

    @cached_property
    def parent_table(self) -> TableLike | None:
        """
        Resolve the parent DAO table for this table.

        This first tries to use a direct inheritance relation from the class diagram. If
        that is not available, it tries to find a parent via MRO. If that is not
        available and this table is an alternative mapping, it resolves the parent
        through the original classes' inheritance and maps back to the correct DAO
        table.

        :return: The parent ``WrappedTable`` or ``None`` if there is no parent.
        """
        # Try finding parent via inheritance graph in class diagram
        try:
            inheritance_parents = (
                self.ormatic.class_dependency_graph._dependency_graph.predecessors(
                    self.wrapped_clazz.index
                )
            )
            for parent_wrapped in inheritance_parents[::-1]:
                # Check if this parent has a wrapped table and if the relation is Inheritance
                # We need to check the actual relation object
                edge_data = (
                    self.ormatic.class_dependency_graph._dependency_graph.get_edge_data(
                        parent_wrapped.index, self.wrapped_clazz.index
                    )
                )

                if isinstance(edge_data, Inheritance) and (
                    parent_wrapped in self.ormatic.wrapped_tables
                    or parent_wrapped in self.ormatic.external_tables
                ):
                    return self.ormatic.table_for(wrapped_class=parent_wrapped)
        except (AttributeError, KeyError):
            pass

        direct_parent = self._find_direct_parent_wrapped()
        if direct_parent is not None:
            return self.ormatic.table_for(wrapped_class=direct_parent)

        if not self.is_alternatively_mapped:
            return None

        original = self._original_wrapped_for_mapping(
            mapping_wrapped=self.wrapped_clazz
        )
        if original is None:
            return None

        original_parent = self._find_original_parent_wrapped(original_wrapped=original)
        if original_parent is None:
            return None

        resolved_parent_wrapped = self._resolve_alternative_parent_wrapped(
            original_parent_wrapped=original_parent
        )
        key = self._to_wrapped_tables_key(wrapped=resolved_parent_wrapped)
        # the resolved parent may itself be mapped by a dependency rather than locally
        if (
            key not in self.ormatic.wrapped_tables
            and key not in self.ormatic.external_tables
        ):
            return None
        return self.ormatic.table_for(wrapped_class=key)

    # %% helper methods

    def _find_direct_parent_wrapped(self) -> WrappedClass | None:
        """
        Return the first mapped parent ``WrappedClass`` from the MRO.

        This iterates through the MRO (Method Resolution Order) of the class and returns
        the first parent class that has a corresponding wrapped table.
        """
        # Get the actual class from wrapped_clazz
        current_class = self.wrapped_clazz.clazz

        # Handle GenericAlias which doesn't have __mro__
        origin = get_origin(current_class)
        if origin is not None and not isinstance(current_class, type):
            mro = origin.__mro__
        else:
            try:
                mro = current_class.__mro__
            except AttributeError:
                return None

        # Iterate through MRO, skipping the first element (the class itself)
        for parent_class in mro[1:]:
            # Skip object base class
            if parent_class is object:
                continue

            # Try to get the wrapped class for this parent
            try:
                parent_wrapped = self.ormatic.class_dependency_graph.get_wrapped_class(
                    clazz=parent_class
                )
            except ClassIsUnMappedInClassDiagram:
                # Skip classes that are not in the class diagram
                continue

            # Check if this wrapped class exists in wrapped_tables or external_tables
            if parent_wrapped is not None and (
                parent_wrapped in self.ormatic.wrapped_tables
                or parent_wrapped in self.ormatic.external_tables
            ):
                return parent_wrapped

        return None

    def _original_wrapped_for_mapping(
        self, *, mapping_wrapped: WrappedClass
    ) -> WrappedClass | None:
        """
        :param mapping_wrapped: The mapping class as a ``WrappedClass``.
        :return: The original class ``WrappedClass`` for an alternative mapping.
        """
        for rel in self.ormatic.alternatively_maps_relations:
            if rel.source == mapping_wrapped:
                return rel.target
        return None

    def _find_original_parent_wrapped(
        self, *, original_wrapped: WrappedClass
    ) -> WrappedClass | None:
        """
        :param original_wrapped: The original class wrapped node.
        :return: The original parent ``WrappedClass`` of the given original class.
        """
        original_parents = self.ormatic.inheritance_graph.predecessors(
            original_wrapped.index
        )
        if len(original_parents) == 0:
            return None
        return self.ormatic.class_dependency_graph._dependency_graph[
            original_parents[0]
        ]

    def _resolve_alternative_parent_wrapped(
        self, *, original_parent_wrapped: WrappedClass
    ) -> WrappedClass:
        """
        Return the mapping parent if available, otherwise the original parent.

        :param original_parent_wrapped: The original parent wrapped node.
        :return: The wrapped node to use as parent in DAO generation.
        """
        alt_parent = self.ormatic.get_alternative_mapping(
            wrapped_class=original_parent_wrapped
        )
        return alt_parent if alt_parent is not None else original_parent_wrapped

    def _to_wrapped_tables_key(self, *, wrapped: WrappedClass) -> WrappedClass:
        """
        Translate a mapping wrapped node to the original wrapped for table lookup.

        ``wrapped_tables`` are keyed by the original class nodes, even if an alternative
        mapping is used. This ensures we always use the correct key.
        """
        origin = get_origin(wrapped.clazz)
        actual_cls = (
            origin
            if (origin is not None and not isinstance(wrapped.clazz, type))
            else wrapped.clazz
        )

        try:
            is_alt_mapping = issubclass(actual_cls, AlternativeMapping)
        except TypeError:
            is_alt_mapping = False

        if is_alt_mapping:
            for rel in self.ormatic.alternatively_maps_relations:
                if rel.source == wrapped:
                    return rel.target
        return wrapped

    @property
    def is_alternatively_mapped(self):

        origin = get_origin(self.wrapped_clazz.clazz)
        actual_cls = (
            origin
            if (origin is not None and not isinstance(self.wrapped_clazz.clazz, type))
            else self.wrapped_clazz.clazz
        )
        try:
            return issubclass(actual_cls, AlternativeMapping)
        except TypeError:
            return False

    @cached_property
    def domain_class(self) -> type:
        """
        :return: The class whose objects this table holds, which for an alternative
            mapping is the class it maps.
        """
        clazz = self.wrapped_clazz.clazz
        if self.is_alternatively_mapped:
            clazz = clazz.original_class()
        return get_origin(clazz) or clazz

    @property
    def own_attribute_names(self) -> set[str]:
        return (
            {column.name for column in self.builtin_columns}
            | {column.name for column in self.custom_columns}
            | {column.name for column in self.foreign_keys}
            | {column.name for column in self.relationships}
        )

    @cached_property
    def mapped_ancestor_tables(self) -> list[TableLike]:
        """
        :return: The tables of the mapped base classes of the domain class, nearest
            first.
        """
        result = []
        for ancestor in self.domain_class.__mro__[1:]:
            table = self.ormatic.tables_by_class.get(ancestor)
            if table is not None and not any(table is known for known in result):
                result.append(table)
        return result

    @cached_property
    def primary_chain_tables(self) -> list[TableLike]:
        """
        :return: The tables this table inherits, from its parent table up to the root
            table of its hierarchy.
        """
        result = []
        table = self.parent_table
        while table is not None:
            result.append(table)
            table = table.parent_table if isinstance(table, WrappedTable) else None
        return result

    @cached_property
    def secondary_base_tables(self) -> list[TableLike]:
        """
        :return: The tables of the mapped base classes outside the primary chain whose
            rows this table's DAO holds as parts; those its parent table's class already
            has are left to the parent table.
        """
        parent_class = (
            None if self.parent_table is None else self.parent_table.domain_class
        )
        return [
            table
            for table in self.mapped_ancestor_tables
            if not any(
                table is chain_table for chain_table in self.primary_chain_tables
            )
            and not (
                parent_class is not None
                and issubclass(parent_class, table.domain_class)
            )
        ]

    @cached_property
    def is_interface(self) -> bool:
        """
        :return: Whether the domain class is an interface, mapped without fields.
        """
        return isclass(self.domain_class) and Interface.is_interface(
            class_=self.domain_class
        )

    @property
    def has_part_class(self) -> bool:
        """
        :return: Whether the domain class is open, so other classes' DAOs hold rows of
            this table through a part class.
        """
        return self.domain_class in self.ormatic.open_classes

    @property
    def part_class_name(self) -> str:
        """
        :return: The name of the part class of this table.
        """
        return f"{self.tablename}Part"

    @cached_property
    def fields(self) -> list[WrappedField]:
        """
        :return: The list of fields specified only in this associated dataclass that should be mapped.
        """
        if self.is_interface:
            return []

        # fields a mapped base class stores live in that base class's table
        inherited_mapped_names: set[str] = set()
        for table in [*self.primary_chain_tables, *self.mapped_ancestor_tables]:
            inherited_mapped_names.update(table.own_attribute_names)

        # Keep only fields that are not inherited by name
        result = [
            f
            for f in self.wrapped_clazz.fields
            if f.field.name not in inherited_mapped_names
        ]

        # If the parent table is alternatively mapped, drop fields that do not exist
        # in the original parent class (compare by name as well). An external parent's
        # `mapped_field_names` above already reflects exactly what it maps, so this
        # reconciliation only applies to locally generated parents.
        if (
            isinstance(self.parent_table, WrappedTable)
            and self.parent_table.is_alternatively_mapped
        ):
            og_parent_class = self.parent_table.wrapped_clazz.clazz.original_class()
            wrapped_og_parent_class = (
                self.ormatic.class_dependency_graph.get_wrapped_class(
                    clazz=og_parent_class
                )
            )

            og_parent_field_names = {
                f.field.name for f in wrapped_og_parent_class.fields
            }
            parent_dao_field_names = {f.field.name for f in self.parent_table.fields}

            # Fields present in original parent class but removed by the DAO mapping
            removed_by_dao = og_parent_field_names - parent_dao_field_names
            result = [r for r in result if r.field.name not in removed_by_dao]

        return result

    @memoize
    def parse_fields(self) -> None:
        """
        Create the columns and relationships of this table, after those of the tables of
        its mapped base classes, which store the fields those classes declare.
        """
        for table in [*self.primary_chain_tables, *self.mapped_ancestor_tables]:
            if isinstance(table, WrappedTable):
                table.parse_fields()

        for f in self.fields:
            logger.info("=" * 80)
            logger.info(
                f"Processing Field {self.wrapped_clazz.clazz.__name__}.{f.field.name}: {f.field.type}."
            )

            # skip private fields
            if f.field.name.startswith("_"):
                logger.info(f"Skipping since the field starts with _.")
                continue

            self.parse_field(wrapped_field=f)

        self.create_mapper_args()

    def parse_field(self, *, wrapped_field: WrappedField):
        """
        Create the column or relationship that stores the field, in the way the ORMatic
        field classifier decides on.

        :param wrapped_field: The field to store.
        """
        storage = self.ormatic.field_classifier.classify(wrapped_field=wrapped_field)
        if storage is FieldStorage.NOT_STORED:
            return
        column_creators = {
            FieldStorage.TYPE: self.create_type_type_column,
            FieldStorage.BUILTIN: self.create_builtin_column,
            FieldStorage.MANY_TO_ONE: self.create_one_to_one_relationship,
            FieldStorage.MANY_TO_MANY: self.create_many_to_many_relationship,
            FieldStorage.CUSTOM_TYPE: self.create_custom_type,
            FieldStorage.JSON: self.create_json_column,
        }
        column_creators[storage](wrapped_field=wrapped_field)

    def create_builtin_column(self, *, wrapped_field: WrappedField):
        """
        Create a column for a builtin value SQLAlchemy maps by itself.

        :param wrapped_field: The field to create the column for.
        """
        self.builtin_columns.append(
            self.value_column(
                name=wrapped_field.field.name,
                value_type=wrapped_field.type_endpoint,
                nullable=wrapped_field.is_optional,
                storage=ValueStorage.COLUMN,
            )
        )

    def create_type_type_column(self, *, wrapped_field: WrappedField):
        """
        Create a column for a field of type `Type`.

        :param wrapped_field: The field to extract type information from.
        :return:
        """
        column_name = wrapped_field.field.name
        column_type = (
            f"Mapped[TypeType]"
            if not wrapped_field.is_optional
            else f"Mapped[{module_and_class_name(Optional)}[TypeType]]"
        )
        column_constructor = f"mapped_column(TypeType, nullable={wrapped_field.is_optional}, use_existing_column=True)"
        self.custom_columns.append(
            ColumnConstructor(
                name=column_name, type=column_type, constructor=column_constructor
            )
        )

    def get_table_of_class(
        self, *, clazz: type, wrapped_field: WrappedField
    ) -> TableLike:
        """
        :param clazz: A class a field refers to.
        :param wrapped_field: The field referring to it.
        :return: The table, generated here or by a dependency, mapping the class.
        :raises WrappedTableNotFound: If no table maps the class.
        """
        table = self.ormatic.tables_by_class.get(clazz)
        if table is None:
            raise WrappedTableNotFound(type_=clazz, wrapped_field=wrapped_field)
        return table

    def create_one_to_one_relationship(self, *, wrapped_field: WrappedField):
        """
        Create a one-to-one relationship with using the given field.

        This adds a foreign key and a relationship to this table.

        :param wrapped_field: The field to get the information from.
        """
        # create foreign key
        foreign_key_name = (
            f"_{wrapped_field.field.name}{self.ormatic.foreign_key_postfix}"
        )
        foreign_key_type = (
            f"Mapped[{module_and_class_name(Optional)}[{module_and_class_name(int)}]]"
            if wrapped_field.is_optional
            else "Mapped[int]"
        )

        # get the target table
        target_wrapped_table = self.get_table_of_class(
            clazz=wrapped_field.type_endpoint, wrapped_field=wrapped_field
        )

        # columns have to be nullable and use_alter=True since the insertion order might be incorrect otherwise
        foreign_key_column_constructor = f"mapped_column(ForeignKey('{target_wrapped_table.full_primary_key_name}', use_alter=True), nullable=True, use_existing_column=True)"

        self.foreign_keys.append(
            ColumnConstructor(
                name=foreign_key_name,
                type=foreign_key_type,
                constructor=foreign_key_column_constructor,
            )
        )

        # create relationship to remote side
        relationship_name = wrapped_field.field.name
        own_foreign_key = f"{self.tablename}.{foreign_key_name}"
        # relationships have to be post updated since since it won't work in the case of subclasses with another ref otherwise
        # they also stay lazy: eager selectin would cascade through many-to-one
        # cycles at query time; from_dao resolves them via the identity map instead
        if target_wrapped_table.domain_class in self.ormatic.open_classes:
            root = module_and_class_name(StoredObjectDAO)
            self.ormatic.imported_modules.add(
                target_wrapped_table.domain_class.__module__
            )
            relationship_type = f"Mapped[{root}]"
            relationship_arguments = (
                f"{root}, uselist=False, "
                f"primaryjoin=lambda: {own_foreign_key} == {root}.database_id, "
                f"foreign_keys=lambda: [{own_foreign_key}], post_update=True, "
                f"info={{{_information_key(RelationshipInformationKey.DECLARED_TYPE)}: "
                f"{module_and_class_name(target_wrapped_table.domain_class)}}}"
            )
        else:
            relationship_type = f"Mapped[{target_wrapped_table.tablename}]"
            relationship_arguments = (
                f"'{target_wrapped_table.tablename}', uselist=False, "
                f"foreign_keys=lambda: [{own_foreign_key}], post_update=True"
            )
        self.add_relationship(
            column_constructor=ColumnConstructor(
                name=relationship_name,
                type=relationship_type,
                constructor=f"relationship({relationship_arguments})",
            )
        )

    def add_relationship(self, *, column_constructor: ColumnConstructor) -> None:
        """
        Add a relationship to this table, and its copy to the table's part class.

        :param column_constructor: The relationship.
        """
        self.relationships.append(column_constructor)
        self.part_relationships.append(
            ColumnConstructor(
                name=column_constructor.name,
                type=column_constructor.type,
                constructor=f"{column_constructor.constructor[:-1]}, overlaps='{column_constructor.name}')",
            )
        )

    def create_many_to_many_relationship(self, *, wrapped_field: WrappedField):
        """
        Store the collection the field holds in an association table, with a row per
        value, and add the relationship reading its rows in the collection's order.

        :param wrapped_field: The field holding the collection.
        """
        field_to_store = FieldToStore(wrapped_field=wrapped_field, ormatic=self.ormatic)
        shape = field_to_store.collection_shape
        association_table_name = (
            f"{self.tablename}_{wrapped_field.field.name}_association"
        )

        key_references = {}
        key_columns = {}
        for level in shape.levels:
            if not isinstance(level, MappingLevel):
                continue
            nullable = level.depth > 1
            key_storage = field_to_store.value_storage(value_type=level.key_type)
            if key_storage is ValueStorage.REFERENCE:
                key_table = self.get_table_of_class(
                    clazz=level.key_type, wrapped_field=wrapped_field
                )
                key_references[level.depth] = self.association_reference(
                    relationship_name=level.column_names.key,
                    foreign_key_name=f"_{level.column_names.key}_{key_table.tablename.lower()}{self.ormatic.foreign_key_postfix}",
                    table=key_table,
                    nullable=nullable,
                )
            else:
                key_columns[level.depth] = self.value_column(
                    name=level.column_names.key,
                    value_type=level.key_type,
                    nullable=nullable,
                    storage=key_storage,
                )

        target_reference = None
        value_column = None
        leaf_storage = field_to_store.value_storage(value_type=shape.leaf_type)
        if leaf_storage is ValueStorage.REFERENCE:
            target_table = self.get_table_of_class(
                clazz=shape.leaf_type, wrapped_field=wrapped_field
            )
            # the source_/target_ prefixes keep the columns of a self-referential
            # collection apart
            target_reference = self.association_reference(
                relationship_name=AssociationLeafAttribute.TARGET,
                foreign_key_name=f"_target_{target_table.tablename.lower()}{self.ormatic.foreign_key_postfix}",
                table=target_table,
                nullable=shape.is_nested,
            )
        else:
            value_column = self.value_column(
                name=AssociationLeafAttribute.VALUE,
                value_type=shape.leaf_type,
                nullable=shape.is_nested,
                storage=leaf_storage,
            )

        association_table = AssociationObject(
            name=association_table_name,
            left_foreign_key=f"_source_{self.tablename.lower()}{self.ormatic.foreign_key_postfix}",
            left_primary_key=self.full_primary_key_name,
            primary_key_name="database_id",
            shape=shape,
            key_references=key_references,
            key_columns=key_columns,
            target_reference=target_reference,
            value_column=value_column,
        )
        self.ormatic.association_objects.append(association_table)

        container_name = module_and_class_name(list)
        collection_class = association_table.collection_class_constructor(
            ormatic=self.ormatic
        )
        ordering_columns = ", ".join(
            f"{association_table.name}.{column_name}"
            for column_name in association_table.index_column_names
        )
        self.add_relationship(
            column_constructor=ColumnConstructor(
                name=wrapped_field.field.name,
                type=f"Mapped[{container_name}[{association_table.name}]]",
                constructor=(
                    f"relationship('{association_table.name}', "
                    f"collection_class={collection_class}, "
                    f"cascade='all, delete-orphan', "
                    f"foreign_keys='[{association_table.name}.{association_table.left_foreign_key}]', "
                    f"order_by='[{ordering_columns}]', "
                    f"lazy='selectin')"
                ),
            )
        )

    def association_reference(
        self,
        *,
        relationship_name: str,
        foreign_key_name: str,
        table: TableLike,
        nullable: bool,
    ) -> AssociationReference:
        """
        :param relationship_name: The name of the relationship reading the object.
        :param foreign_key_name: The name of the foreign key column.
        :param table: The table of the class of the objects referred to.
        :param nullable: Whether a row may refer to no object.
        :return: The reference of an association table of this table to objects of the
            class of *table*.
        """
        is_open = table.domain_class in self.ormatic.open_classes
        if is_open:
            self.ormatic.imported_modules.add(table.domain_class.__module__)
        return AssociationReference(
            relationship_name=relationship_name,
            foreign_key_name=foreign_key_name,
            table=table,
            is_open=is_open,
            nullable=nullable,
        )

    def value_column(
        self, *, name: str, value_type: type, nullable: bool, storage: ValueStorage
    ) -> ColumnConstructor:
        """
        :param name: The name of the column.
        :param value_type: The type of the values the column holds.
        :param nullable: Whether the column may hold None.
        :param storage: How the values are stored, :attr:`ValueStorage.COLUMN` or
            :attr:`ValueStorage.JSON`.
        :return: A column holding the values: for a column of their own type, text for
            a string, the type mapping's column type for a mapped type, else the type
            SQLAlchemy derives.
        """
        if storage is ValueStorage.JSON:
            self.ormatic.imported_modules.add("typing_extensions")
            annotation = module_and_class_name(Any)
            column_type = "JSON"
        else:
            self.ormatic.imported_modules.add(value_type.__module__)
            annotation = module_and_class_name(value_type)
            column_type = self._column_type(value_type=value_type)
        if nullable:
            annotation = f"{module_and_class_name(Optional)}[{annotation}]"
        arguments = [
            *([column_type] if column_type is not None else []),
            f"nullable={nullable}",
            "use_existing_column=True",
        ]
        return ColumnConstructor(
            name=name,
            type=f"Mapped[{annotation}]",
            constructor=f"mapped_column({', '.join(arguments)})",
        )

    def _column_type(self, *, value_type: type) -> str | None:
        """
        :param value_type: A type with a column type of its own.
        :return: The column type to name for values of the type, or None where
            SQLAlchemy derives it from the annotation.
        """
        if value_type is str:
            return module_and_class_name(sqlalchemy.types.Text)
        if value_type in FieldToStore.sqlalchemy_builtins:
            return None
        if value_type not in self.ormatic.type_mappings:
            return None
        custom_type = self.ormatic.type_mappings[value_type]
        # registered under the exact type, so the generated interface maps it
        self.ormatic.type_mappings[value_type] = custom_type
        return module_and_class_name(custom_type)

    def create_json_column(self, *, wrapped_field: WrappedField):
        """
        Create a JSON column for a value, or a collection of values, that krrood's JSON
        serializer can write.

        :param wrapped_field: The field to extract the information from.
        """
        type_endpoint = wrapped_field.type_endpoint
        self.ormatic.imported_modules.add("typing_extensions")
        column_name = wrapped_field.field.name
        shape = FieldToStore(
            wrapped_field=wrapped_field, ormatic=self.ormatic
        ).collection_shape
        is_flat_collection_of_one_class = (
            wrapped_field.is_container
            and shape is not None
            and not shape.is_nested
            and not shape.key_types
            and isclass(type_endpoint)
        )
        if shape is not None and not is_flat_collection_of_one_class:
            column_type = f"Mapped[{module_and_class_name(Any)}]"
        elif not wrapped_field.is_container:
            self.ormatic.imported_modules.add(type_endpoint.__module__)
            value_type = module_and_class_name(type_endpoint)
            column_type = (
                f"Mapped[{module_and_class_name(Optional)}[{value_type}]]"
                if wrapped_field.is_optional
                else f"Mapped[{value_type}]"
            )
        else:
            self.ormatic.imported_modules.add(type_endpoint.__module__)
            value_type = module_and_class_name(type_endpoint)
            container = (
                Set
                if isclass(wrapped_field.container_type)
                and issubclass(wrapped_field.container_type, set)
                else List
            )
            column_type = f"Mapped[{module_and_class_name(container)}[{value_type}]]"
        column_constructor = f"mapped_column(JSON, nullable={wrapped_field.is_optional}, use_existing_column=True)"
        self.custom_columns.append(
            ColumnConstructor(
                name=column_name, type=column_type, constructor=column_constructor
            )
        )

    def create_custom_type(self, *, wrapped_field: WrappedField):
        """
        Create a column of the type the type mappings give for the field's type.

        :param wrapped_field: The field to create the column for.
        """
        self.custom_columns.append(
            self.value_column(
                name=wrapped_field.field.name,
                value_type=wrapped_field.type_endpoint,
                nullable=wrapped_field.is_optional,
                storage=ValueStorage.COLUMN,
            )
        )

    @property
    def base_class_name(self) -> str:
        """
        :return: The reference to the DAO class of the parent table, or the declarative
            base for a root table.
        """
        if self.parent_table is None:
            return "Base"
        return self.parent_table.qualified_reference

    def __hash__(self):
        return hash(self.wrapped_clazz)
