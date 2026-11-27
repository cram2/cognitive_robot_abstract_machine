"""
The data access object that maps a class to its database rows and back.
"""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from itertools import batched

import sqlalchemy.inspection
import sqlalchemy.orm
from sqlalchemy import event
from sqlalchemy.sql.base import ExecutableOption
from sqlalchemy.orm import (
    MANYTOONE,
    MANYTOMANY,
    ONETOMANY,
    RelationshipProperty,
    make_transient_to_detached,
    selectinload,
)
from sortedcontainers import SortedSet
from typing_extensions import get_origin, Any, Callable, Sequence, TypeVar

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute, IndexByValue
from krrood.entity_query_language.query.match import AbstractMatchExpression
from krrood.entity_query_language._monitoring import monitored
from krrood.ormatic.base import Base
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.from_dao import (
    FromDataAccessObjectWorkItem,
    FromDataAccessObjectState,
    ObjectInheritingFromAlternativeMapping,
)
from krrood.ormatic.data_access_objects.helper import (
    get_data_access_object_class,
    get_part_class,
    to_dao,
    clear_dao_lookup_caches,
)
from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState
from krrood.ormatic.exceptions import (
    NoDAOFoundDuringParsingError,
    PartOfDetachedObjectNotLoadedError,
    QueryCannotBePersisted,
    StoredObjectTableNotUniqueError,
)
from krrood.patterns.called_positionally import called_positionally
from krrood.patterns.classproperty import classproperty
from krrood.patterns.exceptions import NoGenericError
from krrood.patterns.has_generic import HasGeneric
from krrood.ormatic.utils import is_data_column, _get_type_hints_cached

logger = logging.getLogger(__name__)
_repr_thread_local = threading.local()

T = TypeVar("T")
_DAO = TypeVar("_DAO", bound="DataAccessObject")


class AssociationDataAccessObject:
    """
    Base class for association objects in the Data Access Object layer.
    Association objects are used to map many-to-many relationships that
    require additional information or identity for each association,
    such as when duplicates are allowed in a collection.
    """

    @property
    def target(self) -> DataAccessObject:
        """
        :return: The target Data Access Object of this association.
        """
        raise NotImplementedError

    @target.setter
    def target(self, value: DataAccessObject) -> None:
        """
        :param value: The target Data Access Object of this association.
        """
        raise NotImplementedError


class RelationshipInformationKey(StrEnum):
    """
    Keys of the information a generated relationship carries for the conversion between
    DAOs and domain objects.
    """

    DECLARED_TYPE = "krrood_declared_type"
    """
    The type a field is declared with, for a relationship that targets the table of
    every stored object because objects of that type may lie outside the type's table
    hierarchy.
    """


def _declared_type(relationship: RelationshipProperty) -> type:
    """
    :param relationship: A relationship of a DAO class.
    :return: The domain type of the objects the relationship holds.
    """
    declared_type = relationship.info.get(RelationshipInformationKey.DECLARED_TYPE)
    if declared_type is not None:
        return declared_type
    return relationship.mapper.class_.original_class()


@dataclass(kw_only=True, frozen=True)
class SingleRelationship:
    """
    Metadata for a single-valued (many-to-one or one-to-one) DAO relationship.
    """

    key: str
    """The attribute name of the relationship on the DAO."""

    domain_type: type
    """The expected domain type of the related object."""

    foreign_key_attribute: str | None = None
    """
    The attribute name of the local foreign key column for many-to-one
    relationships, or ``None`` if the foreign key is on the remote side.
    """

    target_dao_class: type | None = None
    """The DAO class this relationship points to."""


@dataclass(kw_only=True, frozen=True)
class CollectionRelationship:
    """
    Metadata for a collection-valued (one-to-many or many-to-many) DAO relationship.
    """

    key: str
    """The attribute name of the relationship on the DAO."""

    association_class: type | None
    """
    The association DAO class used as an intermediary, or ``None`` for direct
    collection relationships.
    """

    domain_type: type
    """The expected domain type of the items in the collection."""

    association_target_relationship: SingleRelationship | None = None
    """
    The ``target`` relationship of the association class, if any. Used to
    resolve association targets via the identity map instead of lazy loads.
    """

    target_dao_class: type
    """The DAO class of the items in the collection."""


@dataclass(kw_only=True, frozen=True)
class DataAccessObjectSchema:
    """
    Precomputed, class-level metadata used by the to_dao/from_dao hot paths.

    All information in here is derived from the SQLAlchemy mapper and the generic
    arguments of a DAO class. Computing it once per class avoids repeated mapper
    introspection, relationship classification and MRO walks per converted instance.
    """

    data_column_names: tuple[str, ...]
    """
    Names of all data columns (no primary keys, foreign keys or polymorphic markers).
    """

    single_relationships: tuple[SingleRelationship, ...]
    """
    Single-valued relationships of this DAO class.
    """

    collection_relationships: tuple[CollectionRelationship, ...]
    """
    Collection-valued relationships of this DAO class.
    """

    relationship_keys: tuple[str, ...]
    """
    The keys of all relationships of the DAO class.
    """

    part_classes: tuple[type[PartDataAccessObject], ...]
    """
    The classes of the parts holding the rows of the DAO's objects in the tables of
    their mapped base classes outside the DAO class's own tables.
    """

    shares_id_with_stored_object: bool
    """
    Whether the DAO's object takes its id from its row in the table of every stored
    object.
    """

    alternative_base: type | None
    """
    The first base class of the DAO that maps to an AlternativeMapping, if any.
    """

    uses_alternative_mapping: bool
    """
    Whether the DAO class itself maps to an AlternativeMapping.
    """


def _build_single_relationship(
    relationship: RelationshipProperty, domain_type: type
) -> SingleRelationship:
    """
    Build a :class:`SingleRelationship` descriptor from a SQLAlchemy relationship property.

    For many-to-one relationships the local foreign key column is recorded as
    ``foreign_key_attribute``.  During ``from_dao`` this lets
    :func:`_read_single_relationship` resolve the related instance via
    ``Session.get`` (an identity-map lookup) instead of triggering a lazy load.
    SQLAlchemy disables its built-in identity-map shortcut (``use_get``) for
    targets deep in joined-table inheritance hierarchies, so without this
    explicit FK read every many-to-one access on an unloaded attribute would
    emit a separate SELECT.

    :param relationship: The SQLAlchemy relationship property to inspect.
    :param domain_type: The expected domain type of the related object.
    :return: A :class:`SingleRelationship` ready for use in a
        :class:`DataAccessObjectSchema`.
    """
    local_columns = list(relationship.local_columns)
    foreign_key_attribute = (
        local_columns[0].key
        if relationship.direction == MANYTOONE and len(local_columns) == 1
        else None
    )
    return SingleRelationship(
        key=relationship.key,
        domain_type=domain_type,
        foreign_key_attribute=foreign_key_attribute,
        target_dao_class=relationship.mapper.class_,
    )


@lru_cache(maxsize=None)
def get_dao_schema(dao_class: type) -> DataAccessObjectSchema:
    """
    Build (and cache) the conversion plan for a DAO class.

    :param dao_class: The DAO class to build the plan for.
    :return: The conversion plan.
    """
    mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(dao_class)

    single_relationships = []
    collection_relationships = []
    for relationship in mapper.relationships:
        if DataAccessObject._is_single_relationship(relationship=relationship):
            single_relationships.append(
                _build_single_relationship(relationship, _declared_type(relationship))
            )
        elif relationship.direction in (ONETOMANY, MANYTOMANY):
            target_dao_clazz = relationship.mapper.class_
            if issubclass(target_dao_clazz, AssociationDataAccessObject):
                target_relationship = sqlalchemy.inspection.inspect(
                    target_dao_clazz
                ).relationships["target"]
                domain_type = _declared_type(target_relationship)
                collection_relationships.append(
                    CollectionRelationship(
                        key=relationship.key,
                        association_class=target_dao_clazz,
                        domain_type=domain_type,
                        association_target_relationship=_build_single_relationship(
                            target_relationship, domain_type
                        ),
                        target_dao_class=target_relationship.mapper.class_,
                    )
                )
            else:
                collection_relationships.append(
                    CollectionRelationship(
                        key=relationship.key,
                        association_class=None,
                        domain_type=target_dao_clazz.original_class(),
                        target_dao_class=target_dao_clazz,
                    )
                )

    return DataAccessObjectSchema(
        data_column_names=tuple(
            column.name for column in mapper.columns if is_data_column(column)
        ),
        single_relationships=tuple(single_relationships),
        collection_relationships=tuple(collection_relationships),
        relationship_keys=tuple(
            relationship.key for relationship in mapper.relationships
        ),
        part_classes=_part_classes(dao_class=dao_class, mapper=mapper),
        shares_id_with_stored_object=_shares_id_with_stored_object(
            dao_class=dao_class, mapper=mapper
        ),
        alternative_base=dao_class._find_alternative_mapping_base(),
        uses_alternative_mapping=_uses_alternative_mapping(dao_class),
    )


def _shares_id_with_stored_object(
    *, dao_class: type[DataAccessObject], mapper: sqlalchemy.orm.Mapper
) -> bool:
    """
    :param dao_class: A DAO class.
    :param mapper: The mapper of the DAO class.
    :return: Whether the DAO class's objects take their ids from their rows in
        the table of every stored object.
    """
    return not issubclass(dao_class, PartDataAccessObject) and any(
        foreign_key.column.table is StoredObjectRow.mapped_class().__table__
        for column in mapper.base_mapper.local_table.primary_key
        for foreign_key in column.foreign_keys
    )


def _part_classes(
    *, dao_class: type[DataAccessObject], mapper: sqlalchemy.orm.Mapper
) -> tuple[type[PartDataAccessObject], ...]:
    """
    :param dao_class: A DAO class.
    :param mapper: The mapper of the DAO class.
    :return: The part classes of the mapped base classes of the DAO class's domain class
        whose tables lie outside the mapper's tables.
    """
    if not _shares_id_with_stored_object(dao_class=dao_class, mapper=mapper):
        return ()
    domain_class = dao_class.original_class()
    if _uses_alternative_mapping(dao_class):
        domain_class = domain_class.original_class()
    domain_class = get_origin(domain_class) or domain_class
    result = []
    for base in domain_class.__mro__[1:]:
        part_class = get_part_class(base)
        if (
            part_class is not None
            and part_class not in result
            and not any(part_class.__table__ is table for table in mapper.tables)
        ):
            result.append(part_class)
    return tuple(result)


@dataclass(kw_only=True, frozen=True)
class AlternativePartitionPlan:
    """
    Precomputed partition of columns and relationships between a DAO class and its
    alternatively mapped base class.
    """

    parent_data_column_names: tuple[str, ...]
    """
    Data column names that belong to the alternatively mapped base.
    """

    own_data_column_names: tuple[str, ...]
    """
    Data column names that belong to this DAO's own tables.
    """

    intermediate_attribute_keys: tuple[str, ...]
    """
    Column attribute keys of intermediate ancestors not covered by the parent.
    """

    parent_single_relationships: tuple[SingleRelationship, ...]
    """
    Single-valued relationships that belong to the alternatively mapped base.
    """

    parent_collection_relationships: tuple[CollectionRelationship, ...]
    """
    Collection relationships that belong to the alternatively mapped base.
    """

    own_single_relationships: tuple[SingleRelationship, ...]
    """
    Single-valued relationships that belong to this DAO's own tables,
    i.e. those not covered by the alternatively mapped base.
    """

    own_collection_relationships: tuple[CollectionRelationship, ...]
    """
    Collection relationships that belong to this DAO's own tables,
    i.e. those not covered by the alternatively mapped base.
    """


@lru_cache(maxsize=None)
def _get_alternative_partition_plan(
    dao_class: type, alternative_base: type
) -> AlternativePartitionPlan:
    """
    Build (and cache) the column/relationship partition between a DAO class and its
    alternatively mapped base class.

    :param dao_class: The DAO class.
    :param alternative_base: The alternatively mapped base class.
    :return: The partition plan.
    """
    mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(dao_class)
    parent_mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(
        alternative_base
    )
    plan = get_dao_schema(dao_class)
    parent_plan = get_dao_schema(alternative_base)

    parent_column_names = {column.name for column in parent_mapper.columns}
    parent_relationship_keys = set(parent_plan.relationship_keys)

    return AlternativePartitionPlan(
        parent_data_column_names=parent_plan.data_column_names,
        own_data_column_names=tuple(
            column.name
            for column in mapper.columns
            if column.name not in parent_column_names and is_data_column(column)
        ),
        intermediate_attribute_keys=tuple(
            prop.key
            for prop in mapper.column_attrs
            if prop.key not in parent_column_names and is_data_column(prop.columns[0])
        ),
        parent_single_relationships=parent_plan.single_relationships,
        parent_collection_relationships=parent_plan.collection_relationships,
        own_single_relationships=tuple(
            entry
            for entry in plan.single_relationships
            if entry.key not in parent_relationship_keys
        ),
        own_collection_relationships=tuple(
            entry
            for entry in plan.collection_relationships
            if entry.key not in parent_relationship_keys
        ),
    )


@lru_cache(maxsize=None)
def _uses_alternative_mapping(clazz: type) -> bool:
    """
    :param clazz: The class to check.
    :return: Whether the class is a DAO whose original class is an AlternativeMapping.
    """
    try:
        return issubclass(clazz, DataAccessObject) and issubclass(
            clazz.original_class(), AlternativeMapping
        )
    except (AttributeError, TypeError, NoGenericError):
        return False


@lru_cache(maxsize=None)
def _has_post_init(clazz: type) -> bool:
    """
    :param clazz: The class to check.
    :return: Whether the class defines a ``__post_init__``.
    """
    return hasattr(clazz, "__post_init__")


class _SessionInfoKey(StrEnum):
    """Session-info keys written into ``Session.info`` by the krrood ORM layer."""

    BULK_LOADING_FLAG = "krrood_bulk_loading"
    BULK_LOADED_MAPPERS = "krrood_bulk_loaded_mappers"
    PLAIN_LOAD = "krrood_plain_load"
    OWNERS_OF_UNLOADED_PARTS = "krrood_owners_of_unloaded_parts"
    """
    The loaded DAOs, by the classes of the parts they have but whose part of that
    class was not loaded yet.
    """


def _bulk_load_target_table_if_enabled(
    session: sqlalchemy.orm.Session, target_dao_class: type
) -> None:
    """
    Load all rows of the target's root mapper into the identity map, once per
    session and root mapper. Only active inside :func:`selectin_loading`.

    When a whole object graph is reconstructed, every row of a referenced class
    is typically needed, so loading the full table once is far cheaper than one
    SELECT per instance.

    :param session: The session to load into.
    :param target_dao_class: The DAO class about to be fetched.
    """
    if not session.info.get(_SessionInfoKey.BULK_LOADING_FLAG, False):
        return
    root_mapper = sqlalchemy.inspection.inspect(target_dao_class).base_mapper
    loaded = session.info.setdefault(_SessionInfoKey.BULK_LOADED_MAPPERS, {})
    if root_mapper in loaded:
        return
    # keep strong references to the rows: the identity map is weak, so the
    # instances would be garbage collected (and the map emptied) otherwise
    loaded[root_mapper] = session.scalars(
        sqlalchemy.select(root_mapper.class_).execution_options(
            **{_SessionInfoKey.PLAIN_LOAD: True}
        )
    ).all()


def _read_single_relationship(
    instance: Any, relationship: SingleRelationship
) -> DataAccessObject | None:
    """
    Read the value of a single-valued relationship on a DAO or association object.

    For unloaded many-to-one relationships this resolves the target through
    ``Session.get`` using the local foreign key, which hits the identity map.
    Plain attribute access would emit one SELECT per access, because SQLAlchemy
    disables its identity-map shortcut (``use_get``) for targets deep in
    joined-table inheritance hierarchies.

    :param instance: The object to read the relationship from.
    :param relationship: The relationship to read.
    :return: The related DAO instance or None.
    """
    key = relationship.key
    if relationship.foreign_key_attribute is None or key in instance.__dict__:
        return getattr(instance, key)
    session = sqlalchemy.orm.object_session(instance)
    if session is None:
        return getattr(instance, key)
    foreign_key_value = getattr(instance, relationship.foreign_key_attribute)
    if foreign_key_value is not None:
        _bulk_load_target_table_if_enabled(session, relationship.target_dao_class)
    value = (
        session.get(
            relationship.target_dao_class,
            foreign_key_value,
            execution_options={_SessionInfoKey.PLAIN_LOAD: True},
        )
        if foreign_key_value is not None
        else None
    )
    sqlalchemy.orm.attributes.set_committed_value(instance, key, value)
    return value


@lru_cache(maxsize=None)
def _get_set_field_names(clazz: type) -> tuple[str, ...]:
    """
    :param clazz: The domain class to inspect.
    :return: The names of all fields annotated as sets.
    """
    return tuple(
        attr_name
        for attr_name, hint in _get_type_hints_cached(clazz).items()
        if get_origin(hint) is set or hint is set
    )


@lru_cache(maxsize=None)
def _get_containers_of_list_held_fields(clazz: type) -> dict[str, type]:
    """
    :param clazz: The domain class to inspect.
    :return: The container class of every field annotated with one of
        :attr:`DataAccessObject.containers_held_as_lists`, by field name.
    """
    return {
        attr_name: get_origin(hint) or hint
        for attr_name, hint in _get_type_hints_cached(clazz).items()
        if (get_origin(hint) or hint) in DataAccessObject.containers_held_as_lists
    }


class DataAccessObject(HasGeneric[T]):
    """
    Base class for Data Access Objects (DAOs) providing bidirectional conversion between
    domain objects and SQLAlchemy models.

    This class automates the mapping between complex domain object graphs and relational
    database schemas using SQLAlchemy. It supports inheritance, circular references,
    and custom mappings via :class:`AlternativeMapping`.

    Conversion Directions
    ---------------------

    1. **Domain to DAO (to_dao)**:
       Converts a domain object into its DAO representation. It uses an iterative
       BFS approach with a queue of work items to traverse the object graph. New work items
       for nested relationships are added to the queue during processing, ensuring all
       reachable objects are converted while maintaining the BFS order.

    2. **DAO to Domain (from_dao)**:
       Converts a DAO back into a domain object using a Four-Phase Iterative Approach:

       - Phase 1: Allocation & Discovery (DFS):
         Traverses the DAO relationships to identify all reachable DAOs. For each DAO, it
         allocates an uninitialized domain object (or alternative mapping) (using ``__new__``) and records
         the discovery order.
       - Phase 2: Population & Alternative Mapping Resolution (Bottom-Up):
         Populates every field of the domain objects using ``setattr``. This avoids
         the complexities of constructor matching and ensures that circular
         references are handled correctly by using the already allocated identities.
       - Phase 3: For every field, if the value is an ``AlternativeMapping``, it is converted to its final
         domain object representation.
         During this phase, collections are represented as lists.
       - Phase 3: Container Finalization:
         Convert containers that are currently lists but should be something else (e. g. sets) to the container from
         the type hint.
       - Phase 4: Post-Initialization:
         Calls ``__post_init__`` on all fully populated and finalized domain objects but not on the alternative mappings.


    Alternative Mappings
    --------------------

    For domain objects that do not map 1:1 to a single DAO (e.g., those requiring
    special constructor logic) :class:`AlternativeMapping` can be used. The converter recognizes these and
    delegates the creation of the domain object to the mapping's ``create_from_dao``
    method during the Filling Phase.

    """

    stored_object_in_memory: StoredObjectRow | None = None
    """
    The row of the DAO's object in the table of every stored object, once it was asked
    for.
    """

    parts_in_memory: (
        dict[type[PartDataAccessObject], PartDataAccessObject | None] | None
    ) = None
    """
    The parts of the DAO's object by their classes, once created or loaded; None for a
    part that the object does not have.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # New DAO classes invalidate previously failed (None) lookups.
        clear_dao_lookup_caches()

    @classproperty
    def containers_held_as_lists(cls) -> tuple[type, ...]:
        """
        :return: The domain containers that SQLAlchemy cannot instrument as the
            collection of a relationship, so that a relationship holds them as a list
            and loading an object rebuilds them from it.
        """
        return tuple, SortedSet

    @classproperty
    def containers_that_sort_their_elements(cls) -> tuple[type, ...]:
        """
        :return: The :attr:`containers_held_as_lists` that compare their elements, so that
            loading an object rebuilds them only once every loaded object is
            initialised.
        """
        return (SortedSet,)

    # %% conversion to dao routines
    @classmethod
    def to_dao(
        cls,
        *,
        source_object: T,
        state: ToDataAccessObjectState | None = None,
        register: bool = True,
    ) -> _DAO:
        """
        Convert an object to its Data Access Object.

        :param source_object: The object to convert.
        :param state: The conversion state.
        :param register: Whether to register the result in the memo.
        :return: The converted DAO instance.
        :raises QueryCannotBePersisted: If the object is an entity query language query.
        """
        cls._refuse_queries(source_object=source_object)
        state = state or ToDataAccessObjectState()

        # Phase 1: Resolution - Check memo and apply alternative mappings
        existing = state.get(source=source_object)
        if existing is not None:
            return existing

        resolved_source = state.apply_alternative_mapping_if_needed(
            dao_clazz=cls, source_object=source_object
        )

        # Phase 2: Allocation & Registration
        result = cls()

        if register:
            state.register(source=source_object, target=result)
            if id(source_object) != id(resolved_source):
                state.register(source=resolved_source, target=result)

        # Phase 3: Queueing & Processing
        is_entry_call = len(state.work_items) == 0
        alternative_base = get_dao_schema(cls).alternative_base
        state.push_work_item(
            source_object=resolved_source,
            dao_instance=result,
            alternative_base=alternative_base,
        )

        if is_entry_call:
            cls._process_to_dao_queue(state=state)

        return result

    @property
    def stored_object(self) -> StoredObjectRow:
        """
        :return: The row of this DAO's object in the table of every stored object,
            created for an object that was never stored.
        """
        if self.stored_object_in_memory is not None:
            return self.stored_object_in_memory
        stored_object_class = StoredObjectRow.mapped_class()
        if self.database_id is None:
            self.stored_object_in_memory = stored_object_class(
                polymorphic_type=self.__tablename__
            )
            self.stored_object_in_memory.object_dao_in_memory = self
        else:
            self.stored_object_in_memory = sqlalchemy.orm.object_session(self).get(
                stored_object_class, self.database_id
            )
        return self.stored_object_in_memory

    @staticmethod
    def _refuse_queries(*, source_object: Any) -> None:
        """
        :param source_object: An object that is to be stored.
        :raises QueryCannotBePersisted: If the object is an entity query language query.
        """
        if isinstance(source_object, (AbstractMatchExpression, SymbolicExpression)):
            raise QueryCannotBePersisted(query=source_object)

    @classmethod
    def _process_to_dao_queue(cls, *, state: ToDataAccessObjectState) -> None:
        """
        Process the work items for converting objects to DAOs.

        This uses a Breadth-First Search (BFS) approach by processing the deque
        as a FIFO queue (popleft). New work items for nested relationships are
        added to the queue during processing.

        :param state: The conversion state containing the work_items.
        """
        while state.work_items:
            work_item = state.work_items.popleft()
            if work_item.alternative_base is not None:
                work_item.dao_instance.fill_dao_if_subclass_of_alternative_mapping(
                    source_object=work_item.source_object,
                    alternative_base=work_item.alternative_base,
                    state=state,
                )
            else:
                work_item.dao_instance.fill_dao_default(
                    source_object=work_item.source_object, state=state
                )

    @classmethod
    def _find_alternative_mapping_base(cls) -> type[DataAccessObject] | None:
        """
        Find the first base class using an alternative mapping.

        :return: The base class or None.
        """
        for base_clazz in cls.__mro__[1:]:
            try:
                if issubclass(base_clazz, DataAccessObject) and issubclass(
                    base_clazz.original_class(), AlternativeMapping
                ):
                    return base_clazz
            except (AttributeError, TypeError, NoGenericError):
                continue
        return None

    def fill_dao_default(
        self, *, source_object: T, state: ToDataAccessObjectState
    ) -> None:
        """
        Populate the DAO instance from a source object.

        :param source_object: The source object.
        :param state: The conversion state.
        """
        plan = get_dao_schema(type(self))

        for name in plan.data_column_names:
            setattr(self, name, getattr(source_object, name))
        self._fill_relationships_from_plan(
            source_object=source_object,
            single_relationships=plan.single_relationships,
            collection_relationships=plan.collection_relationships,
            state=state,
        )
        self._fill_parts(
            source_object=source_object, part_classes=plan.part_classes, state=state
        )

    def _fill_parts(
        self,
        *,
        source_object: T,
        part_classes: tuple[type[PartDataAccessObject], ...],
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Give this DAO the parts holding its object's rows in the tables of its secondary
        bases, filled from the source object.

        :param source_object: The source object.
        :param part_classes: The classes of the parts to fill.
        :param state: The conversion state.
        """
        self.parts_in_memory = {}
        for part_class in part_classes:
            part_dao = part_class()
            type_column_key = part_class.type_column_key()
            if type_column_key is not None:
                setattr(part_dao, type_column_key, type(self).__tablename__)
            part_dao.fill_dao_default(source_object=source_object, state=state)
            self.parts_in_memory[part_class] = part_dao

    def fill_dao_if_subclass_of_alternative_mapping(
        self,
        *,
        source_object: T,
        alternative_base: type[DataAccessObject],
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Populate the DAO instance for an alternatively mapped subclass.

        :param source_object: The source object.
        :param alternative_base: The base class using alternative mapping.
        :param state: The conversion state.
        """
        # Temporarily remove the object from the memo to allow the parent DAO to be created separately
        temp_dao = state.pop(source=source_object)

        # create dao of alternatively mapped superclass
        parent_dao = alternative_base.original_class().to_dao(
            source_object=source_object, state=state
        )

        # Restore the object in the memo dictionary
        if temp_dao is not None:
            state.register(source=source_object, target=temp_dao)

        partition = _get_alternative_partition_plan(type(self), alternative_base)

        # Copy values from parent DAO and original object
        for name in partition.parent_data_column_names:
            setattr(self, name, getattr(parent_dao, name))
        for name in partition.own_data_column_names:
            setattr(self, name, getattr(source_object, name))

        # Ensure columns on intermediate ancestors are also covered
        for key in partition.intermediate_attribute_keys:
            setattr(self, key, getattr(source_object, key))

        # Fill the partitioned relationships
        self._fill_relationships_from_plan(
            source_object=parent_dao,
            single_relationships=partition.parent_single_relationships,
            collection_relationships=partition.parent_collection_relationships,
            state=state,
        )
        self._fill_relationships_from_plan(
            source_object=source_object,
            single_relationships=partition.own_single_relationships,
            collection_relationships=partition.own_collection_relationships,
            state=state,
        )
        self._fill_parts(
            source_object=source_object,
            part_classes=get_dao_schema(type(self)).part_classes,
            state=state,
        )

    @staticmethod
    def _is_single_relationship(*, relationship: RelationshipProperty) -> bool:
        """
        Check if a relationship is single-valued.

        :param relationship: The relationship to check.
        :return: True if single-valued.
        """
        return relationship.direction == MANYTOONE or (
            relationship.direction == ONETOMANY and not relationship.uselist
        )

    def _fill_relationships_from_plan(
        self,
        *,
        source_object: Any,
        single_relationships: tuple[SingleRelationship, ...],
        collection_relationships: tuple[CollectionRelationship, ...],
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Populate relationships from a source object using a conversion plan.

        :param source_object: The source of relationship values.
        :param single_relationships: The single-valued relationship entries.
        :param collection_relationships: The collection relationship entries.
        :param state: The conversion state.
        """
        for single_relationship in single_relationships:
            self._fill_single_relationship(
                source_object=source_object,
                relationship=single_relationship,
                state=state,
            )

        for collection_relationship in collection_relationships:
            self._fill_collection_relationship(
                source_object=source_object,
                relationship=collection_relationship,
                state=state,
            )

    def _fill_single_relationship(
        self,
        *,
        source_object: Any,
        relationship: SingleRelationship,
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Populate a single-valued relationship attribute from a source object.

        :param source_object: The source of the relationship value.
        :param relationship: The single-valued relationship entry.
        :param state: The conversion state.
        """
        value = getattr(source_object, relationship.key)
        if value is None:
            setattr(self, relationship.key, None)
        else:
            setattr(
                self,
                relationship.key,
                self._get_or_queue_referenced_dao(
                    source_object=value,
                    state=state,
                    expected_type=relationship.domain_type,
                    target_dao_class=relationship.target_dao_class,
                ),
            )

    def _fill_collection_relationship(
        self,
        *,
        source_object: Any,
        relationship: CollectionRelationship,
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Populate a collection-valued relationship attribute from a source object.

        :param source_object: The source of the relationship values.
        :param relationship: The collection-valued relationship entry.
        :param state: The conversion state.
        """
        source_collection = getattr(source_object, relationship.key)

        if relationship.association_class is not None:
            dao_collection = []
            for item in source_collection:
                association_dao = relationship.association_class()
                association_dao.target = self._get_or_queue_referenced_dao(
                    source_object=item,
                    state=state,
                    expected_type=relationship.domain_type,
                    target_dao_class=relationship.target_dao_class,
                )
                dao_collection.append(association_dao)
        else:
            dao_collection = [
                self._get_or_queue_referenced_dao(
                    source_object=item,
                    state=state,
                    expected_type=relationship.domain_type,
                    target_dao_class=relationship.target_dao_class,
                )
                for item in source_collection
            ]

        # The instrumented DAO attribute expects an iterable duck-typed as its own
        # collection class, and the containers SQLAlchemy cannot instrument are held
        # as lists at the ORM layer, so they must be assigned as one here too.
        assignable_container_type = (
            list
            if isinstance(source_collection, self.containers_held_as_lists)
            else type(source_collection)
        )
        setattr(self, relationship.key, assignable_container_type(dao_collection))

    def _get_or_queue_referenced_dao(
        self,
        *,
        source_object: Any,
        state: ToDataAccessObjectState,
        expected_type: type,
        target_dao_class: type | None,
    ) -> DataAccessObject:
        """
        Resolve an object a relationship refers to into the DAO the relationship holds,
        queuing the object's own DAO if necessary.

        :param source_object: The object referred to.
        :param state: The conversion state.
        :param expected_type: The domain type of the field being filled.
        :param target_dao_class: The DAO class the relationship points to; for the
            table of every stored object, the relationship holds the object's row in it.
        :return: The DAO the relationship holds.
        """
        dao = self._get_or_queue_dao(
            source_object=source_object, state=state, expected_type=expected_type
        )
        if target_dao_class is not None and issubclass(
            target_dao_class, StoredObjectRow
        ):
            return dao.stored_object
        return dao

    def _get_or_queue_dao(
        self,
        *,
        source_object: Any,
        state: ToDataAccessObjectState,
        expected_type: type | None = None,
    ) -> DataAccessObject:
        """
        Resolve a source object to a DAO, queuing it if necessary.

        Constructing through a parametrized alias (``GenericClass[float](...)``) leaves
        the alias on the instance, while a bare construction leaves nothing. That alias
        is the object's own type argument, so it is preferred over ``expected_type``,
        which only describes the field the object is reached through: a shared object
        then resolves to the same DAO class no matter which field reaches it first,
        which matters because the first resolution is the one this state keeps.

        :param source_object: The object to resolve.
        :param state: The conversion state.
        :param expected_type: The expected domain type of the field being filled.
        :return: The corresponding DAO instance.
        :raises QueryCannotBePersisted: If the object is an entity query language query.
        """
        self._refuse_queries(source_object=source_object)
        expected_type = getattr(source_object, "__orig_class__", None) or expected_type

        # Check if already built
        existing = state.get(source=source_object)
        if existing is not None:
            return existing

        dao_clazz = get_data_access_object_class(type(source_object), expected_type)
        if dao_clazz is None:
            raise NoDAOFoundDuringParsingError(
                obj=source_object, dao=type(self), relationship=None
            )

        # Check for alternative mapping
        mapped_object = state.apply_alternative_mapping_if_needed(
            dao_clazz=dao_clazz, source_object=source_object
        )
        if isinstance(mapped_object, dao_clazz):
            state.register(source=source_object, target=mapped_object)
            return mapped_object

        # Create new DAO instance
        result = dao_clazz()
        state.register(source=source_object, target=result)
        if id(source_object) != id(mapped_object):
            state.register(source=mapped_object, target=result)

        # Queue for filling
        alternative_base = get_dao_schema(dao_clazz).alternative_base
        state.push_work_item(
            source_object=mapped_object,
            dao_instance=result,
            alternative_base=alternative_base,
        )

        return result

    # %% conversion from dao routines

    def from_dao(
        self,
        *,
        state: FromDataAccessObjectState | None = None,
    ) -> T:
        """
        Convert the DAO back into a domain object instance.

        :param state: The conversion state.
        :return: The converted domain object.
        """
        state = state or FromDataAccessObjectState()

        if state.has(source=self) and state.is_initialized(dao_instance=self):
            return state.resolve_result(result=state.get(source=self))

        if not state.is_processing:
            result = self._perform_from_dao_conversion(state=state)

            # if the instance that started this whole process is alternatively mapped, finally convert it
            return state.resolve_result(result=result)

        return self._register_for_conversion(state=state)

    def _perform_from_dao_conversion(self, *, state: FromDataAccessObjectState) -> T:
        """
        Perform the four-phase conversion process.

        :param state: The conversion state.
        :return: The converted domain object.
        """
        state.is_processing = True
        state.reset_conversion_tracking()
        discovery_order = []

        if not state.has(source=self):
            state.allocate_and_memoize(
                dao_instance=self, original_clazz=self.constructable_original_class()
            )
        state.push_work_item(dao_instance=self, domain_object=state.get(source=self))

        self._discover_dependencies(state=state, discovery_order=discovery_order)
        self._fill_domain_objects(state=state, discovery_order=discovery_order)
        state.convert_alternative_mappings_to_domain_objects()
        state.copy_attributes_inherited_from_alternative_mappings()
        self._finalize_containers(state=state, discovery_order=discovery_order)
        self._call_post_inits(state=state, discovery_order=discovery_order)
        self._finalize_sorting_containers(state=state, discovery_order=discovery_order)

        for work_item in discovery_order:
            state.mark_initialized(dao_instance=work_item.dao_instance)

        state.is_processing = False
        return state.get(source=self)

    def _discover_dependencies(
        self,
        *,
        state: FromDataAccessObjectState,
        discovery_order: list[FromDataAccessObjectWorkItem],
    ) -> None:
        """
        Phase 1: Discovery (DFS) to identify all reachable DAOs.

        :param state: The conversion state.
        :param discovery_order: List to record the discovery order.
        """
        state.discovery_mode = True
        collected_types = set()

        while state.work_items:
            # Use pop() to treat the deque as a stack (LIFO) for DFS
            work_item = state.work_items.pop()
            discovery_order.append(work_item)
            if isinstance(work_item.domain_object, AlternativeMapping):
                collected_types.add(type(work_item.domain_object))
            work_item.dao_instance._fill_from_dao(
                domain_object=work_item.domain_object, state=state
            )

        state._build_class_dependencies(alternative_mapping_types=list(collected_types))
        state.discovery_mode = False

    def _fill_domain_objects(
        self,
        *,
        state: FromDataAccessObjectState,
        discovery_order: list[FromDataAccessObjectWorkItem],
    ):
        """
        Phase 2: Filling (Bottom-Up) to initialize domain objects.

        Populate all relationships and scalars for all discovered instances.
        This ensures that all objects point to each other (even if not yet fully resolved).

        :param state: The conversion state.
        :param discovery_order: The order in which to process the instances.
        """
        for work_item in discovery_order:
            if not state.is_initialized(dao_instance=work_item.dao_instance):
                work_item.dao_instance._populate_relationships_and_scalars_from_dao(
                    domain_object=work_item.domain_object, state=state
                )

    def _finalize_containers(
        self,
        *,
        state: FromDataAccessObjectState,
        discovery_order: list[FromDataAccessObjectWorkItem],
    ) -> None:
        """
        Convert temporary lists to their final container types.
        """
        for domain_object in self._distinct_domain_objects(
            state=state, discovery_order=discovery_order
        ):
            for attr_name in _get_set_field_names(type(domain_object)):
                self._rebuild_container(
                    domain_object=domain_object, attr_name=attr_name, container=set
                )
            for attr_name, container in _get_containers_of_list_held_fields(
                type(domain_object)
            ).items():
                if container not in self.containers_that_sort_their_elements:
                    self._rebuild_container(
                        domain_object=domain_object,
                        attr_name=attr_name,
                        container=container,
                    )

    def _finalize_sorting_containers(
        self,
        *,
        state: FromDataAccessObjectState,
        discovery_order: list[FromDataAccessObjectWorkItem],
    ) -> None:
        """
        Convert temporary lists to the :attr:`containers_that_sort_their_elements` their
        type hints declare, once every loaded object is initialised.
        """
        for domain_object in self._distinct_domain_objects(
            state=state, discovery_order=discovery_order
        ):
            for attr_name, container in _get_containers_of_list_held_fields(
                type(domain_object)
            ).items():
                if container in self.containers_that_sort_their_elements:
                    self._rebuild_container(
                        domain_object=domain_object,
                        attr_name=attr_name,
                        container=container,
                    )

    @staticmethod
    def _distinct_domain_objects(
        *,
        state: FromDataAccessObjectState,
        discovery_order: list[FromDataAccessObjectWorkItem],
    ) -> Iterable[Any]:
        """
        :return: Every domain object loaded, once each, in the order of discovery.
        """
        processed_ids = set()
        for work_item in discovery_order:
            domain_object = state.get(source=work_item.dao_instance)
            if domain_object is not None and id(domain_object) not in processed_ids:
                processed_ids.add(id(domain_object))
                yield domain_object

    @staticmethod
    def _rebuild_container(
        *, domain_object: Any, attr_name: str, container: type
    ) -> None:
        """
        Replace the list a relationship loaded into a field by the container its type hint
        declares.
        """
        value = getattr(domain_object, attr_name, None)
        if isinstance(value, list):
            # object.__setattr__ (not setattr) so this also works on frozen dataclasses.
            object.__setattr__(domain_object, attr_name, container(value))

    def _call_post_inits(
        self,
        *,
        state: FromDataAccessObjectState,
        discovery_order: list[FromDataAccessObjectWorkItem],
    ) -> None:
        """
        Phase 4: Call post_init or __post_init__ on all objects.
        """
        processed_ids = set()
        for work_item in discovery_order:
            # Skip post_init for objects that were created via AlternativeMapping
            # because they are created via their constructor, which already
            # calls __post_init__.
            if issubclass(
                work_item.dao_instance.constructable_original_class(),
                AlternativeMapping,
            ):
                continue

            domain_object = state.get(source=work_item.dao_instance)
            if domain_object is not None and id(domain_object) not in processed_ids:
                if _has_post_init(type(domain_object)):
                    domain_object.__post_init__()
                processed_ids.add(id(domain_object))

    def _register_for_conversion(self, *, state: FromDataAccessObjectState) -> T:
        """
        Register this DAO for conversion if not already present.

        :param state: The conversion state.
        :return: The uninitialized domain object.
        """
        if not state.has(source=self):
            domain_object = state.allocate_and_memoize(
                dao_instance=self, original_clazz=self.constructable_original_class()
            )
            state.push_work_item(dao_instance=self, domain_object=domain_object)
        return state.get(source=self)

    def _populate_relationships_and_scalars_from_dao(
        self, *, domain_object: T, state: FromDataAccessObjectState
    ) -> None:
        """
        Populate the relationships and scalar columns of the domain object.

        :param domain_object: The domain object.
        :param state: The conversion state.
        """
        plan = get_dao_schema(type(self))
        for part_dao, part_plan in self._parts_and_their_plans(plan=plan):
            part_dao._populate_from_columns_and_relationships(
                domain_object=domain_object,
                data_column_names=part_plan.data_column_names,
                single_relationships=part_plan.single_relationships,
                collection_relationships=part_plan.collection_relationships,
                state=state,
            )
        alternative_parent = self._alternative_parent(state=state)
        if alternative_parent is None:
            self._populate_from_columns_and_relationships(
                domain_object=domain_object,
                data_column_names=plan.data_column_names,
                single_relationships=plan.single_relationships,
                collection_relationships=plan.collection_relationships,
                state=state,
            )
            return

        partition = _get_alternative_partition_plan(type(self), plan.alternative_base)
        self._populate_from_columns_and_relationships(
            domain_object=domain_object,
            data_column_names=partition.intermediate_attribute_keys,
            single_relationships=partition.own_single_relationships,
            collection_relationships=partition.own_collection_relationships,
            state=state,
        )
        own_attribute_keys = (
            *partition.intermediate_attribute_keys,
            *(entry.key for entry in partition.own_single_relationships),
            *(entry.key for entry in partition.own_collection_relationships),
        )
        state.objects_inheriting_from_alternative_mappings.append(
            ObjectInheritingFromAlternativeMapping(
                domain_object=domain_object,
                parent=alternative_parent,
                own_attribute_keys=own_attribute_keys,
            )
        )

    def _populate_from_columns_and_relationships(
        self,
        *,
        domain_object: T,
        data_column_names: Iterable[str],
        single_relationships: Iterable[SingleRelationship],
        collection_relationships: Iterable[CollectionRelationship],
        state: FromDataAccessObjectState,
    ) -> None:
        """
        Set the given columns and relationships of this DAO on the domain object.

        :param domain_object: The domain object.
        :param data_column_names: The names of the scalar columns to set.
        :param single_relationships: The single valued relationships to set.
        :param collection_relationships: The collection valued relationships to set.
        :param state: The conversion state.
        """
        for name in data_column_names:
            object.__setattr__(domain_object, name, getattr(self, name))

        for relationship in single_relationships:
            self._populate_single_relationship(
                domain_object=domain_object,
                key=relationship.key,
                value=_read_single_relationship(self, relationship),
                state=state,
            )
        for relationship in collection_relationships:
            self._populate_collection_relationship(
                domain_object=domain_object,
                key=relationship.key,
                value=getattr(self, relationship.key),
                state=state,
                association_class=relationship.association_class,
            )

    def _fill_from_dao(
        self, *, domain_object: T, state: FromDataAccessObjectState
    ) -> T:
        """
        Populate the domain object with data from the DAO.

        :param domain_object: The domain object to populate.
        :param state: The conversion state.
        :return: The populated domain object.
        """
        if state.discovery_mode:
            return self._trigger_discovery(domain_object=domain_object, state=state)

        # Fallback for when _fill_from_dao is called directly (not during Phase 1)
        self._populate_relationships_and_scalars_from_dao(
            domain_object=domain_object, state=state
        )
        return domain_object

    def _trigger_discovery(
        self,
        *,
        domain_object: T,
        state: FromDataAccessObjectState,
    ) -> T:
        """
        Trigger discovery of dependencies without fully populating the object.

        :param domain_object: The domain object.
        :param state: The conversion state.
        :return: The domain object.
        """
        plan = get_dao_schema(type(self))
        self._discover_related_objects(plan=plan, state=state)
        for part_dao, part_plan in self._parts_and_their_plans(plan=plan):
            part_dao._discover_related_objects(plan=part_plan, state=state)

        self._alternative_parent(state=state)
        return domain_object

    def _parts_and_their_plans(
        self, *, plan: DataAccessObjectSchema
    ) -> list[tuple[PartDataAccessObject, DataAccessObjectSchema]]:
        """
        :param plan: The conversion plan of this DAO's class.
        :return: The parts this DAO holds, each with the conversion plan of its class.
        """
        return [
            (part_dao, get_dao_schema(part_class))
            for part_class in plan.part_classes
            if (part_dao := self.part(part_class=part_class)) is not None
        ]

    def part(
        self, *, part_class: type[PartDataAccessObject]
    ) -> PartDataAccessObject | None:
        """
        :param part_class: The class of one of the parts of this DAO's class.
        :return: The part of this DAO's object of that class, if the object has one.
        :raises PartOfDetachedObjectNotLoadedError: If this DAO was loaded, left its
            session and never loaded that part.
        """
        if self.parts_in_memory is not None and part_class in self.parts_in_memory:
            return self.parts_in_memory[part_class]
        instance_state = sqlalchemy.inspect(self)
        if instance_state.detached:
            raise PartOfDetachedObjectNotLoadedError(
                data_access_object=self, part_class=part_class
            )
        session = instance_state.session
        if session is None:
            return None
        owners = session.info.get(_SessionInfoKey.OWNERS_OF_UNLOADED_PARTS, {}).pop(
            part_class, weakref.WeakSet()
        )
        owners.add(self)
        owners_by_id = {owner.database_id: owner for owner in owners}
        parts_by_id = {
            part.database_id: part
            for part in session.scalars(
                sqlalchemy.select(part_class)
                .where(part_class.database_id.in_(owners_by_id))
                .execution_options(**{_SessionInfoKey.PLAIN_LOAD: True})
            )
        }
        for database_id, owner in owners_by_id.items():
            if owner.parts_in_memory is None:
                owner.parts_in_memory = {}
            owner.parts_in_memory[part_class] = parts_by_id.get(database_id)
        return self.parts_in_memory[part_class]

    @staticmethod
    @called_positionally
    def register_owner_of_unloaded_parts(
        target: Any, context: sqlalchemy.orm.QueryContext
    ) -> None:
        """
        When a DAO that has parts is loaded, remember it, so its parts are loaded
        together with those of the other DAOs loaded meanwhile.

        :param target: The loaded object.
        :param context: The query that loaded it.
        """
        if not isinstance(target, DataAccessObject):
            return
        part_classes = get_dao_schema(type(target)).part_classes
        if not part_classes:
            return
        owners = context.session.info.setdefault(
            _SessionInfoKey.OWNERS_OF_UNLOADED_PARTS, {}
        )
        for part_class in part_classes:
            owners.setdefault(part_class, weakref.WeakSet()).add(target)

    def _discover_related_objects(
        self, *, plan: DataAccessObjectSchema, state: FromDataAccessObjectState
    ) -> None:
        """
        Register every object this DAO's relationships refer to for conversion.

        :param plan: The conversion plan of this DAO's class.
        :param state: The conversion state.
        """
        for relationship in plan.single_relationships:
            value = _read_single_relationship(self, relationship)
            if value is not None:
                value.from_dao(state=state)

        for relationship in plan.collection_relationships:
            value = getattr(self, relationship.key)
            if not value:
                continue
            if relationship.association_class is not None:
                target_relationship = relationship.association_target_relationship
                for item in value:
                    target = _read_single_relationship(item, target_relationship)
                    if target is not None:
                        target.from_dao(state=state)
            else:
                for item in value:
                    item.from_dao(state=state)

    def _populate_single_relationship(
        self,
        *,
        domain_object: Any,
        key: str,
        value: Any,
        state: FromDataAccessObjectState,
    ) -> None:
        """
        Populate a single-valued relationship on the domain object.

        :param domain_object: The domain object.
        :param key: The attribute name.
        :param value: The DAO instance.
        :param state: The conversion state.
        """
        if value is None:
            object.__setattr__(domain_object, key, None)
            return
        instance = self._get_or_allocate_domain_object(dao_instance=value, state=state)
        if isinstance(instance, AlternativeMapping):
            state._alternative_mappings_being_referenced[instance].append(
                (domain_object, Attribute(_attribute_name_=key, _child_=None))
            )
        object.__setattr__(domain_object, key, instance)

    def _populate_collection_relationship(
        self,
        *,
        domain_object: Any,
        key: str,
        value: Any,
        state: FromDataAccessObjectState,
        association_class: type | None = None,
    ) -> None:
        """
        Populate a collection relationship on the domain object.

        :param domain_object: The domain object.
        :param key: The attribute name.
        :param value: The collection of DAO instances.
        :param state: The conversion state.
        :param association_class: The association class of the collection, if any.
        """

        # handle empty collections / None
        if not value:
            # copy, so the domain object does not alias the DAO's instrumented collection
            object.__setattr__(
                domain_object, key, list(value) if value is not None else value
            )
            return

        if association_class is not None:
            dao_collection = [item.target for item in value if item.target is not None]
        else:
            dao_collection = list(value)

        instances = [
            self._get_or_allocate_domain_object(dao_instance=v, state=state)
            for v in dao_collection
        ]

        # memorize alternative mapping references
        for index, instance in enumerate(instances):
            if isinstance(instance, AlternativeMapping):
                state._alternative_mappings_being_referenced[instance].append(
                    (
                        domain_object,
                        IndexByValue(
                            _key_=index,
                            _child_=Attribute(_attribute_name_=key, _child_=None),
                        ),
                    )
                )

        object.__setattr__(domain_object, key, list(instances))

    def _get_or_allocate_domain_object(
        self, *, dao_instance: DataAccessObject, state: FromDataAccessObjectState
    ) -> Any:
        """
        Resolve a DAO to a domain object, allocating it if necessary.

        :param dao_instance: The DAO to resolve.
        :param state: The conversion state.
        :return: The corresponding domain object.
        """
        return dao_instance.from_dao(state=state)

    def _alternative_parent(
        self, *, state: FromDataAccessObjectState
    ) -> AlternativeMapping | None:
        """
        The alternative mapping of the alternatively mapped base class that the domain
        object of this DAO inherits from, built from the base's columns of this DAO.

        :param state: The conversion state.
        :return: The alternative mapping, or ``None`` if the DAO has no alternatively
            mapped base class or is an alternative mapping itself.
        """
        plan = get_dao_schema(type(self))
        if plan.alternative_base is None or plan.uses_alternative_mapping:
            return None

        # Keyed by id(self) since a synthetic parent DAO belongs to this DAO only, and
        # cached so discovery and filling see the same parent.
        cache_key = (id(self), plan.alternative_base)
        if cache_key not in state.synthetic_parent_daos:
            state.synthetic_parent_daos[cache_key] = self._create_filled_parent_dao(
                base_clazz=plan.alternative_base
            )
        return state.synthetic_parent_daos[cache_key].from_dao(state=state)

    def _create_filled_parent_dao(
        self, *, base_clazz: type[DataAccessObject]
    ) -> DataAccessObject:
        """
        Create a parent DAO instance populated from the current DAO.

        :param base_clazz: The parent DAO class.
        :return: The populated parent DAO instance.
        """
        parent_dao = base_clazz()
        parent_plan = get_dao_schema(base_clazz)
        for name in parent_plan.data_column_names:
            setattr(parent_dao, name, getattr(self, name))
        for key in parent_plan.relationship_keys:
            setattr(parent_dao, key, getattr(self, key))
        return parent_dao

    def __repr__(self) -> str:
        """
        Return a string representation including columns and relationships.

        :return: The string representation.
        """
        if not hasattr(_repr_thread_local, "seen"):
            _repr_thread_local.seen = set()

        if id(self) in _repr_thread_local.seen:
            return f"{self.__class__.__name__}(...)"

        _repr_thread_local.seen.add(id(self))
        try:
            mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))
            representations = []

            for column in mapper.columns:
                if is_data_column(column):
                    value = getattr(self, column.name)
                    representations.append(f"{column.name}={repr(value)}")

            for relationship in mapper.relationships:
                value = getattr(self, relationship.key)
                if value is not None:
                    representations.append(f"{relationship.key}={repr(value)}")

            return f"{self.__class__.__name__}({', '.join(representations)})"
        finally:
            _repr_thread_local.seen.remove(id(self))


class StoredObjectColumn(StrEnum):
    """
    The columns of the table of every stored object that a selection of the ids and
    types of stored objects gives, under the names of the attributes of its rows.
    """

    DATABASE_ID = "database_id"
    """
    The id of the object, the same in the table of every class it has rows in.
    """

    POLYMORPHIC_TYPE = "polymorphic_type"
    """
    The name of the table of the class the object is loaded as.
    """


class StoredObjectRow:
    """
    The row every stored object has, whatever its class.

    The root table of every generated hierarchy takes its ids from the table of the
    mapped subclass, so an object has the same id in the tables of all its mapped base
    classes, and the row's type column names the DAO class the object is loaded as.
    """

    object_dao_in_memory: DataAccessObject | None = None
    """
    The DAO of the object, while it was created in this process and not loaded.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Insert the rows of new objects into the subclass's table before every flush.
        """
        super().__init_subclass__(**kwargs)
        event.listen(
            sqlalchemy.orm.Session, "before_flush", cls.insert_rows_of_new_objects
        )

    @staticmethod
    def mapped_class() -> type[StoredObjectRow]:
        """
        :return: The class mapping the table of every stored object.
        :raises StoredObjectTableNotUniqueError: If not exactly one class maps it.
        """
        table_classes = StoredObjectRow.__subclasses__()
        if len(table_classes) != 1:
            raise StoredObjectTableNotUniqueError(table_classes=table_classes)
        return table_classes[0]

    @property
    def object_dao(self) -> DataAccessObject:
        """
        :return: The DAO of the object this row belongs to, loaded as its own class.
        """
        if self.object_dao_in_memory is not None:
            return self.object_dao_in_memory
        return sqlalchemy.orm.object_session(self).get(
            self.dao_class_of_table(table_name=self.polymorphic_type), self.database_id
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def dao_class_of_table(*, table_name: str) -> type[DataAccessObject]:
        """
        :param table_name: The name of the table of a generated DAO class.
        :return: The DAO class objects of that table are loaded as.
        """
        return next(
            mapper.class_
            for mapper in Base.registry.mappers
            if mapper.local_table.name == table_name
            and issubclass(mapper.class_, DataAccessObject)
            and not issubclass(mapper.class_, PartDataAccessObject)
        )

    @classmethod
    def load_objects_by_class(
        cls,
        *,
        stored_objects: Sequence[StoredObjectRow | sqlalchemy.Row],
        load_objects_of_class: Callable[
            [type[DataAccessObject], list[int]], Iterable[Any]
        ],
    ) -> list[Any]:
        """
        :param stored_objects: Rows of this table, or rows with the
            :attr:`~krrood.ormatic.stored_object.StoredObjectDAO.database_id` and
            :attr:`~krrood.ormatic.stored_object.StoredObjectDAO.polymorphic_type` of stored objects.
        :param load_objects_of_class: Loads the objects of one DAO class from their ids,
            each of them with its
            :attr:`~krrood.ormatic.stored_object.StoredObjectDAO.database_id`.
        :return: What was loaded for every stored object, in the same order.
        """
        ids_by_class: dict[type[DataAccessObject], list[int]] = {}
        for stored_object in stored_objects:
            dao_class = cls.dao_class_of_table(
                table_name=stored_object.polymorphic_type
            )
            ids_by_class.setdefault(dao_class, []).append(stored_object.database_id)
        loaded_by_id = {
            loaded.database_id: loaded
            for dao_class, ids in ids_by_class.items()
            for loaded in load_objects_of_class(dao_class, ids)
        }
        return [
            loaded_by_id[stored_object.database_id] for stored_object in stored_objects
        ]

    @classmethod
    def object_daos(
        cls,
        *,
        session: sqlalchemy.orm.Session,
        stored_objects: Sequence[StoredObjectRow | sqlalchemy.Row],
        loader_options: tuple[ExecutableOption, ...] = (),
        ids_per_statement: int,
    ) -> list[DataAccessObject]:
        """
        :param session: The session the rows were loaded with.
        :param stored_objects: Rows of this table, or rows with the
            :attr:`~krrood.ormatic.stored_object.StoredObjectDAO.database_id` and
            :attr:`~krrood.ormatic.stored_object.StoredObjectDAO.polymorphic_type` of stored objects.
        :param loader_options: The options of the statements that load the DAOs.
        :param ids_per_statement: The most ids one statement binds.
        :return: The DAOs of the rows' objects, in the same order, loaded with one query
            per DAO class and batch of ids.
        """
        return cls.load_objects_by_class(
            stored_objects=stored_objects,
            load_objects_of_class=lambda dao_class, ids: (
                dao
                for batch in batched(ids, ids_per_statement)
                for dao in session.scalars(
                    sqlalchemy.select(dao_class)
                    .where(dao_class.database_id.in_(batch))
                    .options(*loader_options)
                )
            ),
        )

    def from_dao(self, *, state: FromDataAccessObjectState | None = None) -> Any:
        """
        :param state: The conversion state.
        :return: The object this row belongs to, converted from its DAO as
            :meth:`DataAccessObject.from_dao` converts it.
        """
        return self.object_dao.from_dao(state=state)

    @classmethod
    @called_positionally
    def insert_rows_of_new_objects(
        cls, session: sqlalchemy.orm.Session, flush_context: Any, instances: Any
    ) -> None:
        """
        Before a session flushes, add the DAOs and parts that its new objects hold in
        memory, insert the rows of all new objects, give their DAOs and parts the rows'
        ids, and delete the other rows of the objects it deletes.

        :param session: The session about to flush.
        :param flush_context: The flush's unit of work.
        :param instances: The objects the flush was asked for.
        """
        cls._add_objects_held_in_memory(session=session)
        cls._delete_rows_of_deleted_objects(session=session)
        new_daos = [
            dao
            for dao in session.new
            if isinstance(dao, DataAccessObject)
            and get_dao_schema(type(dao)).shares_id_with_stored_object
            and dao.database_id is None
        ]
        if new_daos:
            cls._insert_rows(session=session, daos=new_daos)

    @classmethod
    def _insert_rows(
        cls, *, session: sqlalchemy.orm.Session, daos: list[DataAccessObject]
    ) -> None:
        """
        Insert the rows of new DAOs' objects in one statement and give the rows, the
        DAOs and their parts the ids the database chose.

        :param session: The session about to flush.
        :param daos: New DAOs whose objects have no row yet.
        """
        rows = [dao.stored_object for dao in daos]
        ids = session.execute(
            sqlalchemy.insert(cls.__table__).returning(
                cls.__table__.c.database_id, sort_by_parameter_order=True
            ),
            [{"polymorphic_type": row.polymorphic_type} for row in rows],
        ).scalars()
        for row, dao, database_id in zip(rows, daos, ids, strict=True):
            dao.database_id = database_id
            for part in (dao.parts_in_memory or {}).values():
                part.database_id = database_id
            cls._make_row_persistent(session=session, row=row, database_id=database_id)

    @classmethod
    def _make_row_persistent(
        cls, *, session: sqlalchemy.orm.Session, row: StoredObjectRow, database_id: int
    ) -> None:
        """
        Turn a row inserted outside the flush into a stored object of the session if a
        field already refers to it, or into a detached one otherwise.

        :param session: The session about to flush.
        :param row: The inserted row.
        :param database_id: The id the database chose for the row.
        """
        referenced = row in session
        if referenced:
            session.expunge(row)
        row.database_id = database_id
        make_transient_to_detached(row)
        if not referenced:
            return
        # a row deleted without the session may still hold the id the database reused
        stale_row = session.identity_map.get(
            cls.__mapper__.identity_key_from_primary_key((database_id,))
        )
        if stale_row is not None:
            session.expunge(stale_row)
        session.add(row)

    @classmethod
    def _add_objects_held_in_memory(cls, *, session: sqlalchemy.orm.Session) -> None:
        """
        Add the DAOs of the objects whose new rows of this table a session holds, since
        a field referring to an object holds only its row, and the parts of the new
        DAOs, together with what they refer to in turn.

        :param session: The session about to flush.
        """
        while True:
            held_in_memory = [
                held
                for new in session.new
                for held in cls._objects_held_in_memory(new_object=new)
                if held not in session
            ]
            if not held_in_memory:
                return
            session.add_all(held_in_memory)

    @classmethod
    def _objects_held_in_memory(cls, *, new_object: Any) -> list[DataAccessObject]:
        """
        :param new_object: An object a session is about to insert.
        :return: The DAO of the object a new row of this table belongs to, or the parts
            of a new DAO.
        """
        if isinstance(new_object, cls):
            owner = new_object.object_dao_in_memory
            return [] if owner is None else [owner]
        if isinstance(new_object, DataAccessObject) and new_object.parts_in_memory:
            return list(new_object.parts_in_memory.values())
        return []

    @staticmethod
    def _delete_rows_of_deleted_objects(*, session: sqlalchemy.orm.Session) -> None:
        """
        Delete the rows that the objects of the DAOs a session is about to delete have
        in the table of every stored object and in the tables of their secondary bases.

        :param session: The session about to flush.
        """
        for dao in list(session.deleted):
            if not isinstance(dao, DataAccessObject):
                continue
            schema = get_dao_schema(type(dao))
            if not schema.shares_id_with_stored_object:
                continue
            rows = [dao.stored_object] + [
                dao.part(part_class=part_class) for part_class in schema.part_classes
            ]
            for row in rows:
                if row is not None and sqlalchemy.inspect(row).persistent:
                    session.delete(row)


event.listen(
    Base, "load", DataAccessObject.register_owner_of_unloaded_parts, propagate=True
)


class PartDataAccessObject(DataAccessObject[T]):
    """
    The row of an object in the table of a base class outside the object's primary
    chain, mapped to that base class's table and held by the object's own DAO.

    It is never the DAO an object is converted to or loaded as.
    """

    @classmethod
    @lru_cache(maxsize=None)
    def type_column_key(cls) -> str | None:
        """
        :return: The attribute of the type column of the base class's hierarchy, if the
            base class's table is the root of that hierarchy and so holds it.
        """
        base_dao = get_data_access_object_class(cls.original_class())
        type_column = sqlalchemy.inspection.inspect(base_dao).polymorphic_on
        if type_column is None or type_column.table is not cls.__table__:
            return None
        return type_column.key


@contextmanager
def selectin_loading(session: sqlalchemy.orm.Session):
    """
    Context manager that optimizes bulk reads of whole object graphs.

    Top-level ORM queries executed within its scope use selectin loading for
    their relationships. Additionally, identity-map misses during
    :meth:`DataAccessObject.from_dao` bulk-load the entire table of the missed
    class instead of fetching one row at a time, so reconstructing a graph that
    spans (most of) the database takes O(tables) instead of O(rows) queries.

    Use this context manager when you want call `from_dao()` on the result, as it is ~3 times faster.

    Usage::

        with selectin_loading(session):
            dao = session.scalars(select(SomeDAO)).one()
            domain_obj = dao.from_dao()

    :param session: The SQLAlchemy session whose queries should use selectin loading.
    """

    def _add_selectin(orm_execute_state: sqlalchemy.orm.ORMExecuteState) -> None:
        if orm_execute_state.execution_options.get(_SessionInfoKey.PLAIN_LOAD, False):
            return None
        if orm_execute_state.is_select and not orm_execute_state.is_relationship_load:
            return orm_execute_state.invoke_statement(
                statement=orm_execute_state.statement.options(selectinload("*"))
            )

    event.listen(session, "do_orm_execute", _add_selectin)
    session.info[_SessionInfoKey.BULK_LOADING_FLAG] = True
    try:
        yield
    finally:
        session.info[_SessionInfoKey.BULK_LOADING_FLAG] = False
        session.info.pop(_SessionInfoKey.BULK_LOADED_MAPPERS, None)
        event.remove(session, "do_orm_execute", _add_selectin)
