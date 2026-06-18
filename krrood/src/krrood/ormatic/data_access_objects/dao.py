from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, fields
from enum import StrEnum
from functools import lru_cache

import sqlalchemy.inspection
import sqlalchemy.orm
from sqlalchemy import event
from sqlalchemy.orm import (
    MANYTOONE,
    MANYTOMANY,
    ONETOMANY,
    RelationshipProperty,
    selectinload,
)
from typing_extensions import (
    Type,
    get_origin,
    Any,
    TypeVar,
    Optional,
    List,
    Tuple,
)

from krrood.entity_query_language.core.mapped_variable import Attribute, Index
from krrood.entity_query_language._monitoring import monitored
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.data_access_objects.base import (
    HasGeneric,
)
from krrood.ormatic.data_access_objects.from_dao import (
    FromDataAccessObjectWorkItem,
    FromDataAccessObjectState,
)
from krrood.ormatic.data_access_objects.helper import (
    get_dao_class,
    to_dao,
    clear_dao_lookup_caches,
)
from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState
from krrood.ormatic.exceptions import (
    NoGenericError,
    NoDAOFoundDuringParsingError,
)
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


@dataclass(frozen=True)
class SingleRelationship:
    """
    Metadata for a single-valued (many-to-one or one-to-one) DAO relationship.
    """

    key: str
    """The attribute name of the relationship on the DAO."""

    domain_type: Type
    """The expected domain type of the related object."""

    foreign_key_attribute: Optional[str] = None
    """
    The attribute name of the local foreign key column for many-to-one
    relationships, or ``None`` if the foreign key is on the remote side.
    """

    target_dao_class: Optional[Type] = None
    """The DAO class this relationship points to."""


@dataclass(frozen=True)
class CollectionRelationship:
    """
    Metadata for a collection-valued (one-to-many or many-to-many) DAO relationship.
    """

    key: str
    """The attribute name of the relationship on the DAO."""

    association_class: Optional[Type]
    """
    The association DAO class used as an intermediary, or ``None`` for direct
    collection relationships.
    """

    domain_type: Type
    """The expected domain type of the items in the collection."""

    association_target_relationship: Optional[SingleRelationship] = None
    """
    The ``target`` relationship of the association class, if any. Used to
    resolve association targets via the identity map instead of lazy loads.
    """


@dataclass(frozen=True)
class DataAccessObjectConversionPlan:
    """
    Precomputed, class-level metadata used by the to_dao/from_dao hot paths.

    All information in here is derived from the SQLAlchemy mapper and the generic
    arguments of a DAO class. Computing it once per class avoids repeated mapper
    introspection, relationship classification and MRO walks per converted instance.
    """

    data_column_names: Tuple[str, ...]
    """
    Names of all data columns (no primary keys, foreign keys or polymorphic markers).
    """

    single_relationships: Tuple[SingleRelationship, ...]
    """
    Single-valued relationships of this DAO class.
    """

    collection_relationships: Tuple[CollectionRelationship, ...]
    """
    Collection-valued relationships of this DAO class.
    """

    relationship_keys: Tuple[str, ...]
    """
    The keys of all relationships of the DAO class.
    """

    alternative_base: Optional[Type]
    """
    The first base class of the DAO that maps to an AlternativeMapping, if any.
    """

    uses_alternative_mapping: bool
    """
    Whether the DAO class itself maps to an AlternativeMapping.
    """


def _build_single_relationship(
    relationship: RelationshipProperty, domain_type: Type
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
        :class:`DataAccessObjectConversionPlan`.
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
def _get_conversion_plan(dao_class: Type) -> DataAccessObjectConversionPlan:
    """
    Build (and cache) the conversion plan for a DAO class.

    :param dao_class: The DAO class to build the plan for.
    :return: The conversion plan.
    """
    mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(dao_class)

    single_relationships = []
    collection_relationships = []
    for relationship in mapper.relationships:
        if DataAccessObject._is_single_relationship(relationship):
            single_relationships.append(
                _build_single_relationship(
                    relationship, relationship.mapper.class_.original_class()
                )
            )
        elif relationship.direction in (ONETOMANY, MANYTOMANY):
            target_dao_clazz = relationship.mapper.class_
            if issubclass(target_dao_clazz, AssociationDataAccessObject):
                target_relationship = sqlalchemy.inspection.inspect(
                    target_dao_clazz
                ).relationships["target"]
                domain_type = target_relationship.mapper.class_.original_class()
                collection_relationships.append(
                    CollectionRelationship(
                        key=relationship.key,
                        association_class=target_dao_clazz,
                        domain_type=domain_type,
                        association_target_relationship=_build_single_relationship(
                            target_relationship, domain_type
                        ),
                    )
                )
            else:
                collection_relationships.append(
                    CollectionRelationship(
                        key=relationship.key,
                        association_class=None,
                        domain_type=target_dao_clazz.original_class(),
                    )
                )

    return DataAccessObjectConversionPlan(
        data_column_names=tuple(
            column.name for column in mapper.columns if is_data_column(column)
        ),
        single_relationships=tuple(single_relationships),
        collection_relationships=tuple(collection_relationships),
        relationship_keys=tuple(
            relationship.key for relationship in mapper.relationships
        ),
        alternative_base=dao_class._find_alternative_mapping_base(),
        uses_alternative_mapping=_uses_alternative_mapping(dao_class),
    )


@dataclass(frozen=True)
class AlternativePartitionPlan:
    """
    Precomputed partition of columns and relationships between a DAO class and its
    alternatively mapped base class.
    """

    parent_data_column_names: Tuple[str, ...]
    """
    Data column names that belong to the alternatively mapped base.
    """

    own_data_column_names: Tuple[str, ...]
    """
    Data column names that belong to this DAO's own tables.
    """

    intermediate_attribute_keys: Tuple[str, ...]
    """
    Column attribute keys of intermediate ancestors not covered by the parent.
    """

    parent_single_relationships: Tuple[SingleRelationship, ...]
    """
    Single-valued relationships that belong to the alternatively mapped base.
    """

    parent_collection_relationships: Tuple[CollectionRelationship, ...]
    """
    Collection relationships that belong to the alternatively mapped base.
    """

    own_single_relationships: Tuple[SingleRelationship, ...]
    """
    Single-valued relationships that belong to this DAO's own tables,
    i.e. those not covered by the alternatively mapped base.
    """

    own_collection_relationships: Tuple[CollectionRelationship, ...]
    """
    Collection relationships that belong to this DAO's own tables,
    i.e. those not covered by the alternatively mapped base.
    """


@lru_cache(maxsize=None)
def _get_alternative_partition_plan(
    dao_class: Type, alternative_base: Type
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
    plan = _get_conversion_plan(dao_class)
    parent_plan = _get_conversion_plan(alternative_base)

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
def _uses_alternative_mapping(clazz: Type) -> bool:
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
def _has_post_init(clazz: Type) -> bool:
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


def _bulk_load_target_table_if_enabled(
    session: sqlalchemy.orm.Session, target_dao_class: Type
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
) -> Optional[DataAccessObject]:
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
def _get_set_field_names(clazz: Type) -> Tuple[str, ...]:
    """
    :param clazz: The domain class to inspect.
    :return: The names of all fields annotated as sets.
    """
    return tuple(
        attr_name
        for attr_name, hint in _get_type_hints_cached(clazz).items()
        if get_origin(hint) is set or hint is set
    )


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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # New DAO classes invalidate previously failed (None) lookups.
        clear_dao_lookup_caches()

    # %% conversion to dao routines
    @classmethod
    def to_dao(
        cls,
        source_object: T,
        state: Optional[ToDataAccessObjectState] = None,
        register: bool = True,
    ) -> _DAO:
        """
        Convert an object to its Data Access Object.

        :param source_object: The object to convert.
        :param state: The conversion state.
        :param register: Whether to register the result in the memo.
        :return: The converted DAO instance.
        """
        state = state or ToDataAccessObjectState()

        # Phase 1: Resolution - Check memo and apply alternative mappings
        existing = state.get(source_object)
        if existing is not None:
            return existing

        resolved_source = state.apply_alternative_mapping_if_needed(cls, source_object)

        # Phase 2: Allocation & Registration
        result = cls()

        if register:
            state.register(source_object, result)
            if id(source_object) != id(resolved_source):
                state.register(resolved_source, result)

        # Phase 3: Queueing & Processing
        is_entry_call = len(state.work_items) == 0
        alternative_base = _get_conversion_plan(cls).alternative_base
        state.push_work_item(resolved_source, result, alternative_base)

        if is_entry_call:
            cls._process_to_dao_queue(state)

        return result

    @classmethod
    def _process_to_dao_queue(cls, state: ToDataAccessObjectState) -> None:
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
    def _find_alternative_mapping_base(cls) -> Optional[Type[DataAccessObject]]:
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
        self, source_object: T, state: ToDataAccessObjectState
    ) -> None:
        """
        Populate the DAO instance from a source object.

        :param source_object: The source object.
        :param state: The conversion state.
        """
        plan = _get_conversion_plan(type(self))

        for name in plan.data_column_names:
            setattr(self, name, getattr(source_object, name))
        self._fill_relationships_from_plan(
            source_object,
            plan.single_relationships,
            plan.collection_relationships,
            state,
        )

    def fill_dao_if_subclass_of_alternative_mapping(
        self,
        source_object: T,
        alternative_base: Type[DataAccessObject],
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Populate the DAO instance for an alternatively mapped subclass.

        :param source_object: The source object.
        :param alternative_base: The base class using alternative mapping.
        :param state: The conversion state.
        """
        # Temporarily remove the object from the memo to allow the parent DAO to be created separately
        temp_dao = state.pop(source_object)

        # create dao of alternatively mapped superclass
        parent_dao = alternative_base.original_class().to_dao(source_object, state)

        # Restore the object in the memo dictionary
        if temp_dao is not None:
            state.register(source_object, temp_dao)

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
            parent_dao,
            partition.parent_single_relationships,
            partition.parent_collection_relationships,
            state,
        )
        self._fill_relationships_from_plan(
            source_object,
            partition.own_single_relationships,
            partition.own_collection_relationships,
            state,
        )

    @staticmethod
    def _is_single_relationship(relationship: RelationshipProperty) -> bool:
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
        source_object: Any,
        single_relationships: Tuple[SingleRelationship, ...],
        collection_relationships: Tuple[CollectionRelationship, ...],
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Populate relationships from a source object using a conversion plan.

        :param source_object: The source of relationship values.
        :param single_relationships: The single-valued relationship entries.
        :param collection_relationships: The collection relationship entries.
        :param state: The conversion state.
        """
        for relationship in single_relationships:
            value = getattr(source_object, relationship.key)
            if value is None:
                setattr(self, relationship.key, None)
            else:
                setattr(
                    self,
                    relationship.key,
                    self._get_or_queue_dao(value, state, relationship.domain_type),
                )

        for relationship in collection_relationships:
            source_collection = getattr(source_object, relationship.key)

            if relationship.association_class is not None:
                dao_collection = []
                for item in source_collection:
                    association_dao = relationship.association_class()
                    association_dao.target = self._get_or_queue_dao(
                        item, state, relationship.domain_type
                    )
                    dao_collection.append(association_dao)
            else:
                dao_collection = [
                    self._get_or_queue_dao(item, state, relationship.domain_type)
                    for item in source_collection
                ]

            setattr(self, relationship.key, type(source_collection)(dao_collection))

    def _get_or_queue_dao(
        self,
        source_object: Any,
        state: ToDataAccessObjectState,
        expected_type: Optional[Type] = None,
    ) -> DataAccessObject:
        """
        Resolve a source object to a DAO, queuing it if necessary.

        :param source_object: The object to resolve.
        :param state: The conversion state.
        :param expected_type: The expected domain type.
        :return: The corresponding DAO instance.
        """
        # Check if already built
        existing = state.get(source_object)
        if existing is not None:
            return existing

        dao_clazz = get_dao_class(type(source_object), expected_type)
        if dao_clazz is None:
            raise NoDAOFoundDuringParsingError(source_object, type(self), None)

        # Check for alternative mapping
        mapped_object = state.apply_alternative_mapping_if_needed(
            dao_clazz, source_object
        )
        if isinstance(mapped_object, dao_clazz):
            state.register(source_object, mapped_object)
            return mapped_object

        # Create new DAO instance
        result = dao_clazz()
        state.register(source_object, result)
        if id(source_object) != id(mapped_object):
            state.register(mapped_object, result)

        # Queue for filling
        alternative_base = _get_conversion_plan(dao_clazz).alternative_base
        state.push_work_item(mapped_object, result, alternative_base)

        return result

    # %% conversion from dao routines

    def from_dao(
        self,
        state: Optional[FromDataAccessObjectState] = None,
    ) -> T:
        """
        Convert the DAO back into a domain object instance.

        :param state: The conversion state.
        :return: The converted domain object.
        """
        state = state or FromDataAccessObjectState()

        if state.has(self) and state.is_initialized(self):
            return state.resolve_result(state.get(self))

        if not state.is_processing:
            result = self._perform_from_dao_conversion(state)

            # if the instance that started this whole process is alternatively mapped, finally convert it
            return state.resolve_result(result)

        return self._register_for_conversion(state)

    def _perform_from_dao_conversion(self, state: FromDataAccessObjectState) -> T:
        """
        Perform the four-phase conversion process.

        :param state: The conversion state.
        :return: The converted domain object.
        """
        state.is_processing = True
        state.reset_conversion_tracking()
        discovery_order = []

        if not state.has(self):
            state.allocate_and_memoize(self, self.constructable_original_class())
        state.push_work_item(self, state.get(self))

        self._discover_dependencies(state, discovery_order)
        self._fill_domain_objects(state, discovery_order)
        state.convert_alternative_mappings_to_domain_objects()
        self._finalize_containers(state, discovery_order)
        self._call_post_inits(state, discovery_order)

        for work_item in discovery_order:
            state.mark_initialized(work_item.dao_instance)

        state.is_processing = False
        return state.get(self)

    def _discover_dependencies(
        self,
        state: FromDataAccessObjectState,
        discovery_order: List[FromDataAccessObjectWorkItem],
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
            work_item.dao_instance._fill_from_dao(work_item.domain_object, state)

        state._build_class_dependencies(list(collected_types))
        state.discovery_mode = False

    def _fill_domain_objects(
        self,
        state: FromDataAccessObjectState,
        discovery_order: List[FromDataAccessObjectWorkItem],
    ):
        """
        Phase 2: Filling (Bottom-Up) to initialize domain objects.

        Populate all relationships and scalars for all discovered instances.
        This ensures that all objects point to each other (even if not yet fully resolved).

        :param state: The conversion state.
        :param discovery_order: The order in which to process the instances.
        """
        for work_item in discovery_order:
            if not state.is_initialized(work_item.dao_instance):
                work_item.dao_instance._populate_relationships_and_scalars_from_dao(
                    work_item.domain_object, state
                )

    def _handle_subclass_of_alternative_mapping_in_from_dao(
        self,
        data_access_object: DataAccessObject,
        domain_object: Any,
        alternatively_mapped_base: Type[AlternativeMapping],
    ):
        """
        Handle the case where the parent class is an alternative mapping in the `from_dqo` algorithm.

        :param data_access_object: The data access object that has an alternative mapping as its parent class.
        :param domain_object: The domain object that is being constructed.
        :param alternatively_mapped_base: The base class that is the alternative mapping.
        :return:
        """
        logger.warning(
            "Subclasses of AlternativeMapping are only partially supported. "
            "If the parent classes alternative mapping has dependencies these are ignored and may yield "
            "inconsistent build orders."
        )
        # create the domain object of the alternatively mapped base
        base_domain_object = alternatively_mapped_base.to_domain_object(
            data_access_object
        )
        for domain_object_field in fields(domain_object):
            if hasattr(base_domain_object, domain_object_field.name):
                setattr(
                    domain_object,
                    domain_object_field.name,
                    getattr(base_domain_object, domain_object_field.name),
                )

    def _finalize_containers(
        self,
        state: FromDataAccessObjectState,
        discovery_order: List[FromDataAccessObjectWorkItem],
    ) -> None:
        """
        Convert temporary lists to their final container types.
        """
        processed_ids = set()
        for work_item in discovery_order:
            domain_object = state.get(work_item.dao_instance)
            if domain_object is not None and id(domain_object) not in processed_ids:
                self._finalize_object_containers(domain_object)
                processed_ids.add(id(domain_object))

    @staticmethod
    def _finalize_object_containers(domain_object: Any) -> None:
        """
        Convert lists to sets based on type hints.
        """
        for attr_name in _get_set_field_names(type(domain_object)):
            value = getattr(domain_object, attr_name, None)
            if isinstance(value, list):
                setattr(domain_object, attr_name, set(value))

    def _call_post_inits(
        self,
        state: FromDataAccessObjectState,
        discovery_order: List[FromDataAccessObjectWorkItem],
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

            domain_object = state.get(work_item.dao_instance)
            if domain_object is not None and id(domain_object) not in processed_ids:
                if _has_post_init(type(domain_object)):
                    domain_object.__post_init__()
                processed_ids.add(id(domain_object))

    def _register_for_conversion(self, state: FromDataAccessObjectState) -> T:
        """
        Register this DAO for conversion if not already present.

        :param state: The conversion state.
        :return: The uninitialized domain object.
        """
        if not state.has(self):
            domain_object = state.allocate_and_memoize(
                self, self.constructable_original_class()
            )
            state.push_work_item(self, domain_object)
        return state.get(self)

    def _populate_relationships_and_scalars_from_dao(
        self, domain_object: T, state: FromDataAccessObjectState
    ) -> None:
        """
        Populate the relationships and scalar columns of the domain object.

        :param domain_object: The domain object.
        :param state: The conversion state.
        """
        plan = _get_conversion_plan(type(self))

        # check if self is a subclass of an alternative mapping and is not alternatively mapped on its own
        if plan.alternative_base is not None and not plan.uses_alternative_mapping:
            self._handle_subclass_of_alternative_mapping_in_from_dao(
                self, domain_object, plan.alternative_base.original_class()
            )
            return

        # Populate scalar columns
        for name in plan.data_column_names:
            object.__setattr__(domain_object, name, getattr(self, name))

        # Populate all relationships
        for relationship in plan.single_relationships:
            self._populate_single_relationship(
                domain_object,
                relationship.key,
                _read_single_relationship(self, relationship),
                state,
            )
        for relationship in plan.collection_relationships:
            self._populate_collection_relationship(
                domain_object,
                relationship.key,
                getattr(self, relationship.key),
                state,
                relationship.association_class,
            )

    def _fill_from_dao(self, domain_object: T, state: FromDataAccessObjectState) -> T:
        """
        Populate the domain object with data from the DAO.

        :param domain_object: The domain object to populate.
        :param state: The conversion state.
        :return: The populated domain object.
        """
        if state.discovery_mode:
            return self._trigger_discovery(domain_object, state)

        # Fallback for when _fill_from_dao is called directly (not during Phase 1)
        self._populate_relationships_and_scalars_from_dao(domain_object, state)
        return domain_object

    def _trigger_discovery(
        self,
        domain_object: T,
        state: FromDataAccessObjectState,
    ) -> T:
        """
        Trigger discovery of dependencies without fully populating the object.

        :param domain_object: The domain object.
        :param state: The conversion state.
        :return: The domain object.
        """
        plan = _get_conversion_plan(type(self))

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

        self._build_base_keyword_arguments_for_alternative_parent(domain_object, state)
        return domain_object

    def _populate_single_relationship(
        self, domain_object: Any, key: str, value: Any, state: FromDataAccessObjectState
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
        instance = self._get_or_allocate_domain_object(value, state)
        if isinstance(instance, AlternativeMapping):
            state._alternative_mappings_being_referenced[instance].append(
                (domain_object, Attribute(_attribute_name_=key, _child_=None))
            )
        object.__setattr__(domain_object, key, instance)

    def _populate_collection_relationship(
        self,
        domain_object: Any,
        key: str,
        value: Any,
        state: FromDataAccessObjectState,
        association_class: Optional[Type] = None,
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
            self._get_or_allocate_domain_object(v, state) for v in dao_collection
        ]

        # memorize alternative mapping references
        for index, instance in enumerate(instances):
            if isinstance(instance, AlternativeMapping):
                state._alternative_mappings_being_referenced[instance].append(
                    (
                        domain_object,
                        Index(
                            _key_=index,
                            _child_=Attribute(_attribute_name_=key, _child_=None),
                        ),
                    )
                )

        object.__setattr__(domain_object, key, list(instances))

    def _get_or_allocate_domain_object(
        self, dao_instance: DataAccessObject, state: FromDataAccessObjectState
    ) -> Any:
        """
        Resolve a DAO to a domain object, allocating it if necessary.

        :param dao_instance: The DAO to resolve.
        :param state: The conversion state.
        :return: The corresponding domain object.
        """
        return dao_instance.from_dao(state=state)

    def _build_base_keyword_arguments_for_alternative_parent(
        self,
        domain_object: T,
        state: FromDataAccessObjectState,
    ) -> None:
        """
        Build keyword arguments from an alternative parent DAO.

        :param domain_object: The domain object to populate.
        :param state: The conversion state.
        """
        base_clazz = self.__class__.__bases__[0]
        if not _uses_alternative_mapping(base_clazz):
            return

        # The cache key uses id(self) because synthetic parent DAOs are only valid
        # for the lifetime of this specific DAO instance and are scoped to the
        # current conversion state to ensure identity consistency between discovery
        # and filling phases.
        cache_key = (id(self), base_clazz)
        if cache_key not in state.synthetic_parent_daos:
            state.synthetic_parent_daos[cache_key] = self._create_filled_parent_dao(
                base_clazz
            )
        parent_dao = state.synthetic_parent_daos[cache_key]

        base_result = parent_dao.from_dao(state=state)

        if state.discovery_mode:
            return

        for key in _get_type_hints_cached(type(domain_object)):
            if not hasattr(self, key) and hasattr(base_result, key):
                object.__setattr__(domain_object, key, getattr(base_result, key))

    def _create_filled_parent_dao(
        self, base_clazz: Type[DataAccessObject]
    ) -> DataAccessObject:
        """
        Create a parent DAO instance populated from the current DAO.

        :param base_clazz: The parent DAO class.
        :return: The populated parent DAO instance.
        """
        parent_dao = base_clazz()
        parent_plan = _get_conversion_plan(base_clazz)
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
