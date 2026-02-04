from __future__ import annotations

import abc
import inspect
import logging
import threading
from dataclasses import dataclass, field, is_dataclass, fields, MISSING
from functools import lru_cache
from typing import _GenericAlias

import sqlalchemy.inspection
import sqlalchemy.orm
from sqlalchemy import Column
from sqlalchemy.orm import MANYTOONE, MANYTOMANY, ONETOMANY, RelationshipProperty
from typing_extensions import (
    Type,
    get_args,
    get_origin,
    get_type_hints,
    Dict,
    Any,
    TypeVar,
    Generic,
    Self,
    Optional,
    List,
    Iterable,
    Tuple,
    Set,
)


@lru_cache(maxsize=None)
def _get_type_hints_cached(clazz: Type) -> Dict[str, Any]:
    """
    Get type hints for a class.
    """
    try:
        return get_type_hints(clazz)
    except Exception:
        return {}


from collections import deque
from .exceptions import (
    NoGenericError,
    NoDAOFoundError,
    NoDAOFoundDuringParsingError,
    UnsupportedRelationshipError,
)
from ..utils import recursive_subclasses

logger = logging.getLogger(__name__)
_repr_thread_local = threading.local()

T = TypeVar("T")
_DAO = TypeVar("_DAO", bound="DataAccessObject")
WorkItemT = TypeVar("WorkItemT", bound="DataAccessObjectWorkItem")

InstanceDict = Dict[int, Any]  # Dictionary that maps object ids to objects
InProgressDict = Dict[int, bool]


def is_data_column(column: Column) -> bool:
    """
    Check if a column contains data.

    :param column: The SQLAlchemy column to check.
    :return: True if it is a data column.
    """
    return (
        not column.primary_key
        and len(column.foreign_keys) == 0
        and column.name != "polymorphic_type"
    )


@dataclass
class DataAccessObjectWorkItem(abc.ABC):
    """
    Abstract base class for conversion work items.
    """

    dao_instance: DataAccessObject


@dataclass
class DataAccessObjectState(Generic[WorkItemT], abc.ABC):
    """
    Abstract base class for conversion states.
    """

    memo: InstanceDict = field(default_factory=dict)
    """
    Cache for converted objects to prevent duplicates and handle circular references.
    """

    work_items: deque[WorkItemT] = field(default_factory=deque)
    """
    Deque of work items to be processed.
    """

    @abc.abstractmethod
    def push_work_item(self, *args: Any, **kwargs: Any) -> None:
        """
        Add a new work item to the processing queue.

        :param args: Positional arguments for the work item.
        :param kwargs: Keyword arguments for the work item.
        """
        pass

    def has(self, source: Any) -> bool:
        """
        Check if the given source object has already been converted.

        :param source: The object to check.
        :return: True if already converted.
        """
        return id(source) in self.memo

    def get(self, source: Any) -> Optional[Any]:
        """
        Get the converted object for the given source object.

        :param source: The source object.
        :return: The converted object if it exists.
        """
        return self.memo.get(id(source))

    def register(self, source: Any, target: Any) -> None:
        """
        Register a conversion result in the memoization store.

        :param source: The source object.
        :param target: The conversion result.
        """
        self.memo[id(source)] = target

    def pop(self, source: Any) -> Optional[Any]:
        """
        Remove and return the conversion result for the given source object.

        :param source: The source object.
        :return: The conversion result if it existed.
        """
        return self.memo.pop(id(source), None)


@dataclass
class ToDataAccessObjectWorkItem(DataAccessObjectWorkItem):
    """
    Work item for converting an object to a Data Access Object.
    """

    source_object: Any
    alternative_base: Optional[Type[DataAccessObject]] = None


@dataclass
class ToDataAccessObjectState(DataAccessObjectState[ToDataAccessObjectWorkItem]):
    """
    State for converting objects to Data Access Objects.
    """

    keep_alive: InstanceDict = field(default_factory=dict)
    """
    Dictionary that prevents objects from being garbage collected.
    """

    def push_work_item(
        self,
        source_object: Any,
        dao_instance: DataAccessObject,
        alternative_base: Optional[Type[DataAccessObject]] = None,
    ):
        """
        Add a new work item to the processing queue.

        :param source_object: The object being converted.
        :param dao_instance: The DAO instance being populated.
        :param alternative_base: Base class for alternative mapping, if any.
        """
        self.work_items.append(
            ToDataAccessObjectWorkItem(
                dao_instance=dao_instance,
                source_object=source_object,
                alternative_base=alternative_base,
            )
        )

    def apply_alternative_mapping_if_needed(
        self, dao_clazz: Type[DataAccessObject], source_object: Any
    ) -> Any:
        """
        Apply an alternative mapping if the DAO class requires it.

        :param dao_clazz: The DAO class to check.
        :param source_object: The object being converted.
        :return: The source object or the result of alternative mapping.
        """
        if issubclass(dao_clazz.original_class(), AlternativeMapping):
            return dao_clazz.original_class().to_dao(source_object, state=self)
        return source_object

    def register(self, source_object: Any, dao_instance: DataAccessObject) -> None:
        """
        Register a partially built DAO in the memoization stores.

        :param source_object: The object being converted.
        :param dao_instance: The partially built DAO.
        """
        super().register(source_object, dao_instance)
        self.keep_alive[id(source_object)] = source_object


@dataclass
class FromDataAccessObjectWorkItem(DataAccessObjectWorkItem):
    """
    Work item for converting a Data Access Object back to a domain object.
    """

    domain_object: Any


@dataclass
class FromDataAccessObjectState(DataAccessObjectState[FromDataAccessObjectWorkItem]):
    """
    State for converting Data Access Objects back to domain objects.
    """

    discovery_mode: bool = False
    """
    Whether the state is currently in discovery mode.
    """

    initialized_ids: Set[int] = field(default_factory=set)
    """
    Set of DAO ids that have been fully initialized.
    """

    is_processing: bool = False
    """
    Whether the state is currently in the processing loop.
    """

    synthetic_parent_daos: Dict[
        Tuple[int, Type[DataAccessObject]], DataAccessObject
    ] = field(default_factory=dict)
    """
    Cache for synthetic parent DAOs to maintain identity across discovery and filling phases.
    """

    def is_initialized(self, dao_instance: DataAccessObject) -> bool:
        """
        Check if the given DAO instance has been fully initialized.

        :param dao_instance: The DAO instance to check.
        :return: True if fully initialized.
        """
        return id(dao_instance) in self.initialized_ids

    def mark_initialized(self, dao_instance: DataAccessObject):
        """
        Mark the given DAO instance as fully initialized.

        :param dao_instance: The DAO instance to mark.
        """
        self.initialized_ids.add(id(dao_instance))

    def push_work_item(self, dao_instance: DataAccessObject, domain_object: Any):
        """
        Add a new work item to the processing queue.

        :param dao_instance: The DAO instance being converted.
        :param domain_object: The domain object being populated.
        """
        self.work_items.append(
            FromDataAccessObjectWorkItem(
                dao_instance=dao_instance, domain_object=domain_object
            )
        )

    def allocate_and_memoize(
        self, dao_instance: DataAccessObject, original_clazz: Type
    ) -> Any:
        """
        Allocate a new instance and store it in the memoization dictionary.
        Initializes default values for dataclass fields.

        :param dao_instance: The DAO instance to register.
        :param original_clazz: The domain class to instantiate.
        :return: The uninitialized domain object instance.
        """
        result = original_clazz.__new__(original_clazz)
        if is_dataclass(original_clazz):
            for f in fields(original_clazz):
                if f.default is not MISSING:
                    object.__setattr__(result, f.name, f.default)
                elif f.default_factory is not MISSING:
                    object.__setattr__(result, f.name, f.default_factory())
        self.register(dao_instance, result)
        return result


class HasGeneric(Generic[T]):
    """
    Base class for classes that carry a generic type argument.
    """

    @classmethod
    @lru_cache(maxsize=None)
    def original_class(cls) -> T:
        """
        Get the concrete generic argument.

        :return: The generic type argument.
        :raises NoGenericError: If no generic argument is found.
        """
        tp = cls._dao_like_argument()
        if tp is None:
            raise NoGenericError(cls)
        return tp

    @classmethod
    def _dao_like_argument(cls) -> Optional[Type]:
        """
        Extract the generic argument from the class hierarchy.

        :return: The generic type or None.
        """
        # filter for instances of generic aliases in the superclasses
        for base in filter(
            lambda x: isinstance(x, _GenericAlias),
            cls.__orig_bases__,
        ):
            return get_args(base)[0]

        # No acceptable base found
        return None


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
         Traverses the DAO graph (its referenced objects) to identify all reachable DAOs. For each DAO, it
         allocates an uninitialized domain object (using ``__new__``) and records
         the discovery order.
       - Phase 2: Population & Alternative Mapping Resolution (Bottom-Up):
         Populates every field of the domain objects using ``setattr``. This avoids
         the complexities of constructor matching and ensures that circular
         references are handled correctly by using the already allocated identities.
         If an object is an ``AlternativeMapping``, it is converted to its final
         domain object representation.
         During this phase, collections are represented as lists.
       - Phase 3: Container Finalization:
         Converts temporary lists back to sets where required by type hints.
       - Phase 4: Post-Initialization:
         Calls ``__post_init__`` on all fully populated and
         finalized domain objects.


    Alternative Mappings
    --------------------

    For domain objects that do not map 1:1 to a single DAO (e.g., those requiring
    special constructor logic) :class:`AlternativeMapping` can be used. The converter recognizes these and
    delegates the creation of the domain object to the mapping's ``create_from_dao``
    method during the Filling Phase.

    """

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
        alternative_base = cls._find_alternative_mapping_base()
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
    def uses_alternative_mapping(cls, class_to_check: Type) -> bool:
        """
        Check if a class uses an alternative mapping.

        :param class_to_check: The class to check.
        :return: True if alternative mapping is used.
        """
        return issubclass(class_to_check, DataAccessObject) and issubclass(
            class_to_check.original_class(), AlternativeMapping
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
        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))

        self.get_columns_from(source_object=source_object, columns=mapper.columns)
        self.fill_relationships_from(
            source_object=source_object,
            relationships=mapper.relationships,
            state=state,
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

        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))
        parent_mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(
            alternative_base
        )

        # Split columns into those from parent and those from this DAO's table
        columns_of_parent = parent_mapper.columns
        parent_column_names = {c.name for c in columns_of_parent}
        columns_of_this_table = [
            c for c in mapper.columns if c.name not in parent_column_names
        ]

        # Copy values from parent DAO and original object
        self.get_columns_from(parent_dao, columns_of_parent)
        self.get_columns_from(source_object, columns_of_this_table)

        # Ensure columns on intermediate ancestors are also covered
        for prop in mapper.column_attrs:
            if prop.key in parent_column_names:
                continue

            col = prop.columns[0]
            if is_data_column(col):
                setattr(self, prop.key, getattr(source_object, prop.key))

        # Partition and fill relationships
        relationships_of_parent, relationships_of_this_table = (
            self._partition_parent_child_relationships(parent_mapper, mapper)
        )
        self.fill_relationships_from(parent_dao, relationships_of_parent, state)
        self.fill_relationships_from(source_object, relationships_of_this_table, state)

    def _partition_parent_child_relationships(
        self, parent: sqlalchemy.orm.Mapper, child: sqlalchemy.orm.Mapper
    ) -> Tuple[
        List[RelationshipProperty[Any]],
        List[RelationshipProperty[Any]],
    ]:
        """
        Partition relationships into parent and child sets.

        :param parent: The parent mapper.
        :param child: The child mapper.
        :return: Tuple of parent and child relationship lists.
        """
        parent_rel_keys = {rel.key for rel in parent.relationships}
        relationships_of_parent = parent.relationships
        relationships_of_child = [
            relationship
            for relationship in child.relationships
            if relationship.key not in parent_rel_keys
        ]
        return relationships_of_parent, relationships_of_child

    def get_columns_from(self, source_object: Any, columns: Iterable[Column]) -> None:
        """
        Assign values from specified columns of a source object to the DAO.

        :param source_object: The source of column values.
        :param columns: The columns to copy.
        """
        for column in columns:
            if is_data_column(column):
                setattr(self, column.name, getattr(source_object, column.name))

    def fill_relationships_from(
        self,
        source_object: Any,
        relationships: Iterable[RelationshipProperty],
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Populate relationships from a source object.

        :param source_object: The source of relationship values.
        :param relationships: The relationships to process.
        :param state: The conversion state.
        """
        for relationship in relationships:
            if self._is_single_relationship(relationship):
                self._extract_single_relationship(source_object, relationship, state)
            elif relationship.direction in (ONETOMANY, MANYTOMANY):
                self._extract_collection_relationship(
                    source_object, relationship, state
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

    def _extract_single_relationship(
        self,
        source_object: Any,
        relationship: RelationshipProperty,
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Extract a single-valued relationship from a source object.

        :param source_object: The source object.
        :param relationship: The relationship property.
        :param state: The conversion state.
        """
        value = getattr(source_object, relationship.key)
        if value is None:
            setattr(self, relationship.key, None)
            return

        dao_instance = self._get_or_queue_dao(value, state)
        setattr(self, relationship.key, dao_instance)

    def _extract_collection_relationship(
        self,
        source_object: Any,
        relationship: RelationshipProperty,
        state: ToDataAccessObjectState,
    ) -> None:
        """
        Extract a collection relationship from a source object.

        :param source_object: The source object.
        :param relationship: The relationship property.
        :param state: The conversion state.
        """
        source_collection = getattr(source_object, relationship.key)
        dao_collection = [self._get_or_queue_dao(v, state) for v in source_collection]
        setattr(self, relationship.key, type(source_collection)(dao_collection))

    def _get_or_queue_dao(
        self, source_object: Any, state: ToDataAccessObjectState
    ) -> DataAccessObject:
        """
        Resolve a source object to a DAO, queuing it if necessary.

        :param source_object: The object to resolve.
        :param state: The conversion state.
        :return: The corresponding DAO instance.
        """
        # Check if already built
        existing = state.get(source_object)
        if existing is not None:
            return existing

        dao_clazz = get_dao_class(type(source_object))
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
        alternative_base = dao_clazz._find_alternative_mapping_base()
        state.push_work_item(mapped_object, result, alternative_base)

        return result

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
            return state.get(self)

        if not state.is_processing:
            return self._perform_from_dao_conversion(state)

        return self._register_for_conversion(state)

    def _perform_from_dao_conversion(self, state: FromDataAccessObjectState) -> T:
        """
        Perform the four-phase conversion process.

        :param state: The conversion state.
        :return: The converted domain object.
        """
        state.is_processing = True
        discovery_order = []
        if not state.has(self):
            state.allocate_and_memoize(self, self.original_class())
        state.push_work_item(self, state.get(self))

        self._discover_dependencies(state, discovery_order)
        self._fill_domain_objects(state, discovery_order)
        self._finalize_containers(state, discovery_order)
        self._call_post_inits(state, discovery_order)

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

        while state.work_items:
            # Use pop() to treat the deque as a stack (LIFO) for DFS
            work_item = state.work_items.pop()
            discovery_order.append(work_item)
            work_item.dao_instance._fill_from_dao(work_item.domain_object, state)

        state.discovery_mode = False

    def _fill_domain_objects(
        self,
        state: FromDataAccessObjectState,
        discovery_order: List[FromDataAccessObjectWorkItem],
    ) -> None:
        """
        Phase 2: Filling (Bottom-Up) to initialize domain objects.

        :param state: The conversion state.
        :param discovery_order: The order in which DAOs were discovered.
        """
        for work_item in reversed(discovery_order):
            if not state.is_initialized(work_item.dao_instance):
                work_item.dao_instance._fill_from_dao(work_item.domain_object, state)
                state.mark_initialized(work_item.dao_instance)

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
        hints = _get_type_hints_cached(type(domain_object))

        for attr_name, hint in hints.items():
            origin = get_origin(hint)
            # Handle both typing.Set[...] and built-in set
            if origin is not set and hint is not set:
                continue
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
            if issubclass(work_item.dao_instance.original_class(), AlternativeMapping):
                continue

            domain_object = state.get(work_item.dao_instance)
            if domain_object is not None and id(domain_object) not in processed_ids:
                if hasattr(domain_object, "__post_init__"):
                    domain_object.__post_init__()
                processed_ids.add(id(domain_object))

    def _register_for_conversion(self, state: FromDataAccessObjectState) -> T:
        """
        Register this DAO for conversion if not already present.

        :param state: The conversion state.
        :return: The uninitialized domain object.
        """
        if not state.has(self):
            domain_object = state.allocate_and_memoize(self, self.original_class())
            state.push_work_item(self, domain_object)
        return state.get(self)

    def _fill_from_dao(self, domain_object: T, state: FromDataAccessObjectState) -> T:
        """
        Populate the domain object with data from the DAO.

        :param domain_object: The domain object to populate.
        :param state: The conversion state.
        :return: The populated domain object.
        """
        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))

        if state.discovery_mode:
            return self._trigger_discovery(domain_object, mapper, state)

        return self._populate_domain_object(domain_object, mapper, state)

    def _trigger_discovery(
        self,
        domain_object: T,
        mapper: sqlalchemy.orm.Mapper,
        state: FromDataAccessObjectState,
    ) -> T:
        """
        Trigger discovery of dependencies without fully populating the object.

        :param domain_object: The domain object.
        :param mapper: The SQLAlchemy mapper.
        :param state: The conversion state.
        :return: The domain object.
        """
        for relationship in mapper.relationships:
            value = getattr(self, relationship.key)
            if self._is_single_relationship(relationship) and value is not None:
                value.from_dao(state=state)
            elif relationship.direction in (ONETOMANY, MANYTOMANY):
                for item in value:
                    item.from_dao(state=state)

        self._build_base_keyword_arguments_for_alternative_parent(domain_object, state)
        return domain_object

    def _populate_domain_object(
        self,
        domain_object: T,
        mapper: sqlalchemy.orm.Mapper,
        state: FromDataAccessObjectState,
    ) -> T:
        """
        Fully populate the domain object using setattr.

        :param domain_object: The domain object to populate.
        :param mapper: The SQLAlchemy mapper.
        :param state: The conversion state.
        :return: The populated domain object.
        """
        # Populate scalar columns
        for column in mapper.columns:
            if is_data_column(column):
                value = getattr(self, column.name)
                object.__setattr__(domain_object, column.name, value)

        # Populate all relationships
        for relationship in mapper.relationships:
            self._populate_relationship(domain_object, relationship, state)

        # Populate from alternative parent if any
        self._build_base_keyword_arguments_for_alternative_parent(domain_object, state)

        if isinstance(domain_object, AlternativeMapping):
            return self._handle_alternative_mapping_result(domain_object, state)

        return domain_object

    def _handle_alternative_mapping_result(
        self, alternative_mapping: AlternativeMapping, state: FromDataAccessObjectState
    ) -> Any:
        """
        Handle the result of an AlternativeMapping.

        :param alternative_mapping: The alternative mapping instance.
        :param state: The conversion state.
        :return: The final domain object.
        """
        final_result = alternative_mapping.to_domain_object()
        # Update memo if AlternativeMapping changed the instance
        state.register(self, final_result)
        return final_result

    def _populate_relationship(
        self,
        domain_object: T,
        relationship: RelationshipProperty,
        state: FromDataAccessObjectState,
    ) -> None:
        """
        Populate a specific relationship on the domain object.

        :param domain_object: The domain object.
        :param relationship: The relationship to populate.
        :param state: The conversion state.
        """
        value = getattr(self, relationship.key)
        if self._is_single_relationship(relationship):
            self._populate_single_relationship(
                domain_object, relationship.key, value, state
            )
        elif relationship.direction in (ONETOMANY, MANYTOMANY):
            self._populate_collection_relationship(
                domain_object, relationship.key, value, state
            )

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
        object.__setattr__(domain_object, key, instance)

    def _populate_collection_relationship(
        self, domain_object: Any, key: str, value: Any, state: FromDataAccessObjectState
    ) -> None:
        """
        Populate a collection relationship on the domain object.

        :param domain_object: The domain object.
        :param key: The attribute name.
        :param value: The collection of DAO instances.
        :param state: The conversion state.
        """
        if not value:
            object.__setattr__(domain_object, key, value)
            return
        instances = [self._get_or_allocate_domain_object(v, state) for v in value]
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

    def _register_for_conversion(self, state: FromDataAccessObjectState) -> T:
        """
        Register this DAO for conversion if not already present.

        :param state: The conversion state.
        :return: The uninitialized domain object.
        """
        if not state.has(self):
            domain_object = state.allocate_and_memoize(self, self.original_class())
            state.push_work_item(self, domain_object)
        return state.get(self)

    def _perform_from_dao_conversion(self, state: FromDataAccessObjectState) -> T:
        """
        Perform the four-phase conversion process.

        :param state: The conversion state.
        :return: The converted domain object.
        """
        state.is_processing = True
        discovery_order = []
        if not state.has(self):
            state.allocate_and_memoize(self, self.original_class())
        state.push_work_item(self, state.get(self))

        self._discover_dependencies(state, discovery_order)
        self._fill_domain_objects(state, discovery_order)
        self._finalize_containers(state, discovery_order)
        self._call_post_inits(state, discovery_order)

        state.is_processing = False

        return state.get(self)

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
        if not self.uses_alternative_mapping(base_clazz):
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
        parent_mapper = sqlalchemy.inspection.inspect(base_clazz)
        parent_dao.get_columns_from(self, parent_mapper.columns)
        for relationship in parent_mapper.relationships:
            setattr(parent_dao, relationship.key, getattr(self, relationship.key))
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


class AlternativeMapping(HasGeneric[T], abc.ABC):
    """
    Base class for alternative mapping implementations.
    """

    @classmethod
    def to_dao(
        cls, source_object: T, state: Optional[ToDataAccessObjectState] = None
    ) -> _DAO:
        """
        Resolve a source object to a DAO.

        :param source_object: The object to convert.
        :param state: The conversion state.
        :return: The converted DAO instance.
        """
        state = state or ToDataAccessObjectState()
        if state.has(source_object):
            return state.get(source_object)
        elif isinstance(source_object, cls):
            return source_object
        else:
            result = cls.from_domain_object(source_object)
            return result

    @classmethod
    @abc.abstractmethod
    def from_domain_object(cls, obj: T) -> Self:
        """
        Create this from a domain object.
        Do not create any DAOs here but the target DAO of `T`.
        The rest of the `to_dao` algorithm will process the fields of the created instance.

        :param obj: The source object.
        :return: A new instance of this mapping class.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_domain_object(self) -> T:
        """
        Create a domain object from this instance.

        :return: The constructed domain object.
        """
        raise NotImplementedError


@lru_cache(maxsize=None)
def _get_clazz_by_original_clazz(
    base_clazz: Type, original_clazz: Type
) -> Optional[Type]:
    """
    Find a subclass that maps to a specific domain class.

    :param base_clazz: The base class to search from.
    :param original_clazz: The domain class to match.
    :return: The matching subclass or None.
    """
    for subclass in recursive_subclasses(base_clazz):
        try:
            if subclass.original_class() == original_clazz:
                return subclass
        except (AttributeError, TypeError, NoGenericError):
            continue
    return None


@lru_cache(maxsize=None)
def get_dao_class(original_clazz: Type) -> Optional[Type[DataAccessObject]]:
    """
    Retrieve the DAO class for a domain class.

    :param original_clazz: The domain class.
    :return: The corresponding DAO class or None.
    """
    alternative_mapping = get_alternative_mapping(original_clazz)
    if alternative_mapping is not None:
        original_clazz = alternative_mapping

    return _get_clazz_by_original_clazz(DataAccessObject, original_clazz)


@lru_cache(maxsize=None)
def get_alternative_mapping(
    original_clazz: Type,
) -> Optional[Type[AlternativeMapping]]:
    """
    Retrieve the alternative mapping for a domain class.

    :param original_clazz: The domain class.
    :return: The corresponding alternative mapping or None.
    """
    return _get_clazz_by_original_clazz(AlternativeMapping, original_clazz)


def to_dao(
    source_object: Any, state: Optional[ToDataAccessObjectState] = None
) -> DataAccessObject:
    """
    Convert an object to its corresponding DAO.

    :param source_object: The object to convert.
    :param state: The conversion state.
    :return: The converted DAO instance.
    """
    dao_clazz = get_dao_class(type(source_object))
    if dao_clazz is None:
        raise NoDAOFoundError(source_object)
    state = state or ToDataAccessObjectState()
    return dao_clazz.to_dao(source_object, state)
