from __future__ import annotations

import abc
import inspect
import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import _GenericAlias

import sqlalchemy.inspection
import sqlalchemy.orm
from sqlalchemy import Column
from sqlalchemy.orm import MANYTOONE, MANYTOMANY, ONETOMANY, RelationshipProperty
from sqlalchemy.util import ReadOnlyProperties
from typing_extensions import (
    Type,
    get_args,
    Dict,
    Any,
    TypeVar,
    Generic,
    Self,
    Optional,
    List,
    Iterable,
    Tuple,
)

from ..utils import recursive_subclasses

logger = logging.getLogger(__name__)
_repr_thread_local = threading.local()

T = TypeVar("T")
_DAO = TypeVar("_DAO", bound="DataAccessObject")
InstanceDict = Dict[int, Any]  # Dictionary that maps object ids to objects
InProgressDict = Dict[int, bool]


@dataclass
class NoGenericError(TypeError):
    """
    Exception raised when the original class for a DataAccessObject subclass cannot
    be determined.

    This exception is typically raised when a DataAccessObject subclass has not
    been parameterized properly, which prevents identifying the original class
    associated with it.
    """

    clazz: Type

    def __post_init__(self):
        super().__init__(
            f"Cannot determine original class for {self.clazz}. "
            "Did you forget to parameterise the DataAccessObject subclass?"
        )


@dataclass
class NoDAOFoundError(TypeError):
    """
    Represents an error raised when no DAO (Data Access Object) class is found for a given class.

    This exception is typically used when an attempt to convert a class into a corresponding DAO fails.
    It provides information about the class and the DAO involved.
    """

    obj: Any
    """
    The class that no dao was found for
    """

    def __post_init__(self):
        super().__init__(
            f"Class {type(self.obj)} does not have a DAO. Did you forget to import your ORM Interface?"
        )


@dataclass
class NoDAOFoundDuringParsingError(NoDAOFoundError):

    dao: Type
    """
    The DAO class that tried to convert the cls to a DAO if any.
    """

    relationship: RelationshipProperty
    """
    The relationship that tried to create the DAO.
    """

    def __init__(self, obj: Any, dao: Type, relationship: RelationshipProperty = None):
        TypeError.__init__(
            self,
            f"Class {type(obj)} does not have a DAO. This happened when trying "
            f"to create a dao for {dao}) on the relationship {relationship} with the "
            f"relationship value {obj}. "
            f"Expected a relationship value of type {relationship.target}.",
        )


def is_data_column(column: Column):
    return (
        not column.primary_key
        and len(column.foreign_keys) == 0
        and column.name != "polymorphic_type"
    )


@dataclass
class ToDAOState:
    """
    Encapsulates the conversion state for to_dao conversions.

    This bundles memoization and keep-alive dictionaries and exposes
    convenience operations used during conversion so that only the state
    needs to be passed around.
    """

    memo: InstanceDict = field(default_factory=dict)
    """
    Dictionary that keeps track of already converted objects during DAO conversion.
    Maps object IDs to their corresponding DAO instances to prevent duplicate conversion
    and handle circular references. Acts as a memoization cache to improve performance
    and maintain object identity.
    """

    keep_alive: InstanceDict = field(default_factory=dict)
    """
    Dictionary that prevents objects from being garbage collected.
    """

    def get_existing(self, obj: Any) -> Any:
        """
        Return an existing DAO for the given object if it was already created.
        """
        return self.memo.get(id(obj))

    def apply_alternative_mapping_if_needed(
        self, dao_cls: Type[DataAccessObject], obj: Any
    ) -> Any:
        """
        Apply AlternativeMapping.to_dao if the dao class uses an alternative mapping.
        """
        if issubclass(dao_cls.original_class(), AlternativeMapping):
            return dao_cls.original_class().to_dao(obj, state=self)
        return obj

    def register(self, obj: Any, result: Any) -> None:
        """
        Register a partially built DAO in the memoization stores to break cycles.
        """
        oid = id(obj)
        self.memo[oid] = result
        self.keep_alive[oid] = obj


@dataclass
class FromDAOState:
    """
    Encapsulates the conversion state for from_dao conversions.

    Bundles memoization and in-progress tracking and provides helpers for
    allocation, circular detection, and fix-ups.
    """

    memo: InstanceDict = field(default_factory=dict)
    """
    Dictionary that keeps track of already converted objects during DAO conversion.
    Maps object IDs to their corresponding DAO instances to prevent duplicate conversion
    and handle circular references. Acts as a memoization cache to improve performance
    and maintain object identity.
    """

    in_progress: InProgressDict = field(default_factory=dict)
    """
    Dictionary that marks objects as currently being processed by the `from_dao` method.
    """

    def has(self, dao_obj: Any) -> bool:
        return id(dao_obj) in self.memo

    def get(self, dao_obj: Any) -> Any:
        return self.memo[id(dao_obj)]

    def allocate_and_memoize(self, dao_obj: Any, original_cls: Type) -> Any:
        """
        Allocates a new instance of the specified class and stores it in a memoization
        dictionary to avoid duplicating object construction for the same identifier.

        :param dao_obj: The data access object whose identifier is used to memoize
            the created instance.
        :param original_cls: The class type to create a new instance for.
        :return: A newly allocated instance of the given class.
        """
        result = original_cls.__new__(original_cls)
        self.memo[id(dao_obj)] = result
        self.in_progress[id(dao_obj)] = True
        return result

    def parse_single(self, value: Any) -> tuple[Any, bool]:
        """
        Parses a single value from the DAO context. This method checks whether the given
        value is `None` and returns a tuple containing the parsed object and a boolean
        flag indicating whether the parsed object exists within the `memo` dictionary,
        based on its unique identifier.

        :param value: The value to be parsed. It can be any type.
        :return: A tuple containing the parsed object and a boolean flag. The boolean
            indicates whether the parsed value exists in the `memo` dictionary.
        """
        if value is None:
            return None, False
        parsed = value.from_dao(state=self)
        return parsed, parsed is self.memo.get(id(value))

    def parse_collection(self, value: Any) -> tuple[Any, List[Any]]:
        """
        Parses a given collection of objects, converting each element using its ``from_dao``
        method and handling circular references.

        This method takes an input collection, processes each element by converting it
        using a state, and identifies circular references if any of the converted instances
        match previously processed elements in the memo dictionary. The method returns the
        processed collection with all objects converted and a list of circular references that
        could not be resolved.

        :param value: The collection to be parsed, which can contain objects with a
            ``from_dao`` method. It may also contain circular references.
        :return: A tuple where the first element is the processed collection with all
            objects converted using ``from_dao``, and the second element is a list of
            circular references that could not be fully resolved.
        """
        if not value:
            return value, []
        instances = []
        circular_values: List[Any] = []
        for v in value:
            instance = v.from_dao(state=self)
            if instance is self.memo.get(id(v)):
                circular_values.append(v)
            instances.append(instance)
        return type(value)(instances), circular_values

    def apply_circular_fixes(self, result: Any, circular_refs: Dict[str, Any]) -> None:
        """
        Fixes circular references in the provided `result` object using the `circular_refs`
        dictionary. This method resolves values in `circular_refs` to objects stored in the
        `memo` dictionary and assigns the resolved reference back to the `result` object.

        :param result: The object whose circular references need to be fixed.
        :param circular_refs: A dictionary mapping attribute names in `result` to circular
            reference values. The keys in the dictionary specify attributes in `result`, and
            the values are either lists or single objects that are resolved using the `memo`
            dictionary.
        """
        for key, value in circular_refs.items():
            if isinstance(value, list):
                fixed_list = []
                for v in value:
                    fixed_list.append(self.memo.get(id(v)))
                setattr(result, key, fixed_list)
            else:
                setattr(result, key, self.memo.get(id(value)))


class HasGeneric(Generic[T]):

    @classmethod
    @lru_cache(maxsize=None)
    def original_class(cls) -> T:
        """
        :return: The concrete generic argument for DAO-like bases.
        :raises NoGenericError: If no generic argument is found.
        """
        tp = cls._dao_like_argument()
        if tp is None:
            raise NoGenericError(cls)
        return tp

    @classmethod
    def _dao_like_argument(cls) -> Optional[Type]:
        """
        :return: The concrete generic argument for DAO-like bases.
        """
        # filter for instances of generic aliases in the superclasses
        for base in filter(
            lambda x: isinstance(x, _GenericAlias),
            cls.__orig_bases__,
        ):
            return get_args(base)[0]

        # No acceptable base found
        return None


@dataclass
class UnsupportedRelationshipError(ValueError):
    """
    Raised when a relationship direction is not supported by the ORM mapping.

    This error indicates that the relationship configuration could not be
    interpreted into a domain mapping.
    """

    relationship: RelationshipProperty

    def __post_init__(self):
        ValueError.__init__(
            self,
            f"Unsupported relationship direction for {self.relationship}.",
        )


class DataAccessObject(HasGeneric[T]):
    """
    This class defines the interfaces the DAO classes should implement.

    ORMatic generates classes from your python code that are derived from the provided classes in your package.
    The generated classes can be instantiated from objects of the given classes and vice versa.
    This class implements the necessary functionality.
    """

    def __init__(self, *args, **kwargs):
        """
        Allow constructing DAO instances with positional arguments that map to
        data columns (non-PK, non-FK, non-polymorphic) in declaration order.

        Falls back to the default SQLAlchemy initialization if positional
        arguments do not match the number of data columns or if keyword
        arguments are provided.
        """
        if args and not kwargs:
            try:
                mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(
                    type(self)
                )
                data_columns = [c for c in mapper.columns if is_data_column(c)]
                if len(args) == len(data_columns):
                    kwargs = {col.name: value for col, value in zip(data_columns, args)}
                    super().__init__(**kwargs)
                    return
            except Exception:
                # If inspection fails or mapping is not aligned, defer to default behavior
                pass
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = getattr(cls, "__init__", None)

        def init_with_positional(self, *args, **kw):
            if args and not kw:
                try:
                    mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(
                        type(self)
                    )
                    data_columns = [c for c in mapper.columns if is_data_column(c)]
                    if len(args) == len(data_columns):
                        built = {
                            col.name: value for col, value in zip(data_columns, args)
                        }
                        return (
                            original_init(self, **built)
                            if original_init
                            else super(cls, self).__init__(**built)
                        )
                except Exception:
                    pass
            return (
                original_init(self, *args, **kw)
                if original_init
                else super(cls, self).__init__(*args, **kw)
            )

        # Inject only if the class did not already define a positional-friendly constructor
        cls.__init__ = init_with_positional

    @classmethod
    def to_dao(
        cls,
        obj: T,
        state: Optional[ToDAOState] = None,
        register=True,
    ) -> _DAO:
        """
        Convert an object to its Data Access Object.

        Ensures memoization to prevent duplicate work, applies alternative
        mappings when needed, and delegates to the appropriate conversion
        strategy based on inheritance.

        :param obj: Object to be converted into its DAO equivalent
        :param state: The state to use as context
        :param register: Whether to register the DAO class in the memo.
        :return: Instance of the DAO class (_DAO) that represents the input object after conversion
        """

        state = state or ToDAOState()

        # check if this object has been build already
        existing = state.get_existing(obj)
        if existing is not None:
            return existing

        dao_obj = state.apply_alternative_mapping_if_needed(cls, obj)

        # Determine the appropriate DAO base to consider for alternative mappings.
        # The previous implementation only looked at the immediate base class, which
        # fails when an alternatively-mapped parent exists higher up the hierarchy
        # (e.g., a grandparent). We now scan the MRO to find the nearest DAO ancestor
        # whose original class uses AlternativeMapping.
        alt_base: Optional[Type[DataAccessObject]] = None
        for b in cls.__mro__[1:]:  # skip cls itself
            try:
                if issubclass(b, DataAccessObject) and issubclass(
                    b.original_class(), AlternativeMapping
                ):
                    alt_base = b
                    break
            except Exception:
                # Some bases may not be DAOs or may not have generic info; skip safely
                continue

        result = cls()

        if register:
            state.register(obj, result)

        # choose the correct building method
        if alt_base is not None:
            result.to_dao_if_subclass_of_alternative_mapping(
                obj=dao_obj, base=alt_base, state=state
            )
        else:
            result.to_dao_default(obj=dao_obj, state=state)

        return result

    @classmethod
    def uses_alternative_mapping(cls, class_to_check: Type) -> bool:
        """
        :param class_to_check: The class to check
        :return: If the class to check uses an alternative mapping to specify the DAO or not.
        """
        return issubclass(class_to_check, DataAccessObject) and issubclass(
            class_to_check.original_class(), AlternativeMapping
        )

    def to_dao_default(self, obj: T, state: ToDAOState):
        """
        Convert the given object into a Data Access Object (DAO) representation.

        The method extracts column and relationship data.

        :param obj: The source object to be converted into a DAO representation.
        :param state: The conversion state for memoization and lifecycle control.
        """
        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))

        self.get_columns_from(obj=obj, columns=mapper.columns)
        self.get_relationships_from(
            obj=obj,
            relationships=mapper.relationships,
            state=state,
        )

    def to_dao_if_subclass_of_alternative_mapping(
        self,
        obj: T,
        base: Type[DataAccessObject],
        state: ToDAOState,
    ):
        """
        Transforms the given object into a corresponding DAO if it is a
        subclass of an alternatively mapped entity. This involves processing both the inherited
        and subclass-specific attributes and relationships of the object.
        The method directly modifies the DAO instance by populating it with attribute
        and relationship data from the source object.

        :param obj: The source object to be transformed into a DAO.
        :param base: The parent class type that defines the base mapping for the DAO.
        :param state: The conversion state.
        """

        # Temporarily remove the object from the memo dictionary to allow the parent DAO to be created
        temp_dao = None
        if id(obj) in state.memo:
            temp_dao = state.memo[id(obj)]
            del state.memo[id(obj)]

        # create dao of alternatively mapped superclass
        parent_dao = base.original_class().to_dao(obj, state)

        # Restore the object in the memo dictionary
        if temp_dao is not None:
            state.memo[id(obj)] = temp_dao

        # Fill super class columns
        parent_mapper = sqlalchemy.inspection.inspect(base)
        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))

        # split up the columns in columns defined by the parent and columns defined by this dao
        all_columns = mapper.columns
        columns_of_parent = parent_mapper.columns
        columns_of_this_table = [
            c for c in all_columns if c.name not in columns_of_parent
        ]

        # copy values from superclass dao
        self.get_columns_from(parent_dao, columns_of_parent)

        # copy values that only occur in this dao (current table)
        self.get_columns_from(obj, columns_of_this_table)

        # Also ensure that columns declared on intermediate ancestors (between the
        # alternatively-mapped base and this DAO) are populated. SQLAlchemy may not
        # include those columns in the current table's column collection, so we walk
        # all column attributes visible on this mapper and fill any data columns that
        # are not owned by the alternatively-mapped base.
        parent_column_names = {c.name for c in columns_of_parent}
        for prop in mapper.column_attrs:
            try:
                col = prop.columns[0]
            except Exception:
                continue
            if is_data_column(col) and prop.key not in parent_column_names:
                # take the value from the original object; attribute names match
                setattr(self, prop.key, getattr(obj, prop.key))

        # split relationships in relationships by parent and relationships by child
        relationships_of_parent, relationships_of_this_table = (
            self.partition_parent_child_relationships(parent_mapper, mapper)
        )

        # get relationships from parent dao
        self.get_relationships_from(parent_dao, relationships_of_parent, state)

        # get relationships from the current table
        self.get_relationships_from(obj, relationships_of_this_table, state)

    def partition_parent_child_relationships(
        self, parent: sqlalchemy.orm.Mapper, child: sqlalchemy.orm.Mapper
    ) -> Tuple[
        List[RelationshipProperty[Any]],
        List[RelationshipProperty[Any]],
    ]:
        """
        Partition the relationships by parent-only and child-only relationships.

        :param parent: The parent mapper to extract relationships from
        :param child: The child mapper to extract relationships from
        :return: A tuple of the relationships that are only in the parent and the relationships that are only in the child
        """
        all_relationships = child.relationships
        relationships_of_parent = parent.relationships
        relationship_names_of_parent = list(
            map(lambda x: x.key, relationships_of_parent)
        )

        relationships_of_child = list(
            filter(
                lambda x: x.key not in relationship_names_of_parent, all_relationships
            )
        )
        return relationships_of_parent, relationships_of_child

    def get_columns_from(self, obj: T, columns: Iterable[Column]) -> None:
        """
        Retrieves and assigns values from specified columns of a given object.

        Assumes that the attribute names of `obj` and `self` are the same.

        :param obj: The object from which the column values are retrieved.
        :param columns: A list of columns to be processed.

        Raises:
            AttributeError: Raised if the provided object or column does not have
                the corresponding attribute during assignment.
        """
        for column in columns:
            if is_data_column(column):
                setattr(self, column.name, getattr(obj, column.name))

    def get_relationships_from(
        self,
        obj: T,
        relationships: Iterable[RelationshipProperty],
        state: ToDAOState,
    ):
        """
        Retrieve and update relationships from an object based on the given relationship
        properties.

        This method delegates to focused helpers for single-valued and collection-valued
        relationships to keep complexity low.

        :param obj: The object from which the relationship values are retrieved.
        :param relationships: A list of relationships to be processed.
        :param state: The conversion state for memoization and lifecycle control.
        """
        for relationship in relationships:
            if relationship.direction == MANYTOONE or (
                relationship.direction == ONETOMANY and not relationship.uselist
            ):
                self._extract_single_relationship(
                    obj=obj,
                    relationship=relationship,
                    state=state,
                )
            elif relationship.direction in (ONETOMANY, MANYTOMANY):
                self._extract_collection_relationship(
                    obj=obj,
                    relationship=relationship,
                    state=state,
                )

    def _extract_single_relationship(
        self,
        obj: T,
        relationship: RelationshipProperty,
        state: ToDAOState,
    ) -> None:
        """
        Extract a single-valued relationship and assign the corresponding DAO.
        Check `get_relationships_from` for more information.
        """
        value_in_obj = getattr(obj, relationship.key)
        if value_in_obj is None:
            setattr(self, relationship.key, None)
            return

        dao_class = get_dao_class(type(value_in_obj))
        if dao_class is None:
            raise NoDAOFoundDuringParsingError(value_in_obj, type(self), relationship)

        dao_of_value = dao_class.to_dao(value_in_obj, state=state)
        setattr(self, relationship.key, dao_of_value)

    def _extract_collection_relationship(
        self,
        obj: T,
        relationship: RelationshipProperty,
        state: "ToDAOState",
    ) -> None:
        """
        Extract a collection-valued relationship and assign a list of DAOs.
        Check `get_relationships_from` for more information.
        """
        result = []
        value_in_obj = getattr(obj, relationship.key)
        for v in value_in_obj:
            dao_class = get_dao_class(type(v))
            if dao_class is None:
                raise NoDAOFoundDuringParsingError(v, type(self), relationship)
            result.append(dao_class.to_dao(v, state=state))
        setattr(self, relationship.key, result)

    def from_dao(
        self,
        state: Optional[FromDAOState] = None,
    ) -> T:
        """
        Convert this Data Access Object into its domain model instance.

        Uses a two-phase approach: allocate and memoize first to break cycles,
        then populate scalars and relationships, handle alternative mapping
        inheritance, initialize, and finally fix circular references.
        """
        state = state or FromDAOState()

        if state.has(self):
            return state.get(self)

        result = self._allocate_uninitialized_and_memoize(state)
        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))

        argument_names = self._argument_names()
        kwargs = self._collect_scalar_kwargs(mapper, argument_names)

        rel_kwargs, circular_refs = self._collect_relationship_kwargs(
            mapper, argument_names, state
        )
        kwargs.update(rel_kwargs)

        base_kwargs = self._build_base_kwargs_for_alternative_parent(
            argument_names, state
        )

        init_args = {**base_kwargs, **kwargs}
        self._call_initializer_or_assign(result, init_args)

        self._apply_circular_fixes(result, circular_refs, state)

        if isinstance(result, AlternativeMapping):
            result = result.create_from_dao()
            state.memo[id(self)] = result

        del state.in_progress[id(self)]
        return result

    def _allocate_uninitialized_and_memoize(self, state: FromDAOState) -> Any:
        """
        Allocate an uninitialized domain object and memoize immediately.
        """
        return state.allocate_and_memoize(self, self.original_class())

    def _argument_names(self) -> List[str]:
        """
        :return: __init__ argument names of the original class (excluding self).
        """
        init_of_original_class = self.original_class().__init__
        return [
            p.name
            for p in inspect.signature(init_of_original_class).parameters.values()
        ][1:]

    def _collect_scalar_kwargs(
        self, mapper: sqlalchemy.orm.Mapper, argument_names: List[str]
    ) -> Dict[str, Any]:
        """
        :return: keyword arguments for scalar columns present in the constructor.
        """
        kwargs: Dict[str, Any] = {}
        for column in mapper.columns:
            if column.name in argument_names and is_data_column(column):
                kwargs[column.name] = getattr(self, column.name)
        return kwargs

    def _collect_relationship_kwargs(
        self,
        mapper: sqlalchemy.orm.Mapper,
        argument_names: List[str],
        state: FromDAOState,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Collect relationship constructor arguments and capture circular references.

        :param mapper: SQLAlchemy mapper object
        :param argument_names: Names of arguments
        :param state: The conversion state.
        :return: A tuple of (relationship_kwargs, circular_references_map).
        """
        rel_kwargs: Dict[str, Any] = {}
        circular_refs: Dict[str, Any] = {}
        for relationship in mapper.relationships:
            if relationship.key not in argument_names:
                continue
            value = getattr(self, relationship.key)
            if relationship.direction == MANYTOONE or (
                relationship.direction == ONETOMANY and not relationship.uselist
            ):
                parsed, is_circular = state.parse_single(value)
                if is_circular:
                    circular_refs[relationship.key] = value
                rel_kwargs[relationship.key] = parsed
            elif relationship.direction in (ONETOMANY, MANYTOMANY):
                parsed_list, circular_list = state.parse_collection(value)
                if circular_list:
                    circular_refs[relationship.key] = circular_list
                rel_kwargs[relationship.key] = parsed_list
            else:
                raise UnsupportedRelationshipError(relationship)
        return rel_kwargs, circular_refs

    def _build_base_kwargs_for_alternative_parent(
        self,
        argument_names: List[str],
        state: "FromDAOState",
    ) -> Dict[str, Any]:
        """
        Build a dictionary of base keyword arguments for an alternative parent DAO and mapping.

        :param argument_names: Constructor argument names of the original class.
        :param state: The conversion state.
        :return: A dictionary of keyword arguments derived from the base DAO and mapping.
        """
        base = self.__class__.__bases__[0]
        base_kwargs: Dict[str, Any] = {}
        if self.uses_alternative_mapping(base):
            parent_dao = base()
            parent_mapper = sqlalchemy.inspection.inspect(base)
            for column in parent_mapper.columns:
                if is_data_column(column):
                    setattr(parent_dao, column.name, getattr(self, column.name))
            for rel in parent_mapper.relationships:
                setattr(parent_dao, rel.key, getattr(self, rel.key))
            base_result = parent_dao.from_dao(state=state)
            for argument in argument_names:
                if argument not in base_kwargs and not hasattr(self, argument):
                    try:
                        base_kwargs[argument] = getattr(base_result, argument)
                    except AttributeError:
                        ...
        return base_kwargs

    @classmethod
    def _call_initializer_or_assign(
        cls, result: Any, init_args: Dict[str, Any]
    ) -> None:
        """
        Call the original __init__. If it fails due to signature mismatch, assign attributes directly.
        """
        try:
            result.__init__(**init_args)
        except TypeError as e:
            logging.getLogger(__name__).debug(
                f"from_dao __init__ call failed with {e}; falling back to manual assignment. "
                f"This might skip side effects of the original initialization."
            )
            for key, val in init_args.items():
                setattr(result, key, val)

    @classmethod
    def _apply_circular_fixes(
        cls, result: Any, circular_refs: Dict[str, Any], state: "FromDAOState"
    ) -> None:
        """
        Replace circular placeholder DAOs with their finalized domain objects using the state.
        """
        state.apply_circular_fixes(result, circular_refs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        try:
            mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))
            # Compare only data columns, ignoring PK/FK/polymorphic columns
            for column in mapper.columns:
                if is_data_column(column):
                    if getattr(self, column.name) != getattr(other, column.name):
                        return False
            return True
        except Exception:
            # Fallback to identity comparison if we cannot inspect
            return self is other

    def __repr__(self):
        if not hasattr(_repr_thread_local, "seen"):
            _repr_thread_local.seen = set()

        if id(self) in _repr_thread_local.seen:
            return f"{self.__class__.__name__}(...)"

        _repr_thread_local.seen.add(id(self))
        try:
            mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(type(self))
            kwargs = []
            for column in mapper.columns:
                value = getattr(self, column.name)
                if is_data_column(column):
                    kwargs.append(f"{column.name}={repr(value)}")

            for relationship in mapper.relationships:
                value = getattr(self, relationship.key)
                if value is not None:
                    if isinstance(value, list):
                        kwargs.append(
                            f"{relationship.key}=[{', '.join(repr(v) for v in value)}]"
                        )
                    else:
                        kwargs.append(f"{relationship.key}={repr(value)}")
                else:
                    kwargs.append(f"{relationship.key}=None")

            return f"{self.__class__.__name__}({', '.join(kwargs)})"
        finally:
            _repr_thread_local.seen.remove(id(self))


class AlternativeMapping(HasGeneric[T], abc.ABC):

    @classmethod
    def to_dao(cls, obj: T, state: Optional[ToDAOState] = None) -> _DAO:
        """
        Create a DAO from the obj if it doesn't exist.

        :param obj: The obj to create the DAO from.
        :param state: The state to use for the conversion.

        :return: An instance of this class created from the obj.
        """
        state = state or ToDAOState()
        if id(obj) in state.memo:
            return state.memo[id(obj)]
        elif isinstance(obj, cls):
            return obj
        else:
            result = cls.create_instance(obj)
            return result

    @classmethod
    @abc.abstractmethod
    def create_instance(cls, obj: T) -> Self:
        """
        Create a DAO from the obj.
        The method needs to be overloaded by the user.

        :param obj: The obj to create the DAO from.
        :return: An instance of this class created from the obj.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_from_dao(self) -> T:
        """
        Creates an object from a Data Access Object (DAO) by using the predefined
        logic and transformations specific to the implementation. This facilitates
        constructing domain-specific objects from underlying data representations.

        :return: The object created from the DAO.
        :rtype: T
        """
        raise NotImplementedError


@lru_cache(maxsize=None)
def get_dao_class(cls: Type) -> Optional[Type[DataAccessObject]]:
    if get_alternative_mapping(cls) is not None:
        cls = get_alternative_mapping(cls)
    for dao in recursive_subclasses(DataAccessObject):
        if dao.original_class() == cls:
            return dao
    return None


@lru_cache(maxsize=None)
def get_alternative_mapping(cls: Type) -> Optional[Type[DataAccessObject]]:
    for alt_mapping in recursive_subclasses(AlternativeMapping):
        if alt_mapping.original_class() == cls:
            return alt_mapping
    return None


def to_dao(obj: Any, state: Optional[ToDAOState] = None) -> DataAccessObject:
    """
    Convert any object to a dao class.

    :param obj: The object to convert to a dao.
    :param state: The state to use for the conversion.
    :return: The DAO version of `obj`.
    """
    dao_class = get_dao_class(type(obj))
    if dao_class is None:
        raise NoDAOFoundError(obj)
    state = state or ToDAOState()
    return dao_class.to_dao(obj, state)
