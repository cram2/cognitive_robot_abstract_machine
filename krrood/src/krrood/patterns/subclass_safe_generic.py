from __future__ import annotations

from abc import ABC
from copy import copy
from dataclasses import dataclass, fields, Field, field
from functools import lru_cache

from typing_extensions import (
    Generic,
    TypeVar,
    Type,
    TYPE_CHECKING,
    Optional,
    Dict,
    Any,
    get_origin,
    get_args,
    List
)

from krrood import logger
from krrood.class_diagrams.utils import (
    get_and_resolve_generic_type_hints_of_object_using_substitutions,
)
from krrood.utils import (
    get_generic_type_param,
    T,
)

if TYPE_CHECKING:
    pass


def _is_strictly_more_specific_bound(current: TypeVar, base: TypeVar) -> bool:
    """Return True if current's bound is strictly narrower than base's bound."""
    current_bound = getattr(current, "__bound__", None)
    base_bound = getattr(base, "__bound__", None)
    if current_bound is None:
        return False
    if base_bound is None:
        return True
    try:
        return issubclass(current_bound, base_bound) and current_bound is not base_bound
    except TypeError:
        # base_bound is a subscripted generic (e.g. Callable[..., Any]) — treat current as more specific.
        return True


@dataclass
class AbstractSubClassSafeGeneric(ABC):
    """
    Base implementation that automatically updates field types when a subclass binds the generic
    type parameters of its generic base to concrete types.

    Concrete subclasses must declare the generic parameters via ``Generic[...]`` and inherit from
    this class. Here it is important that in the inheritance order, ``Generic[...]`` is positioned before
    ``AbstractSubClassSafeGeneric`` similar to how it is done in ``SubClassSafeGeneric``.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Automatically updates the field types that use the generic type parameters with the new
        specified types, before the class is initialized.
        """
        substitutions = cls._get_generic_type_substitutions()
        if not substitutions:
            return
        resolution_results = (
            get_and_resolve_generic_type_hints_of_object_using_substitutions(
                cls, substitutions
            )
        )
        for name, result in resolution_results.items():
            if not result.resolved:
                continue
            cls._update_field_kwargs(name, {"type": result.resolved_type})

    @classmethod
    def _update_field_kwargs(
        cls, name: str, kwargs: Dict[str, Any], type_: Optional[Type] = None
    ):
        """
        Update the field kwargs with the provided keyword arguments.

        :param name: The name of the field.
        :param kwargs: Keyword arguments to update the field with.
        """
        field_ = next((f for f in fields(cls) if f.name == name), None)
        if hasattr(cls, name):
            # First check if there's a new created field that is yet to be processed
            attribute_value = getattr(cls, name)
            if isinstance(attribute_value, Field):
                for key, value in kwargs.items():
                    setattr(attribute_value, key, value)
            else:
                non_type_kwargs = copy(kwargs)
                non_type_kwargs.pop("type", None)
                if non_type_kwargs:
                    setattr(cls, name, field(**non_type_kwargs))
        else:
            # If not, check if there's an existing field that needs to be updated
            field_ = copy(next((f for f in fields(cls) if f.name == name), None))
            if field_ is not None:
                for key, value in kwargs.items():
                    setattr(field_, key, value)
                setattr(cls, field_.name, field_)
            else:
                setattr(cls, name, field(**kwargs))
        if "type" in kwargs:
            cls.__annotations__[name] = kwargs["type"]
        elif type_ is not None:
            cls.__annotations__[name] = type_
        elif field_ is not None:
            cls.__annotations__[name] = field_.type
        else:
            cls.__annotations__[name] = Any

    @classmethod
    def _get_generic_base(cls) -> Optional[Type]:
        """
        :return: The class that declares the generic parameters for this hierarchy, i.e. the
            direct subclass of :class:`AbstractSubClassSafeGeneric` in the MRO.
        """
        for base in cls.__mro__:
            if base in (cls, AbstractSubClassSafeGeneric, object):
                continue
            if not issubclass(base, AbstractSubClassSafeGeneric):
                continue
            if AbstractSubClassSafeGeneric in base.__bases__:
                return base
        return None

    @classmethod
    def _get_generic_type_substitutions(cls) -> Dict[Any, Any]:
        """
        :return: A mapping from each old generic type (as declared on the parent class) to the
            new generic type used by this class, for every position whose binding changed.
        """
        current_types = cls.get_generic_types()
        if not current_types:
            return {}
        generic_base = cls._get_generic_base()
        if generic_base is None:
            return {}
        for base in cls.__bases__:
            if not isinstance(base, type) or not issubclass(base, generic_base):
                continue
            base_types = base.get_generic_types()
            if not base_types or len(base_types) != len(current_types):
                continue
            substitutions: Dict[Any, Any] = {}
            for old_type, new_type in zip(base_types, current_types):
                if old_type is new_type or new_type is None:
                    continue
                substitutions[old_type] = new_type
            if substitutions:
                return substitutions
        return {}

    @classmethod
    @lru_cache
    def get_generic_types(cls) -> Optional[List[Type]]:
        """
        :return: The concrete generic type parameters bound for this class, in declaration order.
        """
        generic_base = cls._get_generic_base()
        if generic_base is None:
            return None
        return get_generic_type_param(cls, generic_base)

@dataclass
class SubClassSafeGeneric(Generic[T], ABC):
    """
    A generic class that can be subclassed safely because it automatically updates the field types that use the generic
     type with the new specified type.
     Example:
         >>> T = TypeVar("T")
         >>> @dataclass
         >>> class MyClass(SubClassSafeGeneric[T]):
         >>>     my_attribute: T
         >>>
         >>> @dataclass
         >>> class MyClass2(SubClassSafeGeneric[int]): ...
         >>> assert next(f for f in fields(MyClass2) if f.name == "my_attribute").type == int)
    """

    def __init_subclass__(cls, **kwargs):
        """
        Automatically updates the field types that use the generic type with the new specified type, before the class is
        initialized.
        """
        old_generic_type = cls._get_old_generic_type_if_different()
        if not old_generic_type:
            return
        try:
            resolution_results = (
                get_and_resolve_generic_type_hints_of_object_using_substitutions(
                    cls, {old_generic_type: cls.get_generic_type()}
                )
            )
        except Exception as e:
            logger.warning(
                f"SubClassSafeGeneric: could not resolve type hints for {cls} — "
                f"field types will not be updated. Cause: {e}"
            )
            return
        for name, result in resolution_results.items():
            if not result.resolved:
                continue
            cls._update_field_kwargs(name, {"type": result.resolved_type})

    @classmethod
    def _update_field_kwargs(
            cls, name: str, kwargs: Dict[str, Any], type_: Optional[Type] = None
    ):
        """
        Update the field kwargs with the provided keyword arguments.

        :param name: The name of the field.
        :param kwargs: Keyword arguments to update the field with.
        """
        field_ = next((f for f in fields(cls) if f.name == name), None)
        if hasattr(cls, name):
            # First check if there's a new created field that is yet to be processed
            attribute_value = getattr(cls, name)
            if isinstance(attribute_value, Field):
                for key, value in kwargs.items():
                    setattr(attribute_value, key, value)
            else:
                non_type_kwargs = copy(kwargs)
                non_type_kwargs.pop("type", None)
                if non_type_kwargs:
                    setattr(cls, name, field(**non_type_kwargs))
        else:
            # If not, check if there's an existing field that needs to be updated.
            # fields(cls) reads only the nearest ancestor's __dataclass_fields__ via
            # MRO lookup; search the full MRO so we don't miss a field defined on a
            # farther ancestor (e.g. objects on HasStorageSpace when cls is Bottle).
            raw_field = next(
                (
                    ancestor.__dict__["__dataclass_fields__"][name]
                    for ancestor in cls.__mro__[1:]
                    if "__dataclass_fields__" in ancestor.__dict__
                       and name in ancestor.__dict__["__dataclass_fields__"]
                ),
                None,
            )
            field_ = copy(raw_field)
            if field_ is not None:
                for key, value in kwargs.items():
                    setattr(field_, key, value)
                setattr(cls, field_.name, field_)
            else:
                non_type_kwargs = copy(kwargs)
                non_type_kwargs.pop("type", None)
                if non_type_kwargs:
                    setattr(cls, name, field(**non_type_kwargs))
        if "type" in kwargs:
            cls.__annotations__[name] = kwargs["type"]
        elif type_ is not None:
            cls.__annotations__[name] = type_
        elif field_ is not None:
            cls.__annotations__[name] = field_.type
        else:
            cls.__annotations__[name] = Any

    @classmethod
    def _get_old_generic_type_if_different(cls) -> Optional[Type[T]]:
        """
        :return: The type of the generic type that was used in the parent class if it was changed in this class.
        """
        current_generic_type = cls.get_generic_type()
        if current_generic_type is None:
            return None
        # True when cls has SubClassSafeGeneric[X] as a direct explicit base, meaning
        # it introduces a fresh TypeVar rather than specialising an inherited one.
        cls_directly_introduces_generic = any(
            get_origin(base) is SubClassSafeGeneric
            for base in getattr(cls, "__orig_bases__", [])
        )
        for base in cls.__bases__:
            if not issubclass(base, SubClassSafeGeneric):
                continue
            base_generic_type = base.get_generic_type()
            if base_generic_type is None:
                continue
            if base_generic_type is not current_generic_type:
                if isinstance(current_generic_type, TypeVar):
                    # Skip when this class directly introduces a new generic or when the base's
                    # generic is already concrete (current TypeVar replaces a concrete type).
                    if cls_directly_introduces_generic or not isinstance(
                            base_generic_type, TypeVar
                    ):
                        continue
                    # Both are TypeVars. Only allow substitution when current has a strictly
                    # more specific bound (e.g. NewVar bound to a subclass of base's bound).
                    if not _is_strictly_more_specific_bound(
                            current_generic_type, base_generic_type
                    ):
                        continue
                return base_generic_type
        return None

    @classmethod
    @lru_cache
    def get_generic_type(cls) -> Optional[Type[T]]:
        """
        :return: The type of the role taker.
        """
        generic_types = get_generic_type_param(cls, SubClassSafeGeneric)
        if generic_types:
            return generic_types[0]
        return None
