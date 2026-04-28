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
    List,
)

from krrood.class_diagrams.utils import (
    get_and_resolve_generic_type_hints_of_object_using_substitutions,
)
from krrood.utils import (
    get_generic_type_param,
    T,
)

if TYPE_CHECKING:
    pass


@dataclass
class AbstractSubClassSafeGeneric(ABC):
    """
    Base implementation that automatically updates field types when a subclass binds the generic
    type parameters of its generic base to concrete types.

    Concrete subclasses must declare the generic parameters via ``Generic[...]`` and inherit from
    this class. The class declaring ``Generic[...]`` (the direct subclass of
    :class:`AbstractSubClassSafeGeneric`) is treated as the "generic base" against which type
    parameters are resolved.
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
class SubClassSafeGeneric(Generic[T], AbstractSubClassSafeGeneric, ABC):
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
