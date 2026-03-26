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
    Optional, Dict, Any,
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
        resolution_results = (
            get_and_resolve_generic_type_hints_of_object_using_substitutions(
                cls, {old_generic_type: cls.get_generic_type()}
            )
        )
        for name, result in resolution_results.items():
            if not result.resolved:
                continue
            cls._update_generic_field_with_resolved_type(name, result.resolved_type)

    @classmethod
    def _update_generic_field_with_resolved_type(cls, name: str, resolved_type: Type | TypeVar):
        """
        Update the field type with the resolved type.

        :param name: The name of the field.
        :param resolved_type: The resolved type that replaces the generic type.
        """
        cls._update_field_kwargs(name, {'type': resolved_type})

    @classmethod
    def _update_field_kwargs(cls, name: str, kwargs: Dict[str, Any], type_: Optional[Type] = None):
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
                setattr(cls, name, field(**kwargs))
        else:
            # If not, check if there's an existing field that needs to be updated
            field_ = copy(next((f for f in fields(cls) if f.name == name), None))
            if field_ is not None:
                for key, value in kwargs.items():
                    setattr(field_, key, value)
                setattr(cls, field_.name, field_)
            else:
                setattr(cls, name, field(**kwargs))
        if 'type' in kwargs:
            cls.__annotations__[name] = kwargs['type']
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
        for base in cls.__bases__:
            if not issubclass(base, SubClassSafeGeneric):
                continue
            base_generic_type = base.get_generic_type()
            if base_generic_type is None:
                continue
            if base_generic_type is not current_generic_type:
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
