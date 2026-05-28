from __future__ import annotations

from abc import ABC
from copy import copy
from dataclasses import dataclass, fields, Field, field
from functools import lru_cache
from inspect import isclass
from typing import Union, Tuple, Hashable

from typing_extensions import (
    Generic,
    TypeVar,
    Type,
    TYPE_CHECKING,
    Optional,
    Dict,
    Any,
    List,
    get_origin,
    get_args,
)

from krrood.class_diagrams.utils import (
    get_and_resolve_generic_type_hints_of_object_using_substitutions,
)
from krrood.entity_query_language.utils import ensure_hashable
from krrood.utils import (
    get_generic_type_params,
    T,
    resolve_union_type,
)

if TYPE_CHECKING:
    pass


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
        # if cls.__name__ != "HSRBGripper":
        #     return
        # if cls.__name__ != "SubClassGenericThatUpdatesGenericTypeToBuiltInType":
        #     return

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
        existing_field = None
        for base in cls.__mro__:
            if hasattr(base, "__dataclass_fields__"):
                if name in base.__dataclass_fields__:
                    existing_field = base.__dataclass_fields__[name]
                    break
        if hasattr(cls, name):
            # First check if there's a new created field that is yet to be processed
            attribute_value = getattr(cls, name)
            if isinstance(attribute_value, Field):
                new_attribute_value = copy(attribute_value)
                for key, value in kwargs.items():
                    setattr(new_attribute_value, key, value)
            else:
                non_type_kwargs = copy(kwargs)
                non_type_kwargs.pop("type", None)
                if non_type_kwargs:
                    setattr(cls, name, field(**non_type_kwargs))
        else:
            if existing_field is not None:
                new_field = copy(existing_field)
                for key, value in kwargs.items():
                    setattr(new_field, key, value)
                setattr(cls, name, new_field)
            else:
                field_kwargs = copy(kwargs)
                field_kwargs.pop("type", None)
                if field_kwargs:
                    setattr(cls, name, field(**field_kwargs))
        if "type" in kwargs:
            cls.__annotations__[name] = kwargs["type"]
        elif type_ is not None:
            cls.__annotations__[name] = type_
        elif existing_field is not None:
            cls.__annotations__[name] = existing_field.type
        else:
            cls.__annotations__[name] = Any

    @classmethod
    def _get_unique_generic_bases(cls) -> List[Type[AbstractSubClassSafeGeneric]]:
        """
        :return: The unique generic bases of this class, excluding itself, AbstractSubClassSafeGeneric and object,
        and excluding any base that is a superclass of another base or not a subclass of AbstractSubClassSafeGeneric.
        """
        unique_bases = []
        for base in cls.__mro__:
            if base in (cls, AbstractSubClassSafeGeneric, object):
                continue
            if not issubclass(base, AbstractSubClassSafeGeneric):
                continue
            if any(issubclass(other_base, base) for other_base in unique_bases):
                continue
            unique_bases.append(base)
        return unique_bases

    @classmethod
    def _get_generic_type_substitutions(cls) -> Dict[Type, Type]:
        """
        :return: A mapping from each old generic type (as declared on the parent class) to the
            new generic type used by this class, for every position whose binding changed.
        """
        unique_generic_bases = cls._get_unique_generic_bases()
        if not unique_generic_bases:
            return {}
        generic_base_to_type_map = {}
        for generic_base in unique_generic_bases:
            generic_base_types = generic_base.get_generic_types(True, False)
            if not generic_base_types:
                generic_base_types = generic_base.get_generic_types(False, True)

            if not generic_base_types:
                continue

            generic_base_to_generic_type_map = (
                cls._get_origin_base_to_generic_types_map()
            )
            generic_base_to_type_map.update(
                cls.get_superclass_generic_type_substitution()
            )
            # if not len(root_generic_base_types) == len(
            #     generic_base_to_generic_type_map[ensure_hashable(generic_base)]
            # ):
            #     raise ValueError(f"Lengths differ")
            for old_type, new_type in zip(
                generic_base_types,
                generic_base_to_generic_type_map[ensure_hashable(generic_base)],
            ):
                if old_type is new_type or new_type is None:
                    continue
                generic_base_to_type_map[old_type] = new_type

        # first resolve all generic types, then afterwards resolve union types
        for base_type, resolved_type in generic_base_to_type_map.items():
            if get_origin(resolved_type) is not Union:
                continue
            generic_base_to_type_map[base_type] = resolve_union_type(
                resolved_type, generic_base_to_type_map
            )

        return generic_base_to_type_map

    @classmethod
    def _get_origin_base_to_generic_types_map(cls) -> Dict[Hashable, Tuple[Type, ...]]:
        """
        :return: A mapping from each generic base of this class to the generic types it uses, for every generic base
        that is a subclass of AbstractSubClassSafeGeneric or is declared with generic type parameters via Generic[...].
        """
        origin_base_to_generic_type: Dict[Hashable, Tuple[Type, ...]] = {}
        for base in getattr(cls, "__orig_bases__", []):
            base_origin = get_origin(base)
            if base_origin is None:
                continue
            if base_origin is not Generic and not issubclass(
                base_origin, AbstractSubClassSafeGeneric
            ):
                continue
            origin_base_to_generic_type[ensure_hashable(base_origin)] = get_args(base)
        return origin_base_to_generic_type

    @classmethod
    def get_generic_types(
        cls,
        from_root_generic_base: bool = True,
        from_specialized_generic_base: bool = True,
    ) -> List[Type]:
        """
        :return: The concrete generic type parameters bound for this class, in declaration order.
        """
        return get_generic_type_params(
            cls,
            AbstractSubClassSafeGeneric,
            from_root_generic_base,
            from_specialized_generic_base,
        )

    @classmethod
    def get_superclass_generic_type_substitution(cls) -> Dict[Type, Type]:

        substitutions = {}

        for base in getattr(cls, "__orig_bases__", []):
            if base is AbstractSubClassSafeGeneric:
                continue
            if isclass(base) and not issubclass(base, AbstractSubClassSafeGeneric):
                continue
            base_origin = get_origin(base)
            if not base_origin or not issubclass(
                base_origin, AbstractSubClassSafeGeneric
            ):
                continue

            specialized_base_types = base_origin.get_generic_types(False, True)
            root_base_types = base_origin.get_generic_types(True, False)
            resolved_types = get_args(base)

            for old_type, new_type in zip(
                base_origin._get_unique_generic_bases(), specialized_base_types
            ):
                if old_type is new_type or new_type is None:
                    continue

                substitutions[
                    ensure_hashable(old_type.get_generic_types(True, False)[0])
                ] = new_type
            for old_type, new_type in zip(root_base_types, resolved_types):
                if old_type is new_type or new_type is None:
                    continue
                substitutions[ensure_hashable(old_type)] = new_type
        return substitutions


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
        generic_types = get_generic_type_params(cls, SubClassSafeGeneric)
        if generic_types:
            return generic_types[0]
        return None
