from __future__ import annotations

from abc import ABC
from copy import copy
from dataclasses import dataclass, fields, Field
from functools import lru_cache
from inspect import isclass
from typing import Tuple

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
    TypeAlias,
)

from krrood.class_diagrams.utils import (
    get_and_resolve_generic_type_hints_of_object_using_substitutions,
)
from krrood.exceptions import MismatchingNumberOfGenericParametersAndResolvedTypes
from krrood.utils import (
    get_generic_type_params,
    T,
    ensure_hashable,
    get_existing_field_by_name,
)

if TYPE_CHECKING:
    pass


ResolvableType: TypeAlias = type | TypeVar | list["ResolvableType"] | Any


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
        :param type_: The type of the field.
        """
        existing_field = get_existing_field_by_name(cls, name)

        # Check if we should update an existing attribute or field on the current class
        target_field = None
        if hasattr(cls, name):
            attr = getattr(cls, name)
            if isinstance(attr, Field):
                target_field = attr

        # If no direct field, but we found one in MRO, we might need to copy it
        if target_field is None and existing_field is not None:
            target_field = existing_field

        if target_field is not None:
            new_field = copy(target_field)
            for key, value in kwargs.items():
                setattr(new_field, key, value)
            setattr(cls, name, new_field)

        # Update annotations
        if "type" in kwargs:
            resolved_type = kwargs["type"]
        elif type_ is not None:
            resolved_type = type_
        elif existing_field is not None:
            resolved_type = existing_field.type
        else:
            resolved_type = Any
        cls.__annotations__[name] = resolved_type

    @classmethod
    def _get_generic_type_substitutions(cls) -> Dict[Any, ResolvableType]:
        """
        Get the generic type substitutions for this class.

        :return: A mapping from each old generic type (as declared on the parent class) to the
            new generic type used by this class, for every position whose binding changed.
        """
        if cls is AbstractSubClassSafeGeneric or not issubclass(
            cls, AbstractSubClassSafeGeneric
        ):
            return {}

        # Use a class-level cache to avoid redundant recursive calculations
        # Check ONLY the current class's dict to avoid using an inherited cache
        if "_subclass_safe_substitutions" in cls.__dict__:
            return cls._subclass_safe_substitutions

        substitutions = {}
        for base in getattr(cls, "__orig_bases__", []):
            base_origin, resolved_types = cls._resolve_base_origin_and_arguments(base)
            if base_origin is None or not issubclass(
                base_origin, AbstractSubClassSafeGeneric
            ):
                continue

            # Map the root TypeVars of the base to the concrete arguments provided here
            if resolved_types:
                root_parameters = get_generic_type_params(
                    base_origin,
                    AbstractSubClassSafeGeneric,
                    include_root_generic_base=True,
                    include_specialized_generic_base=False,
                )
                if len(root_parameters) != len(resolved_types):
                    raise MismatchingNumberOfGenericParametersAndResolvedTypes(
                        affected_class=base_origin,
                        parameters=root_parameters,
                        resolved_types=resolved_types,
                    )

                for old_type, new_type in zip(root_parameters, resolved_types):
                    if (
                        not isinstance(old_type, TypeVar)
                        or old_type is new_type
                        or new_type is None
                    ):
                        continue
                    substitutions[ensure_hashable(old_type)] = new_type

            # Recursively pull substitutions already defined by the parent
            if base_origin is cls:
                continue
            substitutions.update(base_origin._get_generic_type_substitutions())

        if substitutions:
            substitutions = cls._resolve_substitutions_transitively(substitutions)

        cls._subclass_safe_substitutions = substitutions
        return substitutions

    @classmethod
    def _resolve_substitutions_transitively(
        cls, substitutions: Dict[Any, ResolvableType]
    ) -> Dict[Any, ResolvableType]:
        """
        Recursively resolve TypeVars in the substitution map to their most concrete form
        using cycle detection to handle complex generic hierarchies safely.

        :param substitutions: The substitution map to resolve.
        :return: A new substitution map with fully resolved types.
        """
        resolved_substitutions = {}

        def _resolve_recursive(
            current_type: ResolvableType, visited: set[Any]
        ) -> ResolvableType:
            if isinstance(current_type, TypeVar):
                type_key = ensure_hashable(current_type)
                if type_key in visited:
                    return current_type

                if type_key in substitutions:
                    return _resolve_recursive(
                        substitutions[type_key], visited | {type_key}
                    )
                return current_type

            if isinstance(current_type, list):
                return [_resolve_recursive(item, visited) for item in current_type]

            origin = get_origin(current_type)
            if origin is None:
                return current_type

            args = get_args(current_type)
            resolved_args = tuple(_resolve_recursive(arg, visited) for arg in args)

            if resolved_args == args:
                return current_type

            return origin[resolved_args]

        for old_type, new_type in substitutions.items():
            resolved_substitutions[old_type] = _resolve_recursive(new_type, set())

        return resolved_substitutions

    @classmethod
    def _resolve_base_origin_and_arguments(
        cls, base: Type
    ) -> Tuple[Optional[Type], Tuple[Type, ...]]:
        """
        Resolve the origin and generic arguments for a base class.

        :param base: The base to resolve.
        :return: A tuple of the origin class and its generic arguments.
        """
        origin = get_origin(base)
        if origin is None:
            if isclass(base) and issubclass(base, AbstractSubClassSafeGeneric):
                return base, ()
            return None, ()

        # Ensure origin is a class before calling issubclass
        if isclass(origin) and issubclass(origin, AbstractSubClassSafeGeneric):
            return origin, get_args(base)

        return None, ()


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
        :return: The type that is currently bound to the generic type parameter T for this class, or None if T is not bound.
        """
        generic_types = get_generic_type_params(cls, SubClassSafeGeneric)
        if not generic_types:
            return None
        return generic_types[0]
