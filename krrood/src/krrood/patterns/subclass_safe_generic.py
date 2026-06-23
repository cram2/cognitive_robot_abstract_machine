from __future__ import annotations

from abc import ABC
from copy import copy
from dataclasses import dataclass, Field
from functools import lru_cache
from inspect import isclass
from typing import Tuple

from typing_extensions import (
    Generic,
    TypeVar,
    Type,
    Optional,
    Dict,
    Any,
    get_origin,
    get_args,
    TypeAlias,
    TypeVarTuple,
    Unpack,
)

from krrood import logger
from krrood.adapters.json_serializer import list_like_classes
from krrood.class_diagrams.exceptions import CouldNotResolveType
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

TRANSIENT_TYPE_RESOLUTION_ERRORS = (
    ImportError,
    NameError,
    TypeError,
    CouldNotResolveType,
)
"""Exceptions raised while importing and resolving a field's type annotation when that type is
not yet available (typically a circular import during module load). They are transient: the field
is narrowed later, once a subclass binds the parameter to a concrete type. Any other exception is
a genuine fault and is left to propagate."""


ResolvableType: TypeAlias = (
    type
    | TypeVar
    | TypeVarTuple
    | list["ResolvableType"]
    | tuple["ResolvableType", ...]
    | Any
)


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
        try:
            resolution_results = (
                get_and_resolve_generic_type_hints_of_object_using_substitutions(
                    cls, substitutions
                )
            )
        except TRANSIENT_TYPE_RESOLUTION_ERRORS as error:
            # A field's type could not be imported/resolved yet (typically a circular import during
            # module load). The class must still be definable; such fields are narrowed later, when a
            # subclass binds the parameter to a concrete type. Only warn when a concrete binding was
            # actually lost; a failed pure type-variable rename is a no-op, logged at debug to avoid
            # noise.
            log = (
                logger.warning
                if cls._concrete_binding_was_lost(substitutions)
                else logger.debug
            )
            log(
                f"SubClassSafeGeneric: could not resolve type hints for {cls} — field "
                f"types will not be updated. Cause: {error}"
            )
            return
        for name, result in resolution_results.items():
            if not result.resolved:
                continue
            cls._update_field_kwargs(name, {"type": result.resolved_type})

    @staticmethod
    def _concrete_binding_was_lost(
        substitutions: Dict[Any, ResolvableType],
    ) -> bool:
        """
        Decide whether a failed substitution dropped information worth surfacing.

        :param substitutions: The substitution map that could not be applied.
        :return: True if any substitution maps to a concrete type, False when every substitution
            only renames a type variable (a no-op not worth warning about).
        """
        return any(
            not isinstance(value, (TypeVar, TypeVarTuple))
            for value in substitutions.values()
        )

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
                substitutions.update(
                    cls._get_resolved_type_substitutions(base_origin, resolved_types)
                )

            # Recursively pull substitutions already defined by the parent
            if base_origin is cls:
                continue
            substitutions.update(base_origin._get_generic_type_substitutions())

        if substitutions:
            substitutions = cls._resolve_substitutions_transitively(substitutions)

        cls._subclass_safe_substitutions = substitutions
        return substitutions

    @classmethod
    def _get_resolved_type_substitutions(
        cls,
        base_origin: type,
        resolved_types: tuple[type, ...],
    ) -> dict[Any, ResolvableType]:
        """
        Retrieves resolved type substitutions for a given base origin and resolved types.

        :param base_origin: The base origin type for which substitutions are retrieved.
        :param resolved_types: The resolved types to match against the base origin's generic parameters.
        :return: A dictionary of resolved type substitutions.
        """
        root_parameters = get_generic_type_params(
            base_origin,
            AbstractSubClassSafeGeneric,
            include_root_generic_base=True,
            include_specialized_generic_base=False,
        )

        type_var_tuple_index = next(
            (i for i, p in enumerate(root_parameters) if isinstance(p, TypeVarTuple)),
            -1,
        )

        if type_var_tuple_index == -1:
            if len(root_parameters) != len(resolved_types):
                raise MismatchingNumberOfGenericParametersAndResolvedTypes(
                    affected_class=base_origin,
                    parameters=root_parameters,
                    resolved_types=resolved_types,
                )
            matched_pairs = zip(root_parameters, resolved_types)
        else:
            matched_pairs = cls._match_type_var_tuple_variable_to_resolved_type(
                type_var_tuple_index,
                root_parameters,
                resolved_types,
                base_origin,
            )

        return {
            ensure_hashable(old_type): new_type
            for old_type, new_type in matched_pairs
            if cls._fulfills_substitution_condition(old_type, new_type)
        }

    @classmethod
    def _fulfills_substitution_condition(
        cls,
        old_type: Any,
        new_type: type | tuple[type, ...] | None,
    ) -> bool:
        """
        Determines if a substitution condition is fulfilled based on the old and new types.

        :param old_type: The old type to compare against.
        :param new_type: The new type to compare with.
        :return: True if the substitution condition is fulfilled, False otherwise.
        """
        if not isinstance(old_type, (TypeVar, TypeVarTuple)):
            return False

        if old_type is new_type or new_type is None:
            return False

        if not (
            isinstance(old_type, TypeVarTuple)
            and isinstance(new_type, tuple)
            and len(new_type) == 1
        ):
            return True

        inner = new_type[0]
        return not (get_origin(inner) is Unpack and get_args(inner)[0] is old_type)

    @classmethod
    def _match_type_var_tuple_variable_to_resolved_type(
        cls,
        type_var_tuple_index: int,
        root_parameters: list[Any],
        resolved_types: tuple[type, ...],
        base_origin: type,
    ) -> list[tuple[type, type]]:
        """
        Matches type variables in a tuple with resolved types, considering prefix and suffix lengths.

        :param type_var_tuple_index: Index of the TypeVarTuple within root_parameters.
        :param root_parameters: List of root parameters to match against resolved types.
        :param resolved_types: Tuple of resolved types to match with root parameters.
        :param base_origin: Base origin type for substitutions.
        :return: List of matched type variable tuples.
        """
        prefix_length = type_var_tuple_index
        suffix_length = len(root_parameters) - type_var_tuple_index - 1
        if len(resolved_types) < prefix_length + suffix_length:
            raise MismatchingNumberOfGenericParametersAndResolvedTypes(
                affected_class=base_origin,
                parameters=root_parameters,
                resolved_types=resolved_types,
            )
        type_var_tuple_content_length = (
            len(resolved_types) - prefix_length - suffix_length
        )
        matched_pairs = [
            (root_parameters[index], resolved_types[index])
            for index in range(prefix_length)
        ]
        matched_pairs.append(
            (
                root_parameters[type_var_tuple_index],
                tuple(
                    resolved_types[
                        prefix_length : prefix_length + type_var_tuple_content_length
                    ]
                ),
            )
        )
        for index in range(suffix_length):
            matched_pairs.append(
                (
                    root_parameters[type_var_tuple_index + 1 + index],
                    resolved_types[
                        prefix_length + type_var_tuple_content_length + index
                    ],
                )
            )
        return matched_pairs

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
            if isinstance(current_type, (TypeVar, TypeVarTuple)):
                type_key = ensure_hashable(current_type)
                if type_key in visited:
                    return current_type

                if type_key in substitutions:
                    return _resolve_recursive(
                        substitutions[type_key], visited | {type_key}
                    )
                return current_type

            if isinstance(current_type, list_like_classes):
                resolved_items = []
                for item in current_type:
                    result = _resolve_recursive(item, visited)
                    if get_origin(item) is Unpack and isinstance(
                        result, list_like_classes
                    ):
                        resolved_items.extend(result)
                    else:
                        resolved_items.append(result)
                return type(current_type)(resolved_items)

            origin = get_origin(current_type)
            if origin is None:
                return current_type

            if origin is Unpack:
                inner_arg = get_args(current_type)[0]
                resolved_inner = _resolve_recursive(inner_arg, visited)
                if isinstance(resolved_inner, tuple):
                    return resolved_inner
                return Unpack[resolved_inner]

            args = get_args(current_type)
            resolved_args = []
            for arg in args:
                result = _resolve_recursive(arg, visited)
                if get_origin(arg) is Unpack and isinstance(result, tuple):
                    resolved_args.extend(result)
                else:
                    resolved_args.append(result)
            resolved_args = tuple(resolved_args)

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
class SubClassSafeGeneric(Generic[T], AbstractSubClassSafeGeneric):
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

    The field-type updating is performed by :class:`AbstractSubClassSafeGeneric`, which
    resolves every generic parameter (multiple ``TypeVar`` s and ``TypeVarTuple`` s) across
    the whole inheritance chain, rather than only the first generic parameter.
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
