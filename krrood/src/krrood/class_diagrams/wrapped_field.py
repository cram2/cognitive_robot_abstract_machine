from __future__ import annotations

import enum
import inspect
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, Field, MISSING
from datetime import datetime
from functools import cached_property, lru_cache
from types import NoneType
from copy import copy

from typing_extensions import (
    get_type_hints,
    get_origin,
    get_args,
    ClassVar,
    List,
    Type,
    TYPE_CHECKING,
    Optional,
    Union,
)

from .failures import MissingContainedTypeOfContainer
from .utils import behaves_like_a_built_in_class
from ..ormatic.utils import module_and_class_name

if TYPE_CHECKING:
    from .class_diagram import WrappedClass
    from ..ontomatic.property_descriptor import PropertyDescriptor


@dataclass
class TypeResolutionError(TypeError):
    """
    Error raised when a type cannot be resolved, even if searched for manually.
    """

    name: str

    def __post_init__(self):
        super().__init__(f"Could not resolve type for {self.name}")


@dataclass
class WrappedField:
    """
    A class that wraps a field of dataclass and provides some utility functions.
    """

    clazz: WrappedClass
    """
    The wrapped class that the field was created from.
    """

    field: Field
    """
    The dataclass field object that is wrapped.
    """

    public_name: Optional[str] = None
    """
    If the field is a relationship managed field, this is public name of the relationship that manages the field.
    """

    property_descriptor: Optional[PropertyDescriptor] = None
    """
    The property descriptor instance that manages the field.
    """

    container_types: ClassVar[List[Type]] = [list, set, tuple, type, Sequence]
    """
    A list of container types that are supported by the parser.
    """

    def __post_init__(self):
        self.public_name = self.public_name or self.field.name

    @cached_property
    def name(self):
        return self.public_name

    def __hash__(self):
        return hash((self.clazz.clazz, self.field))

    def __eq__(self, other):
        return (self.clazz.clazz, self.field) == (
            other.clazz.clazz,
            other.field,
        )

    def __repr__(self):
        return f"{module_and_class_name(self.clazz.clazz)}.{self.field.name}"

    @cached_property
    def resolved_type(self):
        """
        Resolve the type hint for this field.

        Handles forward references by iteratively building a namespace with
        classes from the class diagram and sys.modules until all references
        are resolved.
        """
        local_namespace = self._build_initial_namespace()

        while True:
            try:
                return get_type_hints(self.clazz.clazz, localns=local_namespace)[
                    self.field.name
                ]
            except NameError as e:
                found_class = self._find_class_by_name(e.name)
                local_namespace[e.name] = found_class

    def _build_initial_namespace(self) -> dict:
        """
        Build the initial namespace for type resolution from the class diagram.
        """
        class_diagram = self.clazz._class_diagram
        if class_diagram is None:
            return {}
        return {cls.clazz.__name__: cls.clazz for cls in class_diagram.wrapped_classes}

    def _find_class_by_name(self, class_name: str) -> Type:
        """
        Find a class by name in the class diagram or sys.modules.
        """
        class_diagram = self.clazz._class_diagram
        if class_diagram is not None:
            for wrapped_class in class_diagram.wrapped_classes:
                if wrapped_class.clazz.__name__ == class_name:
                    return wrapped_class.clazz
        return manually_search_for_class_name(class_name)

    @cached_property
    def is_builtin_type(self) -> bool:
        return self.type_endpoint in [int, float, str, bool, datetime, NoneType]

    @cached_property
    def is_container(self) -> bool:
        return get_origin(self.resolved_type) in self.container_types

    @cached_property
    def container_type(self) -> Optional[Type]:
        if not self.is_container:
            return None
        return get_origin(self.resolved_type)

    @cached_property
    def is_collection_of_builtins(self):
        return self.is_container and all(
            behaves_like_a_built_in_class(field_type)
            for field_type in get_args(self.resolved_type)
        )

    @cached_property
    def is_optional(self):
        origin = get_origin(self.resolved_type)
        if origin not in [Union, Optional]:
            return False
        if origin == Union:
            args = get_args(self.resolved_type)
            return len(args) == 2 and NoneType in args
        return True

    @cached_property
    def contained_type(self):
        if not self.is_container and not self.is_optional:
            raise ValueError("Field is not a container")
        if self.is_optional:
            return get_args(self.resolved_type)[0]
        else:
            try:
                return get_args(self.resolved_type)[0]
            except IndexError:
                if self.resolved_type is Type:
                    return self.resolved_type
                else:
                    raise MissingContainedTypeOfContainer(
                        self.clazz.clazz, self.name, self.container_type
                    )

    @cached_property
    def is_type_type(self) -> bool:
        return get_origin(self.resolved_type) is type

    @cached_property
    def is_enum(self) -> bool:
        if self.is_container or not inspect.isclass(self.resolved_type):
            return False
        if self.is_optional:
            return issubclass(self.contained_type, enum.Enum)

        return issubclass(self.resolved_type, enum.Enum)

    @cached_property
    def is_one_to_one_relationship(self) -> bool:
        return not self.is_container and not self.is_builtin_type

    @cached_property
    def is_one_to_many_relationship(self) -> bool:
        return self.is_container and not self.is_builtin_type and not self.is_optional

    @cached_property
    def is_iterable(self):
        return self.is_one_to_many_relationship and hasattr(
            self.container_type, "__iter__"
        )

    @cached_property
    def type_endpoint(self) -> Type:
        if self.is_container or self.is_optional:
            return self.contained_type
        else:
            return self.resolved_type

    @cached_property
    def is_role_taker(self) -> bool:
        return (
            self.is_one_to_one_relationship
            and not self.is_optional
            and self.field.default == MISSING
            and self.field.default_factory == MISSING
        )


@lru_cache(maxsize=None)
def manually_search_for_class_name(target_class_name: str) -> Type:
    """
    Searches for a class with the specified name in the current module's `globals()` dictionary
    and all loaded modules present in `sys.modules`. This function attempts to find and resolve
    the first class that matches the given name. If multiple classes are found with the same
    name, a warning is logged, and the first one is returned. If no matching class is found,
    an exception is raised.

    :param target_class_name: Name of the class to search for.
    :return: The resolved class with the matching name.

    :raises ValueError: Raised when no class with the specified name can be found.
    """
    found_classes = search_class_in_globals(target_class_name)
    found_classes += search_class_in_sys_modules(target_class_name)

    if len(found_classes) == 0:
        raise TypeResolutionError(target_class_name)
    elif len(found_classes) == 1:
        resolved_class = found_classes[0]
    else:
        logging.warning(
            f"Found multiple classes with name {target_class_name}. Found classes: {found_classes} "
        )
        resolved_class = found_classes[0]

    return resolved_class


def search_class_in_globals(target_class_name: str) -> List[Type]:
    """
    Searches for a class with the given name in the current module's globals.

    :param target_class_name: The name of the class to search for.
    :return: The resolved classes with the matching name.
    """
    return [
        value
        for name, value in globals().items()
        if inspect.isclass(value) and value.__name__ == target_class_name
    ]


def search_class_in_sys_modules(target_class_name: str) -> List[Type]:
    """
    Searches for a class with the given name in all loaded modules (via sys.modules).
    """
    found_classes = []
    for module_name, module in copy(sys.modules).items():
        if module is None or not hasattr(module, "__dict__"):
            continue  # Skip built-in modules or modules without a __dict__

        for name, obj in module.__dict__.items():
            if inspect.isclass(obj) and obj.__name__ == target_class_name:
                # Avoid duplicates if a class is imported into multiple namespaces
                if obj not in found_classes:
                    found_classes.append(obj)
    return found_classes
