from __future__ import annotations

import enum
import importlib
import inspect
import logging
import sys
import typing
from dataclasses import dataclass, Field
from datetime import datetime
from functools import lru_cache
from types import NoneType

from typing_extensions import Type, get_origin, Optional, get_type_hints, Tuple

from krrood.ripple_down_rules.utils import make_tuple


class ParseError(TypeError):
    """
    Error that will be raised when the parser encounters something that can/should not be parsed.

    For instance, Union types
    """
    pass


@dataclass(unsafe_hash=True)
class FieldInfo:
    """
    A class that wraps a field of dataclass and provides some utility functions.
    """

    clazz: Type
    """
    The class that the field is in.
    """

    name: str
    """
    The name of the field.
    """

    type: Tuple[Type, ...]
    """
    The type of the field or inner type of the container if it is a container.
    """

    optional: bool
    """
    True if the field is optional, False otherwise.
    """

    container: Optional[Type]
    """
    The type of the container if it is one (list, set, tuple, etc.). If there is no container this is None
    """

    is_type_field: bool = False
    """
    The field value is a type.
    """

    field: Field = None
    """
    The field instance.
    """

    def __init__(self, clazz: Type, f: Field):
        self.field = f
        self.name = f.name
        self.clazz = clazz

        try:
            type_hints = get_type_hints(clazz)[self.name]
        except NameError as e:
            found_clazz = manually_search_for_class_name(e.name)
            module = importlib.import_module(found_clazz.__module__)
            locals()[e.name] = getattr(module, e.name)
            type_hints = get_type_hints(clazz, localns=locals())[self.name]
        type_args = typing.get_args(type_hints)

        # try to unpack the type if it is a nested type
        if len(type_args) > 0:
            self.optional = NoneType in type_args

            if self.optional:
                self.container = None
            else:
                self.container = get_origin(type_hints)

            if not self.optional and type_hints == Type[type_args]:
                self.is_type_field = True

            self.type = type_args
        else:
            self.optional = False
            self.container = None
            self.type = make_tuple(type_hints)

    @property
    def is_builtin_class(self) -> bool:
        return not self.container and all(t.__module__ == 'builtins' for t in self.type)

    @property
    def is_container_of_builtin(self) -> bool:
        return self.container and all(t.__module__ == 'builtins' for t in self.type)

    @property
    def is_type_type(self) -> bool:
        return self.is_type_field

    @property
    def is_enum(self):
        return len(self.type) == 1 and issubclass(self.type[0], enum.Enum)

    @property
    def is_datetime(self):
        return len(self.type) == 1 and self.type[0] == datetime


def is_container(clazz: Type) -> bool:
    """
    Check if a class is an iterable.

    :param clazz: The class to check
    :return: True if the class is an iterable, False otherwise
    """
    return get_origin(clazz) in [list, set, tuple]


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
    found_classes = []

    # Search 1: In the current module's globals()
    for name, obj in globals().items():
        if inspect.isclass(obj) and obj.__name__ == target_class_name:
            found_classes.append(obj)

    # Search 2: In all loaded modules (via sys.modules)
    for module_name, module in sys.modules.items():
        if module is None or not hasattr(module, '__dict__'):
            continue  # Skip built-in modules or modules without a __dict__

        for name, obj in module.__dict__.items():
            if inspect.isclass(obj) and obj.__name__ == target_class_name:
                # Avoid duplicates if a class is imported into multiple namespaces
                if (obj, f"from module '{module_name}'") not in found_classes:
                    found_classes.append(obj)

    # If you wanted to "resolve" the forward ref based on this
    if len(found_classes) == 0:
        raise ValueError(f"Could not find any class with name {target_class_name} in globals or sys.modules.")
    elif len(found_classes) == 1:
        resolved_class = found_classes[0]
    else:
        warn_multiple_classes(target_class_name, tuple(found_classes))
        resolved_class = found_classes[0]

    return resolved_class


@lru_cache(maxsize=None)
def warn_multiple_classes(target_class_name, found_classes):
    logging.warning(f"Found multiple classes with name {target_class_name}. Found classes: {found_classes} ")
