from __future__ import annotations

import ast
import inspect
import os
import types
from ast import Module
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from inspect import isclass
from typing import Union, Type

from typing_extensions import TypeVar, Type, List, Optional, _SpecialForm

T = TypeVar("T")


def recursive_subclasses(cls: Type[T]) -> List[Type[T]]:
    """
    :param cls: The class.
    :return: A list of the classes subclasses without the class itself.
    """
    return cls.__subclasses__() + [
        g for s in cls.__subclasses__() for g in recursive_subclasses(s)
    ]


@dataclass
class DataclassException(Exception):
    """
    A base exception class for dataclass-based exceptions.
    The way this is used is by inheriting from it and setting the `message` field in the __post_init__ method,
    then calling the super().__post_init__() method.
    """

    message: str = field(kw_only=True, default=None)

    def __post_init__(self):
        super().__init__(self.message)


def get_full_class_name(cls):
    """
    Returns the full name of a class, including the module name.

    :param cls: The class.
    :return: The full name of the class
    """
    return cls.__module__ + "." + cls.__name__


@lru_cache
def inheritance_path_length(child_class: Type, parent_class: Type) -> Optional[int]:
    """
    Calculate the inheritance path length between two classes.
    Every inheritance level that lies between `child_class` and `parent_class` increases the length by one.
    In case of multiple inheritance, the path length is calculated for each branch and the minimum is returned.

    :param child_class: The child class.
    :param parent_class: The parent class.
    :return: The minimum path length between `child_class` and `parent_class` or None if no path exists.
    """
    if not (
        isclass(child_class)
        and isclass(parent_class)
        and issubclass(child_class, parent_class)
    ):
        return None

    return _inheritance_path_length(child_class, parent_class, 0)


def _inheritance_path_length(
    child_class: Type, parent_class: Type, current_length: int = 0
) -> int:
    """
    Helper function for :func:`inheritance_path_length`.

    :param child_class: The child class.
    :param parent_class: The parent class.
    :param current_length: The current length of the inheritance path.
    :return: The minimum path length between `child_class` and `parent_class`.
    """

    if child_class == parent_class:
        return current_length
    else:
        return min(
            _inheritance_path_length(base, parent_class, current_length + 1)
            for base in child_class.__bases__
            if issubclass(base, parent_class)
        )


def module_and_class_name(t: Union[Type, _SpecialForm]) -> str:
    return f"{t.__module__}.{t.__name__}"


def extract_imports(
    module: types.ModuleType, exclude_libraries: Optional[List[str]] = None
) -> List[str]:
    """Extract imports from a module source, handling multiline imports, and merging duplicates."""
    source = inspect.getsource(module)
    tree = ast.parse(source)

    import_modules = set()
    from_imports = defaultdict(set)

    for node in ast.walk(tree):

        # import x, y
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in exclude_libraries:
                    continue
                if alias.asname:
                    import_modules.add(f"{alias.name} as {alias.asname}")
                else:
                    import_modules.add(alias.name)

        # from x import y
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module in exclude_libraries:
                continue
            for alias in node.names:
                if alias.asname:
                    from_imports[node.module].add(f"{alias.name} as {alias.asname}")
                else:
                    from_imports[node.module].add(alias.name)

    result = set()

    for module in import_modules:
        result.add(f"import {module}")

    for module, names in from_imports.items():
        joined = ", ".join(sorted(names))
        result.add(f"from {module} import {joined}")

    return sorted(result)
