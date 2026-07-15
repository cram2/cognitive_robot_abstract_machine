from __future__ import annotations

from functools import lru_cache
from inspect import isclass
from typing import Type, Optional

from krrood.entity_query_language.predicate import symbolic_function


@lru_cache
@symbolic_function
def inheritance_path_length(child_class: Type, parent_class: Type) -> Optional[int]:
    """
    Calculate the inheritance path length between two classes.

    Every inheritance level that lies between `child_class` and `parent_class` increases
    the length by one. In case of multiple inheritance, the path length is calculated
    for each branch and the minimum is returned.

    :param child_class: The child class.
    :param parent_class: The parent class.
    :return: The minimum path length between `child_class` and `parent_class` or None if
        no path exists.
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
