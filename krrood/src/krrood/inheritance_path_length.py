from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from inspect import isclass
from typing import Type, Optional

from krrood.entity_query_language.predicate import (
    NameVerbalized,
    SymbolicFunction,
    functional_form,
)


@dataclass(eq=False)
class InheritancePathLength(NameVerbalized, SymbolicFunction):
    """The inheritance path length between two classes, as a value operation.

    Every inheritance level that lies between :attr:`child_class` and :attr:`parent_class` increases
    the length by one. For multiple inheritance the length is computed per branch and the minimum is
    returned; ``None`` means no path exists.
    """

    child_class: Type
    """The child class."""

    parent_class: Type
    """The parent class."""

    def __call__(self) -> Optional[int]:
        if not (
            isclass(self.child_class)
            and isclass(self.parent_class)
            and issubclass(self.child_class, self.parent_class)
        ):
            return None

        return _inheritance_path_length(self.child_class, self.parent_class, 0)


inheritance_path_length = lru_cache(functional_form(InheritancePathLength))


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
