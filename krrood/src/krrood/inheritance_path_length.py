from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from inspect import isclass
from typing import Type, Optional, TYPE_CHECKING

from krrood.entity_query_language.predicate import (
    RenderedFields,
    SymbolicFunction,
    symbolic_callable_to_function,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.fragments.base import (
        VerbalizationFragment,
    )


@dataclass(eq=False)
class InheritancePathLength(SymbolicFunction):
    """
    The inheritance path length between two classes, as a value operation.

    Every inheritance level that lies between :attr:`child_class` and
    :attr:`parent_class` increases the length by one. For multiple inheritance the
    length is computed per branch and the minimum is returned; ``None`` means no path
    exists.
    """

    child_class: Type
    """
    The child class.
    """

    parent_class: Type
    """
    The parent class.
    """

    def __call__(self) -> Optional[int]:
        if not (
            isclass(self.child_class)
            and isclass(self.parent_class)
            and issubclass(self.child_class, self.parent_class)
        ):
            return None

        return _inheritance_path_length(self.child_class, self.parent_class, 0)

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """:return: the noun phrase *"the inheritance path length between <child> and <parent>"* — a
        custom fragment because the length is BETWEEN its two operands, a relation the name-based
        *"of X and Y"* genitive cannot express."""
        # Imported locally to avoid the core -> verbalization import cycle (as Triple does).
        from krrood.entity_query_language.verbalization.vocabulary.english import (
            Prepositions,
        )
        from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
            FunctionVerbalizationTemplates,
        )

        return FunctionVerbalizationTemplates.custom_relation(
            cls, Prepositions.BETWEEN, *fields.values()
        )


inheritance_path_length = lru_cache(
    symbolic_callable_to_function(InheritancePathLength)
)


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
