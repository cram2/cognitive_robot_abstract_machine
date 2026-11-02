"""
The kinds of collections krrood recognises, for values and for field annotations alike.
"""

from __future__ import annotations

from enum import Enum

from sortedcontainers import SortedSet

# %% collection kinds


class CollectionKind(Enum):
    """
    A kind of collection, each member holding the classes of that kind.

    The classes serve both :func:`isinstance` on values and the origin of a generic
    annotation, since ``get_origin(list[int])`` is :class:`list`.
    """

    ELEMENT_SEQUENCE = (list, tuple, set, frozenset, SortedSet)
    """
    Unkeyed collections of elements, each rebuilt from its elements by calling its class
    on them.
    """

    MAPPING = (dict,)
    """
    Collections of key-value pairs.
    """

    @classmethod
    def all_classes(cls) -> tuple[type, ...]:
        """
        :return: The classes of every kind of collection.
        """
        return tuple(
            collection_class
            for collection_kind in cls
            for collection_class in collection_kind.value
        )
