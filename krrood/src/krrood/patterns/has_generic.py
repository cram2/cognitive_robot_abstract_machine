"""
A base class for classes that carry a generic type argument.
"""

from __future__ import annotations

from functools import lru_cache
from typing import _GenericAlias

from typing_extensions import Generic, get_args, get_origin

from krrood.patterns.exceptions import NoGenericError
from krrood.utils import T


class HasGeneric(Generic[T]):
    """
    Base class for classes that carry a generic type argument.
    """

    @classmethod
    @lru_cache(maxsize=None)
    def original_class(cls) -> T:
        """
        Get the concrete generic argument.

        :return: The generic type argument.
        :raises NoGenericError: If no generic argument is found.
        """
        tp = cls._generic_argument()
        if tp is None:
            raise NoGenericError(clazz=cls)
        return tp

    @classmethod
    @lru_cache(maxsize=None)
    def constructable_original_class(cls) -> T:
        """
        Return the constructable original class.

        Use this for object allocation in from_dao cycles, as Generic Aliases cannot be
        constructed directly.
        """
        original_class = cls.original_class()
        if type(original_class) is _GenericAlias:
            return get_origin(original_class)
        else:
            return original_class

    @classmethod
    def _generic_argument(cls) -> type | None:
        """
        Extract the generic argument from the class hierarchy.

        :return: The generic type or None.
        """
        # filter for instances of generic aliases in the superclasses
        for base in filter(
            lambda x: isinstance(x, _GenericAlias),
            cls.__orig_bases__,
        ):
            return get_args(base)[0]

        # No acceptable base found
        return None
