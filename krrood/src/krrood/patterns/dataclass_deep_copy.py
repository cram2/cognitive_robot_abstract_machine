"""
Deep copies of dataclasses built through their constructors.
"""

from __future__ import annotations

import copy
from dataclasses import fields
from enum import Enum, auto

from typing_extensions import Any, Self

from krrood.patterns.interface import Interface
from krrood.patterns.mixin import Mixin

# %% deep copies of dataclasses


class DeepCopyOfField(Enum):
    """
    How a deep copy by :class:`DataclassDeepcopyMixin` treats a constructor field, given
    as the field's metadata under this enum, as in ``field(metadata={DeepCopyOfField:

    DeepCopyOfField.SHARED})``.
    """

    SHARED = auto()
    """
    The copy refers to the same object as the original instead of a copy of it.
    """


class DataclassDeepcopyMixin(Mixin, Interface):
    """
    Mixin for dataclasses whose deep copy is constructed from deep copies of their
    constructor fields, except those marked :attr:`DeepCopyOfField.SHARED`. An object
    the copied fields reach more than once is copied once.

    .. warning::
        A copied field that reaches back to the dataclass makes it impossible to copy,
        as the copy only exists once its constructor has run; mark such a field shared.
    """

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        """
        :param memo: The copies already made in this deep copy, by the id of the original.
        :return: A copy constructed from the constructor fields, copied or shared.
        """
        # a field known only by its name can only be read with getattr
        return self.__class__(
            **{
                field_.name: (
                    getattr(self, field_.name)
                    if field_.metadata.get(DeepCopyOfField) is DeepCopyOfField.SHARED
                    else copy.deepcopy(getattr(self, field_.name), memo)
                )
                for field_ in fields(self)
                if field_.init
            }
        )
