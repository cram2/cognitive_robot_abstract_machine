from __future__ import annotations

import copy
from dataclasses import fields

from typing_extensions import Any, Self

from krrood.patterns.interface import Interface
from krrood.patterns.mixin import Mixin

# %% deep copies of dataclasses


class DataclassDeepcopyMixin(Mixin, Interface):
    """
    Mixin for dataclasses whose deep copy is constructed from deep copies of their
    constructor fields.

    Fields the constructor does not take are derived by the constructor of the copy, and
    an object the copied fields reach more than once is copied once.

    .. warning::
        A dataclass whose constructor fields reach back to itself cannot be copied, as
        the copy only exists once its constructor has run.
    """

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        # a field known only by its name can only be read with getattr
        return self.__class__(
            **{
                field_.name: copy.deepcopy(getattr(self, field_.name), memo)
                for field_ in fields(self)
                if field_.init
            }
        )
