from __future__ import annotations

from uuid import UUID

from typing_extensions import Any

from krrood.patterns.interface import Interface
from krrood.patterns.mixin import Mixin

# %% what an object is told apart and called by


class HasID(Mixin, Interface):
    """
    Gives an object an identifier, which tells it apart from every other object, so that
    others can refer to it by the identifier instead of holding a copy of it.
    """

    id: UUID
    """
    The identifier that tells this object apart from every other object.
    """


class HasName(Mixin, Interface):
    """
    Gives an object a name that people call it by, which need not tell it apart from
    other objects.
    """

    name: Any
    """
    What the object is called.
    """
