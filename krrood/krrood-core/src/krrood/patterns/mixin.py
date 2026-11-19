from __future__ import annotations

from abc import ABC

# %% mixins


class Mixin(ABC):
    """
    Marks a class as a mixin: a class that adds behaviour to its subclasses without
    deciding what they are.
    """
