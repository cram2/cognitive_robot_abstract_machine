from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

# %% mixins


class Mixin(ABC):
    """
    Marks a class as a mixin: a class that adds behaviour to its subclasses without
    deciding what they are.
    """


# %% parameter mixins


@dataclass(eq=False, repr=False, kw_only=True)
class ParameterMixin(Mixin):
    """
    Marks a mixin that adds parameters, as dataclass fields, to the classes inheriting
    it, so that a parameter has the same name, type and meaning in each of them.
    """
