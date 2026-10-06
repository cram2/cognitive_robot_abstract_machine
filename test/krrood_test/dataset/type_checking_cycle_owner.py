"""
A module whose ``TYPE_CHECKING`` import targets a module that imports this one back.

Mirrors a mixin module annotating its fields with classes from a module that builds on
those mixins: the cycle is harmless at runtime because the import only runs while type
checking.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Generic, Optional, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

if TYPE_CHECKING:
    from fractions import Fraction
    from .type_checking_cycle_target import ContainerOfBoundHolder

TContent = TypeVar("TContent")


@dataclass
class Content:
    """
    The value a holder keeps.
    """


@dataclass
class HolderWithGenericContent(Generic[TContent], SubClassSafeGeneric):
    """
    A holder whose content type is bound by its subclasses.
    """

    content: Optional[TContent] = None
    """
    The value this holder keeps.
    """

    share: Optional[Fraction] = None
    """
    The portion of the content in use, annotated with a name that only exists while type
    checking.
    """

    def place_into(self, container: ContainerOfBoundHolder) -> None:
        """
        :param container: The container, a class from the module that imports this one
            back, that this holder is placed in.
        """
        container.holders.append(self)


@dataclass
class HolderBindingItsContent(HolderWithGenericContent[Content]):
    """
    A holder that binds its content type while this module is still being imported.
    """
