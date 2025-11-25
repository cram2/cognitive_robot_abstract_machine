from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .property_descriptor import PropertyDescriptor


@dataclass
class TransitiveProperty:
    """
    A mixin for descriptors that are transitive.
    """

    ...


@dataclass
class HasInverseProperty(ABC):
    """
    A mixin for descriptors that have an inverse property.
    """

    @classmethod
    @abstractmethod
    def get_inverse(cls) -> Type[PropertyDescriptor]:
        """
        The inverse of the property.
        """
        ...
