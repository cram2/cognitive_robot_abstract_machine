from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from krrood.ontomatic.property_descriptor.property_descriptor import (
        PropertyDescriptor,
    )


@dataclass(kw_only=True)
class TransitiveProperty:
    """
    A mixin for descriptors that are transitive.
    """

    ...


@dataclass(kw_only=True)
class HasInverseProperty(ABC):
    """
    A mixin for descriptors that have an inverse property.
    """

    @classmethod
    @abstractmethod
    def get_inverse(cls) -> type[PropertyDescriptor]:
        """
        The inverse of the property.
        """
        ...
