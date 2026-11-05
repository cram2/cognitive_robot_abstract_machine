"""
Mixins that give property descriptors the characteristics of OWL properties.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from krrood.ontomatic.property_descriptor.property_descriptor import (
        PropertyDescriptor,
    )


# %% property characteristics


@dataclass
class TransitiveProperty:
    """
    A mixin for descriptors that are transitive.
    """

    ...


@dataclass
class SymmetricProperty(ABC):
    """
    A mixin for descriptors that relate the object of every statement back to its
    subject.
    """

    ...


@dataclass
class ASymmetricProperty(ABC):
    """
    A mixin for descriptors that never relate the object of a statement back to its
    subject.
    """

    ...


@dataclass
class ReflexiveProperty(ABC):
    """
    A mixin for descriptors that relate every individual to itself.
    """

    ...


@dataclass
class IrreflexiveProperty(ABC):
    """
    A mixin for descriptors that never relate an individual to itself.
    """

    ...


# %% properties related to other properties


@dataclass
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


@dataclass
class HasEquivalentProperties(ABC):
    """
    A mixin for descriptors whose statements are also statements of other properties,
    and the other way round.
    """

    @classmethod
    @abstractmethod
    def get_equivalent_properties(cls) -> list[type[PropertyDescriptor]]:
        """
        The properties equivalent to this property.
        """
        ...


@dataclass
class HasDisjointProperties(ABC):
    """
    A mixin for descriptors that never relate the same two individuals as certain other
    properties.
    """

    @classmethod
    @abstractmethod
    def get_disjoint_properties(cls) -> list[type[PropertyDescriptor]]:
        """
        The properties disjoint with this property.
        """
        ...


@dataclass
class HasChainAxioms(ABC):
    """
    A mixin for descriptors implied by chains of properties: a statement of each
    property of a chain, each starting where the previous one ends, implies a statement
    of this property from the first subject to the last object.
    """

    @classmethod
    @abstractmethod
    def get_chain_axioms(cls) -> list[tuple[type[PropertyDescriptor], ...]]:
        """
        The chains of properties that imply this property.
        """
        ...


# %% classes


@dataclass
class IsBaseClass(ABC):
    """
    A mixin for the class every class of a generated ontology module inherits from.
    """

    ...
