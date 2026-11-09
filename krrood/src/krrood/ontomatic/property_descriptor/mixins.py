"""
Mixins that give property descriptors the characteristics of OWL properties.
"""

from __future__ import annotations

from abc import abstractmethod

from typing_extensions import TYPE_CHECKING

from krrood.patterns.mixin import Mixin

if TYPE_CHECKING:
    from krrood.ontomatic.property_descriptor.property_descriptor import (
        PropertyChain,
        PropertyDescriptor,
    )


# %% properties


class OntologyProperty(Mixin):
    """
    A mixin for descriptors that are properties of an ontology, with the characteristics
    and the relations to other properties OWL gives them.
    """


# %% property characteristics


class TransitiveProperty(OntologyProperty):
    """
    A mixin for descriptors that are transitive.
    """

    ...


class SymmetricProperty(OntologyProperty):
    """
    A mixin for descriptors that relate the object of every statement back to its
    subject.
    """

    ...


class ASymmetricProperty(OntologyProperty):
    """
    A mixin for descriptors that never relate the object of a statement back to its
    subject.
    """

    ...


class ReflexiveProperty(OntologyProperty):
    """
    A mixin for descriptors that relate every individual to itself.
    """

    ...


class IrreflexiveProperty(OntologyProperty):
    """
    A mixin for descriptors that never relate an individual to itself.
    """

    ...


# %% properties related to other properties


class HasInverseProperty(OntologyProperty):
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


class HasEquivalentProperties(OntologyProperty):
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


class HasDisjointProperties(OntologyProperty):
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


class HasChainAxioms(OntologyProperty):
    """
    A mixin for descriptors implied by chains of properties: a statement of each
    property of a chain, each starting where the previous one ends, implies a statement
    of this property from the first subject to the last object.
    """

    @classmethod
    @abstractmethod
    def get_chain_axioms(cls) -> list[PropertyChain]:
        """
        The chains of properties that imply this property.
        """
        ...


# %% classes


class OntologyBaseClass(Mixin):
    """
    A mixin for the class every class of a generated ontology module inherits from.
    """

    ...
