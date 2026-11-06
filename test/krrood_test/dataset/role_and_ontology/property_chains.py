"""
A property implied by chains of other properties: ``parentOf o parentOf`` and ``parentOf o ancestorOf`` imply
``ancestorOf``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set

from krrood.entity_query_language.factories import a
from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.mixins import HasChainAxioms
from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyChain,
    PropertyDescriptor,
)


@dataclass(eq=False)
class PersonWithAncestors(Symbol):
    """
    A person related to its children and to its descendants.
    """

    name: str = ""
    """
    The name of the person.
    """
    parent_of: Set[PersonWithAncestors] = field(default_factory=set)
    """
    The children of the person.
    """
    ancestor_of: Set[PersonWithAncestors] = field(default_factory=set)
    """
    The descendants of the person.
    """


@dataclass
class ParentOf(PropertyDescriptor):
    """
    Relates a person to its children.
    """


@dataclass
class AncestorOf(PropertyDescriptor, HasChainAxioms):
    """
    Relates a person to its descendants, implied by every chain of children.
    """

    @classmethod
    def get_chain_axioms(cls) -> list[PropertyChain]:
        return [
            PropertyChain((ParentOf, ParentOf)),
            PropertyChain((ParentOf, AncestorOf)),
        ]


PersonWithAncestors.parent_of = ParentOf.from_eql_attribute(
    a(PersonWithAncestors).parent_of
)
PersonWithAncestors.ancestor_of = AncestorOf.from_eql_attribute(
    a(PersonWithAncestors).ancestor_of
)
