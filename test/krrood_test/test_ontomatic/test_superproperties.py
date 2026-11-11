"""
Statements implied by the superproperties of a property, up its whole hierarchy.
"""

from __future__ import annotations

from krrood.ontomatic.property_descriptor.property_descriptor_relation import (
    InferredThrough,
    PropertyDescriptorRelation,
    RelationExplanation,
)
from ..dataset.role_and_ontology.superproperties import (
    Knows,
    Likes,
    Loves,
    PersonWithFeelings,
)


def test_a_statement_of_a_property_is_a_statement_of_every_superproperty_above_it():
    """
    Loving someone implies liking them, and liking them implies knowing them, although
    knows is no direct superproperty of loves.
    """
    romeo = PersonWithFeelings(name="romeo")
    juliet = PersonWithFeelings(name="juliet")

    romeo.loves.add(juliet)

    assert romeo.likes == {juliet}
    assert romeo.knows == {juliet}


def test_a_statement_of_a_superproperty_is_explained_one_step_up_the_hierarchy():
    """
    Each implied statement is explained by the statement of the property directly below
    its own, so the explanation follows the hierarchy.
    """
    romeo = PersonWithFeelings(name="romeo")
    juliet = PersonWithFeelings(name="juliet")

    romeo.loves.add(juliet)

    loves = PropertyDescriptorRelation.find_relation_for_triple(
        source_instance=romeo, property_descriptor_class=Loves, target_instance=juliet
    )
    likes = PropertyDescriptorRelation.find_relation_for_triple(
        source_instance=romeo, property_descriptor_class=Likes, target_instance=juliet
    )
    knows = PropertyDescriptorRelation.find_relation_for_triple(
        source_instance=romeo, property_descriptor_class=Knows, target_instance=juliet
    )
    assert likes.explanation == RelationExplanation(
        rule=InferredThrough.SUPER, premises=(loves,)
    )
    assert knows.explanation == RelationExplanation(
        rule=InferredThrough.SUPER, premises=(likes,)
    )
