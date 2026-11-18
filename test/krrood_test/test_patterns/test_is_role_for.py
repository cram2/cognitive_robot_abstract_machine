"""
A role for a class is a role for that class's superclasses and for every role along its
taker chain.
"""

from __future__ import annotations

from krrood.patterns.role import Role

from ..dataset.role_and_ontology.roles_of_subclasses import (
    Animal,
    CircusPerformer,
    HeadlineAct,
    Tiger,
    Vehicle,
)

# %% superclasses of the role taker


def test_a_role_for_a_class_is_a_role_for_the_class_itself():
    """
    The role takers of a role are of its role-taker type.
    """
    assert CircusPerformer.is_role_for(taker_classes=(Tiger,))


def test_a_role_for_a_subclass_is_a_role_for_its_superclass():
    """
    A role for a tiger is a role for an animal.
    """
    assert CircusPerformer.is_role_for(taker_classes=(Animal,))


def test_a_role_for_a_class_is_not_one_for_an_unrelated_class():
    """
    A role is not a role for a class its role takers do not inherit from.
    """
    assert not CircusPerformer.is_role_for(taker_classes=(Vehicle,))


def test_a_role_is_one_for_a_tuple_when_it_is_one_for_any_member():
    """
    A role is a role for a tuple of classes when it is one for any of them.
    """
    assert CircusPerformer.is_role_for(taker_classes=(Vehicle, Animal))


# %% taker chain


def test_a_role_taken_by_a_role_is_one_for_that_role():
    """
    A role whose taker is a role is a role for that role.
    """
    assert HeadlineAct.is_role_for(taker_classes=(CircusPerformer,))


def test_a_role_taken_by_a_role_follows_the_taker_chain_to_its_end():
    """
    A role whose taker is a role is a role for the superclasses at the end of the chain.
    """
    assert HeadlineAct.is_role_for(taker_classes=(Animal,))


def test_a_role_taken_by_a_role_is_one_for_role():
    """
    Every role taker along a chain of roles is a :class:`Role`.
    """
    assert HeadlineAct.is_role_for(taker_classes=(Role,))
