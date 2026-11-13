"""
A class is an OWL subclass of another when it inherits from it, or when it is a role
whose role takers are.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import an, entity, variable_from
from krrood.entity_query_language.role_predicates import IsSubClassOrRole
from ..dataset.role_and_ontology.university_ontology_like_classes import (
    CEOAsFirstRole,
    Company,
    Person,
    RecognizedGroup,
    RepresentativeAsSecondRole,
)

# %% concrete classes


def test_a_subclass_is_a_subclass_or_role():
    """
    A class that inherits from another is a subclass or role of it.
    """
    assert IsSubClassOrRole.holds(sub_class=Company, super_class=RecognizedGroup)


def test_a_role_is_a_subclass_or_role_of_the_type_of_its_role_taker():
    """
    A role is a subclass or role of the type of its role taker.
    """
    assert IsSubClassOrRole.holds(sub_class=CEOAsFirstRole, super_class=Person)


def test_a_role_of_a_role_is_a_subclass_or_role_of_the_root_role_taker_type():
    """
    A role of a role is a subclass or role of the type at the end of its role-taker
    chain.
    """
    assert IsSubClassOrRole.holds(
        sub_class=RepresentativeAsSecondRole, super_class=Person
    )


def test_an_unrelated_class_is_not_a_subclass_or_role():
    """
    A role is not a subclass or role of a class its role takers do not inherit from.
    """
    assert not IsSubClassOrRole.holds(
        sub_class=CEOAsFirstRole, super_class=RecognizedGroup
    )


def test_a_class_is_a_subclass_or_role_of_a_tuple_when_it_is_one_of_any_member():
    """
    A class is a subclass or role of a tuple of classes when it is one of any of them.
    """
    assert IsSubClassOrRole.holds(
        sub_class=CEOAsFirstRole, super_class=(RecognizedGroup, Person)
    )


# %% queries


def test_a_query_selects_the_classes_that_are_subclasses_or_roles():
    """
    Used in a query, the predicate keeps the classes that are subclasses or roles.
    """
    candidate_class = variable_from([CEOAsFirstRole, Company])
    query = an(
        entity(candidate_class).where(
            IsSubClassOrRole(sub_class=candidate_class, super_class=Person)
        )
    )
    assert list(query.evaluate()) == [CEOAsFirstRole]
