"""
Objects hashed by their identity, which stay distinct and findable whatever their fields
are.
"""

from __future__ import annotations

from krrood.patterns.hashed_by_identity import HashedByIdentity
from krrood.patterns.mixin import Mixin

from ..dataset.identity_hashed_classes import Visitor

# %% hashing


def test_objects_with_the_same_fields_are_distinct_set_members():
    """
    Two objects with equal fields are two members of a set.
    """
    assert len({Visitor(name="Ada"), Visitor(name="Ada")}) == 2


def test_an_object_is_found_after_its_fields_change():
    """
    An object stays findable in a set after one of its fields changes.
    """
    visitor = Visitor(name="Ada")
    visitors = {visitor}
    visitor.visited_rooms.append("kitchen")
    assert visitor in visitors


# %% marking


def test_hashing_by_identity_is_a_mixin():
    """
    Hashing by identity adds behaviour to a class without deciding what it is.
    """
    assert issubclass(HashedByIdentity, Mixin)
