"""
What a role read back out of the database still is.
"""

from sqlalchemy import select

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.patterns.role import Role
from ..dataset.example_classes import TextJSONSerializableClass
from ..dataset.ormatic_interface import RoleOverAValueStoredAsJsonDAO
from ..dataset.role_and_ontology.roles_over_a_value_stored_as_json import (
    RoleOverAValueStoredAsJson,
)


def test_a_role_read_back_from_the_database_keeps_the_taker_it_was_stored_with(
    session, database
):
    """
    A role taker kept as a value rather than in a table of its own is still what the
    restored role is about.
    """
    taker = TextJSONSerializableClass(label="colours")
    session.add(to_dao(RoleOverAValueStoredAsJson(role_taker=taker, note="asked")))
    session.commit()

    restored = session.scalars(select(RoleOverAValueStoredAsJsonDAO)).one().from_dao()

    assert restored.role_taker == taker
    assert restored.note == "asked"


def test_a_role_read_back_from_the_database_is_registered_for_its_taker(
    session, database
):
    """
    Reconstruction runs the role's ``__post_init__``, so the restored role answers a
    membership query from its taker just as a constructed one does.
    """
    taker = TextJSONSerializableClass(label="colours")
    session.add(to_dao(RoleOverAValueStoredAsJson(role_taker=taker, note="asked")))
    session.commit()

    restored = session.scalars(select(RoleOverAValueStoredAsJsonDAO)).one().from_dao()

    assert Role.roles_for(restored.role_taker) == [restored]
