"""
A field whose type is a generic left unparameterized still has one thing to store where
the ORM keeps that type as a value: the value's own JSON names which subclass it is.
"""

from sqlalchemy import select

from krrood.ormatic.data_access_objects.helper import to_dao
from ..dataset.example_classes import GenericJSONWrapper, TextJSONSerializableClass
from ..dataset.ormatic_interface import GenericJSONWrapperDAO


def test_a_field_typed_as_an_unparameterized_generic_is_mapped():
    """
    The free type parameter says nothing about how the value is stored, so the field
    gets its column rather than being dropped as underspecified.
    """
    assert hasattr(GenericJSONWrapperDAO, "json_serializable_object")


def test_a_value_stored_as_json_under_an_unparameterized_generic_field_round_trips(
    session, database
):
    """
    The subclass that was stored comes back, not only the base the field is typed with.
    """
    value = TextJSONSerializableClass(label="colours")
    session.add(to_dao(GenericJSONWrapper(json_serializable_object=value)))
    session.commit()

    restored = session.scalars(select(GenericJSONWrapperDAO)).one().from_dao()

    assert restored.json_serializable_object == value
