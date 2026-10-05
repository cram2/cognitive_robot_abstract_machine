import logging

import pytest

from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.ormatic.field_storage import CustomTypeRule, FieldStorage, FieldToStore
from krrood.ormatic.ormatic import ORMatic
from ..dataset.field_storage_classes import FieldsOfEveryStorage, StorageLeaf

# %% helpers


@pytest.fixture
def ormatic() -> ORMatic:
    return ORMatic(ClassDiagram([StorageLeaf, FieldsOfEveryStorage]))


def field_named(ormatic: ORMatic, name: str) -> WrappedField:
    """
    :return: The field of :class:`FieldsOfEveryStorage` with the given name.
    """
    wrapped_class = ormatic.class_dependency_graph.get_wrapped_class(
        FieldsOfEveryStorage
    )
    return next(f for f in wrapped_class.fields if f.field.name == name)


# %% classification


@pytest.mark.parametrize(
    "field_name, storage",
    [
        ("number", FieldStorage.BUILTIN),
        ("kind", FieldStorage.TYPE),
        ("leaf", FieldStorage.MANY_TO_ONE),
        ("leaves", FieldStorage.MANY_TO_MANY),
        ("day", FieldStorage.CUSTOM_TYPE),
        ("complex_number", FieldStorage.CUSTOM_TYPE),
        ("span", FieldStorage.JSON),
        ("numbers", FieldStorage.JSON),
        ("days", FieldStorage.JSON),
        ("lookup", FieldStorage.NOT_STORED),
        ("anything", FieldStorage.NOT_STORED),
    ],
)
def test_field_is_stored_the_way_its_type_calls_for(ormatic, field_name, storage):
    assert (
        ormatic.field_classifier.classify(field_named(ormatic, field_name)) is storage
    )


def test_classification_logs_how_the_field_is_stored(ormatic, caplog):
    day = field_named(ormatic, "day")

    with caplog.at_level(logging.INFO, logger="krrood.ormatic.field_storage"):
        ormatic.field_classifier.classify(day)

    assert CustomTypeRule.describe(FieldToStore(day, ormatic)) in caplog.messages
