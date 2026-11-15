import pytest

from krrood.class_diagrams.causal_relation import AttributeEndpoint
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.class_diagrams.exceptions import FieldIsUnMappedInClassDiagram
from krrood.entity_query_language.factories import variable

from ..dataset.example_classes import KRROODPosition, SceneObject, SceneRoom

# %% a diagram of a class with an attribute behind another class


@pytest.fixture
def class_diagram() -> ClassDiagram:
    return ClassDiagram(classes=[SceneRoom, SceneObject, KRROODPosition])


# %% attributes reached through fields


def test_an_attribute_is_named_as_its_symbolic_attribute(class_diagram):
    attribute = variable(SceneRoom, []).position.x

    assert (
        AttributeEndpoint.along(
            class_diagram=class_diagram, access_path=attribute._access_path_
        ).name
        == attribute._name_
    )


def test_the_fields_to_an_attribute_start_at_the_class(class_diagram):
    attribute = variable(SceneRoom, []).position.x

    assert AttributeEndpoint.along(
        class_diagram=class_diagram, access_path=attribute._access_path_
    ).fields == (
        class_diagram.get_wrapped_class(clazz=SceneRoom).get_field(name="position"),
        class_diagram.get_wrapped_class(clazz=KRROODPosition).get_field(name="x"),
    )


def test_a_field_the_class_lacks_raises(class_diagram):
    with pytest.raises(FieldIsUnMappedInClassDiagram):
        class_diagram.get_wrapped_class(clazz=SceneRoom).get_field(name="height")
