import pytest

import semantic_digital_twin.orm.ormatic_interface  # type: ignore  # noqa: F401
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cup, Pot
from semantic_digital_twin.world import World

from experiments.confidence_aware_eql.feature_pipeline import extract_feature_dataframe
from random_events.variable import Continuous, Symbolic

from experiments.confidence_aware_eql.feature_pipeline import (
    ObjectClass,
    infer_feature_variables,
)

MASS_COLUMN = "mass"


@pytest.fixture
def kitchen_objects():
    """Three kitchen objects of known mass: two cups and a heavier pot."""
    world = World.create_with_root_body("map")
    with world.modify_world():
        light_cup = Cup.create_with_new_body_in_world(name="light_cup", world=world)
        heavy_cup = Cup.create_with_new_body_in_world(name="heavy_cup", world=world)
        pot = Pot.create_with_new_body_in_world(name="pot", world=world)
    light_cup.root.inertial.mass = 0.25
    heavy_cup.root.inertial.mass = 0.30
    pot.root.inertial.mass = 2.50
    return [light_cup, heavy_cup, pot]


def test_extracted_dataframe_contains_the_object_masses(kitchen_objects):
    """The feature dataframe carries the mass set on each object."""
    dataframe = extract_feature_dataframe(kitchen_objects)
    masses = set(dataframe[MASS_COLUMN].round(2))
    assert masses == {0.25, 0.30, 2.50}


CLASS_COLUMN = "class"


def test_dataframe_labels_each_object_with_its_class(kitchen_objects):
    """The feature dataframe names the class of each object as an `ObjectClass` enum."""
    dataframe = extract_feature_dataframe(kitchen_objects)
    assert list(dataframe[CLASS_COLUMN]) == [
        ObjectClass.CUP,
        ObjectClass.CUP,
        ObjectClass.POT,
    ]


def test_class_is_symbolic_and_mass_is_continuous(kitchen_objects):
    """Mass is inferred as a continuous variable and class as a symbolic one."""
    dataframe = extract_feature_dataframe(kitchen_objects)
    annotated_variables = infer_feature_variables(dataframe)
    by_name = {
        annotated.variable.name: annotated.variable for annotated in annotated_variables
    }
    assert isinstance(by_name["class"], Symbolic)
    assert isinstance(by_name["mass"], Continuous)
