import numpy as np
import pytest

import semantic_digital_twin.orm.ormatic_interface  # type: ignore  # noqa: F401
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cup, Pot
from semantic_digital_twin.world import World

from experiments.confidence_aware_eql.confidence_model import ConfidenceModel


@pytest.fixture
def familiar_kitchen_objects():
    """Forty light cups and forty heavier pots with familiar masses."""
    world = World.create_with_root_body("map")
    generator = np.random.default_rng(0)
    cups = []
    pots = []
    with world.modify_world():
        for index in range(40):
            cups.append(
                Cup.create_with_new_body_in_world(name=f"cup_{index}", world=world)
            )
        for index in range(40):
            pots.append(
                Pot.create_with_new_body_in_world(name=f"pot_{index}", world=world)
            )
    for cup in cups:
        cup.root.inertial.mass = float(generator.normal(0.25, 0.03))
    for pot in pots:
        pot.root.inertial.mass = float(generator.normal(2.50, 0.20))
    return cups + pots


def _light_cup(mass):
    world = World.create_with_root_body("map")
    with world.modify_world():
        cup = Cup.create_with_new_body_in_world(name="probe_cup", world=world)
    cup.root.inertial.mass = mass
    return cup


def test_familiar_cup_scores_higher_than_impossible_cup(familiar_kitchen_objects):
    """A normal cup is more likely under the model than a fifty kilogram cup."""
    model = ConfidenceModel.fit_from_instances(familiar_kitchen_objects)
    familiar = model.log_likelihood_of(_light_cup(0.25))
    impossible = model.log_likelihood_of(_light_cup(50.0))
    assert familiar > impossible


def test_familiar_cup_is_accepted_and_impossible_cup_is_flagged(
    familiar_kitchen_objects,
):
    """A normal cup is familiar and a fifty kilogram cup is unfamiliar."""
    model = ConfidenceModel.fit_from_instances(familiar_kitchen_objects)
    assert model.is_familiar(_light_cup(0.25))
    assert not model.is_familiar(_light_cup(50.0))


def test_every_familiar_object_scores_a_finite_log_likelihood(familiar_kitchen_objects):
    """No familiar object is scored as impossible (-inf) under its own fitted model.

    Regression test for the class-column hashing mismatch between training and
    scoring: before the pipeline aligned both sides, every familiar object,
    including the ones the model was fitted on, scored -inf.
    """
    model = ConfidenceModel.fit_from_instances(familiar_kitchen_objects)
    log_likelihoods = [
        model.log_likelihood_of(instance) for instance in familiar_kitchen_objects
    ]
    assert all(np.isfinite(log_likelihood) for log_likelihood in log_likelihoods)
