import numpy as np
import pytest

import semantic_digital_twin.orm.ormatic_interface  # type: ignore  # noqa: F401
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.rules.conclusion import Add
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cup
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.world_entity import Body

from experiments.confidence_aware_eql.confidence_guard import (
    UnfamiliarObjectError,
    evaluate_with_confidence_guard,
)
from experiments.confidence_aware_eql.confidence_model import ConfidenceModel

# %% fixtures


@pytest.fixture
def confidence_model():
    """A confidence model fitted on forty cups with a familiar mass around 0.25kg."""
    np.random.seed(0)
    cups = [
        Cup(root=Body(inertial=Inertial(mass=float(mass))))
        for mass in np.random.normal(0.25, 0.03, 40)
    ]
    return ConfidenceModel.fit_from_instances(cups)


def _concluding_rule_for(cup: Cup) -> SymbolicExpression:
    """A trivial always-true rule whose conclusion binds ``cup`` to a result variable."""
    cup_variable = variable(Cup, domain=[cup])
    condition = HasType(cup_variable, Cup)
    concluded = variable(Cup, domain=[])
    with condition:
        Add(concluded, cup_variable)
    return condition


# %% behaviour


def test_familiar_object_lets_the_rule_proceed(confidence_model):
    """A rule concluding on a familiar cup evaluates to its normal result."""
    familiar_cup = Cup(root=Body(inertial=Inertial(mass=0.25)))
    condition = _concluding_rule_for(familiar_cup)

    results = list(evaluate_with_confidence_guard(condition, confidence_model))

    assert len(results) == 1


def test_unfamiliar_object_raises_unfamiliar_object_error(confidence_model):
    """A rule concluding on a fifty kilogram cup raises before yielding a result."""
    outlier_cup = Cup(root=Body(inertial=Inertial(mass=50.0)))
    condition = _concluding_rule_for(outlier_cup)

    with pytest.raises(UnfamiliarObjectError) as excinfo:
        list(evaluate_with_confidence_guard(condition, confidence_model))
    assert excinfo.value.instance is outlier_cup
