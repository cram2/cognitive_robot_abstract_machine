"""
Grounding a scalar confounder alongside an exchangeable relation.

The scenario: a :class:`~test.krrood_test.dataset.example_classes.PickingRobot` has a
``skill`` level that confounds its :class:`~test.krrood_test.dataset.example_classes.GraspAttempt`
attempts — skill affects both the arm position chosen (the cause) and the grasp
outcome (the effect) directly, so the naive (non-adjusted) correlation between arm and
outcome is biased and only backdoor-adjusting for skill recovers the true causal
effect. This is the shape any relational causal query needs before a
:class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`
could be built on top of it: a class-level scalar alongside an exchangeable relation
whose own aggregation statistics may or may not be determined by the query.

These tests exercise ``RelationalProbabilisticCircuit.ground()`` for that shape,
independently of any further question about backdoor-criterion validity.
"""

from __future__ import annotations

import numpy as np
import pytest

from krrood.entity_query_language.factories import a
from krrood.ormatic.data_access_objects.helper import to_dao
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import GraspAttempt, PickingRobot

ATTEMPTS_PER_ROBOT = 4
TRAINING_ROBOT_COUNT = 400


# %% confounded synthetic population
#
# skill in {0.0, 1.0}, P(skill=1) = 0.5
# arm in {0.0, 1.0}: P(arm=1|skill=1)=0.9, P(arm=1|skill=0)=0.1   -- skill confounds arm
# grasped ~ Bernoulli(0.2 + 0.5*arm + 0.3*skill)                  -- arm causes grasped,
#                                                                     skill also affects
#                                                                     it directly
#
# True P(grasped=True | do(arm=1)) = 0.5*0.7 + 0.5*1.0 = 0.85
# Naive P(grasped=True | arm=1), uncorrected for skill, is biased toward ~0.97.


def _sample_confounded_robot(rng: np.random.Generator) -> PickingRobot:
    skill = 1.0 if rng.random() < 0.5 else 0.0
    arm_probability = 0.9 if skill == 1.0 else 0.1
    attempts = []
    for _ in range(ATTEMPTS_PER_ROBOT):
        arm = 1.0 if rng.random() < arm_probability else 0.0
        grasp_probability = 0.2 + 0.5 * arm + 0.3 * skill
        attempts.append(
            GraspAttempt(arm=arm, grasped=bool(rng.random() < grasp_probability))
        )
    return PickingRobot(skill=skill, attempts=attempts)


@pytest.fixture
def confounded_model() -> RelationalProbabilisticCircuit:
    rng = np.random.default_rng(0)
    robots = [_sample_confounded_robot(rng) for _ in range(TRAINING_ROBOT_COUNT)]
    model = RelationalProbabilisticCircuit(PickingRobot)
    model.fit([to_dao(robot) for robot in robots])
    return model


def _attempt_variable_names(count: int) -> set[str]:
    names = set()
    for index in range(count):
        names.add(f"PickingRobot.attempts[{index}].arm")
        names.add(f"PickingRobot.attempts[{index}].grasped")
    return names


# %% the class-level circuit does capture the confound (sanity check on the fit itself)


def test_class_circuit_models_both_skill_and_the_aggregate_it_confounds(
    confounded_model,
):
    names = {v.name for v in confounded_model.class_probabilistic_circuit.variables}
    assert "PickingRobot.skill" in names
    assert "PickingRobotAggregations.success_count()" in names


# %% grounding a scalar confounder alongside a fully determined exchangeable relation


def test_grounding_a_fully_observed_query_yields_one_connected_circuit(
    confounded_model,
):
    """
    A backdoor-adjustment query needs the adjustment set concrete, so every attempt is
    given a concrete arm/grasped value here (making ``success_count`` fully determined,
    avoiding the Monte-Carlo integration path).
    """
    query = a(PickingRobot)(
        skill=...,
        attempts=[
            a(GraspAttempt)(arm=1.0, grasped=True),
            a(GraspAttempt)(arm=1.0, grasped=True),
            a(GraspAttempt)(arm=1.0, grasped=False),
            a(GraspAttempt)(arm=0.0, grasped=True),
        ],
    )
    query.resolve()

    grounded = confounded_model.ground(query)

    assert grounded.is_valid()
    names = {v.name for v in grounded.variables}
    assert names == _attempt_variable_names(ATTEMPTS_PER_ROBOT) | {
        "PickingRobot.skill",
        "PickingRobotAggregations.success_count()",
        "PickingRobotAggregations.total_count()",
    }


# %% grounding a scalar confounder alongside an underspecified exchangeable relation


def test_grounding_an_underspecified_query_yields_one_connected_circuit(
    confounded_model,
):
    """
    Every attempt field is left underspecified, so ``success_count`` cannot be
    determined from the query and must be integrated out via the Monte-Carlo path, while
    ``total_count`` is determined directly from the query's attempt count.
    """
    query = a(PickingRobot)(
        skill=...,
        attempts=[
            a(GraspAttempt)(arm=..., grasped=...) for _ in range(ATTEMPTS_PER_ROBOT)
        ],
    )
    query.resolve()

    np.random.seed(0)
    grounded = confounded_model.ground(query)

    assert grounded.is_valid()
    names = {v.name for v in grounded.variables}
    assert names == _attempt_variable_names(ATTEMPTS_PER_ROBOT) | {
        "PickingRobot.skill",
        "PickingRobotAggregations.total_count()",
    }
    assert "PickingRobotAggregations.success_count()" not in names
