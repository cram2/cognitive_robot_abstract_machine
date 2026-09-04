"""
Experiment: can :class:`~probabilistic_model.probabilistic_circuit.relational.rspn.RelationalProbabilisticCircuit`
ground a query whose backdoor-adjustment set is a confounder shared across an
exchangeable relation, so that the result could be wrapped in
:class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`
for relational ``do()`` reasoning?

The scenario: a :class:`~test.krrood_test.dataset.example_classes.PickingRobot` has a
``skill`` level that confounds its :class:`~test.krrood_test.dataset.example_classes.GraspAttempt`
attempts — skill affects both the arm position chosen (the cause) and the grasp
outcome (the effect) directly, so the naive (non-adjusted) correlation between arm and
outcome is biased and only backdoor-adjusting for skill recovers the true causal effect.

Grounding this relation — the step every relational causal query would need before a
:class:`CausalCircuit` could even be built — never reaches the causal machinery at all:
the resulting circuit is either structurally disconnected or the grounding call itself
raises, in both cases from ``RelationalProbabilisticCircuit.ground()``'s own
node-mounting logic. A relational ``do()`` query is therefore not currently possible
through this API, independently of any further question about backdoor-criterion
validity.
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
        attempts.append(GraspAttempt(arm=arm, grasped=bool(rng.random() < grasp_probability)))
    return PickingRobot(skill=skill, attempts=attempts)


@pytest.fixture
def confounded_model() -> RelationalProbabilisticCircuit:
    rng = np.random.default_rng(0)
    robots = [_sample_confounded_robot(rng) for _ in range(TRAINING_ROBOT_COUNT)]
    model = RelationalProbabilisticCircuit(PickingRobot)
    model.fit([to_dao(robot) for robot in robots])
    return model


# %% the class-level circuit does capture the confound (sanity check on the fit itself)


def test_class_circuit_models_both_skill_and_the_aggregate_it_confounds(
    confounded_model,
):
    names = {v.name for v in confounded_model.class_probabilistic_circuit.variables}
    assert "PickingRobot.skill" in names
    assert "PickingRobotAggregations.success_count()" in names


# %% grounding breaks before a CausalCircuit could ever be built


def test_grounding_a_fully_observed_query_yields_a_disconnected_circuit(
    confounded_model,
):
    """
    A backdoor-adjustment query needs the adjustment set concrete, so every attempt is
    given a concrete arm/grasped value here (making ``success_count`` fully
    determined, avoiding the Monte-Carlo integration path). ``ground()`` does not
    raise, but the circuit it returns has two disconnected roots instead of one.
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

    with pytest.raises(ValueError, match="More than one root"):
        grounded.variables


def test_grounding_an_underspecified_query_raises_immediately(confounded_model):
    """
    The Monte-Carlo path (every attempt field left underspecified) is the pattern the
    RSPN test suite already exercises successfully for a richer class circuit
    (``SceneRoom``). For this relation, mounting a sampled exchangeable instance onto
    the conditioned class circuit fails outright.
    """
    query = a(PickingRobot)(
        skill=...,
        attempts=[a(GraspAttempt)(arm=..., grasped=...) for _ in range(ATTEMPTS_PER_ROBOT)],
    )
    query.resolve()

    np.random.seed(0)
    with pytest.raises(AttributeError, match="add_edge"):
        confounded_model.ground(query)
