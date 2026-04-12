"""
Tests for the pouring domain (D_pour) of the BMP framework.

Verifies that Causes and SatisfiesRequest correctly model pouring out a fraction
of a container's fill level. The cup has a revolute tilt joint (the actuated DOF)
and a virtual prismatic fill-level joint (the derived state) governed by a
TorricelliEquation that is explicitly part of the world model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.body_motion_problem import (
    Motion,
    TaskRequest,
)
from semantic_digital_twin.reasoning.body_motion_problem.pouring import (
    PouringCauses,
    PouringEffect,
    PouringMSCModel,
    PouringSatisfiesRequest,
    TorricelliEquation,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, HasRootBody
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap


@dataclass(eq=False)
class PourableContainer(HasRootBody, HasFillLevel):
    """
    Minimal pourable container for testing.

    Connected to its parent via a revolute joint representing the tilt angle.
    """

    @classproperty
    def _parent_connection_type(self):
        return RevoluteConnection


@pytest.fixture
def world_with_cup():
    """World containing a single pourable container with a tilt joint, filled to 100%."""
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
    with world.modify_world():
        cup = PourableContainer.create_with_new_body_in_world(
            name=PrefixedName("cup"),
            world=world,
            active_axis=Vector3(0, 1, 0),
            connection_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=math.pi / 2, velocity=2.0),
            ),
        )
    cup.initialize_fill_level(world=world, parent_body=cup.root, initial_fill=1.0)
    cup.fill_equation = TorricelliEquation(
        tilt_connection=cup.root.parent_connection,
        fill_connection=cup.fill_connection,
    )
    return world, cup


def test_pouring_satisfies_request(world_with_cup):
    world, cup = world_with_cup
    effect = PouringEffect(
        target_object=cup,
        property_getter=lambda c: c.fill_level,
        goal_value=0.6,
    )
    task = TaskRequest(task_type="pour", name="cup")
    assert PouringSatisfiesRequest(task=task, effect=effect)()


def test_pouring_satisfies_request_rejects_wrong_task_type(world_with_cup):
    world, cup = world_with_cup
    effect = PouringEffect(
        target_object=cup,
        property_getter=lambda c: c.fill_level,
        goal_value=0.6,
    )
    task = TaskRequest(task_type="open", name="cup")
    assert not PouringSatisfiesRequest(task=task, effect=effect)()


def test_causes_pours_out_40_percent(world_with_cup, rclpy_node):
    """BMP Causes predicate generates a trajectory that reduces fill level by 40%."""
    world, cup = world_with_cup
    viz = VizMarkerPublisher(_world=world, node=rclpy_node)
    viz.with_tf_publisher()
    goal_fill = 0.6
    effect = PouringEffect(
        target_object=cup,
        property_getter=lambda c: c.fill_level,
        goal_value=goal_fill,
    )
    physics = PouringMSCModel(fill_equation=cup.fill_equation)
    motion = Motion(
        trajectory=[],
        actuator=cup.fill_equation.tilt_connection,
        motion_model=physics,
    )
    task = TaskRequest(task_type="pour", name="cup")

    assert PouringSatisfiesRequest(task=task, effect=effect)()
    causes = PouringCauses(
        effect=effect,
        environment=world,
        motion=motion,
    )
    result = causes()
    assert (
        result
    ), "Causes predicate must confirm the trajectory achieves the pouring effect"
    assert len(motion.trajectory) > 0, "Physics model must have generated a trajectory"
    # Tilt ramps from 0 to theta_max — values must be strictly increasing then constant
    assert (
        motion.trajectory[0] < motion.trajectory[-1] or len(set(motion.trajectory)) > 1
    )
    # Replay the trajectory so RViz shows the cup tilting and fill level dropping
    causes.replay(step_delay=0.1)
    assert cup.fill_level == pytest.approx(
        goal_fill, abs=0.05
    ), "Fill level must be within tolerance of goal"


def test_physics_model_resets_world_state(world_with_cup):
    """World state is restored to its pre-simulation value after physics model runs."""
    world, cup = world_with_cup
    fill_before = cup.fill_level
    effect = PouringEffect(
        target_object=cup,
        property_getter=lambda c: c.fill_level,
        goal_value=0.6,
    )
    physics = PouringMSCModel(fill_equation=cup.fill_equation)
    physics.run(effect=effect, world=world)

    assert cup.fill_level == pytest.approx(
        fill_before
    ), "World state must be reset after physics model run"
    assert cup.fill_equation.tilt_connection.position == pytest.approx(
        0.0
    ), "Tilt joint must be reset after physics model run"


def test_causes_does_not_hold_when_effect_already_achieved(world_with_cup):
    """Causes returns False if the effect is already satisfied before pouring."""
    world, cup = world_with_cup
    world.set_positions_1DOF_connection({cup.fill_connection: 0.5})
    effect = PouringEffect(
        target_object=cup,
        property_getter=lambda c: c.fill_level,
        goal_value=0.6,
    )
    motion = Motion(
        trajectory=[],
        actuator=cup.fill_equation.tilt_connection,
        motion_model=PouringMSCModel(fill_equation=cup.fill_equation),
    )
    assert not PouringCauses(
        effect=effect,
        environment=world,
        motion=motion,
    )()
