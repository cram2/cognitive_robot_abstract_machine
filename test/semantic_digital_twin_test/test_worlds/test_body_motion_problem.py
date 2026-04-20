"""
Tests for the Body Motion Problem (BMP) framework.

Organised from unit-level predicate tests to fully integrated EQL queries:

  1. TestContainerManipulationPredicates — SatisfiesRequest / Causes / CanPerform for open/close
  2. TestPouringPredicates              — SatisfiesRequest / Causes / physics-model resets for pouring
  3. TestContainerManipulationQueries  — EQL queries over the container manipulation domain
  4. TestPouringQueries                 — EQL queries over pouring and coupled-pouring domains
  5. TestRobotIntegration               — long-running robot-specific queries (Stretch / Tiago / PR2)
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass

import pytest
import rclpy
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from krrood.entity_query_language.factories import an, set_of, variable
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.body_motion_problem import (
    Effect,
    Motion,
    TaskRequest,
    SatisfiesRequest,
    Causes,
)
from semantic_digital_twin.reasoning.body_motion_problem.container_manipulation import (
    ContainerCanPerform,
    ContainerSatisfiesRequest,
    ClosedEffect,
    OpenedEffect,
    RunMSCModel,
)
from semantic_digital_twin.reasoning.body_motion_problem.pouring import (
    ArticulatedPouringEquation,
    ContainerGeometry,
    CoupledPouringMSCModel,
    PouringCanPerform,
    PouringEffect,
    PouringMSCModel,
    ReceiverFillEffect,
)
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door, Drawer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


# ---------------------------------------------------------------------------
# Shared domain types
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class PourableContainer(HasRootBody, HasFillLevel):
    """
    Minimal pourable container for testing.

    Connected to its parent via a revolute joint representing the tilt angle.
    """

    @classproperty
    def _parent_connection_type(self):
        return RevoluteConnection


@dataclass(eq=False)
class ReceiverContainer(HasRootBody, HasFillLevel):
    """Receiver cup: fixed in world, initially empty."""

    @classproperty
    def _parent_connection_type(self):
        return FixedConnection


# ---------------------------------------------------------------------------
# Coupled-pouring geometry constants
# ---------------------------------------------------------------------------

SOURCE_GEOMETRY = ContainerGeometry(height=0.12, half_width=0.04)
RECEIVER_GEOMETRY = ContainerGeometry(height=0.10, half_width=0.05)

RECEIVER_TOP_Z = 0.70
HEIGHT_ABOVE_RECEIVER = 0.10

_SOURCE_RADIUS = SOURCE_GEOMETRY.half_width
_SOURCE_HEIGHT = SOURCE_GEOMETRY.height


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def mutable_model_world(pr2_apartment_world):
    return deepcopy(pr2_apartment_world)


@pytest.fixture
def stretch_kitchen_world(stretch_world, kitchen_world):
    world = deepcopy(stretch_world)
    world.merge_world(deepcopy(kitchen_world))
    return world


@pytest.fixture
def tiago_kitchen_world(tiago_world, kitchen_world):
    world = deepcopy(tiago_world)
    world.merge_world(deepcopy(kitchen_world))
    return world


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
    cup.container_geometry = ContainerGeometry(height=0.12, half_width=0.04)
    cup.initialize_fill_level(world=world, parent_body=cup.root, initial_fill=1.0)
    cup.fill_equation = ArticulatedPouringEquation(
        tilt_connection=cup.root.parent_connection,
        fill_connection=cup.fill_connection,
        container_geometry=cup.container_geometry,
    )
    return world, cup


@pytest.fixture(scope="function")
def pr2_world_with_cup(pr2_world_setup):
    """PR2 world with a pourable cup placed within arm reach at (0.7, 0.0, 0.85)."""
    world = deepcopy(pr2_world_setup)
    robot = PR2.from_world(world)
    with world.modify_world():
        cup = PourableContainer.create_with_new_body_in_world(
            name=PrefixedName("cup"),
            world=world,
            active_axis=Vector3(0, 1, 0),
            connection_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=math.pi / 2, velocity=2.0),
            ),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.7,
                y=0.0,
                z=0.85,
                reference_frame=world.root,
            ),
        )
    cup.container_geometry = ContainerGeometry(height=0.12, half_width=0.04)
    cup.initialize_fill_level(world=world, parent_body=cup.root, initial_fill=1.0)
    cup.fill_equation = ArticulatedPouringEquation(
        tilt_connection=cup.root.parent_connection,
        fill_connection=cup.fill_connection,
        container_geometry=cup.container_geometry,
    )
    return world, cup, robot


@pytest.fixture
def world_with_source_and_receiver():
    """
    World with source cup (kinematic chain: PrismaticX → PrismaticZ → RevoluteY)
    positioned so its rim starts directly above the receiver opening.
    """
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))

    with world.modify_world():
        receiver = ReceiverContainer.create_with_new_body_in_world(
            name=PrefixedName("receiver"),
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.2,
                y=0.0,
                z=RECEIVER_TOP_Z - RECEIVER_GEOMETRY.height / 2,
                reference_frame=world.root,
            ),
            scale=Scale(
                2 * RECEIVER_GEOMETRY.half_width,
                2 * RECEIVER_GEOMETRY.half_width,
                RECEIVER_GEOMETRY.height,
            ),
        )
    receiver.container_geometry = RECEIVER_GEOMETRY
    receiver.initialize_fill_level(
        world=world, parent_body=receiver.root, initial_fill=0.0
    )

    receiver_center = receiver.root.global_pose.to_position().to_np()[:3]
    receiver_opening = receiver_center.copy()
    receiver_opening[2] += RECEIVER_GEOMETRY.height / 2
    x_dof_initial = float(receiver_opening[0]) - _SOURCE_RADIUS
    z_dof_initial = float(receiver_opening[2]) + HEIGHT_ABOVE_RECEIVER - _SOURCE_HEIGHT

    x_dof_initial = x_dof_initial * -2
    z_dof_initial = z_dof_initial * 1.2

    x_body = Body(name=PrefixedName("source_x_translation"))
    with world.modify_world():
        world.add_body(x_body)
        x_connection = PrismaticConnection.create_with_dofs(
            world=world,
            parent=world.root,
            child=x_body,
            axis=Vector3(1, 0, 0),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=-1.0, velocity=-2.0),
                upper=DerivativeMap(position=1.0, velocity=2.0),
            ),
        )
        world.add_connection(x_connection)
    world.set_positions_1DOF_connection({x_connection: x_dof_initial})

    z_body = Body(name=PrefixedName("source_z_translation"))
    with world.modify_world():
        world.add_body(z_body)
        z_connection = PrismaticConnection.create_with_dofs(
            world=world,
            parent=x_body,
            child=z_body,
            axis=Vector3(0, 0, 1),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=2.0, velocity=2.0),
            ),
        )
        world.add_connection(z_connection)
    world.set_positions_1DOF_connection({z_connection: z_dof_initial})

    cup_body = Body(name=PrefixedName("source_cup"))
    cup_shape = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=cup_body),
        scale=Scale(2 * _SOURCE_RADIUS, 2 * _SOURCE_RADIUS, _SOURCE_HEIGHT),
    )
    cup_body.visual = ShapeCollection(shapes=[cup_shape])
    cup_body.collision = ShapeCollection(shapes=[cup_shape])
    with world.modify_world():
        world.add_body(cup_body)
        tilt_connection = RevoluteConnection.create_with_dofs(
            world=world,
            parent=z_body,
            child=cup_body,
            axis=Vector3(0, 1, 0),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=math.pi / 2, velocity=2.0),
            ),
        )
        world.add_connection(tilt_connection)

    source = PourableContainer(name=PrefixedName("source_cup"), root=cup_body)
    with world.modify_world():
        world.add_semantic_annotation(source)
    source.container_geometry = SOURCE_GEOMETRY
    source.initialize_fill_level(world=world, parent_body=cup_body, initial_fill=0.8)
    source.fill_equation = ArticulatedPouringEquation(
        tilt_connection=tilt_connection,
        fill_connection=source.fill_connection,
        container_geometry=SOURCE_GEOMETRY,
    )

    return world, source, receiver


# ---------------------------------------------------------------------------
# Shared helpers for container manipulation tests
# ---------------------------------------------------------------------------


def _get_msc_model_for_open_goal(handle_body, actuator, goal_value) -> RunMSCModel:
    """Create a motion statechart model that drives a joint to goal_value."""
    msc = MotionStatechart()
    goal = Open(
        tip_link=handle_body,
        environment_link=handle_body,
        goal_joint_state=goal_value,
    )
    msc.add_node(goal)
    msc.add_node(EndMotion.when_true(goal))
    return RunMSCModel(msc=msc, actuator=actuator, timeout=500)


def _extend_world(
    world: World,
    only_drawers: bool = False,
    only_doors: bool = False,
    include_close: bool = True,
    half_door_opening: bool = False,
) -> tuple:
    """
    Infer semantic annotations, attach them to the world, and build matching
    effects, motions, and task requests for all drawers and/or doors.

    :param only_drawers: restrict to drawers only (ignore doors).
    :param only_doors: restrict to doors only (ignore drawers).
    :param include_close: also create closed effects/motions and a close task.
    :param half_door_opening: limit door opened-effect goal to half the upper limit.
    :return: (effects, motions, open_task, close_task_or_None, drawers)
    """
    world_reasoner = WorldReasoner(world)
    inferred = world_reasoner.infer_semantic_annotations()
    with world.modify_world():
        world.add_semantic_annotations(inferred)

    drawers = [] if only_doors else world.get_semantic_annotations_by_type(Drawer)
    doors = [] if only_drawers else world.get_semantic_annotations_by_type(Door)
    annotations = drawers + doors

    property_getter = lambda obj: obj.root.parent_connection.position
    effects = []
    motions = []
    for annotation in annotations:
        act = annotation.root.parent_connection
        upper = act.active_dofs[0].limits.upper.position
        effect_goal = (
            upper / 2 if (half_door_opening and isinstance(annotation, Door)) else upper
        )

        effects.append(
            OpenedEffect(
                target_object=annotation,
                goal_value=effect_goal,
                property_getter=property_getter,
            )
        )
        motions.append(
            Motion(
                trajectory=[],
                actuator=act,
                motion_model=_get_msc_model_for_open_goal(
                    annotation.handle, act, upper
                ),
            )
        )

        if include_close:
            lower = act.active_dofs[0].limits.lower.position
            effects.append(
                ClosedEffect(
                    target_object=annotation,
                    goal_value=lower,
                    property_getter=property_getter,
                )
            )
            motions.append(
                Motion(
                    trajectory=[],
                    actuator=act,
                    motion_model=_get_msc_model_for_open_goal(
                        annotation.handle, act, lower
                    ),
                )
            )

    open_task = TaskRequest(task_type="open", name="open_container")
    close_task = (
        TaskRequest(task_type="close", name="close_container")
        if include_close
        else None
    )
    return effects, motions, open_task, close_task, drawers


# ---------------------------------------------------------------------------
# 1. Unit tests: container manipulation predicates
# ---------------------------------------------------------------------------


class TestContainerManipulationPredicates:
    def test_satisfies_request(self, mutable_model_world):
        """ContainerSatisfiesRequest holds for matching task type and rejects mismatched type."""
        world = mutable_model_world
        effects, _, open_task, _, _ = _extend_world(world)

        effect = next(e for e in effects if isinstance(e, OpenedEffect))
        assert ContainerSatisfiesRequest(task=open_task, effect=effect)()

        close_task = TaskRequest(task_type="close", name="close_container")
        assert not ContainerSatisfiesRequest(task=close_task, effect=effect)()

    def test_causes(self, mutable_model_world):
        """Causes holds when motion actuator matches effect actuator, and not otherwise."""
        world = mutable_model_world
        effects, motions, _, _, _ = _extend_world(world)

        # effects[0] = OpenedEffect, motions[0] = open motion — same actuator
        assert Causes(effect=effects[0], motion=motions[0], environment=world)()

        # effects[0] = OpenedEffect, motions[1] = close motion — direction mismatch
        assert not Causes(effect=effects[0], motion=motions[1], environment=world)()

    def test_can_execute(self, mutable_model_world, rclpy_node):
        """ContainerCanPerform returns False for an empty trajectory and a bool for a non-empty one."""
        world = mutable_model_world
        viz = VizMarkerPublisher(_world=world, node=rclpy_node)
        viz.with_tf_publisher()

        _, motions, _, _, drawers = _extend_world(world, only_drawers=True)
        robot = PR2.from_world(world)
        motion = motions[0]

        assert not ContainerCanPerform(motion=motion, robot=robot)()

        act = drawers[0].root.parent_connection
        upper = act.active_dofs[0].limits.upper.position
        motion.trajectory = [i * upper / 8 for i in range(9)]
        result = ContainerCanPerform(motion=motion, robot=robot)()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 2. Unit tests: pouring predicates and physics model
# ---------------------------------------------------------------------------


class TestPouringPredicates:
    def test_pouring_satisfies_request(self, world_with_cup):
        """SatisfiesRequest holds for a pour task paired with a PouringEffect."""
        world, cup = world_with_cup
        effect = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.6
        )
        task = TaskRequest(task_type="pour", name="cup")
        assert SatisfiesRequest(
            task=task, effect=effect, task_type="pour", effect_type=PouringEffect
        )()

    def test_pouring_satisfies_request_rejects_wrong_task_type(self, world_with_cup):
        """SatisfiesRequest rejects a task whose type does not match the expected pour type."""
        world, cup = world_with_cup
        effect = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.6
        )
        task = TaskRequest(task_type="open", name="cup")
        assert not SatisfiesRequest(
            task=task, effect=effect, task_type="pour", effect_type=PouringEffect
        )()

    def test_physics_model_resets_world_state(self, world_with_cup):
        """World state is restored to its pre-simulation value after the physics model runs."""
        world, cup = world_with_cup
        fill_before = cup.fill_level
        effect = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.6
        )
        physics = PouringMSCModel(fill_equation=cup.fill_equation)
        physics.run(effect=effect, world=world)

        assert cup.fill_level == pytest.approx(fill_before)
        assert cup.fill_equation.tilt_connection.position == pytest.approx(0.0)

    def test_causes_does_not_hold_when_effect_already_achieved(self, world_with_cup):
        """Causes returns False when the fill level is already at or below the goal."""
        world, cup = world_with_cup
        world.set_positions_1DOF_connection({cup.fill_connection: 0.5})
        effect = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.6
        )
        motion = Motion(
            trajectory=[],
            actuator=cup.fill_equation.tilt_connection,
            motion_model=PouringMSCModel(fill_equation=cup.fill_equation),
        )
        assert not Causes(effect=effect, environment=world, motion=motion)()

    def test_world_state_reset_after_coupled_model(
        self, world_with_source_and_receiver, rclpy_node
    ):
        """Source and receiver fill levels are restored after CoupledPouringMSCModel.run()."""
        world, source, receiver = world_with_source_and_receiver
        viz = VizMarkerPublisher(_world=world, node=rclpy_node)
        viz.with_tf_publisher()

        source_fill_before = source.fill_level
        receiver_fill_before = receiver.fill_level

        receiver_effect = ReceiverFillEffect(
            target_object=receiver,
            property_getter=lambda c: c.fill_level,
            goal_value=0.3,
            tolerance=0.05,
        )
        coupled_model = CoupledPouringMSCModel(
            source=source,
            receiver=receiver,
            root_link=world.root,
            theta_max=math.pi / 2,
        )
        coupled_model.run(receiver_effect, world)

        assert source.fill_level == pytest.approx(source_fill_before)
        assert receiver.fill_level == pytest.approx(receiver_fill_before)


# ---------------------------------------------------------------------------
# 3. EQL integration tests: container manipulation queries
# ---------------------------------------------------------------------------


class TestContainerManipulationQueries:
    def test_query_motion_satisfying_task_request(self, mutable_model_world):
        """An EQL query returns at least one motion that satisfies the open task request."""
        world = mutable_model_world
        effects, motions, open_task, close_task, _ = _extend_world(world)

        task_sym = variable(TaskRequest, domain=[open_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions)

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                ContainerSatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
            )
        )
        results = list(query.evaluate())
        assert len(results) > 0

    def test_query_motion_satisfying_task_request_not_all(self, mutable_model_world):
        """EQL query adapts to world state: randomly opened drawers reduce the result set."""
        world = mutable_model_world
        effects, motions, open_task, _, _ = _extend_world(
            world, include_close=False, half_door_opening=True
        )

        for drawer in world.get_semantic_annotations_by_type(Drawer):
            if random.randint(0, 5) == 4:
                max_position = drawer.root.parent_connection.active_dofs[
                    0
                ].limits.upper.position
                drawer.root.parent_connection.position = max_position

        task_sym = variable(TaskRequest, domain=[open_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions)

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                ContainerSatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
            )
        )
        results = list(query.evaluate())
        assert len(results) > 0

    def test_query_task_and_effect_satisfying_motion(self, mutable_model_world):
        """Given a fixed motion, the EQL query recovers the matching task and effect."""
        world = mutable_model_world
        effects, _, open_task, close_task, drawers = _extend_world(world)

        motion = Motion(
            trajectory=[0.0, 0.1, 0.2, 0.3, 0.4],
            actuator=drawers[0].root.parent_connection,
        )
        task_sym = variable(TaskRequest, domain=[open_task, close_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=[motion])

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                ContainerSatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
            )
        )
        results = list(query.evaluate())
        assert len(results) > 0

    def test_query_motion_if_drawers_open(self, mutable_model_world):
        """Query results switch from open to close tasks when all drawers are moved to open position."""
        world = mutable_model_world
        effects, motions, open_task, close_task, drawers = _extend_world(
            world, only_drawers=True
        )

        task_sym = variable(TaskRequest, domain=[open_task, close_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions)

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                ContainerSatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
            )
        )

        results = list(query.evaluate())
        assert all(res.data[task_sym].task_type == "open" for res in results)

        for drawer in drawers:
            drawer.root.parent_connection.position = (
                drawer.root.parent_connection.active_dofs[0].limits.upper.position
            )
        world.notify_state_change()

        results = list(query.evaluate())
        assert all(res.data[task_sym].task_type == "close" for res in results)


# ---------------------------------------------------------------------------
# 4. EQL integration tests: pouring queries
# ---------------------------------------------------------------------------


class TestPouringQueries:
    def test_causes_pours_out_40_percent(self, world_with_cup, rclpy_node):
        """Causes predicate generates a trajectory that reduces fill level by 40%."""
        world, cup = world_with_cup
        viz = VizMarkerPublisher(_world=world, node=rclpy_node)
        viz.with_tf_publisher()

        goal_fill = 0.6
        effect = PouringEffect(
            target_object=cup,
            property_getter=lambda c: c.fill_level,
            goal_value=goal_fill,
        )
        motion = Motion(
            trajectory=[],
            actuator=cup.fill_equation.tilt_connection,
            motion_model=PouringMSCModel(fill_equation=cup.fill_equation),
        )
        task = TaskRequest(task_type="pour", name="cup")

        assert SatisfiesRequest(
            task=task, effect=effect, task_type="pour", effect_type=PouringEffect
        )()
        causes = Causes(effect=effect, environment=world, motion=motion)
        assert (
            causes()
        ), "Causes predicate must confirm the trajectory achieves the pouring effect"
        assert len(motion.trajectory) > 0

        causes.replay(step_delay=0.1)
        assert cup.fill_level == pytest.approx(goal_fill, abs=0.05)

    def test_pouring_can_perform(self, pr2_world_with_cup, rclpy_node):
        """PouringCanPerform confirms the PR2 can execute the tilt trajectory from Causes."""
        world, cup, robot = pr2_world_with_cup
        viz = VizMarkerPublisher(_world=world, node=rclpy_node)
        viz.with_tf_publisher()

        goal_fill = 0.6
        effect = PouringEffect(
            target_object=cup,
            property_getter=lambda c: c.fill_level,
            goal_value=goal_fill,
        )
        motion = Motion(
            trajectory=[],
            actuator=cup.fill_equation.tilt_connection,
            motion_model=PouringMSCModel(fill_equation=cup.fill_equation),
        )

        causes = Causes(effect=effect, environment=world, motion=motion)
        assert causes()
        causes.replay(step_delay=0.1)
        assert PouringCanPerform(motion=motion, robot=robot)()

    def test_eql_query_all_three_predicates(self, pr2_world_with_cup, rclpy_node):
        """EQL query resolves task, effect, and motion simultaneously across all three BMP predicates."""
        world, cup, robot = pr2_world_with_cup
        viz = VizMarkerPublisher(_world=world, node=rclpy_node)
        viz.with_tf_publisher()

        goal_fill = 0.6
        task = TaskRequest(task_type="pour", name="cup")
        effect = PouringEffect(
            target_object=cup,
            property_getter=lambda c: c.fill_level,
            goal_value=goal_fill,
        )
        motion = Motion(
            trajectory=[],
            actuator=cup.fill_equation.tilt_connection,
            motion_model=PouringMSCModel(fill_equation=cup.fill_equation),
        )

        task_sym = variable(TaskRequest, domain=[task])
        effect_sym = variable(Effect, domain=[effect])
        motion_sym = variable(Motion, domain=[motion])

        query = an(
            set_of(task_sym, effect_sym, motion_sym).where(
                SatisfiesRequest(
                    task=task_sym,
                    effect=effect_sym,
                    task_type="pour",
                    effect_type=PouringEffect,
                ),
                Causes(effect=effect_sym, environment=world, motion=motion_sym),
                PouringCanPerform(motion=motion_sym, robot=robot),
            )
        )

        results = list(query.evaluate())
        assert len(results) == 1
        result = results[0]
        assert result.data[task_sym].task_type == "pour"
        assert result.data[effect_sym].goal_value == goal_fill
        assert len(result.data[motion_sym].trajectory) > 0

    def test_receiver_fill_increases(self, world_with_source_and_receiver, rclpy_node):
        """CoupledPouringMSCModel produces a tilt trajectory that fills the receiver to the setpoint."""
        world, source, receiver = world_with_source_and_receiver
        viz = VizMarkerPublisher(_world=world, node=rclpy_node)
        viz.with_tf_publisher()

        goal_receiver_fill = 0.2
        receiver_effect = ReceiverFillEffect(
            target_object=receiver,
            property_getter=lambda c: c.fill_level,
            goal_value=goal_receiver_fill,
            tolerance=0.05,
        )
        coupled_model = CoupledPouringMSCModel(
            source=source,
            receiver=receiver,
            root_link=world.root,
            theta_max=math.pi / 2,
        )
        motion = Motion(
            trajectory=[],
            actuator=source.fill_equation.tilt_connection,
            motion_model=coupled_model,
        )

        causes = Causes(effect=receiver_effect, environment=world, motion=motion)
        assert causes()
        assert len(motion.trajectory) > 0

        receiver_fill_trajectory = coupled_model.last_receiver_fill_trajectory
        assert len(receiver_fill_trajectory) > 0
        assert (
            receiver_fill_trajectory[-1]
            >= goal_receiver_fill - receiver_effect.tolerance
        )

        causes.replay(step_delay=0.2)


# ---------------------------------------------------------------------------
# 5. Long-running robot integration tests (not executed in CI)
# ---------------------------------------------------------------------------


def _present_results(results, robot: AbstractRobot) -> None:
    """Print a summary of query results to the console."""
    for result in results:
        task, motion, effect = result.values()
        print("-" * 20)
        print(f"Task: {task}")
        print(f"Robot: {robot.name}")
        print(
            f"Effect: {effect.__class__.__name__} for target: "
            f"{effect.target_object.__class__.__name__}"
            f"(name={effect.target_object.name} "
            f"body={effect.target_object.root.name} "
            f"handle={effect.target_object.handle.name})"
        )


class TestRobotIntegration:
    def test_query_motion_satisfying_task_request_stretch(
        self, stretch_kitchen_world, rclpy_node
    ):
        """Motion querying for open task using Stretch robot in the kitchen world (drawers only)."""
        world = stretch_kitchen_world
        viz = VizMarkerPublisher(node=rclpy_node, _world=world)
        viz.with_tf_publisher()

        effects, motions, open_task, _, drawers = _extend_world(
            world, only_drawers=True
        )

        task_sym = variable(TaskRequest, domain=[open_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions)

        robot = Stretch.from_world(world)
        query = an(
            set_of(task_sym, motion_sym, effect_sym).where(
                ContainerSatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
                ContainerCanPerform(motion=motion_sym, robot=robot),
            )
        )

        results = list(query.evaluate())
        _present_results(results, robot)
        assert len(results) == len(drawers)

    def test_query_motion_satisfying_task_request_tiago(
        self, tiago_kitchen_world, rclpy_node
    ):
        """Motion querying for open task using Tiago robot in the kitchen world (doors only)."""
        world = tiago_kitchen_world
        viz = VizMarkerPublisher(_world=world, node=rclpy_node)
        viz.with_tf_publisher()

        effects, motions, open_task, _, _ = _extend_world(world, only_doors=True)

        task_sym = variable(TaskRequest, domain=[open_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions)

        robot = Tiago.from_world(world)
        query = an(
            set_of(task_sym, motion_sym, effect_sym).where(
                ContainerSatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
                ContainerCanPerform(motion=motion_sym, robot=robot),
            )
        )

        results = list(query.evaluate())
        _present_results(results, robot)

    def test_query_task_and_effect_satisfying_motion_pr2(
        self, mutable_model_world, rclpy_node
    ):
        """Given a fixed motion on the first drawer, query recovers task and effect using PR2."""
        world = mutable_model_world
        viz = VizMarkerPublisher(_world=world, node=rclpy_node)
        viz.with_tf_publisher()

        effects, _, open_task, close_task, drawers = _extend_world(world)

        motion = Motion(
            trajectory=[0.0, 0.1, 0.2, 0.3, 0.4],
            actuator=drawers[0].root.parent_connection,
        )
        task_sym = variable(TaskRequest, domain=[open_task, close_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=[motion])

        robot = PR2.from_world(world)
        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                ContainerSatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
                ContainerCanPerform(motion=motion_sym, robot=robot),
            )
        )
        results = list(query.evaluate())
        assert len(results) > 0
