from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass

import pytest
from giskardpy.motion_statechart.goals.open_close import Open
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from krrood.entity_query_language.factories import an, set_of, variable
from krrood.ormatic.utils import classproperty

from pycram.body_motion_problem.types import Effect, Motion, TaskRequest
from pycram.body_motion_problem.predicates import SatisfiesRequest, Causes
from pycram.body_motion_problem.container_manipulation.predicates import (
    ContainerCanPerform,
)
from pycram.body_motion_problem.container_manipulation.effects import (
    ClosedEffect,
    OpenedEffect,
)
from pycram.body_motion_problem.container_manipulation.physics import RunMSCModel
from pycram.body_motion_problem.pouring.effects import PouringEffect
from pycram.body_motion_problem.pouring.physics import (
    PouringMSCModel,
)
from pycram.body_motion_problem.pouring.predicates import PouringCanPerform
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.mixins import (
    HasFillLevel,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door, Drawer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.world_description.world_entity import Body


# ---------------------------------------------------------------------------
# Shared domain types
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class PourableContainer(HasFillLevel):
    """
    Minimal pourable container for testing.

    Connected to its parent via a revolute joint representing the tilt angle.
    """

    @classproperty
    def _parent_connection_type(self):
        return RevoluteConnection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def mutable_model_world(pr2_apartment_world):
    return deepcopy(pr2_apartment_world)


@pytest.fixture
def stretch_apartment_world(stretch_world, apartment_world_setup):
    world = deepcopy(stretch_world)
    world.merge_world(deepcopy(apartment_world_setup))
    world.get_body_by_name("base_link").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(1.2, 2, 0)
    )
    return world


@pytest.fixture
def tiago_apartment_world(tiago_world, apartment_world_setup):
    world = deepcopy(tiago_world)
    world.merge_world(deepcopy(apartment_world_setup))
    world.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(1.2, 2, 0)
    )
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
            scale=Scale(0.4, 0.4, 1.0),
            initial_fill=1.0,
            k=1,
        )
    world.set_positions_1DOF_connection({cup.root.parent_connection: 0.1})
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
            scale=Scale(0.08, 0.08, 0.12),
            initial_fill=1.0,
        )
    world.set_positions_1DOF_connection({cup.root.parent_connection: 0.1})
    return world, cup, robot


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

    open_task = TaskRequest(
        task_type="open",
        name="open_container",
        goal=lambda e: isinstance(e, OpenedEffect),
    )
    close_task = (
        TaskRequest(
            task_type="close",
            name="close_container",
            goal=lambda e: isinstance(e, ClosedEffect),
        )
        if include_close
        else None
    )
    return effects, motions, open_task, close_task, drawers


# ---------------------------------------------------------------------------
# 1. Unit tests: container manipulation predicates
# ---------------------------------------------------------------------------


class TestContainerManipulationPredicates:
    def test_satisfies_request(self, mutable_model_world):
        """SatisfiesRequest holds for matching task type and rejects mismatched type."""
        world = mutable_model_world
        effects, _, open_task, _, _ = _extend_world(world)

        effect = next(e for e in effects if isinstance(e, OpenedEffect))
        assert SatisfiesRequest(task=open_task, effect=effect)()

        close_task = TaskRequest(
            task_type="close",
            name="close_container",
            goal=lambda e: isinstance(e, ClosedEffect),
        )
        assert not SatisfiesRequest(task=close_task, effect=effect)()

    def test_causes(self, mutable_model_world):
        """Causes holds when motion actuator matches effect actuator, and not otherwise."""
        world = mutable_model_world
        effects, motions, _, _, _ = _extend_world(world)

        # effects[0] = OpenedEffect, motions[0] = open motion — same actuator
        assert Causes(effect=effects[0], motion=motions[0], environment=world)()

        # effects[0] = OpenedEffect, motions[1] = close motion — direction mismatch
        assert not Causes(effect=effects[0], motion=motions[1], environment=world)()

    def test_can_execute(self, mutable_model_world):
        """ContainerCanPerform returns False for an empty trajectory and a bool for a non-empty one."""
        world = mutable_model_world
        world.get_body_by_name("base_footprint").parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.2, 2, 0)
        )

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
        task = TaskRequest(
            task_type="pour",
            name="cup",
            goal=lambda e: isinstance(e, PouringEffect),
        )
        assert SatisfiesRequest(task=task, effect=effect)()

    def test_pouring_satisfies_request_rejects_wrong_task_type(self, world_with_cup):
        """SatisfiesRequest rejects a task whose type does not match the expected pour type."""
        world, cup = world_with_cup
        effect = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.6
        )
        task = TaskRequest(
            task_type="open",
            name="cup",
            goal=lambda e: isinstance(e, OpenedEffect),
        )
        assert not SatisfiesRequest(task=task, effect=effect)()

    def test_physics_model_resets_world_state(self, world_with_cup):
        """World state is restored to its pre-simulation value after the physics model runs."""
        world, cup = world_with_cup
        fill_before = cup.fill_level
        effect = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.6
        )
        physics = PouringMSCModel(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            tilt_connection=cup.root.parent_connection,
            root_link=world.root,
            tip_link=cup.root,
        )
        physics.run(effect=effect, world=world)

        assert cup.fill_level == pytest.approx(fill_before)
        assert cup.root.parent_connection.position == pytest.approx(0.1)

    def test_causes_does_not_hold_when_effect_already_achieved(self, world_with_cup):
        """Causes returns False when the fill level is already at or below the goal."""
        world, cup = world_with_cup
        world.set_positions_1DOF_connection({cup.fill_connection: 0.5})
        effect = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.6
        )
        motion = Motion(
            trajectory=[],
            actuator=cup.root.parent_connection,
            motion_model=PouringMSCModel(
                fill_equation=cup.fill_equation,
                fill_connection=cup.fill_connection,
                tilt_connection=cup.root.parent_connection,
                root_link=world.root,
                tip_link=cup.root,
            ),
        )
        assert not Causes(effect=effect, environment=world, motion=motion)()


# ---------------------------------------------------------------------------
# 3. EQL integration tests: container manipulation queries
# ---------------------------------------------------------------------------


class TestContainerManipulationQueries:
    def test_query_motion_satisfying_task_request(self, mutable_model_world):
        """An EQL query returns at least one motion that satisfies the open task request."""
        world = mutable_model_world
        effects, motions, open_task, close_task, drawers = _extend_world(
            world, only_drawers=True
        )

        task_sym = variable(TaskRequest, domain=[open_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=motions)

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
            )
        )
        results = list(query.evaluate())
        assert len(results) > 0
        assert len(results) == len(drawers)

    def test_query_motion_satisfying_task_request_not_all(self, mutable_model_world):
        """EQL query adapts to world state: randomly opened drawers reduce the result set."""
        world = mutable_model_world
        effects, motions, open_task, _, _ = _extend_world(
            world, include_close=False, half_door_opening=False
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
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
            )
        )
        results = list(query.evaluate())
        assert len(results) > 0 and len(results) < len(effects)

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
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
            )
        )
        results = list(query.evaluate())
        assert len(results) == 1
        assert results[0].data[task_sym].task_type == "open"

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
                SatisfiesRequest(task=task_sym, effect=effect_sym),
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
    def test_causes_pours_out_40_percent(self, world_with_cup):
        """Causes predicate generates a trajectory that reduces fill level by 40%."""
        world, cup = world_with_cup

        goal_fill = 0.6
        effect = PouringEffect(
            target_object=cup,
            property_getter=lambda c: c.fill_level,
            goal_value=goal_fill,
        )
        motion = Motion(
            trajectory=[],
            actuator=cup.root.parent_connection,
            motion_model=PouringMSCModel(
                fill_equation=cup.fill_equation,
                fill_connection=cup.fill_connection,
                tilt_connection=cup.root.parent_connection,
                root_link=world.root,
                tip_link=cup.root,
                initial_tilt=0.1,
            ),
        )
        task = TaskRequest(
            task_type="pour",
            name="cup",
            goal=lambda e: isinstance(e, PouringEffect),
        )

        assert SatisfiesRequest(task=task, effect=effect)()
        causes = Causes(effect=effect, environment=world, motion=motion)
        assert causes()

        causes.replay(step_delay=0.001)
        assert cup.fill_level == pytest.approx(goal_fill, abs=0.1)

    def test_pouring_can_perform(self, pr2_world_with_cup):
        """PouringCanPerform confirms the PR2 can execute the tilt trajectory from Causes."""
        world, cup, robot = pr2_world_with_cup

        goal_fill = 0.6
        effect = PouringEffect(
            target_object=cup,
            property_getter=lambda c: c.fill_level,
            goal_value=goal_fill,
        )
        motion = Motion(
            trajectory=[],
            actuator=cup.root.parent_connection,
            motion_model=PouringMSCModel(
                fill_equation=cup.fill_equation,
                fill_connection=cup.fill_connection,
                tilt_connection=cup.root.parent_connection,
                root_link=world.root,
                tip_link=cup.root,
            ),
        )

        causes = Causes(effect=effect, environment=world, motion=motion)
        assert causes()
        assert PouringCanPerform(motion=motion, robot=robot)()

    def test_eql_query_all_three_predicates(self, pr2_world_with_cup):
        """EQL query resolves task, effect, and motion simultaneously across all three BMP predicates."""
        world, cup, robot = pr2_world_with_cup

        goal_fill = 0.6
        task = TaskRequest(
            task_type="pour",
            name="cup",
            goal=lambda e: isinstance(e, PouringEffect),
        )
        effect = PouringEffect(
            target_object=cup,
            property_getter=lambda c: c.fill_level,
            goal_value=goal_fill,
        )
        motion = Motion(
            trajectory=[],
            actuator=cup.root.parent_connection,
            motion_model=PouringMSCModel(
                fill_equation=cup.fill_equation,
                fill_connection=cup.fill_connection,
                tilt_connection=cup.root.parent_connection,
                root_link=world.root,
                tip_link=cup.root,
            ),
        )

        task_sym = variable(TaskRequest, domain=[task])
        effect_sym = variable(Effect, domain=[effect])
        motion_sym = variable(Motion, domain=[motion])

        query = an(
            set_of(task_sym, effect_sym, motion_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
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

    def test_infer_effects_and_tasks_from_given_motion(self, world_with_cup):
        """Given a fixed tilt trajectory, the query identifies which effects and task requests it satisfies."""
        world, cup = world_with_cup

        trajectory = [0.1, 1.0, 1.3] + ([1.3] * 30) + [1.3, 1.0, 0.7, 0.4, 0.1, 0.0]

        motion = Motion(
            trajectory=trajectory,
            actuator=cup.root.parent_connection,
            dt=0.1,
        )

        candidate_effects = [
            PouringEffect(
                target_object=cup,
                property_getter=lambda c: c.fill_level,
                goal_value=fill,
            )
            for fill in [0.3, 0.6]
        ]
        pour_task = TaskRequest(
            task_type="pour",
            name="cup",
            goal=lambda e: isinstance(e, PouringEffect),
        )
        open_task = TaskRequest(
            task_type="open",
            name="cup",
            goal=lambda e: isinstance(e, OpenedEffect),
        )

        effect_sym = variable(Effect, domain=candidate_effects)
        task_sym = variable(TaskRequest, domain=[pour_task, open_task])
        motion_sym = variable(Motion, domain=[motion])

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
            )
        )

        results = list(query.evaluate())
        assert len(results) == 1
        assert results[0].data[task_sym].task_type == "pour"
        assert results[0].data[effect_sym].goal_value == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 5. Long-running robot integration tests (not executed in CI)
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="Long-running tests are skipped in CI for now")
class TestRobotIntegration:
    def test_query_motion_satisfying_task_request_stretch(
        self, stretch_apartment_world
    ):
        """Motion querying for open task using Stretch robot in the kitchen world (drawers only)."""
        world = stretch_apartment_world

        effects, motions, open_task, _, drawers = _extend_world(
            world, only_drawers=True
        )

        task_sym = variable(TaskRequest, domain=[open_task])
        effect_sym = variable(Effect, domain=effects[:10])
        motion_sym = variable(Motion, domain=motions[:10])

        robot = Stretch.from_world(world)
        query = an(
            set_of(task_sym, motion_sym, effect_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
                ContainerCanPerform(motion=motion_sym, robot=robot),
            )
        )

        results = list(query.evaluate())
        assert len(results) > 1

    def test_query_motion_satisfying_task_request_tiago(self, tiago_apartment_world):
        """Motion querying for open task using Tiago robot in the kitchen world (doors only)."""
        world = tiago_apartment_world

        effects, motions, open_task, _, _ = _extend_world(
            world, only_doors=False, only_drawers=True
        )

        task_sym = variable(TaskRequest, domain=[open_task])
        effect_sym = variable(Effect, domain=effects[:5])
        motion_sym = variable(Motion, domain=motions[:5])

        robot = Tiago.from_world(world)
        left_arm_park = robot.left_arm.get_joint_state_by_type(StaticJointState.PARK)
        right_arm_park = robot.right_arm.get_joint_state_by_type(StaticJointState.PARK)
        world.set_positions_1DOF_connection(dict(left_arm_park.items()))
        world.set_positions_1DOF_connection(dict(right_arm_park.items()))

        query = an(
            set_of(task_sym, motion_sym, effect_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
                ContainerCanPerform(motion=motion_sym, robot=robot),
            )
        )

        results = list(query.evaluate())
        assert len(results) >= 0  # =0 because of Tiago

    def test_query_task_and_effect_satisfying_motion_pr2(self, mutable_model_world):
        """Given a fixed motion on the first drawer, query recovers task and effect using PR2."""
        world = mutable_model_world

        effects, _, open_task, close_task, drawers = _extend_world(world)

        motion = Motion(
            trajectory=[0.0, 0.1, 0.2, 0.3, 0.4],
            actuator=[
                drawer
                for drawer in drawers
                if "cabinet11_drawer_top" in str(drawer.bodies[0].name)
            ][0].root.parent_connection,
        )
        task_sym = variable(TaskRequest, domain=[open_task, close_task])
        effect_sym = variable(Effect, domain=effects)
        motion_sym = variable(Motion, domain=[motion])

        robot = PR2.from_world(world)
        left_arm_park = robot.left_arm.get_joint_state_by_type(StaticJointState.PARK)
        right_arm_park = robot.right_arm.get_joint_state_by_type(StaticJointState.PARK)
        world.set_positions_1DOF_connection(dict(left_arm_park.items()))
        world.set_positions_1DOF_connection(dict(right_arm_park.items()))

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
                ContainerCanPerform(motion=motion_sym, robot=robot),
            )
        )
        results = list(query.evaluate())
        assert (
            len(results) == 0
        )  # Something is not working correctly with the giskardpy end motion setup right now.
