# %% imports
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pytest

from krrood.entity_query_language.factories import an, set_of, variable
from krrood.entity_query_language.operators.core_logical_operators import Not
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression

from giskardpy.body_motion_problem.container_physics import (
    ContainerManipulationPhysicsModel,
)
from giskardpy.body_motion_problem.pouring_physics import PouringMSCModel
from giskardpy.data_types.exceptions import DegreeOfFreedomNotRecordedError
from coraplex.body_motion_problem.predicates import MotionStatechartCanPerform
from coraplex.exceptions import UntrackedMotionConnection
from semantic_digital_twin.reasoning.bmp_predicates import Causes, SatisfiesRequest
from semantic_digital_twin.semantic_annotations.effects import (
    ClosedEffect,
    OpenedEffect,
    PouringEffect,
)
from semantic_digital_twin.world_description.effects import (
    Effect,
    TaskRequest,
    TaskType,
)
from semantic_digital_twin.world_description.motion import Motion, MotionTrajectory
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door, Drawer
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Vector3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.world_description.world_entity import Body, Region

from ..giskardpy_test.test_motion_statechart.single_cup_world import (
    PourableContainer,
    build_single_cup_world,
)

# %% fixtures


@pytest.fixture(scope="function")
def pr2_apartment_world_copy(pr2_apartment_world):
    """
    Function-scoped mutable copy of the session PR2 apartment world.

    Unlike the coraplex conftest fixture ``mutable_model_world``, this returns only the
    :class:`~semantic_digital_twin.world.World` copy.
    """
    return deepcopy(pr2_apartment_world)


@pytest.fixture
def stretch_apartment_world_copy(_stretch_world_setup, _apartment_world_setup):
    """
    Function-scoped mutable Stretch apartment world with the base placed at (1.2, 2, 0),
    unlike the shared session fixture of the same world.
    """
    world = deepcopy(_stretch_world_setup)
    world.merge_world(deepcopy(_apartment_world_setup))
    world.get_body_by_name("base_link").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(1.2, 2, 0)
    )
    return world


@pytest.fixture
def tiago_apartment_world_copy(_tiago_world_setup, _apartment_world_setup):
    """
    Function-scoped mutable Tiago apartment world with the base placed at (1.2, 2, 0);
    the session fixture of the same world returns a shared (world, robot) tuple instead.
    """
    world = deepcopy(_tiago_world_setup)
    world.merge_world(deepcopy(_apartment_world_setup))
    world.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(1.2, 2, 0)
    )
    return world


@pytest.fixture
def world_with_cup():
    """
    World containing a single pourable container with a tilt joint, filled to 100%.
    """
    return build_single_cup_world()


def _add_pourable_cup(
    world: World, world_root_T_cup: HomogeneousTransformationMatrix
) -> PourableContainer:
    """
    Add a small pourable cup with a tilt joint to the world at the given pose.
    """
    with world.modify_world():
        cup = PourableContainer.create_with_new_body_in_world(
            name=PrefixedName("cup"),
            world=world,
            active_axis=Vector3(0, 1, 0),
            connection_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-2.0),
                upper=DerivativeMap(position=math.pi / 2, velocity=2.0),
            ),
            world_root_T_self=world_root_T_cup,
            scale=Scale(0.08, 0.08, 0.12),
        )
    cup.initialize_fill_level(
        world=world,
        initial_fill=1.0,
    )
    world.set_positions_1DOF_connection({cup.root.parent_connection: 0.1})
    return cup


@pytest.fixture(scope="function")
def pr2_world_with_cup(pr2_world_copy):
    """
    PR2 world with a pourable cup placed within arm reach at (0.7, 0.0, 0.85).
    """
    world = pr2_world_copy
    [robot] = world.get_semantic_annotations_by_type(PR2)
    cup = _add_pourable_cup(
        world,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.7,
            y=0.0,
            z=0.85,
            reference_frame=world.root,
        ),
    )
    return world, cup, robot


# %% container scenario helpers


class ContainerSelection(Enum):
    """
    Which kinds of container annotations to build effects and motions for.
    """

    DRAWERS = auto()
    DOORS = auto()
    DRAWERS_AND_DOORS = auto()


@dataclass
class ContainerScenario:
    """
    Effects, motions, and task requests built for the containers of a world.
    """

    effects: list[Effect]
    """
    Opened (and optionally closed) effects, paired index-wise with motions.
    """

    motions: list[Motion]
    """
    Motions driving each container joint, paired index-wise with effects.
    """

    open_task: TaskRequest
    """
    Task request matched by the opened effects.
    """

    close_task: TaskRequest | None
    """
    Task request matched by the closed effects, or None when close effects were not
    built.
    """

    drawers: list[Drawer]
    """
    Drawer annotations found in the world, empty for a doors-only selection.
    """


def _build_container_manipulation_model(
    handle: Body, actuator: ActiveConnection1DOF, goal_value: float
) -> ContainerManipulationPhysicsModel:
    """
    Create a physics model that drives a container joint to goal_value.
    """
    return ContainerManipulationPhysicsModel(
        handle=handle,
        actuator=actuator,
        goal_joint_state=goal_value,
        timeout=500,
    )


def _extend_world(
    world: World,
    selection: ContainerSelection = ContainerSelection.DRAWERS_AND_DOORS,
    include_close: bool = True,
) -> ContainerScenario:
    """
    Infer semantic annotations, attach them to the world, and build matching effects,
    motions, and task requests for the selected container kinds.

    :param world: World to annotate and build the scenario for.
    :param selection: Which container annotation kinds to include.
    :param include_close: Also create closed effects/motions and a close task.
    :return: The assembled container scenario.
    """
    world_reasoner = WorldReasoner(world)
    inferred = world_reasoner.infer_semantic_annotations()
    with world.modify_world():
        world.add_semantic_annotations(inferred)

    drawers = (
        []
        if selection is ContainerSelection.DOORS
        else world.get_semantic_annotations_by_type(Drawer)
    )
    doors = (
        []
        if selection is ContainerSelection.DRAWERS
        else world.get_semantic_annotations_by_type(Door)
    )
    annotations = drawers + doors

    property_getter = lambda obj: obj.root.parent_connection.position
    effects = []
    motions = []
    for annotation in annotations:
        actuator = annotation.root.parent_connection
        upper = actuator.active_dofs[0].limits.upper.position
        effect_goal = upper * 0.5

        effects.append(
            OpenedEffect(
                target_object=annotation,
                goal_value=effect_goal,
                property_getter=property_getter,
            )
        )
        motions.append(
            Motion(
                connection=actuator,
                motion_model=_build_container_manipulation_model(
                    annotation.handle.root, actuator, upper
                ),
            )
        )

        if include_close:
            lower = actuator.active_dofs[0].limits.lower.position
            effects.append(
                ClosedEffect(
                    target_object=annotation,
                    goal_value=lower,
                    property_getter=property_getter,
                )
            )
            motions.append(
                Motion(
                    connection=actuator,
                    motion_model=_build_container_manipulation_model(
                        annotation.handle.root, actuator, lower
                    ),
                )
            )

    open_task = TaskRequest(
        task_type=TaskType.OPEN,
        name="open_container",
        goal=lambda e: isinstance(e, OpenedEffect),
    )
    close_task = (
        TaskRequest(
            task_type=TaskType.CLOSE,
            name="close_container",
            goal=lambda e: isinstance(e, ClosedEffect),
        )
        if include_close
        else None
    )
    return ContainerScenario(
        effects=effects,
        motions=motions,
        open_task=open_task,
        close_task=close_task,
        drawers=drawers,
    )


def _drawer_opening_motion(scenario: ContainerScenario) -> Motion:
    """
    :return: The open motion of the scenario's first drawer, equipped with a
             trajectory of nine evenly spaced steps up to the joint's upper limit.
    """
    motion = scenario.motions[0]
    actuator = scenario.drawers[0].root.parent_connection
    upper = actuator.active_dofs[0].limits.upper.position
    motion.motion_trajectory = MotionTrajectory(
        {actuator: [step * upper / 8 for step in range(9)]}
    )
    return motion


# %% fixed-base planning mimics


@dataclass(eq=False)
class GripperPlaceholder:
    """
    Stands in for an end effector where only its identity and collision bodies matter.
    """

    name: str
    """
    Identifier of the placeholder gripper.
    """

    bodies_with_collision: list[Body] = field(default_factory=list)
    """
    Bodies with collision geometry reported for this gripper.
    """


@dataclass(eq=False)
class FixedBaseRobot:
    """
    Mimic of a robot without a mobile base, exposing only the attributes the base-
    placement dispatch of MotionStatechartCanPerform relies on.
    """

    root: Body
    """
    Root body of the robot.
    """

    _world: World
    """
    World the robot lives in.
    """

    end_effectors: list[GripperPlaceholder]
    """
    Grippers reported by :meth:`get_end_effectors`.
    """

    def get_end_effectors(self) -> list[GripperPlaceholder]:
        """
        :return: The placeholder grippers of this robot.
        """
        return self.end_effectors


@dataclass
class ScriptedPlanningCanPerform(MotionStatechartCanPerform):
    """
    MotionStatechartCanPerform whose planning leg is scripted, recording which grippers
    were attempted instead of running the QP planner.
    """

    planning_outcome: bool = True
    """
    Result every scripted planning attempt reports.
    """

    attempted_grippers: list[GripperPlaceholder] = field(default_factory=list)
    """
    Grippers for which a planning attempt was made, in order.
    """

    def _gripper_can_follow_trajectory(
        self, gripper: GripperPlaceholder, target: Body, trajectory: list[Pose]
    ) -> bool:
        self.attempted_grippers.append(gripper)
        return self.planning_outcome


# %% unit tests: container manipulation predicates


class TestContainerManipulationPredicates:
    def test_satisfies_request(self, pr2_apartment_world_copy):
        """
        SatisfiesRequest holds for matching task type and rejects mismatched type.
        """
        world = pr2_apartment_world_copy
        scenario = _extend_world(world)

        effect = next(e for e in scenario.effects if isinstance(e, OpenedEffect))
        assert SatisfiesRequest(task=scenario.open_task, effect=effect)()

        close_task = TaskRequest(
            task_type=TaskType.CLOSE,
            name="close_container",
            goal=lambda e: isinstance(e, ClosedEffect),
        )
        assert not SatisfiesRequest(task=close_task, effect=effect)()

    def test_causes(self, pr2_apartment_world_copy):
        """
        Causes holds when motion actuator matches effect actuator, and not otherwise.
        """
        world = pr2_apartment_world_copy
        scenario = _extend_world(world)

        # effects[0] = OpenedEffect, motions[0] = open motion — same actuator
        assert Causes(
            effect=scenario.effects[0], motion=scenario.motions[0], environment=world
        )()

        # effects[0] = OpenedEffect, motions[1] = close motion — direction mismatch
        assert not Causes(
            effect=scenario.effects[0], motion=scenario.motions[1], environment=world
        )()

    def test_can_execute(self, pr2_apartment_world_copy, rclpy_node):
        """
        MotionStatechartCanPerform returns False for a missing trajectory and True for a
        reachable drawer trajectory.
        """
        world = pr2_apartment_world_copy
        world.get_body_by_name("base_footprint").parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.2, 2, 0)
        )

        scenario = _extend_world(world, ContainerSelection.DRAWERS)
        [robot] = world.get_semantic_annotations_by_type(PR2)

        assert not MotionStatechartCanPerform(motion=scenario.motions[0], robot=robot)()

        left_arm_park = robot.left_arm.get_joint_state_by_type(StaticJointState.PARK)
        right_arm_park = robot.right_arm.get_joint_state_by_type(StaticJointState.PARK)
        world.set_positions_1DOF_connection(dict(left_arm_park.items()))
        world.set_positions_1DOF_connection(dict(right_arm_park.items()))

        reachable_drawer = next(
            drawer
            for drawer in scenario.drawers
            if "cabinet11_drawer_top" in str(drawer.bodies[0].name)
        )
        actuator = reachable_drawer.root.parent_connection
        motion = Motion(
            connection=actuator,
            motion_trajectory=MotionTrajectory({actuator: [0.0, 0.1, 0.2, 0.3]}),
        )
        assert MotionStatechartCanPerform(motion=motion, robot=robot)() is True


# %% unit tests: can-perform contracts


class TestCanPerformContracts:
    """
    State-restoration, guard, and infeasibility contracts of MotionStatechartCanPerform.
    """

    def test_untracked_motion_connection_raises(self, world_with_cup):
        """
        A trajectory that tracks a different connection than the motion's must fail
        loudly.
        """
        world, cup = world_with_cup
        robot = FixedBaseRobot(
            root=cup.root,
            _world=world,
            end_effectors=[GripperPlaceholder(name="gripper")],
        )
        motion = Motion(
            connection=cup.root.parent_connection,
            motion_trajectory=MotionTrajectory({cup.fill_connection: [0.9, 0.8]}),
        )
        with pytest.raises(UntrackedMotionConnection):
            MotionStatechartCanPerform(motion=motion, robot=robot)()

    def test_collision_rules_skip_regions_in_kinematic_chain(self, world_with_cup):
        """A Region in the chain between world root and target is ignored when building allow-rules."""
        world, cup = world_with_cup
        region = Region(name=PrefixedName("handle_zone"))
        target_body = Body(
            name=PrefixedName("target"),
            collision=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.05))]),
        )
        with world.modify_world():
            world.add_connection(FixedConnection(parent=cup.root, child=region))
            world.add_connection(FixedConnection(parent=region, child=target_body))

        gripper_body = Body(
            name=PrefixedName("gripper_body"),
            collision=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.05))]),
        )
        gripper = GripperPlaceholder(
            name="gripper", bodies_with_collision=[gripper_body]
        )
        predicate = MotionStatechartCanPerform(motion=None, robot=None)

        [rule] = predicate._build_collision_rules(gripper, target_body)
        assert rule.body_group_a == [gripper_body]
        assert rule.body_group_b == [cup.root, target_body]

    @pytest.mark.slow
    def test_state_is_restored_after_call(self, pr2_apartment_world_copy, rclpy_node):
        """
        Temporary rules, the collision matrix, and the base origin are unchanged by a
        MotionStatechartCanPerform call.
        """
        world = pr2_apartment_world_copy
        world.get_body_by_name("base_footprint").parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.2, 2, 0)
        )
        scenario = _extend_world(world, ContainerSelection.DRAWERS)
        [robot] = world.get_semantic_annotations_by_type(PR2)
        motion = _drawer_opening_motion(scenario)

        collision_manager = world.collision_manager
        collision_manager.update_collision_matrix()
        temporary_rules_before = list(collision_manager.temporary_rules)
        collision_matrix_before = collision_manager.collision_matrix
        world_T_base_before = robot.root.parent_connection.origin.to_np().copy()

        MotionStatechartCanPerform(motion=motion, robot=robot)()

        assert collision_manager.temporary_rules == temporary_rules_before
        assert collision_manager.collision_matrix == collision_matrix_before
        assert np.allclose(
            robot.root.parent_connection.origin.to_np(), world_T_base_before
        )

    @pytest.mark.slow
    def test_infeasible_motion_is_false(self, pr2_world_copy):
        """
        The predicate reports infeasibility when the cup floats far above the robot's
        workspace.
        """
        world = pr2_world_copy
        [robot] = world.get_semantic_annotations_by_type(PR2)
        cup = _add_pourable_cup(
            world,
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.7,
                y=0.0,
                z=2.5,
                reference_frame=world.root,
            ),
        )
        tilt_connection = cup.root.parent_connection
        motion = Motion(
            connection=tilt_connection,
            motion_trajectory=MotionTrajectory({tilt_connection: [0.1, 0.3, 0.5]}),
        )
        assert MotionStatechartCanPerform(motion=motion, robot=robot)() is False


class TestFixedBasePlacementDispatch:
    """
    Covers the fixed-base branch of MotionStatechartCanPerform._execute_for_any_gripper.
    """

    @staticmethod
    def _scripted_predicate(
        world: World, cup: PourableContainer, planning_outcome: bool
    ) -> ScriptedPlanningCanPerform:
        robot = FixedBaseRobot(
            root=cup.root,
            _world=world,
            end_effectors=[GripperPlaceholder(name="gripper")],
        )
        motion = Motion(
            connection=cup.root.parent_connection,
            motion_trajectory=MotionTrajectory(
                {cup.root.parent_connection: [0.1, 0.2, 0.3]}
            ),
        )
        return ScriptedPlanningCanPerform(
            motion=motion, robot=robot, planning_outcome=planning_outcome
        )

    def test_reachable_trajectory_is_true_from_current_placement(self, world_with_cup):
        """
        A fixed-base robot is tested once per gripper from its current placement.
        """
        world, cup = world_with_cup
        predicate = self._scripted_predicate(world, cup, planning_outcome=True)
        assert predicate() is True
        assert predicate.attempted_grippers == predicate.robot.get_end_effectors()

    def test_infeasible_planning_is_false_after_trying_every_gripper(
        self, world_with_cup
    ):
        """
        A fixed-base robot whose planning attempts all fail yields False.
        """
        world, cup = world_with_cup
        predicate = self._scripted_predicate(world, cup, planning_outcome=False)
        assert predicate() is False
        assert predicate.attempted_grippers == predicate.robot.get_end_effectors()

    def test_empty_trajectory_raises_before_dispatch(self, world_with_cup):
        """
        An untracked connection raises instead of silently reporting feasibility.
        """
        world, cup = world_with_cup
        predicate = self._scripted_predicate(world, cup, planning_outcome=True)
        predicate.motion.motion_trajectory = MotionTrajectory(
            {cup.fill_connection: [0.9, 0.8]}
        )
        with pytest.raises(UntrackedMotionConnection):
            predicate()
        assert predicate.attempted_grippers == []


# %% unit tests: trajectory subsampling and costmap sampling


class TestTrajectorySubsampling:
    """
    Covers the waypoint cap of MotionStatechartCanPerform._subsample_trajectory.
    """

    @pytest.mark.parametrize("length", [21, 39, 200])
    def test_subsample_respects_waypoint_cap(self, length: int):
        """
        Any over-long trajectory is reduced to at most the waypoint cap, keeping both
        endpoints.
        """
        predicate = MotionStatechartCanPerform(motion=None, robot=None)
        trajectory = [
            HomogeneousTransformationMatrix.from_xyz_rpy(x=step * 0.01).to_pose()
            for step in range(length)
        ]
        subsampled = predicate._subsample_trajectory(trajectory)
        assert len(subsampled) <= predicate._max_trajectory_waypoints
        assert subsampled[0] is trajectory[0]
        assert subsampled[-1] is trajectory[-1]

    def test_short_trajectory_is_returned_unchanged(self):
        """
        A trajectory within the cap must not lose any waypoint.
        """
        predicate = MotionStatechartCanPerform(motion=None, robot=None)
        trajectory = [
            HomogeneousTransformationMatrix.from_xyz_rpy(x=step * 0.01).to_pose()
            for step in range(5)
        ]
        assert predicate._subsample_trajectory(trajectory) == trajectory


class TestCostmapSampleScaling:
    """
    Covers MotionStatechartCanPerform._scaled_number_of_samples.
    """

    def test_few_segments_keep_configured_sample_count(self):
        """
        With fewer segments than samples, the configured sample count is kept.
        """
        predicate = MotionStatechartCanPerform(motion=None, robot=None)
        assert predicate._scaled_number_of_samples(3) == predicate._costmap_samples

    def test_more_segments_than_samples_get_one_sample_each(self):
        """
        With more segments than samples, the count grows to one sample per segment.
        """
        predicate = MotionStatechartCanPerform(motion=None, robot=None)
        number_of_segments = predicate._costmap_samples + 15
        assert (
            predicate._scaled_number_of_samples(number_of_segments)
            == number_of_segments
        )


# %% unit tests: pouring predicates and physics model


class TestPouringPredicates:
    def test_pouring_satisfies_request(self, world_with_cup):
        """
        SatisfiesRequest holds for a pour task paired with a PouringEffect.
        """
        world, cup = world_with_cup
        effect = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.6
        )
        task = TaskRequest(
            task_type=TaskType.POUR,
            name="cup",
            goal=lambda e: isinstance(e, PouringEffect),
        )
        assert SatisfiesRequest(task=task, effect=effect)()

    def test_pouring_satisfies_request_rejects_wrong_task_type(self, world_with_cup):
        """
        SatisfiesRequest rejects a task whose type does not match the expected pour
        type.
        """
        world, cup = world_with_cup
        effect = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.6
        )
        task = TaskRequest(
            task_type=TaskType.OPEN,
            name="cup",
            goal=lambda e: isinstance(e, OpenedEffect),
        )
        assert not SatisfiesRequest(task=task, effect=effect)()

    def test_physics_model_resets_world_state(self, world_with_cup):
        """
        World state is restored to its pre-simulation value after the physics model
        runs.
        """
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

    def test_run_reports_convergence(self, world_with_cup):
        """
        The returned trajectory records whether the statechart reached its end
        condition.
        """
        world, cup = world_with_cup
        physics_arguments = dict(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            tilt_connection=cup.root.parent_connection,
            root_link=world.root,
            tip_link=cup.root,
        )

        achievable = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.6
        )
        trajectory = PouringMSCModel(**physics_arguments).run(
            effect=achievable, world=world
        )
        assert trajectory.converged is True

        out_of_tick_budget = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.2
        )
        truncated = PouringMSCModel(**physics_arguments, timeout=5).run(
            effect=out_of_tick_budget, world=world
        )
        assert truncated.converged is False

    def test_extracting_unrecorded_dof_raises(self, world_with_cup, pr2_world_with_cup):
        """
        Asking for a degree of freedom the simulation never recorded must fail loudly.
        """
        world, cup = world_with_cup
        _, foreign_cup, _ = pr2_world_with_cup
        physics = PouringMSCModel(
            fill_equation=cup.fill_equation,
            fill_connection=cup.fill_connection,
            tilt_connection=cup.root.parent_connection,
            root_link=world.root,
            tip_link=cup.root,
            timeout=30,
        )
        effect = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.6
        )
        physics.run(effect=effect, world=world)
        with pytest.raises(DegreeOfFreedomNotRecordedError):
            physics._extract_dof_positions(
                physics._recorded_trajectory, foreign_cup.fill_connection
            )

    def test_causes_does_not_hold_when_effect_already_achieved(self, world_with_cup):
        """
        Causes returns False when the fill level is already at or below the goal.
        """
        world, cup = world_with_cup
        world.set_positions_1DOF_connection({cup.fill_connection: 0.5})
        effect = PouringEffect(
            target_object=cup, property_getter=lambda c: c.fill_level, goal_value=0.6
        )
        motion = Motion(
            connection=cup.root.parent_connection,
            motion_model=PouringMSCModel(
                fill_equation=cup.fill_equation,
                fill_connection=cup.fill_connection,
                tilt_connection=cup.root.parent_connection,
                root_link=world.root,
                tip_link=cup.root,
            ),
        )
        assert not Causes(effect=effect, environment=world, motion=motion)()

    @pytest.mark.slow
    def test_pouring_can_perform(self, pr2_world_with_cup):
        """
        MotionStatechartCanPerform confirms the PR2 can execute the tilt trajectory from
        Causes.
        """
        world, cup, robot = pr2_world_with_cup

        goal_fill = 0.6
        effect = PouringEffect(
            target_object=cup,
            property_getter=lambda c: c.fill_level,
            goal_value=goal_fill,
        )
        motion = Motion(
            connection=cup.root.parent_connection,
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
        assert MotionStatechartCanPerform(motion=motion, robot=robot)()


# %% integration tests: container manipulation queries


class TestContainerManipulationQueries:
    def test_query_motion_satisfying_task_request(self, pr2_apartment_world_copy):
        """
        An EQL query returns exactly one motion per drawer for the open task request.
        """
        world = pr2_apartment_world_copy
        scenario = _extend_world(world, ContainerSelection.DRAWERS)

        task_sym = variable(TaskRequest, domain=[scenario.open_task])
        effect_sym = variable(Effect, domain=scenario.effects)
        motion_sym = variable(Motion, domain=scenario.motions)

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
            )
        )
        results = list(query.evaluate())
        assert len(results) == len(scenario.drawers)

    def test_query_motion_satisfying_task_request_not_all(
        self, pr2_apartment_world_copy
    ):
        """
        EQL query adapts to world state: already-opened drawers drop out of the result
        set.
        """
        world = pr2_apartment_world_copy
        scenario = _extend_world(world, include_close=False)

        opened_drawers = scenario.drawers[:4]
        for drawer in opened_drawers:
            drawer.root.parent_connection.position = (
                drawer.root.parent_connection.active_dofs[0].limits.upper.position
            )
        world.notify_state_change()

        task_sym = variable(TaskRequest, domain=[scenario.open_task])
        effect_sym = variable(Effect, domain=scenario.effects)
        motion_sym = variable(Motion, domain=scenario.motions)

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
            )
        )
        results = list(query.evaluate())
        assert len(results) == len(scenario.effects) - len(opened_drawers)

    def test_query_task_and_effect_satisfying_motion(self, pr2_apartment_world_copy):
        """
        Given a fixed motion, the EQL query recovers the matching task and effect.
        """
        world = pr2_apartment_world_copy
        scenario = _extend_world(world)

        actuator = scenario.drawers[0].root.parent_connection
        motion = Motion(
            connection=actuator,
            motion_trajectory=MotionTrajectory({actuator: [0.0, 0.1, 0.2, 0.3, 0.4]}),
        )
        task_sym = variable(
            TaskRequest, domain=[scenario.open_task, scenario.close_task]
        )
        effect_sym = variable(Effect, domain=scenario.effects)
        motion_sym = variable(Motion, domain=[motion])

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
            )
        )
        results = list(query.evaluate())
        assert len(results) == 1
        assert results[0].data[task_sym].task_type == TaskType.OPEN

    def test_query_motion_if_drawers_open(self, pr2_apartment_world_copy):
        """
        Query results switch from open to close tasks when all drawers are moved to open
        position.
        """
        world = pr2_apartment_world_copy
        scenario = _extend_world(world, ContainerSelection.DRAWERS)
        drawers = scenario.drawers

        task_sym = variable(
            TaskRequest, domain=[scenario.open_task, scenario.close_task]
        )
        effect_sym = variable(Effect, domain=scenario.effects)
        motion_sym = variable(Motion, domain=scenario.motions)

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
            )
        )

        results = list(query.evaluate())
        assert len(results) == len(drawers)
        assert all(
            result.data[task_sym].task_type == TaskType.OPEN for result in results
        )

        for drawer in drawers:
            drawer.root.parent_connection.position = (
                drawer.root.parent_connection.active_dofs[0].limits.upper.position
            )
        world.notify_state_change()

        results = list(query.evaluate())
        assert len(results) == len(drawers)
        assert all(
            result.data[task_sym].task_type == TaskType.CLOSE for result in results
        )


# %% integration tests: pouring queries


class TestPouringQueries:
    def test_causes_pours_out_40_percent(self, world_with_cup):
        """
        Causes predicate generates a trajectory that reduces fill level by 40%.
        """
        world, cup = world_with_cup

        goal_fill = 0.6
        effect = PouringEffect(
            target_object=cup,
            property_getter=lambda c: c.fill_level,
            goal_value=goal_fill,
        )
        motion = Motion(
            connection=cup.root.parent_connection,
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
            task_type=TaskType.POUR,
            name="cup",
            goal=lambda e: isinstance(e, PouringEffect),
        )

        assert SatisfiesRequest(task=task, effect=effect)()
        causes = Causes(effect=effect, environment=world, motion=motion)
        assert causes()

        causes.replay(step_delay=0.001)
        assert cup.fill_level < goal_fill

    @pytest.mark.slow
    def test_eql_query_all_three_predicates(self, pr2_world_with_cup):
        """
        EQL query resolves task, effect, and motion simultaneously across all three BMP
        predicates.
        """
        world, cup, robot = pr2_world_with_cup

        goal_fill = 0.6
        task = TaskRequest(
            task_type=TaskType.POUR,
            name="cup",
            goal=lambda e: isinstance(e, PouringEffect),
        )
        effect = PouringEffect(
            target_object=cup,
            property_getter=lambda c: c.fill_level,
            goal_value=goal_fill,
        )
        motion = Motion(
            connection=cup.root.parent_connection,
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
                MotionStatechartCanPerform(motion=motion_sym, robot=robot),
            )
        )

        results = list(query.evaluate())
        assert len(results) == 1
        result = results[0]
        assert result.data[task_sym].task_type == TaskType.POUR
        assert result.data[effect_sym].goal_value == goal_fill
        assert not result.data[motion_sym].motion_trajectory.is_empty()

    def test_infer_effects_and_tasks_from_given_motion(self, world_with_cup):
        """
        Given a fixed tilt trajectory, the query identifies which effects and task
        requests it satisfies.
        """
        world, cup = world_with_cup

        tilt_positions = [0.1, 1.0, 1.3] + ([1.3] * 30) + [1.3, 1.0, 0.7, 0.4, 0.1, 0.0]
        actuator = cup.root.parent_connection
        motion = Motion(
            connection=actuator,
            motion_trajectory=MotionTrajectory({actuator: tilt_positions}),
            time_step=0.1,
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
            task_type=TaskType.POUR,
            name="cup",
            goal=lambda e: isinstance(e, PouringEffect),
        )
        open_task = TaskRequest(
            task_type=TaskType.OPEN,
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
        assert results[0].data[task_sym].task_type == TaskType.POUR
        assert results[0].data[effect_sym].goal_value == pytest.approx(0.6)


# %% integration tests: long-running robot planning


@pytest.mark.slow
class TestRobotIntegration:
    def test_query_motion_satisfying_task_request_stretch(
        self, stretch_apartment_world_copy, rclpy_node
    ):
        """
        Motion querying for open task using Stretch robot in the apartment world
        (drawers only).
        """
        world = stretch_apartment_world_copy
        scenario = _extend_world(world, ContainerSelection.DRAWERS)

        task_sym = variable(TaskRequest, domain=[scenario.open_task])
        effect_sym = variable(Effect, domain=scenario.effects[:5])
        motion_sym = variable(Motion, domain=scenario.motions[:5])

        [robot] = world.get_semantic_annotations_by_type(Stretch)
        query = an(
            set_of(task_sym, motion_sym, effect_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
                MotionStatechartCanPerform(motion=motion_sym, robot=robot),
            )
        )

        results = list(query.evaluate())
        assert len(results) >= 1

    def test_query_motion_satisfying_task_request_tiago(
        self, tiago_apartment_world_copy, rclpy_node
    ):
        """
        Motion querying for open task using Tiago robot in the apartment world.
        """
        world = tiago_apartment_world_copy
        scenario = _extend_world(world, ContainerSelection.DRAWERS)

        task_sym = variable(TaskRequest, domain=[scenario.open_task])
        effect_sym = variable(Effect, domain=scenario.effects[:5])
        motion_sym = variable(Motion, domain=scenario.motions[:5])

        [robot] = world.get_semantic_annotations_by_type(Tiago)
        left_arm_park = robot.get_left_arm_if_specified().get_joint_state_by_type(
            StaticJointState.PARK
        )
        right_arm_park = robot.get_right_arm_if_specified().get_joint_state_by_type(
            StaticJointState.PARK
        )
        world.set_positions_1DOF_connection(dict(left_arm_park.items()))
        world.set_positions_1DOF_connection(dict(right_arm_park.items()))

        query = an(
            set_of(task_sym, motion_sym, effect_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
                MotionStatechartCanPerform(motion=motion_sym, robot=robot),
            )
        )

        results = list(query.evaluate())
        assert len(results) >= 1

    def test_query_task_and_effect_satisfying_motion_pr2(
        self, pr2_apartment_world_copy, rclpy_node
    ):
        """
        Given a fixed motion on the first drawer, query recovers task and effect using
        PR2.
        """
        world = pr2_apartment_world_copy
        scenario = _extend_world(world)

        actuator = [
            drawer
            for drawer in scenario.drawers
            if "cabinet11_drawer_top" in str(drawer.bodies[0].name)
        ][0].root.parent_connection
        motion = Motion(
            connection=actuator,
            motion_trajectory=MotionTrajectory({actuator: [0.0, 0.1, 0.2, 0.3]}),
        )
        task_sym = variable(
            TaskRequest, domain=[scenario.open_task, scenario.close_task]
        )
        effect_sym = variable(Effect, domain=scenario.effects)
        motion_sym = variable(Motion, domain=[motion])

        [robot] = world.get_semantic_annotations_by_type(PR2)
        left_arm_park = robot.left_arm.get_joint_state_by_type(StaticJointState.PARK)
        right_arm_park = robot.right_arm.get_joint_state_by_type(StaticJointState.PARK)
        world.set_positions_1DOF_connection(dict(left_arm_park.items()))
        world.set_positions_1DOF_connection(dict(right_arm_park.items()))

        query = an(
            set_of(motion_sym, effect_sym, task_sym).where(
                SatisfiesRequest(task=task_sym, effect=effect_sym),
                Causes(effect=effect_sym, motion=motion_sym, environment=world),
                MotionStatechartCanPerform(motion=motion_sym, robot=robot),
            )
        )
        results = list(query.evaluate())
        assert len(results) >= 1


# %% verbalization of the body-motion-problem predicates


class TestBodyMotionProblemVerbalization:
    def test_predicate_clauses(self):
        """
        Each body-motion-problem predicate verbalizes as its affirmative present-tense
        clause, and a wrapping ``Not`` negates it inline with do-support.
        """
        task = variable(TaskRequest, [])
        effect = variable(Effect, [])
        motion = variable(Motion, [])
        robot = variable(AbstractRobot, [])
        world = World()

        assert (
            verbalize_expression(SatisfiesRequest(task=task, effect=effect))
            == "an Effect satisfies a TaskRequest"
        )
        assert (
            verbalize_expression(Not(SatisfiesRequest(task=task, effect=effect)))
            == "an Effect does not satisfy a TaskRequest"
        )
        assert (
            verbalize_expression(
                Causes(effect=effect, motion=motion, environment=world)
            )
            == "a Motion causes an Effect"
        )
        assert (
            verbalize_expression(
                Not(Causes(effect=effect, motion=motion, environment=world))
            )
            == "a Motion does not cause an Effect"
        )
        assert (
            verbalize_expression(MotionStatechartCanPerform(motion=motion, robot=robot))
            == "an AbstractRobot performs a Motion"
        )
        assert (
            verbalize_expression(
                Not(MotionStatechartCanPerform(motion=motion, robot=robot))
            )
            == "an AbstractRobot does not perform a Motion"
        )

    def test_law_query_reads_as_a_sentence(self):
        """
        The full Law of Task-Achieving Body Motion query verbalizes as one readable
        sentence.
        """
        task = variable(TaskRequest, [])
        effect = variable(Effect, [])
        motion = variable(Motion, [])
        robot = variable(AbstractRobot, [])
        world = World()

        query = an(
            set_of(motion, effect, task).where(
                SatisfiesRequest(task=task, effect=effect),
                Causes(effect=effect, motion=motion, environment=world),
                MotionStatechartCanPerform(motion=motion, robot=robot),
            )
        )

        assert verbalize_expression(query) == (
            "Find a Motion, an Effect, and a TaskRequest such that "
            "the Effect satisfies the TaskRequest, "
            "the Motion causes the Effect, "
            "and an AbstractRobot performs the Motion"
        )
