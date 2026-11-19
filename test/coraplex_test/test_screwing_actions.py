"""
Tests for joint-driven screwing actions and geometry helpers in
:mod:`coraplex.robot_plans.actions.core.screwing`.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pytest

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.core.exceptions import (
    MissingScrewDisplacement,
    MissingSingleDofConnection,
    NegativeScrewTurns,
    NonToolGraspForScrewing,
    UnsupportedScrewConnection,
    ZeroAlignmentAxis,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.actions.core.screwing import (
    InsertScrewAction,
    ScrewJointAction,
    ScrewThreadAction,
    align_local_z_rotation,
    compute_screw_joint_goal_angle,
    offset_pose_along_local_z,
)
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.executor import NoPacing
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from giskardpy.motion_statechart.tasks.gripper import MoveFingers
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Vector3,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForBodies,
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bolt,
    Screwdriver,
)
from semantic_digital_twin.specifications.connections import (
    RevoluteConnectionSpecification,
    ScrewConnectionSpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    ScrewConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Cylinder, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from ..plan_running import simulated_executor, statechart_of

# %% world builders


def _build_world_with_root() -> World:
    """
    Create a minimal world initialized with a single root body.

    :return: The world with ``world.root`` populated.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName(name="world")))
    return world


def _build_tracy_world() -> tuple[World, Tracy]:
    """
    Create a simulation world containing Tracy with both arms parked.

    :return: The world and Tracy robot annotation.
    """
    world = URDFParser.from_file(Tracy.get_ros_file_path()).create_world()
    tracy = Tracy.from_world(world)
    for arm in tracy.all_arms:
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    return world, tracy


def _add_fixed_box(
    *,
    world: World,
    name: str,
    position: tuple[float, float, float],
    scale: Scale,
) -> Body:
    """
    Add a fixed box body to ``world`` at ``position``.

    :param world: World to add the box to.
    :param name: Name of the spawned body.
    :param position: World ``(x, y, z)`` position in metres.
    :param scale: Box dimensions.
    :return: The spawned body.
    """
    shapes = ShapeCollection(shapes=[Box(scale=scale)])
    body = Body(name=PrefixedName(name=name), collision=shapes, visual=shapes)
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=position[0], y=position[1], z=position[2]
                ),
            )
        )
    return body


def _add_threaded_bolt(
    *,
    world: World,
    parent: Body,
    name: str = "bolt",
    position: tuple[float, float, float] = (0.022, 0.0, 0.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    screw_pitch: float = 0.002,
    turns: float = 1.0,
    axis: Vector3 | None = None,
    annotate_bolt: bool = True,
) -> tuple[Body, ScrewConnection]:
    """
    Spawn a procedural box bolt connected to ``parent`` by a :class:`ScrewConnection`.

    :param world: World to spawn the bolt into.
    :param parent: Parent body the bolt threads into.
    :param name: Body name.
    :param position:``(x, y, z)`` offset in ``parent``'s frame.
    :param rotation:``(yaw, pitch, roll)`` orientation in ``parent``'s frame.
    :param screw_pitch: Thread pitch in metres per revolution.
    :param turns: Revolutions defining the joint's lower limit.
    :param axis: Thread axis; defaults to +X.
    :param annotate_bolt: Whether to add a :class:`Bolt` semantic annotation.
    :return: The bolt body and its screw connection.
    """
    thread_axis = Vector3.X() if axis is None else axis
    shapes = ShapeCollection(shapes=[Box(scale=Scale(0.05, 0.015, 0.015))])
    body = Body(name=PrefixedName(name=name), collision=shapes, visual=shapes)
    dof_limits = DegreeOfFreedomLimits.from_position_range_and_speed(
        lower_position=-2.0 * math.pi * turns,
        upper_position=0.0,
        maximum_speed=1.0,
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        connection = ScrewConnectionSpecification(
            axis=thread_axis,
            screw_pitch=screw_pitch,
            dof_limits=dof_limits,
        ).connect(
            world,
            child=body,
            parent=parent,
            parent_T_connection=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=position[0],
                y=position[1],
                z=position[2],
                yaw=rotation[0],
                pitch=rotation[1],
                roll=rotation[2],
            ),
        )
        if annotate_bolt:
            world.add_semantic_annotations([Bolt(root=body)])
    return body, connection


def _bolt_annotation(*, world: World, body: Body) -> Bolt:
    """
    Find the :class:`Bolt` semantic annotation attached to ``body`` in ``world``.

    :param world: World holding the semantic annotation.
    :param body: Bolt root body to look up.
    :return: Matching :class:`Bolt` annotation.
    """
    return next(
        bolt
        for bolt in world.get_semantic_annotations_by_type(Bolt)
        if bolt.root is body
    )


def _add_screwdriver(
    *,
    world: World,
    position: tuple[float, float, float],
    length: float = 0.10,
    diameter: float = 0.03,
) -> Screwdriver:
    """
    Spawn a procedural screwdriver with a dedicated tip body in ``world``.

    :param world: World to spawn the screwdriver into.
    :param position: Starting ``(x, y, z)`` position in world coordinates.
    :param length: Screwdriver length in metres.
    :param diameter: Screwdriver handle diameter in metres.
    :return: The spawned :class:`Screwdriver` semantic annotation.
    """
    shapes = ShapeCollection(shapes=[Cylinder(width=diameter, height=length)])
    handle = Body(
        name=PrefixedName(name="screwdriver"), collision=shapes, visual=shapes
    )
    tip = Body(name=PrefixedName(name="screwdriver_end_effector"))
    with world.modify_world():
        world.add_kinematic_structure_entity(handle)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=handle,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=position[0], y=position[1], z=position[2]
                ),
            )
        )
        world.add_kinematic_structure_entity(tip)
        world.add_connection(
            FixedConnection(
                parent=handle,
                child=tip,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-length / 2.0,
                    pitch=math.pi,
                    reference_frame=handle,
                ),
            )
        )
        screwdriver = Screwdriver(root=handle, grasp_height=length / 4.0)
        world.add_semantic_annotations([screwdriver])
    return screwdriver


# %% geometry and displacement helpers


def test_offset_pose_along_local_z_translates_in_local_frame() -> None:
    """
    Verify :func:`offset_pose_along_local_z` moves along the pose's rotated local Z
    axis.
    """
    world = _build_world_with_root()
    rotated = Pose.from_xyz_rpy(
        x=1.0, y=2.0, z=3.0, pitch=math.pi / 2.0, reference_frame=world.root
    )
    shifted = offset_pose_along_local_z(pose=rotated, distance=-0.05)

    assert shifted.reference_frame is world.root
    np.testing.assert_allclose(
        shifted.position.to_np()[:3], [0.95, 2.0, 3.0], atol=1e-9
    )
    np.testing.assert_allclose(
        shifted.rotation_matrix.to_np(), rotated.rotation_matrix.to_np(), atol=1e-9
    )


@pytest.mark.parametrize(
    "axis",
    [
        Vector3(0.0, 0.0, 1.0),
        Vector3(0.0, 0.0, -1.0),
        Vector3(1.0, 0.0, 0.0),
        Vector3(1.0, 1.0, 1.0),
    ],
    ids=["+z", "-z", "+x", "diagonal"],
)
def test_align_local_z_rotation_maps_z_to_target_axis(axis: Vector3) -> None:
    """
    Verify :func:`align_local_z_rotation` maps local +Z onto the normalized target axis.
    """
    rotation = align_local_z_rotation(axis=axis).rotation_matrix.to_np()[:3, :3]
    axis_np = axis.to_np()[:3]
    expected_z = axis_np / np.linalg.norm(axis_np)
    np.testing.assert_allclose(
        rotation @ np.array([0.0, 0.0, 1.0]), expected_z, atol=1e-9
    )


def test_align_local_z_rotation_rejects_zero_vector() -> None:
    """
    Verify :func:`align_local_z_rotation` raises :class:`ZeroAlignmentAxis` for zero
    vectors.
    """
    with pytest.raises(ZeroAlignmentAxis):
        align_local_z_rotation(axis=Vector3(0.0, 0.0, 0.0))


def test_compute_screw_joint_goal_angle_handles_turns_travel_and_clamping() -> None:
    """
    Verify :func:`compute_screw_joint_goal_angle` computes angles from turns or travel
    distance and clamps against the lower joint limit.
    """
    world = _build_world_with_root()
    plate = _add_fixed_box(
        world=world,
        name="plate",
        position=(0.5, 0.0, 1.0),
        scale=Scale(0.05, 0.05, 0.05),
    )
    _, connection = _add_threaded_bolt(
        world=world, parent=plate, screw_pitch=0.002, turns=1.0
    )

    assert compute_screw_joint_goal_angle(
        connection=connection, turns=0.5
    ) == pytest.approx(-math.pi)
    assert compute_screw_joint_goal_angle(
        connection=connection, turns=0.5, travel_distance=-0.0005
    ) == pytest.approx(-math.pi / 2.0)
    assert compute_screw_joint_goal_angle(
        connection=connection, turns=5.0
    ) == pytest.approx(-2.0 * math.pi)


def test_compute_screw_joint_goal_angle_error_cases() -> None:
    """
    Verify :func:`compute_screw_joint_goal_angle` raises custom exceptions for invalid
    inputs.
    """
    world = _build_world_with_root()
    plate = _add_fixed_box(
        world=world,
        name="plate",
        position=(0.5, 0.0, 1.0),
        scale=Scale(0.05, 0.05, 0.05),
    )
    _, screw_connection = _add_threaded_bolt(world=world, parent=plate)

    hinged = Body(name=PrefixedName(name="hinged"))
    with world.modify_world():
        world.add_kinematic_structure_entity(hinged)
        revolute = RevoluteConnectionSpecification(axis=Vector3.Z()).connect(
            world, child=hinged, parent=plate
        )

    with pytest.raises(UnsupportedScrewConnection):
        compute_screw_joint_goal_angle(connection=revolute, turns=1.0)

    with pytest.raises(MissingScrewDisplacement):
        compute_screw_joint_goal_angle(connection=screw_connection)

    with pytest.raises(NegativeScrewTurns):
        compute_screw_joint_goal_angle(connection=screw_connection, turns=-0.25)


# %% action structure and error handling


def test_screw_joint_action_error_collision_rules_and_goal(
    pr2_apartment_context: tuple,
) -> None:
    """
    Verify :class:`ScrewJointAction` checks parent connections, collision rules, and
    goal.
    """
    world, robot, _ = pr2_apartment_context
    arm = robot.right_arm
    fixed_plate = _add_fixed_box(
        world=world,
        name="plate",
        position=(2.4, 2.2, 1.0),
        scale=Scale(0.05, 0.05, 0.05),
    )

    missing = ScrewJointAction(arm=arm, screw_body=fixed_plate)
    with pytest.raises(MissingSingleDofConnection):
        _ = missing.screw_connection

    bolt, _ = _add_threaded_bolt(world=world, parent=fixed_plate)
    no_collision = ScrewJointAction(
        arm=arm, screw_body=bolt, allow_gripper_collision=False
    )
    assert no_collision.create_temporary_collision_rules() == []

    held_box = _add_fixed_box(
        world=world,
        name="held_tool",
        position=(0.0, 0.0, 0.0),
        scale=Scale(0.02, 0.02, 0.02),
    )
    with world.modify_world():
        world.move_branch(held_box, arm.end_effector.tool_frame)

    action = ScrewJointAction(arm=arm, screw_body=bolt, turns=0.5)
    rules = action.create_temporary_collision_rules()
    assert len(rules) == 2
    assert isinstance(rules[0], AllowCollisionForEndEffector)
    assert rules[0].end_effector is arm.end_effector
    assert isinstance(rules[1], AllowCollisionForBodies)
    assert held_box in rules[1].allowed_collision_bodies

    goal = action.create_goal()
    assert isinstance(goal, ArticulateGraspedMechanism)
    assert goal.tip_link is arm.end_effector.tool_frame
    assert goal.environment_link is bolt
    assert goal.goal_joint_state == pytest.approx(-math.pi)


def test_screw_thread_action_pose_properties_tool_validation_and_steps(
    pr2_apartment_context: tuple,
) -> None:
    """
    Verify :class:`ScrewThreadAction` pose calculations, tool validation, and step
    lists.
    """
    world, robot, extensions = pr2_apartment_context
    arm = robot.right_arm
    plate = _add_fixed_box(
        world=world,
        name="plate",
        position=(2.4, 2.2, 1.0),
        scale=Scale(0.05, 0.05, 0.05),
    )
    bolt_body, _ = _add_threaded_bolt(world=world, parent=plate, annotate_bolt=True)
    plain_bolt, _ = _add_threaded_bolt(
        world=world, parent=plate, name="plain_bolt", annotate_bolt=False
    )

    action = ScrewThreadAction(arm=arm, screw_body=bolt_body, turns=0.5)
    np.testing.assert_allclose(
        action.insertion_axis.to_np()[:3], [-1.0, 0.0, 0.0], atol=1e-9
    )
    assert action.tool_tip_T_tool_frame is None
    np.testing.assert_allclose(
        action.default_grasp_local_pose.position.to_np()[:3],
        [0.025, 0.0, 0.0],
        atol=1e-9,
    )

    explicit = ScrewThreadAction(arm=arm, screw_body=bolt_body, grasp_offset=0.035)
    np.testing.assert_allclose(
        explicit.default_grasp_local_pose.position.to_np()[:3],
        [0.035, 0.0, 0.0],
        atol=1e-9,
    )

    no_bolt_annotation = ScrewThreadAction(arm=arm, screw_body=plain_bolt)
    np.testing.assert_allclose(
        no_bolt_annotation.default_grasp_local_pose.position.to_np()[:3],
        [0.0, 0.0, 0.0],
        atol=1e-9,
    )

    custom_pose = Pose.from_xyz_rpy(x=0.01, reference_frame=bolt_body)
    with_custom_pose = ScrewThreadAction(
        arm=arm, screw_body=bolt_body, grasp_local_pose=custom_pose
    )
    assert with_custom_pose.engage_pose is custom_pose

    bolt_annotation = _bolt_annotation(world=world, body=bolt_body)
    invalid_tool_action = ScrewThreadAction(
        arm=arm,
        screw_body=bolt_body,
        tool_grasp=bolt_annotation.grasp_candidates()[0],
    )
    with pytest.raises(NonToolGraspForScrewing):
        _ = invalid_tool_action.tool_tip_T_tool_frame

    statechart_of(simulated_executor(extensions), action)
    assert any(isinstance(node, ScrewJointAction) for node in action.nodes)
    assert any(
        isinstance(node, MoveFingers)
        for step in action.nodes
        for node in [step, *step.descendants]
    )

    screwdriver = _add_screwdriver(world=world, position=(2.2, 2.2, 1.0))
    tool_action = ScrewThreadAction(
        arm=arm,
        screw_body=bolt_body,
        tool_grasp=screwdriver.grasp_candidates()[0],
        turns=0.5,
    )
    assert tool_action.tool_tip_T_tool_frame is not None
    statechart_of(simulated_executor(extensions), tool_action)
    assert not any(
        isinstance(node, MoveFingers)
        for step in tool_action.nodes
        for node in [step, *step.descendants]
    )


def test_insert_screw_action_steps_respect_turn_by_hand_flag(
    pr2_apartment_context: tuple,
) -> None:
    """
    Verify :class:`InsertScrewAction` toggles the hand-turn step and allows gripper
    collision.
    """
    world, robot, extensions = pr2_apartment_context
    arm = robot.right_arm
    plate = _add_fixed_box(
        world=world,
        name="plate",
        position=(2.4, 2.2, 1.0),
        scale=Scale(0.05, 0.05, 0.05),
    )
    bolt_body, _ = _add_threaded_bolt(world=world, parent=plate)
    grasp = GraspCandidate.from_body_origin(
        _bolt_annotation(world=world, body=bolt_body)
    )
    target = Pose.from_xyz_rpy(x=0.02, reference_frame=plate)

    with_turn = InsertScrewAction(
        arm=arm,
        grasp=grasp,
        counterpart=plate,
        target_location=target,
        turn_by_hand=True,
    )
    statechart_of(simulated_executor(extensions), with_turn)
    assert any(isinstance(node, ScrewJointAction) for node in with_turn.nodes)
    [rule] = with_turn.create_temporary_collision_rules()
    assert isinstance(rule, AllowCollisionForEndEffector)
    assert rule.end_effector is arm.end_effector

    without_turn = InsertScrewAction(
        arm=arm,
        grasp=grasp,
        counterpart=plate,
        target_location=target,
        turn_by_hand=False,
    )
    statechart_of(simulated_executor(extensions), without_turn)
    assert not any(isinstance(node, ScrewJointAction) for node in without_turn.nodes)


# %% kinematic execution


def test_screw_thread_action_drives_screw_connection_kinematically() -> None:
    """
    Verify :class:`ScrewThreadAction` drives a threaded bolt to the expected joint
    angle.
    """
    world, tracy = _build_tracy_world()
    plate = _add_fixed_box(
        world=world,
        name="nut_plate",
        position=(0.528, 0.0, 1.0),
        scale=Scale(0.04, 0.04, 0.04),
    )
    bolt, connection = _add_threaded_bolt(
        world=world, parent=plate, position=(0.022, 0.0, 0.0), turns=0.5
    )

    context = StatechartContext.with_extensions(
        world=world, extensions=[RobotAccess(robot=tracy)]
    )
    statechart = Statechart(context=context)
    statechart.add_node(
        node=Sequence(
            nodes=[
                ParkArmsAction(arms=tracy.all_arms),
                ScrewThreadAction(
                    arm=tracy.right_arm,
                    screw_body=bolt,
                    turns=0.5,
                    grasp_offset=0.030,
                ),
                ParkArmsAction(arms=tracy.all_arms),
            ]
        )
    )

    executor = KinematicallySimulatedPlanExecutor(
        context=context,
        simulation_time_per_action=timedelta(minutes=5),
        pacer=NoPacing(),
    )
    try:
        executor.compile(statechart=statechart)
        executor.execute()
    finally:
        executor.close()

    assert connection.position == pytest.approx(-math.pi, abs=1e-2)


def test_pick_insert_and_tighten_bolt_with_screwdriver_kinematically() -> None:
    """
    Verify picking, inserting, and driving a bolt with a :class:`Screwdriver` reattaches
    the bolt and drives its screw joint to completion.
    """
    world, tracy = _build_tracy_world()
    plate = _add_fixed_box(
        world=world,
        name="target_plate",
        position=(0.55, 0.0, 0.88),
        scale=Scale(0.08, 0.08, 0.04),
    )
    holder = _add_fixed_box(
        world=world,
        name="bolt_holder",
        position=(0.55, -0.25, 0.88),
        scale=Scale(0.06, 0.06, 0.04),
    )
    bolt, connection = _add_threaded_bolt(
        world=world,
        parent=holder,
        position=(-0.025, 0.0, 0.025),
        rotation=(0.0, -math.pi / 2.0, 0.0),
        turns=0.5,
    )
    screwdriver = _add_screwdriver(
        world=world, position=(0.575, -0.25, 0.88 + 0.025 + 0.05)
    )
    bolt_grasp = _bolt_annotation(world=world, body=bolt).grasp_candidates()[0]
    screwdriver_grasp = screwdriver.grasp_candidates()[0]
    target_location = Pose.from_xyz_rpy(
        0.0, -0.032, 0.06, pitch=-math.pi / 2.0, reference_frame=plate
    )

    context = StatechartContext.with_extensions(
        world=world, extensions=[RobotAccess(robot=tracy)]
    )
    statechart = Statechart(context=context)
    statechart.add_node(
        node=Sequence(
            nodes=[
                ParkArmsAction(arms=tracy.all_arms),
                PickUpAction(grasp=bolt_grasp, arm=tracy.right_arm),
                InsertScrewAction(
                    arm=tracy.right_arm,
                    grasp=bolt_grasp,
                    counterpart=plate,
                    target_location=target_location,
                    turn_by_hand=True,
                    hand_turns=0.1,
                ),
                PickUpAction(grasp=screwdriver_grasp, arm=tracy.right_arm),
                ScrewThreadAction(
                    arm=tracy.right_arm,
                    screw_body=bolt,
                    tool_grasp=screwdriver_grasp,
                    turns=0.5,
                ),
                ParkArmsAction(arms=tracy.all_arms),
            ]
        )
    )

    executor = KinematicallySimulatedPlanExecutor(
        context=context,
        simulation_time_per_action=timedelta(minutes=5),
        pacer=NoPacing(),
    )
    try:
        executor.compile(statechart=statechart)
        executor.execute()
    finally:
        executor.close()

    assert bolt.parent_kinematic_structure_entity is plate
    assert isinstance(bolt.parent_connection, ScrewConnection)
    assert bolt.parent_connection.position == pytest.approx(-math.pi, abs=1e-2)
