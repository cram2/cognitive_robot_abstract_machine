from copy import deepcopy

import numpy as np
import pytest
from coraplex.alternative_motion_mappings.stretch_motion_mapping import (
    StretchMoveReal,
    StretchMoveSim,
    StretchMoveToolCenterPoint,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    MovementType,
)
from coraplex.execution_environment import real_robot, semi_real_robot, simulated_robot
from coraplex.plans.executables import MoveBranchExecutable
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.plan_node import ActionNode, MotionNode
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import GraspingAction, PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from coraplex.robot_plans.motions.container import ClosingMotion, OpeningMotion
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveTCPWaypointsAlignedMotion,
    MoveTCPWaypointsMotion,
)
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.cartesian_goals import (
    CartesianPoseStraight,
    DifferentialDriveBaseGoal,
)
from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPose,
    CartesianPositionTrajectory,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from giskardpy.motion_statechart.tasks.pointing import Pointing
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.griplink_gripper import (
    GriplinkFlexConfiguration,
    GriplinkGripPreset,
    GriplinkPresetConfiguration,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose

from ..conftest import left_or_only_arm

try:
    from coraplex.alternative_motion_mappings.hsrb_motion_mapping import *  # noqa: F403
    from coraplex.alternative_motion_mappings.hsrb_motion_mapping import (
        HSRBMoveMotion,
    )
    from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
        NavigateActionServerTask,
    )

    skip_tests = False
except (ImportError, ModuleNotFoundError, AttributeError):
    skip_tests = True

try:
    from coraplex.alternative_motion_mappings.daisy_motion_mapping import (
        DAiSyFlexGripMotion,
        DAiSyGripMotion,
    )
    from coraplex.exceptions import NoGriplinkEndpoint
    from giskardpy.motion_statechart.ros2_nodes.griplink import (
        GriplinkFlexActionServerTask,
        GriplinkPresetActionServerTask,
    )

    daisy_mappings_available = True
except (ImportError, ModuleNotFoundError, AttributeError):
    daisy_mappings_available = False


def _chart_nodes(motion_chart):
    """
    :return: The nodes of ``motion_chart``: the tasks a :class:`Parallel` groups, or the
        single task a motion that needs no accompanying node builds on its own.
    """
    if isinstance(motion_chart, Parallel):
        return list(motion_chart.nodes)
    return [motion_chart]


@pytest.mark.skipif(skip_tests, reason="Alternative motion mappings not available")
def test_pick_up_motion(pr2_apartment_context):
    world, view, context = pr2_apartment_context
    test_world = deepcopy(world)
    milk = test_world.get_semantic_annotations_by_type(Milk)[0]
    test_context = Context.from_world(test_world)
    pick_up = PickUpAction(milk.grasp_candidates()[0], test_context.robot.left_arm)

    root = sequential(
        children=[
            ActionNode(
                designator=NavigateAction(
                    Pose(
                        Point3.from_iterable([1.7, 1.5, 0]),
                        Quaternion.from_iterable([0, 0, 0, 1]),
                        test_world.root,
                    ),
                )
            ),
            MoveTorsoAction(TorsoState.HIGH),
            pick_up,
        ],
        context=test_context,
    )
    assert pick_up.plan is not None
    with simulated_robot:
        root.perform()

    pick_up_node = root.plan.get_nodes_by_designator_type(PickUpAction)[0]

    motion_nodes = list(
        filter(lambda x: isinstance(x, MotionNode), pick_up_node.descendants)
    )

    assert len(motion_nodes) == 5

    motion_chart_task_types = {
        type(node)
        for motion_node in motion_nodes
        for node in _chart_nodes(motion_node.designator.motion_chart)
    }
    assert CartesianPose in motion_chart_task_types
    assert JointPositionList in motion_chart_task_types


def test_move_motion_chart(pr2_apartment_context):
    world, view, context = pr2_apartment_context
    motion = MoveMotion(
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)
    )
    plan = execute_single(
        motion,
        context=context,
    )

    msc = motion.motion_chart

    assert msc
    np.testing.assert_equal(msc.goal_pose.position.to_np(), np.array([1, 1, 1, 1]))


def test_move_tool_center_point_motion_uses_tight_threshold(pr2_apartment_context):
    """
    MoveToolCenterPointMotion drives grasp approaches, so it must not fall back to
    Giskard's loose default CartesianPose/CartesianPosition threshold (0.01m): that
    tolerance is wide enough to let the gripper stop a centimeter away from a small
    object, e.g. missing or off-center grasps.
    """
    world, view, context = pr2_apartment_context
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    cartesian_motion = MoveToolCenterPointMotion(
        target, left_or_only_arm(context.robot), movement_type=MovementType.CARTESIAN
    )
    execute_single(cartesian_motion, context=context)
    assert isinstance(cartesian_motion.motion_chart, CartesianPose)
    assert (
        cartesian_motion.motion_chart.translation_threshold
        == context.motion_tolerances.default_tcp_position_threshold
    )

    translation_motion = MoveToolCenterPointMotion(
        target, left_or_only_arm(context.robot), movement_type=MovementType.TRANSLATION
    )
    execute_single(translation_motion, context=context)
    assert (
        translation_motion.motion_chart.threshold
        == context.motion_tolerances.default_tcp_position_threshold
    )


def test_move_tcp_waypoints_motion_forwards_thresholds(pr2_apartment_context):
    """
    MoveTCPWaypointsMotion must forward an explicit position/orientation threshold to
    its per-waypoint CartesianPose tasks.
    """
    world, view, context = pr2_apartment_context
    waypoints = [Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)]

    motion = MoveTCPWaypointsMotion(
        waypoints,
        left_or_only_arm(context.robot),
        position_threshold=0.001,
        orientation_threshold=0.05,
    )
    execute_single(motion, context=context)

    nodes = motion.motion_chart.nodes
    assert len(nodes) == 1
    assert isinstance(nodes[0], CartesianPose)
    assert nodes[0].translation_threshold == 0.001
    assert nodes[0].orientation_threshold == 0.05


def test_move_tcp_waypoints_motion_uses_giskard_defaults_when_unset(
    pr2_apartment_context,
):
    """
    MoveTCPWaypointsMotion follows waypoints rather than grasping, so leaving the
    thresholds unset must fall back to Giskard's own task defaults instead of the
    tighter grasp tolerance.
    """
    world, view, context = pr2_apartment_context
    waypoints = [Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)]

    motion = MoveTCPWaypointsMotion(waypoints, left_or_only_arm(context.robot))
    execute_single(motion, context=context)

    nodes = motion.motion_chart.nodes
    assert isinstance(nodes[0], CartesianPose)
    assert (
        nodes[0].translation_threshold
        != context.motion_tolerances.default_tcp_position_threshold
    )
    assert (
        nodes[0].orientation_threshold
        != context.motion_tolerances.tool_orientation_threshold
    )


def test_move_tcp_waypoints_aligned_motion_forwards_position_threshold(
    pr2_apartment_context,
):
    """
    MoveTCPWaypointsAlignedMotion must forward an explicit position threshold to its
    CartesianPositionTrajectory task.
    """
    world, view, context = pr2_apartment_context
    waypoints = [Point3.from_iterable([1, 1, 1])]

    motion = MoveTCPWaypointsAlignedMotion(
        waypoints, left_or_only_arm(context.robot), position_threshold=0.001
    )
    execute_single(motion, context=context)

    trajectory = next(
        node
        for parallel in motion.motion_chart.nodes
        for node in parallel.nodes
        if isinstance(node, CartesianPositionTrajectory)
    )
    assert trajectory.threshold == 0.001


def test_move_tool_center_point_motion_without_max_velocity_returns_bare_task(
    pr2_apartment_context,
):
    """
    MoveToolCenterPointMotion must not add any velocity-limit constraint when neither
    ``max_linear_velocity`` nor ``max_angular_velocity`` is set, so a caller that never
    mentions them keeps relying on the robot's own hardware velocity limits only.
    """
    world, view, context = pr2_apartment_context
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    motion = MoveToolCenterPointMotion(
        target, left_or_only_arm(context.robot), movement_type=MovementType.CARTESIAN
    )
    execute_single(motion, context=context)
    assert isinstance(motion.motion_chart, CartesianPose)


def test_move_tool_center_point_motion_max_linear_velocity_adds_real_limit(
    pr2_apartment_context,
):
    """
    An explicit ``max_linear_velocity`` must add a real
    :class:`CartesianPositionVelocityLimit` constraint alongside the goal task via
    ``Parallel``, instead of tuning the goal task's own reference velocity -- per review
    feedback, reference velocities are for QP normalization only and must not be exposed
    as a caller-tunable speed limit.
    """
    world, view, context = pr2_apartment_context
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    motion = MoveToolCenterPointMotion(
        target,
        left_or_only_arm(context.robot),
        movement_type=MovementType.CARTESIAN,
        max_linear_velocity=0.05,
    )
    execute_single(motion, context=context)
    assert isinstance(motion.motion_chart, Parallel)
    node_types = [type(node) for node in motion.motion_chart.nodes]
    assert CartesianPose in node_types
    assert CartesianPositionVelocityLimit in node_types
    velocity_limit_node = next(
        node
        for node in motion.motion_chart.nodes
        if isinstance(node, CartesianPositionVelocityLimit)
    )
    assert velocity_limit_node.max_linear_velocity == 0.05


def test_move_tool_center_point_motion_max_angular_velocity_adds_real_limit(
    pr2_apartment_context,
):
    """
    An explicit ``max_angular_velocity`` must add a real
    :class:`CartesianRotationVelocityLimit` constraint, only meaningful for the non-
    translation (full 6D pose) movement type.
    """
    world, view, context = pr2_apartment_context
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    motion = MoveToolCenterPointMotion(
        target,
        left_or_only_arm(context.robot),
        movement_type=MovementType.CARTESIAN,
        max_angular_velocity=0.2,
    )
    execute_single(motion, context=context)
    assert isinstance(motion.motion_chart, Parallel)
    velocity_limit_node = next(
        node
        for node in motion.motion_chart.nodes
        if isinstance(node, CartesianRotationVelocityLimit)
    )
    assert velocity_limit_node.max_angular_velocity == 0.2


def test_move_gripper_motion_finger_velocity_adds_real_limit(pr2_apartment_context):
    """
    An explicit ``finger_velocity`` must add a real
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`
    constraint alongside the goal task, instead of tuning the goal task's own
    reference/normalization velocity.
    """
    world, view, context = pr2_apartment_context

    close_motion = MoveGripperMotion(
        configuration=left_or_only_arm(view).end_effector.default_configuration(
            GripperState.CLOSE, finger_velocity=0.03
        )
    )
    execute_single(close_motion, context=context)
    assert isinstance(close_motion.motion_chart, Parallel)
    node_types = [type(node) for node in close_motion.motion_chart.nodes]
    assert JointPositionList in node_types
    assert JointVelocityLimit in node_types
    velocity_limit_node = next(
        node
        for node in close_motion.motion_chart.nodes
        if isinstance(node, JointVelocityLimit)
    )
    assert velocity_limit_node.max_velocity == 0.03


def test_move_gripper_motion_tolerate_stall_and_finger_velocity_combine(
    pr2_apartment_context,
):
    """
    ``tolerate_stall`` and ``finger_velocity`` set together must nest correctly: the
    motion is done once (goal reached OR stalled) AND the finger velocity stayed within
    its limit -- not a single flat ``Parallel`` that conflates OR and AND semantics.
    """
    world, view, context = pr2_apartment_context

    close_motion = MoveGripperMotion(
        configuration=left_or_only_arm(view).end_effector.default_configuration(
            GripperState.CLOSE, finger_velocity=0.03
        ),
        tolerate_stall=True,
    )
    execute_single(close_motion, context=context)
    assert isinstance(close_motion.motion_chart, Parallel)
    outer_node_types = [type(node) for node in close_motion.motion_chart.nodes]
    assert JointVelocityLimit in outer_node_types
    inner_parallel = next(
        node for node in close_motion.motion_chart.nodes if isinstance(node, Parallel)
    )
    assert inner_parallel.minimum_success == 1
    inner_node_types = [type(node) for node in inner_parallel.nodes]
    assert JointPositionList in inner_node_types
    assert LocalMinimumReached in inner_node_types


def test_move_gripper_motion_tolerate_stall_defaults_to_false(pr2_apartment_context):
    """
    MoveGripperMotion must not tolerate a stall by default, for either OPEN or CLOSE --
    stalling before reaching the target is a real failure that should be surfaced,
    unless a caller (e.g. PickUpAction, grasping a real object) explicitly opts in via
    ``tolerate_stall=True``.

    A caller that never mentions this field must keep relying on the original,
    unmodified default behaviour: the plain goal task, not wrapped in any stall-tolerant
    monitor.
    """
    world, view, context = pr2_apartment_context

    close_motion = MoveGripperMotion(
        configuration=left_or_only_arm(view).end_effector.default_configuration(
            GripperState.CLOSE
        )
    )
    execute_single(close_motion, context=context)
    assert isinstance(close_motion.motion_chart, JointPositionList)

    open_motion = MoveGripperMotion(
        configuration=left_or_only_arm(view).end_effector.default_configuration(
            GripperState.OPEN
        )
    )
    execute_single(open_motion, context=context)
    assert isinstance(open_motion.motion_chart, JointPositionList)


def test_move_gripper_motion_tolerate_stall_can_be_explicitly_enabled(
    pr2_apartment_context,
):
    """
    An explicit ``tolerate_stall=True`` must wrap the goal task together with a
    :class:`LocalMinimumReached` monitor in a :class:`Parallel` (with
    ``minimum_success=1``), so the motion is considered done as soon as either the goal
    is reached or the fingers have stalled -- without changing what the goal task's own
    observation means (goal reached, nothing else).
    """
    world, view, context = pr2_apartment_context

    close_motion = MoveGripperMotion(
        configuration=left_or_only_arm(view).end_effector.default_configuration(
            GripperState.CLOSE
        ),
        tolerate_stall=True,
    )
    execute_single(close_motion, context=context)
    assert isinstance(close_motion.motion_chart, Parallel)
    assert close_motion.motion_chart.minimum_success == 1
    node_types = [type(node) for node in close_motion.motion_chart.nodes]
    assert JointPositionList in node_types
    assert LocalMinimumReached in node_types


def _close_motion_of(pick_up: PickUpAction) -> MoveGripperMotion:
    """
    :return: The motion that closes the gripper on what a pick-up grasps.

    A pick-up closes the gripper through the grasp it is built from, and the reach
    inside that grasp opens it first, so the closing motion is picked out of an
    expanded plan rather than read off the pick-up's own children.
    """
    pick_up.plan_node.notify()
    [close_motion] = [
        node.designator
        for node in pick_up.plan_node.plan.get_nodes_by_designator_type(
            MoveGripperMotion
        )
        if node.designator.configuration.joint_state.state_type is GripperState.CLOSE
    ]
    return close_motion


def test_pick_up_action_close_motion_stall_tolerance_defaults_to_false(
    pr2_apartment_context,
):
    """
    PickUpAction's grasp-closing motion must not tolerate a stall unless explicitly
    asked to: building the stall monitor needs a velocity variable for every one of the
    gripper's connections, which not every robot has, so it must stay opt-in rather than
    always on (it crashes on Tracy's real-execution gripper otherwise).
    """
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_up = PickUpAction(milk.grasp_candidates()[0], left_or_only_arm(context.robot))
    sequential([pick_up], context=context)

    assert _close_motion_of(pick_up).tolerate_stall is False


def test_pick_up_action_close_motion_tolerates_stall_when_enabled(
    pr2_apartment_context,
):
    """
    PickUpAction's ``tolerate_grasp_stall`` must reach the grasp's CLOSE motion, so a
    grasped object's fingers physically stopping before the nominal fully-closed target
    is correctly treated as a real grasp, not a failed motion, once explicitly enabled.
    """
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_up = PickUpAction(
        milk.grasp_candidates()[0],
        left_or_only_arm(context.robot),
        tolerate_grasp_stall=True,
    )
    sequential([pick_up], context=context)

    assert _close_motion_of(pick_up).tolerate_stall is True


def test_pick_up_action_velocity_fields_default_to_none(pr2_apartment_context):
    """
    PickUpAction's velocity/timing/friction fields must all default to ``None`` when not
    explicitly set, so an existing caller that never mentions them keeps relying on
    Giskard's own task defaults instead of a new, silently-injected value -- these
    physics fields are opt-in additions, not a change to the action's default behaviour.
    """
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_up = PickUpAction(milk.grasp_candidates()[0], left_or_only_arm(context.robot))

    assert pick_up.pre_approach_linear_velocity is None
    assert pick_up.final_approach_linear_velocity is None
    assert pick_up.grasp_closing_velocity is None
    assert pick_up.lift_linear_velocity is None
    assert pick_up.grasp_stall_minimum_time is None
    assert pick_up.object_friction is None


def test_place_action_velocity_fields_default_to_none(pr2_apartment_context):
    """
    PlaceAction's velocity/timing fields must all default to ``None`` when not
    explicitly set, matching PickUpAction's own opt-in design: an existing caller that
    never mentions them keeps relying on Giskard's own task defaults.
    """
    world, view, context = pr2_apartment_context
    target_location = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    place = PlaceAction(
        world.get_semantic_annotations_by_type(Milk)[0], target_location
    )

    assert place.placing_linear_velocity is None
    assert place.transport_linear_velocity is None
    assert place.release_opening_velocity is None
    assert place.retract_linear_velocity is None


# %% allowing the gripper to touch what it manipulates


def _collision_rule_nodes(motion_chart):
    """
    :return: The nodes of ``motion_chart`` that hand temporary collision rules to the
        collision manager.
    """
    return [
        node
        for node in _chart_nodes(motion_chart)
        if isinstance(node, UpdateTemporaryCollisionRules)
    ]


def test_move_tool_center_point_motion_frees_the_manipulator_it_reaches_with(
    pr2_apartment_context,
):
    """
    ``allow_gripper_collision`` must reach the collision manager: without a rule that
    frees the manipulator, collision avoidance holds the fingers a buffer zone away from
    whatever they reach for and the reach never converges on its goal.
    """
    world, view, context = pr2_apartment_context
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    motion = MoveToolCenterPointMotion(
        target,
        left_or_only_arm(context.robot),
        movement_type=MovementType.CARTESIAN,
        allow_gripper_collision=True,
    )
    execute_single(motion, context=context)

    rule_nodes = _collision_rule_nodes(motion.motion_chart)
    assert len(rule_nodes) == 1
    (rule,) = rule_nodes[0].temporary_rules
    assert rule.end_effector is left_or_only_arm(context.robot).end_effector


def test_move_tool_center_point_motion_frees_what_the_manipulator_grasps_later(
    pr2_apartment_context,
):
    """
    The lift that carries a grasped body away is built before the grasp attaches it, so
    the rule must free whatever the manipulator holds when it runs rather than what it
    held when the chart was built.
    """
    world, view, context = pr2_apartment_context
    end_effector = left_or_only_arm(context.robot).end_effector
    held_body = world.get_body_by_name("milk.stl")

    motion = MoveToolCenterPointMotion(
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root),
        left_or_only_arm(context.robot),
        movement_type=MovementType.CARTESIAN,
        allow_gripper_collision=True,
    )
    execute_single(motion, context=context)
    (rule,) = _collision_rule_nodes(motion.motion_chart)[0].temporary_rules

    MoveBranchExecutable(
        context=context, body=held_body, new_parent=end_effector.tool_frame
    ).execute()
    rule.update(world)

    assert held_body in rule.allowed_collision_bodies


def test_move_tool_center_point_motion_keeps_the_manipulator_clear_by_default(
    pr2_apartment_context,
):
    """
    Without ``allow_gripper_collision`` the motion adds no collision rule of its own, so
    the robot's own rules keep deciding how close the gripper may come.
    """
    world, view, context = pr2_apartment_context
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    motion = MoveToolCenterPointMotion(
        target, left_or_only_arm(context.robot), movement_type=MovementType.CARTESIAN
    )
    execute_single(motion, context=context)

    assert _collision_rule_nodes(motion.motion_chart) == []


def test_move_gripper_motion_frees_the_fingers_it_closes(pr2_apartment_context):
    """
    Fingers closing on an object touch it, so ``allow_gripper_collision`` must reach the
    collision manager here too: otherwise the buffer zone kept around the object stops
    the fingers before they hold it.
    """
    world, view, context = pr2_apartment_context

    close_motion = MoveGripperMotion(
        configuration=left_or_only_arm(view).end_effector.default_configuration(
            GripperState.CLOSE
        ),
        allow_gripper_collision=True,
    )
    execute_single(close_motion, context=context)

    rule_nodes = _collision_rule_nodes(close_motion.motion_chart)
    assert len(rule_nodes) == 1
    (rule,) = rule_nodes[0].temporary_rules
    assert rule.end_effector is left_or_only_arm(context.robot).end_effector


def test_move_gripper_motion_keeps_the_fingers_clear_by_default(pr2_apartment_context):
    """
    Without ``allow_gripper_collision`` the gripper motion adds no collision rule of its
    own.
    """
    world, view, context = pr2_apartment_context

    close_motion = MoveGripperMotion(
        configuration=left_or_only_arm(view).end_effector.default_configuration(
            GripperState.CLOSE
        )
    )
    execute_single(close_motion, context=context)

    assert _collision_rule_nodes(close_motion.motion_chart) == []


def test_pick_up_action_closes_the_gripper_on_what_it_grasps(pr2_apartment_context):
    """
    PickUpAction's grasp-closing motion must allow the gripper collision it is about to
    make: the fingers meeting the object are the grasp, not a collision to give up on.
    """
    world, view, context = pr2_apartment_context
    pick_up = PickUpAction(
        world.get_semantic_annotations_by_type(Milk)[0].grasp_candidates()[0],
        left_or_only_arm(context.robot),
    )
    sequential([pick_up], context=context)

    assert _close_motion_of(pick_up).allow_gripper_collision is True


def test_place_action_lets_the_carried_object_touch_what_it_lands_on(
    pr2_apartment_context,
):
    """
    A carried body hangs below the tool frame and is therefore freed together with the
    manipulator, so the motions that carry it and the one that releases it must allow
    the gripper collision.

    The retract afterwards holds nothing and keeps the default.
    """
    world, view, context = pr2_apartment_context
    target_location = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    milk = world.get_semantic_annotations_by_type(Milk)[0]
    with world.modify_world():
        world.move_branch_with_fixed_connection(
            milk.root, view.left_arm.end_effector.tool_frame
        )

    place = PlaceAction(milk, target_location)
    sequential([place], context=context)
    plan = place._action_plan.plan

    tool_center_point_allowances = [
        node.designator.allow_gripper_collision
        for node in plan.get_nodes_by_designator_type(MoveToolCenterPointMotion)
    ]
    assert tool_center_point_allowances.count(True) == 2
    assert tool_center_point_allowances.count(None) == 1

    release_nodes = plan.get_nodes_by_designator_type(MoveGripperMotion)
    assert len(release_nodes) == 1
    assert release_nodes[0].designator.allow_gripper_collision is True


@pytest.mark.skipif(skip_tests, reason="Alternative motion mappings not available")
def test_alternative_mapping(hsr_apartment_world):
    world, view, context = hsr_apartment_world
    context.alternative_motion_mappings = [HSRBMoveMotion]
    move_motion = MoveMotion(
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)
    )

    plan = execute_single(move_motion, context=context)

    with real_robot:
        assert move_motion.get_alternative_motion()
        msc = move_motion.motion_chart
        assert NavigateActionServerTask == type(msc)


# %% looking


def test_looking_motion_pointing_parameters(pr2_apartment_context):
    """
    The looking motion aims the camera's forward axis at the target, moving the head
    relative to the torso so the rest of the body stays where it is.
    """
    world, view, context = pr2_apartment_context
    camera = view.get_default_camera()
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)
    motion = LookingMotion(target=target, camera=camera)
    execute_single(motion, context=context)

    pointing = motion.motion_chart

    assert isinstance(pointing, Pointing)
    assert pointing.root_link is view.get_torso().root
    assert pointing.tip_link is camera.root
    assert np.array_equal(
        pointing.pointing_axis.to_np(), camera.forward_facing_axis.to_np()
    )
    assert pointing.pointing_axis.reference_frame is camera.root
    assert pointing.goal_point.reference_frame is world.root
    assert np.array_equal(pointing.goal_point.to_np(), target.position.to_np())


# %% stretch tool center point


@pytest.mark.skipif(skip_tests, reason="Alternative motion mappings not available")
def test_stretch_tool_center_point_holds_the_base_heading(
    stretch_apartment_context,
):
    """
    The base orientation is held alongside the cartesian goal rather than before it, so
    the base keeps the heading it started with while the arm reaches.
    """
    world, robot, context = stretch_apartment_context
    context.alternative_motion_mappings = [StretchMoveToolCenterPoint]
    motion = MoveToolCenterPointMotion(
        target=Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root),
        arm=left_or_only_arm(context.robot),
    )
    execute_single(motion, context=context)

    with real_robot:
        heading_stage = motion.motion_chart.nodes[0]

    assert isinstance(heading_stage, CartesianOrientation)
    assert heading_stage.root_link is world.root
    assert heading_stage.tip_link is robot.root
    assert heading_stage.binding_policy is GoalBindingPolicy.Bind_on_start
    assert heading_stage.goal_orientation.reference_frame is robot.root


@pytest.mark.skipif(skip_tests, reason="Alternative motion mappings not available")
def test_stretch_tool_center_point_accepts_a_local_minimum(
    stretch_apartment_context,
):
    """
    The arm regularly settles just short of the goal pose, so converging into a local
    minimum counts as success alongside reaching the pose.
    """
    world, robot, context = stretch_apartment_context
    context.alternative_motion_mappings = [StretchMoveToolCenterPoint]
    motion = MoveToolCenterPointMotion(
        target=Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root),
        arm=left_or_only_arm(context.robot),
    )
    execute_single(motion, context=context)

    with real_robot:
        reaching_stage = motion.motion_chart.nodes[1]

    assert isinstance(reaching_stage, Parallel)
    assert reaching_stage.minimum_success == 1
    assert {type(node) for node in reaching_stage.nodes} == {
        CartesianPoseStraight,
        LocalMinimumReached,
    }
    local_minimum = next(
        node for node in reaching_stage.nodes if isinstance(node, LocalMinimumReached)
    )
    assert local_minimum.joint_convergence_threshold == 0.025


# %% stretch base motion


@pytest.mark.skipif(skip_tests, reason="Alternative motion mappings not available")
def test_stretch_base_motion_follows_the_execution_environment(
    stretch_apartment_context,
):
    """
    One base motion resolves to a different mapping per execution environment, so a run
    on the robot drives the real base rather than silently simulating it.
    """
    world, robot, context = stretch_apartment_context
    context.alternative_motion_mappings = [StretchMoveSim, StretchMoveReal]
    motion = MoveMotion(Pose.from_xyz_rpy(1, 1, 0, reference_frame=world.root))
    execute_single(motion, context=context)

    with real_robot:
        assert motion.get_alternative_motion() is StretchMoveReal
        assert isinstance(motion.motion_chart, DifferentialDriveBaseGoal)

    with simulated_robot:
        assert motion.get_alternative_motion() is StretchMoveSim


# %% driving a container's own degree of freedom


def test_opening_motion_yields_to_collision_avoidance(pr2_apartment_context):
    """
    Pulling a drawer contorts the arm against the robot's own body, so the goal driving
    the container must not outrank collision avoidance: at a higher weight the solver
    buys the drawer trajectory by pushing the arm through whatever is in its way.
    """
    world, view, context = pr2_apartment_context
    handle = world.get_body_by_name("handle_cab3_door_top")

    motion = OpeningMotion(object_part=handle, arm=left_or_only_arm(context.robot))
    execute_single(motion, context=context)

    assert (
        motion.motion_chart.mechanism_weight
        == DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
    )


def test_opening_motion_keeps_the_gripper_on_the_handle(pr2_apartment_context):
    """
    Only the container's own degree of freedom yields to collision avoidance.

    The goal holding the gripper on the handle stays above it, because at a lower weight
    the solver buys clearance by letting the gripper drift off the handle, and handle
    and container move independently.
    """
    world, view, context = pr2_apartment_context
    handle = world.get_body_by_name("handle_cab3_door_top")

    motion = OpeningMotion(object_part=handle, arm=left_or_only_arm(context.robot))
    execute_single(motion, context=context)

    assert (
        motion.motion_chart.grasp_weight
        == DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    )


def test_closing_motion_yields_to_collision_avoidance(pr2_apartment_context):
    """
    Pushing a drawer shut is the same motion run backwards and needs the same weight.
    """
    world, view, context = pr2_apartment_context
    handle = world.get_body_by_name("handle_cab3_door_top")

    motion = ClosingMotion(object_part=handle, arm=left_or_only_arm(context.robot))
    execute_single(motion, context=context)

    assert (
        motion.motion_chart.mechanism_weight
        == DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
    )


def test_closing_motion_keeps_the_gripper_on_the_handle(pr2_apartment_context):
    """
    Closing holds the handle the same way opening does.
    """
    world, view, context = pr2_apartment_context
    handle = world.get_body_by_name("handle_cab3_door_top")

    motion = ClosingMotion(object_part=handle, arm=left_or_only_arm(context.robot))
    execute_single(motion, context=context)

    assert (
        motion.motion_chart.grasp_weight
        == DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    )


def test_grasping_action_frees_the_gripper_for_its_whole_approach(
    pr2_apartment_context,
):
    """
    Both halves of a grasp end up inside the buffer zone kept around what is grasped:
    the pre-pose is placed off the body's own geometry, so holding the gripper clear
    there stalls the approach before it ever reaches the object, the same way it would
    at the grasp itself.
    """
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    grasping = GraspingAction(
        GraspCandidate.from_body_origin(milk), left_or_only_arm(context.robot)
    )
    sequential([grasping], context=context)

    grasping.plan_node.notify()
    reach_nodes = grasping.plan_node.plan.get_nodes_by_designator_type(
        MoveToolCenterPointMotion
    )
    assert len(reach_nodes) == 2
    assert all(node.designator.allow_gripper_collision is True for node in reach_nodes)


@pytest.mark.skipif(
    not daisy_mappings_available,
    reason="DAiSy motion mappings not available",
)
class TestDAiSyGripMotion:
    def _grip_motion(
        self,
        immutable_daisy_world,
        state_type=GripperState.CLOSE,
    ):
        _, robot, context = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        motion_obj = DAiSyGripMotion(
            configuration=end_effector.default_configuration(state_type)
        )
        execute_single(motion_obj, context=context)
        return motion_obj

    def test_semi_real_returns_joint_position_list(self, immutable_daisy_world):
        motion = self._grip_motion(immutable_daisy_world)
        with semi_real_robot:
            chart = motion._motion_chart
        assert isinstance(chart, JointPositionList)
        assert chart.name == "left_gripper_close"

    def test_semi_real_open_returns_joint_position_list(self, immutable_daisy_world):
        motion = self._grip_motion(immutable_daisy_world, state_type=GripperState.OPEN)
        with semi_real_robot:
            chart = motion._motion_chart
        assert isinstance(chart, JointPositionList)
        assert chart.name == "left_gripper_open"

    def test_simulated_returns_joint_position_list(self, immutable_daisy_world):
        motion = self._grip_motion(immutable_daisy_world)
        with simulated_robot:
            chart = motion._motion_chart
        assert isinstance(chart, JointPositionList)
        assert chart.name == "left_gripper_close"

    def test_simulated_open_returns_joint_position_list(self, immutable_daisy_world):
        motion = self._grip_motion(immutable_daisy_world, state_type=GripperState.OPEN)
        with simulated_robot:
            chart = motion._motion_chart
        assert isinstance(chart, JointPositionList)
        assert chart.name == "left_gripper_open"

    def test_real_returns_griplink_action_server_task(self, immutable_daisy_world):
        motion = self._grip_motion(immutable_daisy_world)
        with real_robot:
            chart = motion._motion_chart
        assert isinstance(chart, Parallel)
        assert len(chart.nodes) == 1
        assert isinstance(chart.nodes[0], GriplinkPresetActionServerTask)

    def test_flex_state_raises_no_griplink_endpoint(self, immutable_daisy_world):
        """
        The preset motion commands no flex states, so building its chart for one raises
        the endpoint exception instead of silently building a wrong task.
        """
        _, robot, _ = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        motion = DAiSyGripMotion(
            configuration=GriplinkFlexConfiguration.from_state_type(
                end_effector, GripperState.FLEXCLOSE
            )
        )
        with real_robot, pytest.raises(NoGriplinkEndpoint) as caught:
            motion._motion_chart
        assert caught.value.end_effector is end_effector
        assert caught.value.state_type is GripperState.FLEXCLOSE


@pytest.mark.skipif(
    not daisy_mappings_available,
    reason="DAiSy motion mappings not available",
)
class TestDAiSyFlexGripMotion:
    def _flex_motion(
        self,
        immutable_daisy_world,
        state_type=GripperState.FLEXCLOSE,
        **grip_parameters,
    ):
        _, robot, context = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        motion_obj = DAiSyFlexGripMotion(
            configuration=GriplinkFlexConfiguration.from_state_type(
                end_effector, state_type, **grip_parameters
            )
        )
        execute_single(motion_obj, context=context)
        return motion_obj

    def test_semi_real_returns_joint_position_list(self, immutable_daisy_world):
        motion = self._flex_motion(immutable_daisy_world)
        with semi_real_robot:
            chart = motion._motion_chart
        assert isinstance(chart, JointPositionList)
        assert chart.name == "flexgrip"

    def test_semi_real_flexopen_returns_joint_position_list(
        self, immutable_daisy_world
    ):
        motion = self._flex_motion(
            immutable_daisy_world, state_type=GripperState.FLEXOPEN
        )
        with semi_real_robot:
            chart = motion._motion_chart
        assert isinstance(chart, JointPositionList)
        assert chart.name == "flexgrip"

    def test_simulated_returns_joint_position_list(self, immutable_daisy_world):
        motion = self._flex_motion(immutable_daisy_world)
        with simulated_robot:
            chart = motion._motion_chart
        assert isinstance(chart, JointPositionList)
        assert chart.name == "flexgrip"

    def test_semi_real_target_within_declared_states(self, immutable_daisy_world):
        motion = self._flex_motion(
            immutable_daisy_world,
            state_type=GripperState.FLEXCLOSE,
            grip_position=60,
        )
        with semi_real_robot:
            chart = motion._motion_chart
        _, robot, _ = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        open_state = end_effector.get_joint_state_by_type(GripperState.OPEN)
        close_state = end_effector.get_joint_state_by_type(GripperState.CLOSE)
        close_targets = dict(close_state.items())
        expected = [
            open_target + 0.5 * (close_targets[connection] - open_target)
            for connection, open_target in open_state.items()
        ]
        assert chart.goal_state.target_values == pytest.approx(expected)

    def test_semi_real_full_open_maps_to_the_open_state(self, immutable_daisy_world):
        motion = self._flex_motion(
            immutable_daisy_world,
            state_type=GripperState.FLEXOPEN,
            grip_position=120,
        )
        with semi_real_robot:
            chart = motion._motion_chart
        _, robot, _ = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        open_state = end_effector.get_joint_state_by_type(GripperState.OPEN)
        for connection, target in chart.goal_state.items():
            expected = dict(open_state.items())[connection]
            assert abs(target - expected) < 0.001, (
                f"Expected open state {expected}, got {target}"
            )

    def test_semi_real_full_close_maps_to_the_close_state(self, immutable_daisy_world):
        """
        The fully closed flex target is the declared close state, not the connection's
        upper position limit: past the close state the fingers cross and separate
        again, so a limit-anchored target would open the gripper instead of closing it.
        """
        motion = self._flex_motion(
            immutable_daisy_world,
            state_type=GripperState.FLEXCLOSE,
            grip_position=0,
        )
        with semi_real_robot:
            chart = motion._motion_chart
        _, robot, _ = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        close_state = end_effector.get_joint_state_by_type(GripperState.CLOSE)
        for connection, target in chart.goal_state.items():
            expected = dict(close_state.items())[connection]
            assert abs(target - expected) < 0.001, (
                f"Expected close state {expected}, got {target}"
            )

    def test_real_returns_griplink_action_server_task(self, immutable_daisy_world):
        motion = self._flex_motion(immutable_daisy_world)
        with real_robot:
            chart = motion._motion_chart
        assert isinstance(chart, Parallel)
        assert len(chart.nodes) == 1
        assert isinstance(chart.nodes[0], GriplinkFlexActionServerTask)


# %% DAiSy configuration routing and parameter forwarding


@pytest.mark.skipif(
    not daisy_mappings_available,
    reason="DAiSy motion mappings not available",
)
class TestDAiSyGripperConfigurationRouting:
    """
    A base :class:`MoveGripperMotion` with both DAiSy alternatives registered routes to
    the correct alternative by configuration type, and the configuration's hardware
    parameters reach the griplink action server task.
    """

    def _motion_through_dispatch(
        self,
        immutable_daisy_world,
        configuration,
    ):
        _, _, context = immutable_daisy_world
        context.alternative_motion_mappings = [
            DAiSyGripMotion,
            DAiSyFlexGripMotion,
        ]
        motion_obj = MoveGripperMotion(configuration=configuration)
        execute_single(motion_obj, context=context)
        return motion_obj

    def test_preset_configuration_routes_to_grip_motion(self, immutable_daisy_world):
        _, robot, _ = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        motion = self._motion_through_dispatch(
            immutable_daisy_world,
            end_effector.default_configuration(GripperState.CLOSE),
        )
        with real_robot:
            assert motion.get_alternative_motion() is DAiSyGripMotion

    def test_flex_configuration_routes_to_flex_motion(self, immutable_daisy_world):
        _, robot, _ = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        motion = self._motion_through_dispatch(
            immutable_daisy_world,
            GriplinkFlexConfiguration.from_state_type(
                end_effector, GripperState.FLEXCLOSE
            ),
        )
        with real_robot:
            assert motion.get_alternative_motion() is DAiSyFlexGripMotion

    def test_simulated_dispatch_builds_joint_position_goal(self, immutable_daisy_world):
        """
        The full dispatch, not only the alternative's own chart, routes to the joint
        position goal for simulated execution.
        """
        _, robot, _ = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        motion = self._motion_through_dispatch(
            immutable_daisy_world,
            end_effector.default_configuration(GripperState.CLOSE),
        )
        with simulated_robot:
            chart = motion.motion_chart
        assert isinstance(chart, JointPositionList)
        assert chart.name == "left_gripper_close"

    def test_semi_real_dispatch_builds_joint_position_goal(self, immutable_daisy_world):
        _, robot, _ = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        motion = self._motion_through_dispatch(
            immutable_daisy_world,
            end_effector.default_configuration(GripperState.CLOSE),
        )
        with semi_real_robot:
            chart = motion.motion_chart
        assert isinstance(chart, JointPositionList)
        assert chart.name == "left_gripper_close"

    def test_grip_motion_forwards_default_preset(self, immutable_daisy_world):
        _, robot, _ = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        configuration = end_effector.default_configuration(GripperState.CLOSE)
        motion = self._motion_through_dispatch(immutable_daisy_world, configuration)
        with real_robot:
            chart = motion.motion_chart
        task = chart.nodes[0]
        assert isinstance(task, GriplinkPresetActionServerTask)
        assert task.grip_preset is configuration.grip_preset

    def test_grip_motion_explicit_configuration_overrides_default(
        self, immutable_daisy_world
    ):
        _, robot, _ = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        configuration = GriplinkPresetConfiguration.from_state_type(
            end_effector,
            GripperState.CLOSE,
            grip_preset=GriplinkGripPreset.PRESET_3,
        )
        motion = self._motion_through_dispatch(immutable_daisy_world, configuration)
        with real_robot:
            chart = motion.motion_chart
        task = chart.nodes[0]
        assert isinstance(task, GriplinkPresetActionServerTask)
        assert task.grip_preset is configuration.grip_preset

    def test_flexgrip_motion_forwards_position_force_speed_acceleration(
        self, immutable_daisy_world
    ):
        _, robot, _ = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        configuration = GriplinkFlexConfiguration.from_state_type(
            end_effector,
            GripperState.FLEXCLOSE,
            grip_position=60,
            grip_force=100,
            grip_velocity=50,
            grip_acceleration=2000,
        )
        motion = self._motion_through_dispatch(immutable_daisy_world, configuration)
        with real_robot:
            chart = motion.motion_chart
        task = chart.nodes[0]
        assert isinstance(task, GriplinkFlexActionServerTask)
        assert task.grip_position == configuration.grip_position
        assert task.grip_force == configuration.grip_force
        assert task.grip_velocity == configuration.grip_velocity
        assert task.grip_acceleration == configuration.grip_acceleration

    def test_flexrelease_motion_forwards_position_speed_acceleration(
        self, immutable_daisy_world
    ):
        _, robot, _ = immutable_daisy_world
        end_effector = robot.left_arm.end_effector
        configuration = GriplinkFlexConfiguration.from_state_type(
            end_effector,
            GripperState.FLEXOPEN,
            grip_position=30,
            grip_velocity=80,
            grip_acceleration=1500,
        )
        motion = self._motion_through_dispatch(immutable_daisy_world, configuration)
        with real_robot:
            chart = motion.motion_chart
        task = chart.nodes[0]
        assert isinstance(task, GriplinkFlexActionServerTask)
        assert task.grip_position == configuration.grip_position
        assert task.grip_velocity == configuration.grip_velocity
        assert task.grip_acceleration == configuration.grip_acceleration
