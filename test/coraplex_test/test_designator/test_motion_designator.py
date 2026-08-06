from copy import deepcopy

import numpy as np
import pytest

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    ApproachDirection,
    VerticalAlignment,
    Arms,
    MovementType,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot, real_robot
from coraplex.plans.factories import sequential, execute_single
from coraplex.plans.plan_node import MotionNode, ActionNode
from coraplex.robot_plans import MoveMotion, MoveToolCenterPointMotion
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose

try:
    from coraplex.alternative_motion_mappings.hsrb_motion_mapping import *
    from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
        NavigateActionServerTask,
    )

    skip_tests = False
except (ImportError, ModuleNotFoundError, AttributeError):
    skip_tests = True


@pytest.mark.skipif(skip_tests, reason="Alternative motion mappings not available")
def test_pick_up_motion(immutable_model_world):
    world, view, context = immutable_model_world
    test_world = deepcopy(world)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        view.left_arm.end_effector,
    )
    pick_up = PickUpAction(
        test_world.get_body_by_name("milk.stl"), Arms.LEFT, grasp_description
    )

    root = sequential(
        children=[
            ActionNode(
                designator=NavigateAction(
                    Pose(
                        Point3.from_iterable([1.7, 1.5, 0]),
                        Quaternion.from_iterable([0, 0, 0, 1]),
                        test_world.root,
                    ),
                    True,
                )
            ),
            MoveTorsoAction(TorsoState.HIGH),
            pick_up,
        ],
        context=Context.from_world(test_world),
    )
    assert pick_up.plan is not None
    with simulated_robot:
        root.perform()

    pick_up_node = root.plan.get_nodes_by_designator_type(PickUpAction)[0]

    motion_nodes = list(
        filter(lambda x: isinstance(x, MotionNode), pick_up_node.descendants)
    )

    assert len(motion_nodes) == 5

    motion_charts = [type(m.designator.motion_chart) for m in motion_nodes]
    assert all(mc is not None for mc in motion_charts)
    assert CartesianPose in motion_charts
    assert JointPositionList in motion_charts


def test_move_motion_chart(immutable_model_world):
    world, view, context = immutable_model_world
    motion = MoveMotion(
        Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)
    )
    plan = execute_single(
        motion,
        context=context,
    )

    msc = motion.motion_chart

    assert msc
    np.testing.assert_equal(msc.goal_pose.to_position().to_np(), np.array([1, 1, 1, 1]))


def test_move_tool_center_point_motion_uses_tight_threshold(immutable_model_world):
    """
    MoveToolCenterPointMotion drives grasp approaches, so it must not fall back to
    Giskard's loose default CartesianPose/CartesianPosition threshold (0.01m): that
    tolerance is wide enough to let the gripper stop a centimeter away from a small
    object, e.g. missing or off-center grasps.
    """
    world, view, context = immutable_model_world
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    cartesian_motion = MoveToolCenterPointMotion(
        target, Arms.LEFT, movement_type=MovementType.CARTESIAN
    )
    execute_single(cartesian_motion, context=context)
    assert isinstance(cartesian_motion.motion_chart, CartesianPose)
    assert (
        cartesian_motion.motion_chart.translation_threshold
        == context.motion_tolerances.default_tcp_position_threshold
    )

    translation_motion = MoveToolCenterPointMotion(
        target, Arms.LEFT, movement_type=MovementType.TRANSLATION
    )
    execute_single(translation_motion, context=context)
    assert (
        translation_motion.motion_chart.threshold
        == context.motion_tolerances.default_tcp_position_threshold
    )


def test_move_tool_center_point_motion_without_max_velocity_returns_bare_task(
    immutable_model_world,
):
    """
    MoveToolCenterPointMotion must not add any velocity-limit constraint when neither
    ``max_linear_velocity`` nor ``max_angular_velocity`` is set, so a caller that never
    mentions them keeps relying on the robot's own hardware velocity limits only.
    """
    world, view, context = immutable_model_world
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    motion = MoveToolCenterPointMotion(
        target, Arms.LEFT, movement_type=MovementType.CARTESIAN
    )
    execute_single(motion, context=context)
    assert isinstance(motion.motion_chart, CartesianPose)


def test_move_tool_center_point_motion_max_linear_velocity_adds_real_limit(
    immutable_model_world,
):
    """
    An explicit ``max_linear_velocity`` must add a real
    :class:`CartesianPositionVelocityLimit` constraint alongside the goal task via
    ``Parallel``, instead of tuning the goal task's own reference velocity -- per review
    feedback, reference velocities are for QP normalization only and must not be exposed
    as a caller-tunable speed limit.
    """
    world, view, context = immutable_model_world
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    motion = MoveToolCenterPointMotion(
        target,
        Arms.LEFT,
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
    immutable_model_world,
):
    """
    An explicit ``max_angular_velocity`` must add a real
    :class:`CartesianRotationVelocityLimit` constraint, only meaningful for the non-
    translation (full 6D pose) movement type.
    """
    world, view, context = immutable_model_world
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    motion = MoveToolCenterPointMotion(
        target,
        Arms.LEFT,
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


def test_move_gripper_motion_finger_velocity_adds_real_limit(immutable_model_world):
    """
    An explicit ``finger_velocity`` must add a real
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointVelocityLimit`
    constraint alongside the goal task, instead of tuning the goal task's own
    reference/normalization velocity.
    """
    world, view, context = immutable_model_world

    close_motion = MoveGripperMotion(
        motion=GripperState.CLOSE, gripper=Arms.LEFT, finger_velocity=0.03
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
    immutable_model_world,
):
    """
    ``tolerate_stall`` and ``finger_velocity`` set together must nest correctly: the
    motion is done once (goal reached OR stalled) AND the finger velocity stayed within
    its limit -- not a single flat ``Parallel`` that conflates OR and AND semantics.
    """
    world, view, context = immutable_model_world

    close_motion = MoveGripperMotion(
        motion=GripperState.CLOSE,
        gripper=Arms.LEFT,
        tolerate_stall=True,
        finger_velocity=0.03,
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


def test_move_gripper_motion_tolerate_stall_defaults_to_false(immutable_model_world):
    """
    MoveGripperMotion must not tolerate a stall by default, for either OPEN or CLOSE --
    stalling before reaching the target is a real failure that should be surfaced,
    unless a caller (e.g. PickUpAction, grasping a real object) explicitly opts in via
    ``tolerate_stall=True``.

    A caller that never mentions this field must keep relying on the original,
    unmodified default behaviour: the plain goal task, not wrapped in any stall-tolerant
    monitor.
    """
    world, view, context = immutable_model_world

    close_motion = MoveGripperMotion(motion=GripperState.CLOSE, gripper=Arms.LEFT)
    execute_single(close_motion, context=context)
    assert isinstance(close_motion.motion_chart, JointPositionList)

    open_motion = MoveGripperMotion(motion=GripperState.OPEN, gripper=Arms.LEFT)
    execute_single(open_motion, context=context)
    assert isinstance(open_motion.motion_chart, JointPositionList)


def test_move_gripper_motion_tolerate_stall_can_be_explicitly_enabled(
    immutable_model_world,
):
    """
    An explicit ``tolerate_stall=True`` must wrap the goal task together with a
    :class:`LocalMinimumReached` monitor in a :class:`Parallel` (with
    ``minimum_success=1``), so the motion is considered done as soon as either the goal
    is reached or the fingers have stalled -- without changing what the goal task's own
    observation means (goal reached, nothing else).
    """
    world, view, context = immutable_model_world

    close_motion = MoveGripperMotion(
        motion=GripperState.CLOSE, gripper=Arms.LEFT, tolerate_stall=True
    )
    execute_single(close_motion, context=context)
    assert isinstance(close_motion.motion_chart, Parallel)
    assert close_motion.motion_chart.minimum_success == 1
    node_types = [type(node) for node in close_motion.motion_chart.nodes]
    assert JointPositionList in node_types
    assert LocalMinimumReached in node_types


def test_pick_up_action_close_motion_stall_tolerance_defaults_to_false(
    immutable_model_world,
):
    """
    PickUpAction's grasp-closing motion must not tolerate a stall unless explicitly
    asked to: building the stall monitor needs a velocity variable for every one of the
    gripper's connections, which not every robot has, so it must stay opt-in rather than
    always on (it crashes on Tracy's real-execution gripper otherwise).
    """
    world, view, context = immutable_model_world
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        view.left_arm.end_effector,
    )
    pick_up = PickUpAction(
        world.get_body_by_name("milk.stl"), Arms.LEFT, grasp_description
    )
    sequential([pick_up], context=context)

    close_motion_nodes = pick_up._action_plan.plan.get_nodes_by_designator_type(
        MoveGripperMotion
    )
    assert len(close_motion_nodes) == 1
    assert close_motion_nodes[0].designator.tolerate_stall is False


def test_pick_up_action_close_motion_tolerates_stall_when_enabled(
    immutable_model_world,
):
    """
    PickUpAction's ``tolerate_grasp_stall`` must reach the grasp's CLOSE motion, so a
    grasped object's fingers physically stopping before the nominal fully-closed target
    is correctly treated as a real grasp, not a failed motion, once explicitly enabled.
    """
    world, view, context = immutable_model_world
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        view.left_arm.end_effector,
    )
    pick_up = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        grasp_description,
        tolerate_grasp_stall=True,
    )
    sequential([pick_up], context=context)

    close_motion_nodes = pick_up._action_plan.plan.get_nodes_by_designator_type(
        MoveGripperMotion
    )
    assert len(close_motion_nodes) == 1
    assert close_motion_nodes[0].designator.tolerate_stall is True


def test_pick_up_action_velocity_fields_default_to_none(immutable_model_world):
    """
    PickUpAction's velocity/timing/friction fields must all default to ``None`` when not
    explicitly set, so an existing caller that never mentions them keeps relying on
    Giskard's own task defaults instead of a new, silently-injected value -- these
    physics fields are opt-in additions, not a change to the action's default behaviour.
    """
    world, view, context = immutable_model_world
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        view.left_arm.end_effector,
    )

    pick_up = PickUpAction(
        world.get_body_by_name("milk.stl"), Arms.LEFT, grasp_description
    )

    assert pick_up.pre_approach_linear_velocity is None
    assert pick_up.final_approach_linear_velocity is None
    assert pick_up.grasp_closing_velocity is None
    assert pick_up.lift_linear_velocity is None
    assert pick_up.grasp_stall_minimum_time is None
    assert pick_up.object_friction is None


def test_place_action_velocity_fields_default_to_none(immutable_model_world):
    """
    PlaceAction's velocity/timing fields must all default to ``None`` when not
    explicitly set, matching PickUpAction's own opt-in design: an existing caller that
    never mentions them keeps relying on Giskard's own task defaults.
    """
    world, view, context = immutable_model_world
    target_location = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    place = PlaceAction(world.get_body_by_name("milk.stl"), target_location, Arms.LEFT)

    assert place.placing_linear_velocity is None
    assert place.transport_linear_velocity is None
    assert place.release_opening_velocity is None
    assert place.retract_linear_velocity is None


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
