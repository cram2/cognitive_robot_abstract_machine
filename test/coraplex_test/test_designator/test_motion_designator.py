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
    ApproachDirection,
    VerticalAlignment,
    Arms,
    MovementType,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot, real_robot, no_execution
from coraplex.plans.factories import sequential, execute_single
from coraplex.plans.plan_node import MotionNode, ActionNode
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from coraplex.robot_plans.motions.container import (
    ClosingMotion,
    OpeningMotion,
)
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from coraplex.view_manager import ViewManager
from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.goals.collision_avoidance import (
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.open_close import Close, Open
from giskardpy.motion_statechart.data_types import DefaultWeights
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
from giskardpy.motion_statechart.tasks.pointing import Pointing
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.spatial_types.spatial_types import Pose

try:
    from coraplex.alternative_motion_mappings.hsrb_motion_mapping import *
    from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
        NavigateActionServerTask,
    )

    skip_tests = False
except (ImportError, ModuleNotFoundError, AttributeError):
    skip_tests = True


def _nodes_of(chart):
    """
    :return: Every node a motion chart holds, looking through the wrappers a motion puts
        its goal in when it also carries collision rules, velocity limits, a stall
        monitor or a hold.
    """
    children = getattr(chart, "nodes", [])
    return [chart] + [node for child in children for node in _nodes_of(child)]


def _goal_types(chart):
    """
    :return: The types of every node in a motion chart.
    """
    return [type(node) for node in _nodes_of(chart)]


def _container_goal(motion):
    """
    :return: The goal driving the container in a container motion's chart.
    """
    return next(
        node for node in _nodes_of(motion.motion_chart) if isinstance(node, Open)
    )


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

    goal_types = [
        goal_type
        for node in motion_nodes
        for goal_type in _goal_types(node.designator.motion_chart)
    ]
    assert CartesianPose in goal_types
    assert JointPositionList in goal_types


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


def test_move_tool_center_point_motion_allows_the_gripper_to_touch_what_it_grasps(
    immutable_model_world,
):
    """
    ``allow_gripper_collision`` must reach the collision manager: without the rule that
    frees the manipulator, external collision avoidance keeps the gripper a buffer zone
    away from whatever it reaches for, and the grasp never closes on it.
    """
    world, view, context = immutable_model_world
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    motion = MoveToolCenterPointMotion(
        target,
        Arms.LEFT,
        allow_gripper_collision=True,
        movement_type=MovementType.CARTESIAN,
    )
    execute_single(motion, context=context)

    rules_node = next(
        node
        for node in motion.motion_chart.nodes
        if isinstance(node, UpdateTemporaryCollisionRules)
    )
    allowed_bodies = set(
        ViewManager().get_end_effector_view(Arms.LEFT, view).bodies_with_collision
    )
    assert set(rules_node.temporary_rules[0].body_group_b) == allowed_bodies


def test_move_tool_center_point_motion_frees_what_the_gripper_carries(
    mutable_model_world,
):
    """
    What the gripper carries is not one of the end effector's own bodies, it only hangs
    below its tool frame, so freeing the manipulator has to free it too -- otherwise
    external collision avoidance keeps the carried object a buffer zone away from the
    surface it is being put down on and the placing aborts on the contact it was asked
    to make.
    """
    world, view, context = mutable_model_world
    end_effector = ViewManager().get_end_effector_view(Arms.LEFT, view)
    carried = world.get_body_by_name("milk.stl")
    with world.modify_world():
        world.move_branch(carried, end_effector.tool_frame)
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    motion = MoveToolCenterPointMotion(target, Arms.LEFT, allow_gripper_collision=True)
    execute_single(motion, context=context)

    rules_node = next(
        node
        for node in motion.motion_chart.nodes
        if isinstance(node, UpdateTemporaryCollisionRules)
    )
    assert carried in rules_node.temporary_rules[0].body_group_b


def test_move_tool_center_point_motion_keeps_the_gripper_clear_by_default(
    immutable_model_world,
):
    """
    Without ``allow_gripper_collision`` the motion adds no collision rule of its own, so
    the robot's own rules decide how close the gripper may come.
    """
    world, view, context = immutable_model_world
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    motion = MoveToolCenterPointMotion(target, Arms.LEFT)
    execute_single(motion, context=context)

    assert isinstance(motion.motion_chart, CartesianPose)


def test_move_tool_center_point_motion_outranks_buffers_when_it_may_touch(
    immutable_model_world,
):
    """
    A move that is allowed to touch what it manipulates must outweigh the buffer zones
    kept around it, or collision avoidance gives the goal up at the edge of one and the
    reach stops short of what it was allowed to reach.
    """
    world, view, context = immutable_model_world
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)

    touching = MoveToolCenterPointMotion(
        target, Arms.LEFT, allow_gripper_collision=True
    )
    execute_single(touching, context=context)
    keeping_clear = MoveToolCenterPointMotion(target, Arms.LEFT)
    execute_single(keeping_clear, context=context)

    goal = next(
        node for node in touching.motion_chart.nodes if isinstance(node, CartesianPose)
    )
    assert goal.weight == DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    assert (
        keeping_clear.motion_chart.weight
        == DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
    )


def _motions_of(action, context):
    """
    :return: The motions an action expands into, without executing any of them.
    """
    plan = sequential([action], context=context)
    with no_execution:
        plan.perform()
    return [
        node.designator
        for node in plan.plan.get_nodes_by_designator_type(type(action))[0].descendants
        if isinstance(node, MotionNode)
    ]


def test_reaching_for_an_object_lets_the_gripper_touch_it(immutable_model_world):
    """
    A gripper that may not touch what it reaches for cannot close on it once external
    collision avoidance is on, so both moves of a reach have to allow it.
    """
    world, view, context = immutable_model_world
    test_world = deepcopy(world)
    milk = test_world.get_body_by_name("milk.stl")
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        test_world.get_semantic_annotations_by_type(PR2)[0].left_arm.end_effector,
    )

    motions = _motions_of(
        ReachAction(
            target_pose=Pose(reference_frame=milk),
            object_designator=milk,
            arm=Arms.LEFT,
            grasp_description=grasp_description,
        ),
        Context.from_world(test_world),
    )

    reaches = [m for m in motions if isinstance(m, MoveToolCenterPointMotion)]
    assert len(reaches) == 2
    assert all(motion.allow_gripper_collision for motion in reaches)


def test_placing_lets_the_gripper_touch_what_it_sets_down(immutable_model_world):
    """
    The hand holds the object it is putting down, so the moves that carry it there and
    lower it must allow the contact that placing is.
    """
    world, view, context = immutable_model_world
    test_world = deepcopy(world)
    milk = test_world.get_body_by_name("milk.stl")
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=test_world.root)

    motions = _motions_of(
        PlaceAction(milk, target, Arms.LEFT), Context.from_world(test_world)
    )

    carrying, lowering = [
        m for m in motions if isinstance(m, MoveToolCenterPointMotion)
    ][:2]
    assert carrying.allow_gripper_collision
    assert lowering.allow_gripper_collision

    release = next(m for m in motions if isinstance(m, MoveGripperMotion))
    assert release.allow_gripper_collision


def test_a_gripper_that_may_touch_says_which_bodies_may(immutable_model_world):
    """
    Fingers that close on an object, or open around one standing on a surface, are in
    contact by the time they move.

    Collision avoidance has to be told, or the motion is ended over the very contact it
    was asked to make.
    """
    world, view, context = immutable_model_world

    touching = MoveGripperMotion(GripperState.CLOSE, Arms.LEFT, True)
    execute_single(touching, context=context)
    keeping_clear = MoveGripperMotion(GripperState.CLOSE, Arms.LEFT)
    execute_single(keeping_clear, context=context)

    rules_node = next(
        node
        for node in _nodes_of(touching.motion_chart)
        if isinstance(node, UpdateTemporaryCollisionRules)
    )
    assert set(rules_node.temporary_rules[0].body_group_b) == set(
        ViewManager().get_end_effector_view(Arms.LEFT, view).bodies_with_collision
    )
    assert not any(
        isinstance(node, UpdateTemporaryCollisionRules)
        for node in _nodes_of(keeping_clear.motion_chart)
    )


def test_move_gripper_motion_holds_the_tool_center_point(immutable_model_world):
    """
    Nothing else asks the arm to stay while a hand opens or closes, so without a hold of
    its own the arm drifts off what the motion before it reached -- and whatever is
    being put down is let go somewhere else.
    """
    world, view, context = immutable_model_world

    motion = MoveGripperMotion(GripperState.OPEN, Arms.LEFT)
    execute_single(motion, context=context)

    hold = next(
        node for node in motion.motion_chart.nodes if isinstance(node, CartesianPose)
    )
    tool_frame = ViewManager().get_end_effector_view(Arms.LEFT, view).tool_frame
    assert hold.tip_link == tool_frame
    assert hold.goal_pose.reference_frame == tool_frame
    assert hold.weight == DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE


def test_a_hand_keeps_its_goal_until_the_motion_ends(immutable_model_world):
    """
    The chart keeps ticking until the world settles.

    A motion that retires the moment its goal is observed leaves the robot free for
    those ticks, so the one that has to leave the robot where it put it must say that it
    holds on.
    """
    assert MoveGripperMotion.holds_its_goal_until_the_motion_ends
    assert not MoveToolCenterPointMotion.holds_its_goal_until_the_motion_ends


def test_container_motions_hold_the_handle_they_grasped(immutable_model_world):
    """
    The hand is on the handle while the container swings, so it has to be allowed to
    touch it -- otherwise collision avoidance keeps the gripper off what it is holding.
    """
    world, view, context = immutable_model_world
    handle = world.get_body_by_name("handle_cab10_m")

    opening = OpeningMotion(handle, Arms.LEFT)
    execute_single(opening, context=context)
    closing = ClosingMotion(handle, Arms.LEFT)
    execute_single(closing, context=context)

    for motion in (opening, closing):
        rules_node = next(
            node
            for node in _nodes_of(motion.motion_chart)
            if isinstance(node, UpdateTemporaryCollisionRules)
        )
        allowed = set(rules_node.temporary_rules[0].body_group_b)
        assert allowed == set(
            ViewManager().get_end_effector_view(Arms.LEFT, view).bodies_with_collision
        )


def test_a_container_does_not_outrank_keeping_the_robot_clear(immutable_model_world):
    """
    The hand stays on the handle while the container travels, so a container goal that
    outranks collision avoidance drags the whole robot after it -- into the very
    furniture the container is part of.
    """
    world, view, context = immutable_model_world
    handle = world.get_body_by_name("handle_cab10_m")

    opening = OpeningMotion(handle, Arms.LEFT)
    execute_single(opening, context=context)
    closing = ClosingMotion(handle, Arms.LEFT)
    execute_single(closing, context=context)

    for motion in (opening, closing):
        assert (
            _container_goal(motion).weight
            == DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
        )


def test_container_motions_drive_to_the_goal_state_they_were_given(
    immutable_model_world,
):
    """
    How far a container is driven is the caller's to say, and a motion that has an angle
    of its own opens every door equally far no matter what it was asked for.
    """
    world, view, context = immutable_model_world
    handle = world.get_body_by_name("handle_cab10_m")
    goal_state = 0.25

    opening = OpeningMotion(handle, Arms.LEFT, goal_state)
    execute_single(opening, context=context)
    closing = ClosingMotion(handle, Arms.LEFT, goal_state)
    execute_single(closing, context=context)
    as_far_as_it_goes = OpeningMotion(handle, Arms.LEFT)
    execute_single(as_far_as_it_goes, context=context)

    assert _container_goal(opening).goal_joint_state == goal_state
    assert _container_goal(closing).goal_joint_state == goal_state
    assert _container_goal(as_far_as_it_goes).goal_joint_state is None


def test_container_motions_tolerate_the_last_stretch_onto_the_limit(
    immutable_model_world,
):
    """
    A mechanism driven onto its own limit is only approached asymptotically, so a motion
    that insists on arriving exactly never reports that a shut container is shut.

    It is done once the container arrives *or* stops moving, which is as far as it goes.
    """
    world, view, context = immutable_model_world
    handle = world.get_body_by_name("handle_cab10_m")

    closing = ClosingMotion(handle, Arms.LEFT)
    execute_single(closing, context=context)

    stall_tolerant = next(
        node
        for node in _nodes_of(closing.motion_chart)
        if isinstance(node, Parallel) and node.minimum_success == 1
    )
    assert [type(node) for node in stall_tolerant.nodes] == [
        Close,
        LocalMinimumReached,
    ]


def test_container_motions_wait_out_the_stall_time_they_were_given(
    immutable_model_world,
):
    """
    How long a standstill has to last before it counts as as far as the container goes
    is the caller's to say, since it depends on how slowly the mechanism creeps onto its
    own limit.
    """
    world, view, context = immutable_model_world
    handle = world.get_body_by_name("handle_cab10_m")

    closing = ClosingMotion(handle, Arms.LEFT, stall_time=2.5)
    execute_single(closing, context=context)

    stall_monitor = next(
        node
        for node in _nodes_of(closing.motion_chart)
        if isinstance(node, LocalMinimumReached)
    )
    assert stall_monitor.minimum_time == closing.stall_time


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
    assert JointPositionList in _goal_types(close_motion.motion_chart)
    assert LocalMinimumReached not in _goal_types(close_motion.motion_chart)

    open_motion = MoveGripperMotion(motion=GripperState.OPEN, gripper=Arms.LEFT)
    execute_single(open_motion, context=context)
    assert JointPositionList in _goal_types(open_motion.motion_chart)
    assert LocalMinimumReached not in _goal_types(open_motion.motion_chart)


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
    stall_tolerant = next(
        node
        for node in _nodes_of(close_motion.motion_chart)
        if isinstance(node, Parallel) and node.minimum_success == 1
    )
    node_types = [type(node) for node in stall_tolerant.nodes]
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


# %% looking


def test_looking_motion_pointing_parameters(immutable_model_world):
    """
    The looking motion aims the camera's forward axis at the target, moving the head
    relative to the torso so the rest of the body stays where it is.
    """
    world, view, context = immutable_model_world
    camera = view.get_default_camera()
    target = Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root)
    motion = LookingMotion(target=target, camera=camera)
    execute_single(motion, context=context)

    pointing = motion.motion_chart

    assert isinstance(pointing, Pointing)
    assert pointing.root_link is view.get_torso().root
    assert pointing.tip_link is camera.root
    assert pointing.pointing_axis is camera.forward_facing_axis
    assert pointing.pointing_axis.reference_frame is camera.root
    assert pointing.goal_point.reference_frame is world.root
    assert np.array_equal(pointing.goal_point.to_np(), target.to_position().to_np())


# %% stretch tool center point


@pytest.mark.skipif(skip_tests, reason="Alternative motion mappings not available")
def test_stretch_tool_center_point_straightens_wrist_while_turning(
    immutable_stretch_apartment_world,
):
    """
    Straightening the wrist runs alongside the base rotation, so the gripper is already
    aligned once the cartesian goal takes over.
    """
    world, robot, context = immutable_stretch_apartment_world
    context.alternative_motion_mappings = [StretchMoveToolCenterPoint]
    motion = MoveToolCenterPointMotion(
        target=Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root),
        arm=Arms.LEFT,
    )
    execute_single(motion, context=context)

    with real_robot:
        turning_stage = motion.motion_chart.nodes[0]

    assert isinstance(turning_stage, Parallel)
    assert {type(node) for node in turning_stage.nodes} == {Pointing, JointPositionList}
    wrist_goal = next(
        node for node in turning_stage.nodes if isinstance(node, JointPositionList)
    )
    assert [
        connection.name.name for connection in wrist_goal.goal_state.connections
    ] == ["joint_wrist_yaw"]
    assert wrist_goal.goal_state.target_values == [0.0]


@pytest.mark.skipif(skip_tests, reason="Alternative motion mappings not available")
def test_stretch_tool_center_point_accepts_a_local_minimum(
    immutable_stretch_apartment_world,
):
    """
    The arm regularly settles just short of the goal pose, so converging into a local
    minimum counts as success alongside reaching the pose.
    """
    world, robot, context = immutable_stretch_apartment_world
    context.alternative_motion_mappings = [StretchMoveToolCenterPoint]
    motion = MoveToolCenterPointMotion(
        target=Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root),
        arm=Arms.LEFT,
    )
    execute_single(motion, context=context)

    with real_robot:
        reaching_stage = motion.motion_chart.nodes[1]

    assert isinstance(reaching_stage, Parallel)
    assert reaching_stage.minimum_success == 1
    assert {type(node) for node in reaching_stage.nodes} == {
        CartesianPose,
        LocalMinimumReached,
    }
    local_minimum = next(
        node for node in reaching_stage.nodes if isinstance(node, LocalMinimumReached)
    )
    assert local_minimum.joint_convergence_threshold == 0.025


@pytest.mark.skipif(skip_tests, reason="Alternative motion mappings not available")
def test_stretch_tool_center_point_allows_the_gripper_to_touch_what_it_grasps(
    immutable_stretch_apartment_world,
):
    """
    A robot with a mapping of its own runs the same plan as one without, so the mapping
    has to honour ``allow_gripper_collision`` too: dropping the rule leaves external
    collision avoidance to abort the very contact the motion was allowed to make.
    """
    world, robot, context = immutable_stretch_apartment_world
    context.alternative_motion_mappings = [StretchMoveToolCenterPoint]
    motion = MoveToolCenterPointMotion(
        target=Pose(Point3.from_iterable([1, 1, 1]), reference_frame=world.root),
        arm=Arms.LEFT,
        allow_gripper_collision=True,
    )
    execute_single(motion, context=context)

    with real_robot:
        chart = motion.motion_chart

    rules_node = next(
        node for node in chart.nodes if isinstance(node, UpdateTemporaryCollisionRules)
    )
    assert set(rules_node.temporary_rules[0].body_group_b) == set(
        ViewManager().get_end_effector_view(Arms.LEFT, robot).bodies_with_collision
    )


# %% stretch base motion


@pytest.mark.skipif(skip_tests, reason="Alternative motion mappings not available")
def test_stretch_base_motion_follows_the_execution_environment(
    immutable_stretch_apartment_world,
):
    """
    One base motion resolves to a different mapping per execution environment, so a run
    on the robot drives the real base rather than silently simulating it.
    """
    world, robot, context = immutable_stretch_apartment_world
    context.alternative_motion_mappings = [StretchMoveSim, StretchMoveReal]
    motion = MoveMotion(Pose.from_xyz_rpy(1, 1, 0, reference_frame=world.root))
    execute_single(motion, context=context)

    with real_robot:
        assert motion.get_alternative_motion() is StretchMoveReal
        assert isinstance(motion.motion_chart, DifferentialDriveBaseGoal)

    with simulated_robot:
        assert motion.get_alternative_motion() is StretchMoveSim
