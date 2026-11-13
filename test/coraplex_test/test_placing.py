import numpy as np
import pytest

from krrood.entity_query_language.factories import a
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from cramph.composites import Sequence
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.robot_parts import Arm
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from krrood_spatial_math.spatial_types import Pose

from ..conftest import SAMPLING_SEED
from ..plan_running import expand, simulated_executor, statechart_of

# %% fixtures

HELD_AT = (0.03, -0.02, 0.05)
"""
Where the held object's origin sits relative to the tool frame, in meters; off-centre,
as a rim grasp would hold it.
"""

HELD_YAW = np.pi / 3
"""
How far the held object is turned about the tool frame's z-axis.
"""


@pytest.fixture
def pr2_holding_milk(simple_pr2_context):
    """
    A PR2 whose left tool frame holds the milk off-centre, at :data:`HELD_AT`, and the
    extensions it acts in.
    """
    world, robot, extensions = simple_pr2_context
    milk_body = world.get_body_by_name("milk.stl")
    milk = Milk(root=milk_body)
    tool_frame = robot.left_arm.end_effector.tool_frame
    with world.modify_world():
        world.move_branch(milk_body, tool_frame)
        world.add_semantic_annotation(milk)
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        *HELD_AT, yaw=HELD_YAW, reference_frame=tool_frame
    )
    world.notify_state_change()
    return world, robot, milk, extensions


def _tool_goals_of(place: PlaceAction) -> list[CartesianPose]:
    """
    :return: The tool center point goals the expanded `place` runs, in order.
    """
    return [node for node in place.descendants if isinstance(node, CartesianPose)]


def _start(place: PlaceAction, extensions) -> PlaceAction:
    """
    Compile `place` in a statechart of its own and start it, which measures the grasp
    its goals release.

    :return: `place`, started.
    """
    executor = simulated_executor(extensions)
    executor.compile(statechart=statechart_of(executor, Sequence(nodes=[place])))
    place.on_start()
    return place


# %% releasing what is actually held


def test_a_place_puts_the_held_object_on_the_target_however_it_is_held(
    pr2_holding_milk,
):
    """
    Sending the tool frame to the target pose itself would place the object wherever
    the grasp happens to hold it, which is beside the target unless the object is held
    at its own origin. The release has to account for the transform the gripper
    actually holds it at.
    """
    world, robot, milk, extensions = pr2_holding_milk
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, yaw=np.pi / 4, reference_frame=world.root)
    place = _start(
        PlaceAction(arm=robot.left_arm, object_designator=milk, target_location=target),
        extensions,
    )

    end_effector = robot.left_arm.end_effector
    tool_T_milk = world.transform(milk.root.global_transform, end_effector.tool_frame)
    descent = _tool_goals_of(place)[1]
    placed_milk = descent.goal_pose.evaluate() @ tool_T_milk.to_np()

    np.testing.assert_allclose(
        placed_milk, target.homogeneous_matrix.to_np(), atol=1e-9
    )


def test_a_place_runs_the_grasp_backwards(pr2_holding_milk):
    """
    A release comes down onto the target from where a pick-up would lift the object to,
    and leaves the way a pick-up would approach it.
    """
    world, robot, milk, extensions = pr2_holding_milk
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, reference_frame=world.root)
    place = _start(
        PlaceAction(arm=robot.left_arm, object_designator=milk, target_location=target),
        extensions,
    )
    end_effector = robot.left_arm.end_effector
    grasp = GraspCandidate(milk, end_effector.held_body_T_grasp)
    poses = place.grasp_pose_sequence(grasp.moved_to(target), end_effector, grasp)

    tool_goals = [goal.goal_pose for goal in _tool_goals_of(place)]

    # The grasp pose is commanded twice: to come down onto the target, and to stay there
    # while the gripper opens.
    assert len(tool_goals) == 4
    for expected, actual in zip(
        [poses.retreat, poses.grasp, poses.grasp, poses.pre_grasp], tool_goals
    ):
        np.testing.assert_allclose(expected.to_np(), actual.evaluate(), atol=1e-9)


# %% releasing what has not been picked up yet


def test_a_place_after_an_underspecified_pick_up_joins_the_statechart(
    pr2_apartment_context,
):
    """
    An underspecified pick-up is grounded only once it starts, so while the plan is
    built nothing holds the object and no pick-up says how it will be held.
    """
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, reference_frame=world.root)

    pick_up = a(PickUpAction)(grasp=milk.grasp_candidates()[0], arm=robot.left_arm)
    place = PlaceAction(
        arm=robot.left_arm, object_designator=milk, target_location=target
    )
    expand(Sequence(nodes=[pick_up, place]), extensions)

    assert _arms_moved_by(place, robot) == {robot.left_arm}


# %% the arm that places


def _arms_moved_by(place: PlaceAction, robot) -> set[Arm]:
    """
    :return: The arms of `robot` whose tool frames the expanded `place` moves.
    """
    moved_tips = {goal.tip_link for goal in _tool_goals_of(place)}
    return {arm for arm in robot.all_arms if arm.end_effector.tool_frame in moved_tips}


def test_place_moves_only_the_arm_it_is_given(pr2_apartment_context):
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, reference_frame=world.root)
    place = PlaceAction(
        arm=robot.right_arm, object_designator=milk, target_location=target
    )
    expand(Sequence(nodes=[place]), extensions)

    assert _arms_moved_by(place, robot) == {robot.right_arm}
