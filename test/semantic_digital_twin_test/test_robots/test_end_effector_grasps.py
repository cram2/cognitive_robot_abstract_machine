import numpy as np
import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import MoreThanOneBodyHeld, NothingHeld
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasGraspPoses
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import (
    Point3,
    Pose,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures

MISALIGNMENT_LEVER_ARM = 0.1
"""
How many meters of reach one radian of misaligned approach is worth in these tests.
"""

HELD_BODY_OFFSET = HomogeneousTransformationMatrix.from_xyz_rpy(0.01, 0.02, 0.03)
"""
Where the body a gripper holds sits relative to the tool frame.
"""


@pytest.fixture
def pr2_gripper(pr2_world_copy) -> EndEffector:
    """
    The left gripper of a PR2 standing in an otherwise empty world.
    """
    return pr2_world_copy.get_semantic_annotations_by_type(PR2)[0].left_arm.end_effector


@pytest.fixture
def graspable_box(pr2_world_copy) -> HasGraspPoses:
    """
    A box within the PR2's reach that offers the default ring of grasps.
    """
    body = Body(
        name=PrefixedName("graspable_box"),
        collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.2))]),
    )
    annotation = HasGraspPoses(root=body)
    with pr2_world_copy.modify_world():
        pr2_world_copy.add_connection(
            FixedConnection(
                parent=pr2_world_copy.root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.6, z=0.9
                ),
            )
        )
        pr2_world_copy.add_semantic_annotation(annotation)
    return annotation


def hold_body(end_effector: EndEffector, name: str = "held_body") -> Body:
    """
    Attach a body below ``end_effector``'s tool frame, as grasping one does.

    :param end_effector: The gripper that should hold the body.
    :param name: The name of the body it should hold.
    :return: The body it now holds.
    """
    world = end_effector._world
    body = Body(name=PrefixedName(name))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=end_effector.tool_frame,
                child=body,
                parent_T_connection_expression=HELD_BODY_OFFSET,
            )
        )
    return body


# %% the direction a gripper approaches from


def test_front_facing_axis_is_the_grasp_frames_approach_direction(pr2_gripper):
    """
    Rotating the axis into the grasp frame has to yield the direction a grasp frame is
    approached along, which is its x-axis.
    """
    grasp_R_tool = RotationMatrix.from_quaternion(pr2_gripper.front_facing_orientation)

    approach_in_grasp_frame = grasp_R_tool @ pr2_gripper.front_facing_axis

    np.testing.assert_allclose(
        approach_in_grasp_frame.to_np()[:3], Vector3.X().to_np()[:3], atol=1e-9
    )


def test_front_facing_axis_follows_the_grippers_own_convention(
    pr2_gripper, tracy_world
):
    """
    A PR2 gripper points along its tool frame's x-axis and Tracy's along its z-axis, so
    the very same grasp frame is approached along a different local axis.
    """
    tracy_gripper = tracy_world.get_semantic_annotations_by_type(Tracy)[
        0
    ].left_arm.end_effector

    np.testing.assert_allclose(
        pr2_gripper.front_facing_axis.to_np()[:3], [1, 0, 0], atol=1e-9
    )
    np.testing.assert_allclose(
        tracy_gripper.front_facing_axis.to_np()[:3], [0, 0, 1], atol=1e-9
    )


# %% tool frame goals


def test_tool_frame_goal_keeps_the_grasp_position(pr2_gripper, graspable_box):
    grasp = next(iter(graspable_box.grasp_poses()))

    goal = pr2_gripper.tool_frame_goal(grasp)

    np.testing.assert_allclose(goal.to_np()[:3, 3], grasp.to_np()[:3, 3], atol=1e-9)


def test_tool_frame_goal_applies_the_end_effectors_own_orientation(
    pr2_gripper, graspable_box
):
    """
    Two grippers pointing different ways must be sent different orientations for one
    and the same grasp.
    """
    grasp = next(iter(graspable_box.grasp_poses()))

    goal = pr2_gripper.tool_frame_goal(grasp)

    expected = grasp.to_rotation_matrix() @ RotationMatrix.from_quaternion(
        pr2_gripper.front_facing_orientation
    )
    np.testing.assert_allclose(
        goal.to_rotation_matrix().to_np(), expected.to_np(), atol=1e-9
    )


# %% the body a gripper holds


def test_an_empty_gripper_holds_nothing(pr2_gripper):
    assert pr2_gripper.held_body is None


def test_the_held_body_is_the_one_below_the_tool_frame(pr2_gripper):
    body = hold_body(pr2_gripper)

    assert pr2_gripper.held_body is body


def test_a_gripper_with_two_bodies_attached_holds_no_single_one(pr2_gripper):
    """
    Which of them the grasp is on cannot be answered, so it is not guessed at.
    """
    hold_body(pr2_gripper, name="first_body")
    hold_body(pr2_gripper, name="second_body")

    with pytest.raises(MoreThanOneBodyHeld):
        pr2_gripper.held_body


def test_the_grasp_of_an_empty_gripper_cannot_be_read(pr2_gripper):
    with pytest.raises(NothingHeld):
        pr2_gripper.held_body_T_grasp


def test_the_held_grasp_is_turned_the_way_the_gripper_faces(pr2_gripper):
    """
    The grasp is what the tool frame reached, so applying the gripper's own orientation
    to it has to lead back to the way the tool frame points.
    """
    body = hold_body(pr2_gripper)

    body_R_grasp = pr2_gripper.held_body_T_grasp.to_rotation_matrix()

    grasp_R_tool = RotationMatrix.from_quaternion(pr2_gripper.front_facing_orientation)
    body_T_tool = pr2_gripper._world.transform(
        pr2_gripper.tool_frame.global_transform, body
    )
    np.testing.assert_allclose(
        (body_R_grasp @ grasp_R_tool).to_np(),
        body_T_tool.to_rotation_matrix().to_np(),
        atol=1e-9,
    )


def test_the_held_grasp_is_the_offset_the_body_hangs_at(pr2_gripper):
    body = hold_body(pr2_gripper)

    np.testing.assert_allclose(
        pr2_gripper.held_body_T_grasp.to_np()[:3, 3],
        -HELD_BODY_OFFSET.to_np()[:3, 3],
        atol=1e-9,
    )


# %% ranking the grasps an object offers


def grasp_reached_from(
    end_effector: EndEffector, world_V_travel: Vector3, misalignment: float = 0.0
) -> Pose:
    """
    A grasp the gripper reaches by travelling ``world_V_travel``.

    :param end_effector: The gripper the grasp is aimed at.
    :param world_V_travel: The way from the tool frame to the grasp, in world axes.
    :param misalignment: How far the grasp's approach direction is turned away from
        that way, about the world z-axis.
    :return: The grasp frame, in the world's frame.
    """
    world = end_effector._world
    world_V_approach = RotationMatrix.from_rpy(yaw=misalignment) @ world_V_travel
    world_R_grasp = RotationMatrix.from_vectors(
        x=world_V_approach,
        z=Vector3.Z() if world_V_approach.to_np()[0] else Vector3.X(),
    )
    world_P_tool = end_effector.tool_frame.global_transform.to_position()
    return Pose(
        position=Point3.from_iterable(
            world_P_tool.to_np()[:3] + world_V_travel.to_np()[:3]
        ),
        orientation=world_R_grasp.to_quaternion(),
        reference_frame=world.root,
    )


def test_distance_to_grasp_counts_the_way_to_the_grasp(pr2_gripper):
    """
    A grasp entered straight on costs the distance to it and nothing else.
    """
    reach = 0.4

    distance = pr2_gripper.distance_to_grasp(
        grasp_reached_from(pr2_gripper, Vector3(reach, 0, 0)),
        MISALIGNMENT_LEVER_ARM,
    )

    assert distance == pytest.approx(reach, abs=1e-6)


def test_distance_to_grasp_counts_the_turn_onto_the_approach(pr2_gripper):
    """
    A grasp the gripper has to enter sideways costs the detour on top of the distance.
    """
    reach = 0.4
    misalignment = np.pi / 4

    distance = pr2_gripper.distance_to_grasp(
        grasp_reached_from(
            pr2_gripper, Vector3(reach, 0, 0), misalignment=misalignment
        ),
        MISALIGNMENT_LEVER_ARM,
    )

    assert distance == pytest.approx(
        reach + MISALIGNMENT_LEVER_ARM * misalignment, abs=1e-6
    )


def test_a_grasp_facing_back_at_the_gripper_is_the_dearest_of_all(pr2_gripper):
    """
    A grasp whose approach runs against the way there has to be entered from behind the
    object, which is the most a misalignment can cost.
    """
    reach = 0.4

    around_the_back = pr2_gripper.distance_to_grasp(
        grasp_reached_from(pr2_gripper, Vector3(reach, 0, 0), misalignment=np.pi),
        MISALIGNMENT_LEVER_ARM,
    )

    assert around_the_back == pytest.approx(
        reach + MISALIGNMENT_LEVER_ARM * np.pi, abs=1e-6
    )


def test_a_lever_arm_of_zero_prices_the_misalignment_out(pr2_gripper):
    """
    Without a lever arm the ranking is the plain distance to the grasp.
    """
    reach = 0.4
    sideways = grasp_reached_from(
        pr2_gripper, Vector3(reach, 0, 0), misalignment=np.pi / 2
    )

    assert pr2_gripper.distance_to_grasp(sideways, 0.0) == pytest.approx(
        reach, abs=1e-6
    )


def test_the_misalignment_is_priced_even_when_no_lever_arm_is_given(pr2_gripper):
    """
    The default has to price the detour, or a grasp facing back at the gripper would
    rank level with one it can drive straight into.
    """
    reach = 0.4
    straight_on = grasp_reached_from(pr2_gripper, Vector3(reach, 0, 0))
    around_the_back = grasp_reached_from(
        pr2_gripper, Vector3(reach, 0, 0), misalignment=np.pi
    )

    assert pr2_gripper.distance_to_grasp(
        around_the_back
    ) > pr2_gripper.distance_to_grasp(straight_on)


def test_grasp_poses_by_distance_offers_every_grasp_the_object_has(
    pr2_gripper, graspable_box
):
    ranked = pr2_gripper.grasp_poses_by_distance(graspable_box, MISALIGNMENT_LEVER_ARM)

    offered = [pose.to_np() for pose in graspable_box.grasp_poses()]
    assert len(ranked) == len(offered)
    for pose in ranked:
        assert any(np.allclose(pose.to_np(), other, atol=1e-9) for other in offered)


def test_grasp_poses_by_distance_puts_the_closest_grasp_first(
    pr2_gripper, graspable_box
):
    ranked = pr2_gripper.grasp_poses_by_distance(graspable_box, MISALIGNMENT_LEVER_ARM)

    distances = [
        pr2_gripper.distance_to_grasp(pose, MISALIGNMENT_LEVER_ARM) for pose in ranked
    ]
    assert distances == sorted(distances)


def test_the_best_grasp_is_the_one_the_gripper_faces(pr2_gripper, graspable_box):
    """
    The whole point of the ranking: of a ring of grasps that share one position, the one
    entered from the gripper's own side wins, never one reached around the far side.
    """
    best = pr2_gripper.grasp_poses_by_distance(graspable_box, MISALIGNMENT_LEVER_ARM)[0]

    world = graspable_box._world
    world_T_grasp = world.transform(best.to_homogeneous_matrix(), world.root)
    world_V_travel = (
        world_T_grasp.to_position()
        - pr2_gripper.tool_frame.global_transform.to_position()
    )
    world_V_approach = world_T_grasp.to_rotation_matrix() @ Vector3.X()
    assert float(world_V_approach.angle_between(world_V_travel)) < np.pi / 2
