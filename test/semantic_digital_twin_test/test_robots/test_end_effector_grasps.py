from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pytest

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.exceptions import (
    GripperAxesNotPerpendicular,
    MoreThanOneBodyHeld,
    NothingHeld,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_part_mixins import HasTwoFingers
from semantic_digital_twin.robots.robot_parts import Camera, EndEffector
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

# %% fixtures

POSITION_TOLERANCE = 0.005
"""
How close two distances have to be, in meters, to count as the same distance here.
"""

HELD_BODY_OFFSET = HomogeneousTransformationMatrix.from_xyz_rpy(0.01, 0.02, 0.03)
"""
Where the body a gripper holds sits relative to the tool frame.
"""


@pytest.fixture
def pr2_gripper(pr2_world_copy) -> EndEffector:
    """
    The left gripper of a PR2, in a world built from the robot's description alone.
    """
    return pr2_world_copy.get_semantic_annotations_by_type(PR2)[0].left_arm.end_effector


@pytest.fixture
def graspable_box(pr2_world_copy) -> Milk:
    """
    A box-shaped milk carton within the PR2's reach, offering the default ring of
    grasps.
    """
    body = Body(
        name=PrefixedName("graspable_box"),
        collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.2))]),
    )
    annotation = Milk(root=body)
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


# %% the axes a gripper states in its tool frame


@dataclass(eq=False)
class EndEffectorWithSkewedAxes(EndEffector):
    """
    An end effector whose closing axis is not perpendicular to its approach axis.
    """

    @property
    def approach_axis(self) -> Vector3:
        return Vector3.X(reference_frame=self.tool_frame)

    @property
    def closing_axis(self) -> Vector3:
        return Vector3(x=1, y=1, z=0, reference_frame=self.tool_frame)

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> list[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ):
        raise NotImplementedError


def test_the_approach_axis_is_the_grasp_frames_x_axis(pr2_gripper) -> None:
    np.testing.assert_allclose(
        (pr2_gripper.tool_R_grasp @ Vector3.X()).to_np()[:3],
        pr2_gripper.approach_axis.to_np()[:3],
        atol=1e-9,
    )


def test_the_closing_axis_is_the_grasp_frames_y_axis(pr2_gripper) -> None:
    np.testing.assert_allclose(
        (pr2_gripper.tool_R_grasp @ Vector3.Y()).to_np()[:3],
        pr2_gripper.closing_axis.to_np()[:3],
        atol=1e-9,
    )


def test_the_approach_axis_follows_the_grippers_own_convention(
    pr2_gripper, tracy_world
) -> None:
    """
    A PR2 gripper points along its tool frame's x-axis and Tracy's along its z-axis, so
    the very same grasp frame is approached along a different local axis.
    """
    tracy_gripper = tracy_world.get_semantic_annotations_by_type(Tracy)[
        0
    ].left_arm.end_effector

    np.testing.assert_allclose(
        pr2_gripper.approach_axis.to_np()[:3], Vector3.X().to_np()[:3], atol=1e-9
    )
    np.testing.assert_allclose(
        tracy_gripper.approach_axis.to_np()[:3], Vector3.Z().to_np()[:3], atol=1e-9
    )


def test_a_gripper_whose_axes_are_not_perpendicular_is_refused(pr2_gripper) -> None:
    with pytest.raises(GripperAxesNotPerpendicular):
        EndEffectorWithSkewedAxes(
            name=PrefixedName("skewed_gripper"),
            root=pr2_gripper.root,
            tool_frame=pr2_gripper.tool_frame,
        )


def test_every_robot_states_its_axes_in_the_frame_they_belong_to(
    supported_abstract_robots,
) -> None:
    """
    A gripper's axes are read in its tool frame and a camera's in its root, for every
    robot there is.
    """
    for abstract_robot in supported_abstract_robots:
        world = URDFParser.from_file(abstract_robot.get_ros_file_path()).parse()
        abstract_robot.from_world(world)

        for end_effector in world.get_semantic_annotations_by_type(EndEffector):
            assert end_effector.approach_axis.reference_frame is end_effector.tool_frame
            assert end_effector.closing_axis.reference_frame is end_effector.tool_frame
        for camera in world.get_semantic_annotations_by_type(Camera):
            assert camera.forward_facing_axis.reference_frame is camera.root


# %% tool frame goals


def test_tool_frame_goal_keeps_the_grasp_position(pr2_gripper, graspable_box) -> None:
    grasp = graspable_box.grasp_candidates()[0].grasp_pose

    goal = pr2_gripper.tool_frame_goal(grasp)

    np.testing.assert_allclose(goal.to_np()[:3, 3], grasp.to_np()[:3, 3], atol=1e-9)


def test_tool_frame_goal_applies_the_end_effectors_own_orientation(
    pr2_gripper, graspable_box
) -> None:
    """
    Two grippers pointing different ways must be sent different orientations for one
    and the same grasp.
    """
    grasp = graspable_box.grasp_candidates()[0].grasp_pose

    goal = pr2_gripper.tool_frame_goal(grasp)

    expected = grasp.rotation_matrix @ pr2_gripper.tool_R_grasp.inverse()
    np.testing.assert_allclose(
        goal.rotation_matrix.to_np(), expected.to_np(), atol=1e-9
    )


# %% the body a gripper holds


def test_an_empty_gripper_holds_nothing(pr2_gripper) -> None:
    assert pr2_gripper.held_body is None


def test_the_held_body_is_the_one_below_the_tool_frame(pr2_gripper) -> None:
    body = hold_body(pr2_gripper)

    assert pr2_gripper.held_body is body


def test_a_gripper_with_two_bodies_attached_holds_no_single_one(pr2_gripper) -> None:
    """
    Which of them the grasp is on cannot be answered, so it is not guessed at.
    """
    hold_body(pr2_gripper, name="first_body")
    hold_body(pr2_gripper, name="second_body")

    with pytest.raises(MoreThanOneBodyHeld):
        pr2_gripper.held_body


def test_the_grasp_of_an_empty_gripper_cannot_be_read(pr2_gripper) -> None:
    with pytest.raises(NothingHeld):
        pr2_gripper.held_body_T_grasp


def test_the_held_grasp_is_turned_the_way_the_gripper_faces(pr2_gripper) -> None:
    """
    The grasp is what the tool frame reached, so applying the gripper's own orientation
    to it has to lead back to the way the tool frame points.
    """
    body = hold_body(pr2_gripper)

    body_R_grasp = pr2_gripper.held_body_T_grasp.rotation_matrix

    grasp_R_tool = pr2_gripper.tool_R_grasp.inverse()
    body_T_tool = pr2_gripper._world.transform(
        pr2_gripper.tool_frame.global_transform, body
    )
    np.testing.assert_allclose(
        (body_R_grasp @ grasp_R_tool).to_np(),
        body_T_tool.rotation_matrix.to_np(),
        atol=1e-9,
    )


def test_the_held_grasp_sits_where_the_gripper_holds_the_object(pr2_gripper):
    """
    The grasp is read at the point the gripper holds an object, so it sits at that
    point, shifted by how the body hangs off the tool frame.
    """
    body = hold_body(pr2_gripper)

    np.testing.assert_allclose(
        pr2_gripper.held_body_T_grasp.to_np()[:3, 3],
        pr2_gripper.tool_P_grasp_when_closed.to_np()[:3].ravel()
        - HELD_BODY_OFFSET.to_np()[:3, 3],
        atol=1e-9,
    )
    assert pr2_gripper.held_body_T_grasp.reference_frame is body


def test_the_held_grasp_can_be_read_at_any_point_of_the_gripper(pr2_gripper):
    """
    Read at the tool frame itself, the grasp is only the offset the body hangs at.
    """
    hold_body(pr2_gripper)

    body_T_grasp = pr2_gripper.held_body_T_grasp_at(
        Point3(0.0, 0.0, 0.0, reference_frame=pr2_gripper.tool_frame)
    )

    np.testing.assert_allclose(
        body_T_grasp.to_np()[:3, 3], -HELD_BODY_OFFSET.to_np()[:3, 3], atol=1e-9
    )


# %% where a two-finger gripper's fingers meet


def test_the_fingertips_are_the_tips_of_the_two_fingers(pr2_gripper):
    assert pr2_gripper.fingertips.both == (
        pr2_gripper.thumb.tip,
        pr2_gripper.finger.tip,
    )


def test_the_grasp_point_lies_midway_between_the_fingertips(pr2_gripper):
    thumb_tip, finger_tip = (
        pr2_gripper.fingertips._bounding_box(fingertip).center.to_np()[:3].ravel()
        for fingertip in pr2_gripper.fingertips.both
    )

    np.testing.assert_allclose(
        pr2_gripper.fingertips.tool_P_grasp.to_np()[:3].ravel(),
        (thumb_tip + finger_tip) / 2,
        atol=1e-9,
    )


def test_the_grasp_point_is_read_in_the_frame_it_belongs_to(pr2_gripper):
    assert pr2_gripper.fingertips.tool_P_grasp.reference_frame is pr2_gripper.tool_frame


def test_every_two_finger_gripper_can_measure_where_its_fingers_meet(
    supported_abstract_robots,
):
    """
    The measurements are read off the robot's own description, so every two-finger
    gripper there is answers them rather than only the one they were written against.
    """
    for abstract_robot in supported_abstract_robots:
        world = URDFParser.from_file(abstract_robot.get_ros_file_path()).parse()
        abstract_robot.from_world(world)

        for gripper in world.get_semantic_annotations_by_type(EndEffector):
            if not isinstance(gripper, HasTwoFingers):
                continue
            assert gripper.fingertips.tool_P_grasp.reference_frame is gripper.tool_frame
            assert gripper.fingertips.surface_depth >= 0.0
            assert np.linalg.norm(
                gripper.fingertips.finger_P_thumb(0.05).to_np()[:3].ravel()
            ) > np.linalg.norm(
                gripper.fingertips.finger_P_thumb(0.02).to_np()[:3].ravel()
            )


# %% aiming the fingers rather than the tool frame


def test_the_tool_frame_goal_for_a_grasp_puts_the_grasp_point_on_it(
    pr2_gripper, graspable_box
):
    """
    Reaching the returned pose with the tool frame leaves the point where the fingers
    meet on the grasp frame, which is what the plain tool frame goal does not do.
    """
    grasp = graspable_box.grasp_candidates()[0].grasp_pose

    goal = pr2_gripper.tool_frame_goal_for_grasp(grasp)

    reached = goal.homogeneous_matrix.to_np()
    grasp_point_once_reached = (
        reached[:3, :3] @ pr2_gripper.fingertips.tool_P_grasp.to_np()[:3].ravel()
        + reached[:3, 3]
    )
    np.testing.assert_allclose(
        grasp_point_once_reached, grasp.position.to_np()[:3].ravel(), atol=1e-9
    )


def test_the_goal_for_a_grasp_turns_the_gripper_the_same_way_as_the_tool_frame_goal(
    pr2_gripper, graspable_box
):
    grasp = graspable_box.grasp_candidates()[0].grasp_pose

    np.testing.assert_allclose(
        pr2_gripper.tool_frame_goal_for_grasp(grasp).rotation_matrix.to_np(),
        pr2_gripper.tool_frame_goal(grasp).rotation_matrix.to_np(),
        atol=1e-9,
    )


# %% how wide the fingers are sent


def test_a_wider_object_sends_the_fingertip_frames_further_apart(pr2_gripper):
    narrower = np.linalg.norm(
        pr2_gripper.fingertips.finger_P_thumb(0.02).to_np()[:3].ravel()
    )
    wider = np.linalg.norm(
        pr2_gripper.fingertips.finger_P_thumb(0.05).to_np()[:3].ravel()
    )

    assert wider - narrower == pytest.approx(0.03)


def test_the_separation_a_goal_is_built_for_is_the_one_asked_for(pr2_gripper):
    """
    The surfaces stand where they were asked to, which is the distance between the
    fingertip frames less what each surface sits inside its own frame.
    """
    separation = 0.04

    frames_apart = np.linalg.norm(
        pr2_gripper.fingertips.finger_P_thumb(separation).to_np()[:3].ravel()
    )

    assert frames_apart - 2 * pr2_gripper.fingertips.surface_depth == pytest.approx(
        separation
    )


def test_the_thumb_tip_goal_is_read_in_the_other_fingertips_frame(pr2_gripper):
    goal = pr2_gripper.fingertips.finger_P_thumb(0.04)

    assert goal.reference_frame is pr2_gripper.finger.tip


# %% reading where the fingers meet for a state the world does not stand in


def test_the_meeting_point_can_be_read_for_another_state_and_the_world_stands_back_up(
    pr2_gripper,
):
    """
    Standing the gripper in a state inside a reset context is how a state the world
    does not hold is read, and the positions it held come back afterwards.
    """
    world = pr2_gripper._world
    positions_before = world.state.positions.copy()
    as_it_stands = pr2_gripper.fingertips.tool_P_grasp

    with world.reset_state_context():
        pr2_gripper.get_joint_state_by_type(GripperState.OPEN).apply_to(world)
        while_open = pr2_gripper.fingertips.tool_P_grasp

    np.testing.assert_array_equal(world.state.positions, positions_before)
    assert not np.allclose(while_open.to_np(), as_it_stands.to_np())
    np.testing.assert_allclose(
        pr2_gripper.fingertips.tool_P_grasp.to_np(), as_it_stands.to_np(), atol=1e-9
    )


# %% how far closing carries the point the fingers meet at


def test_swinging_fingers_carry_the_point_they_meet_at_further_out(pr2_gripper):
    """
    A PR2's jaws swing on a linkage rather than sliding, so the point they meet at
    travels along the approach axis as they close.
    """
    assert pr2_gripper.travel_when_closing > POSITION_TOLERANCE


def test_the_travel_is_read_between_the_grippers_own_open_and_closed_states(
    pr2_gripper,
):
    """
    It is the distance the meeting point covers over the gripper's whole range, which
    is what a grasp aimed while the gripper is open has to correct for.
    """
    world = pr2_gripper._world
    with world.reset_state_context():
        reached = []
        for state_type in (GripperState.OPEN, GripperState.CLOSE):
            pr2_gripper.get_joint_state_by_type(state_type).apply_to(world)
            reached.append(pr2_gripper.fingertips.tool_P_grasp)
    open_P_grasp, closed_P_grasp = reached
    expected = (closed_P_grasp - open_P_grasp).dot(pr2_gripper.approach_axis)

    assert pr2_gripper.travel_when_closing == pytest.approx(float(expected))


def test_reading_the_travel_stands_the_world_back_up(tracy_world):
    """
    Reading the travel puts the gripper through its own open and closed states, and the
    positions the world held come back afterwards. The state version moves, since a
    reset context announces that it put the state back.
    """
    gripper = tracy_world.get_semantic_annotations_by_type(Tracy)[
        0
    ].left_arm.end_effector
    positions_before = tracy_world.state.positions.copy()

    gripper.travel_when_closing

    np.testing.assert_array_equal(tracy_world.state.positions, positions_before)
