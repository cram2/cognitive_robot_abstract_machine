import numpy as np
import pytest

from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import (
    Pose,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures

BOX_SCALE = Scale(0.1, 0.2, 0.3)
"""
Extents of the box the approach sequence has to clear.
"""


@pytest.fixture
def boxed_pr2_world(mutable_simple_pr2_world):
    """
    A PR2 next to a box of known extents, standing a meter above the world root.
    """
    world, robot, context = mutable_simple_pr2_world
    with world.modify_world():
        box = Body(
            name=PrefixedName("approach_box"),
            collision=ShapeCollection([Box(scale=BOX_SCALE)]),
        )
        connection = Connection6DoF.create_with_dofs(world, world.root, box)
        world.add_connection(connection)
        connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            1, 0, 1, reference_frame=world.root
        )
    return world, robot, box


def grasp_at_origin(body) -> Pose:
    """
    :param body: The body to grasp.
    :return: A grasp frame at the body's origin, approaching along the body's x-axis.
    """
    return Pose(reference_frame=body)


# %% approach sequences


def test_pre_grasp_pose_clears_the_body_it_grasps(boxed_pr2_world):
    """
    A grasp at the body's own origin has to be approached from outside the body, so the
    pre-grasp pose stands off by half of it plus the clearance.
    """
    _, robot, box = boxed_pr2_world
    action = HasApproachesGraspPoses()

    origin_grasp = grasp_at_origin(box)
    pre_grasp, grasp, _ = action.grasp_pose_sequence(
        origin_grasp,
        robot.left_arm.end_effector,
        origin_grasp,
    )

    expected_standoff = BOX_SCALE.x / 2 + action.approach_clearance
    np.testing.assert_allclose(
        pre_grasp.to_np()[:3, 3], [-expected_standoff, 0, 0], atol=1e-9
    )


def test_pre_grasp_pose_of_a_surface_grasp_only_adds_the_clearance(boxed_pr2_world):
    """
    A grasp already on the body's surface, approached from outside it, needs nothing
    beyond the clearance -- this is what lets a bowl be grasped at its rim.
    """
    _, robot, box = boxed_pr2_world
    action = HasApproachesGraspPoses()
    surface_grasp = Pose(
        position=Vector3(0, 0, BOX_SCALE.z / 2).to_point3(),
        orientation=RotationMatrix.from_vectors(
            x=Vector3.NEGATIVE_Z(), y=Vector3.X()
        ).to_quaternion(),
        reference_frame=box,
    )

    pre_grasp, _, _ = action.grasp_pose_sequence(
        surface_grasp,
        robot.left_arm.end_effector,
        surface_grasp,
    )

    np.testing.assert_allclose(
        pre_grasp.to_np()[:3, 3],
        [0, 0, BOX_SCALE.z / 2 + action.approach_clearance],
        atol=1e-9,
    )


def test_grasp_pose_is_the_middle_of_the_sequence(boxed_pr2_world):
    _, robot, box = boxed_pr2_world
    end_effector = robot.left_arm.end_effector
    grasp = grasp_at_origin(box)

    approach = HasApproachesGraspPoses()
    _, middle, _ = approach.grasp_pose_sequence(grasp, end_effector, grasp)

    np.testing.assert_allclose(
        middle.to_np(),
        end_effector.tool_frame_goal(grasp).to_np(),
        atol=1e-9,
    )


def test_retreat_pose_rises_along_the_world_z_axis(boxed_pr2_world):
    world, robot, box = boxed_pr2_world
    action = HasApproachesGraspPoses()
    grasp = grasp_at_origin(box)

    _, _, retreat = action.grasp_pose_sequence(
        grasp,
        robot.left_arm.end_effector,
        grasp,
    )

    world_P_grasp = world.transform(grasp.to_homogeneous_matrix(), world.root).to_np()
    world_P_retreat = world.transform(
        retreat.to_homogeneous_matrix(), world.root
    ).to_np()
    np.testing.assert_allclose(
        world_P_retreat[:3, 3] - world_P_grasp[:3, 3],
        [0, 0, action.retreat_distance],
        atol=1e-9,
    )


def test_retreat_pose_keeps_the_grasp_orientation(boxed_pr2_world):
    _, robot, box = boxed_pr2_world

    approach = HasApproachesGraspPoses()
    origin_grasp = grasp_at_origin(box)
    _, grasp, retreat = approach.grasp_pose_sequence(
        origin_grasp,
        robot.left_arm.end_effector,
        origin_grasp,
    )

    np.testing.assert_allclose(
        retreat.to_rotation_matrix().to_np(),
        grasp.to_rotation_matrix().to_np(),
        atol=1e-9,
    )


def test_reversing_turns_the_grasp_into_a_release(boxed_pr2_world):
    _, robot, box = boxed_pr2_world
    action = HasApproachesGraspPoses()
    grasp = grasp_at_origin(box)

    forward = action.grasp_pose_sequence(grasp, robot.left_arm.end_effector, grasp)
    backward = action.grasp_pose_sequence(
        grasp, robot.left_arm.end_effector, grasp, reverse=True
    )

    for expected, actual in zip(reversed(forward), backward):
        np.testing.assert_allclose(expected.to_np(), actual.to_np(), atol=1e-9)


def test_a_grasp_given_in_another_frame_still_clears_the_body(boxed_pr2_world):
    """
    The clearance is read off the body the grasp is aimed at, so a caller who wrote the
    grasp in the world's frame gets the same stand-off as one who wrote it in the box's.
    """
    world, robot, box = boxed_pr2_world
    action = HasApproachesGraspPoses()
    origin_grasp = grasp_at_origin(box)
    world_grasp = world.transform(
        origin_grasp.to_homogeneous_matrix(), world.root
    ).to_pose()

    pre_grasp, _, _ = action.grasp_pose_sequence(
        world_grasp,
        robot.left_arm.end_effector,
        action._grasp_in_body_frame(world_grasp, box),
    )

    np.testing.assert_allclose(
        world.transform(pre_grasp.to_homogeneous_matrix(), box).to_np()[:3, 3],
        [-(BOX_SCALE.x / 2 + action.approach_clearance), 0, 0],
        atol=1e-9,
    )


def test_sequence_without_a_body_stands_off_by_the_clearance_alone(boxed_pr2_world):
    _, robot, box = boxed_pr2_world
    action = HasApproachesGraspPoses()

    pre_grasp, _, _ = action.grasp_pose_sequence(
        grasp_at_origin(box), robot.left_arm.end_effector
    )

    np.testing.assert_allclose(
        pre_grasp.to_np()[:3, 3], [-action.approach_clearance, 0, 0], atol=1e-9
    )
