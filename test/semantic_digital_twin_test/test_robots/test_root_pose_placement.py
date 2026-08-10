import numpy as np
import pytest

from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World


def _pr2_behind_an_odom(
    world_T_odom: HomogeneousTransformationMatrix,
) -> tuple[World, AbstractRobot]:
    """
    A world holding nothing but a PR2, reached through an odom frame at
    ``world_T_odom``.

    The specification path is what puts the odom between the world root and the robot,
    so the placement is exercised against the same chain a specification builds.
    """
    world = World.create_with_root_body("root")
    try:
        robot = RobotSpecification(
            semantic_annotation_type=PR2, world_T_odom=world_T_odom
        ).spawn(world)
    except ParsingError as error:
        pytest.skip(f"PR2 URDF not available: {error}")
    return world, robot


# %% placing the root through a displaced odom

# The drive is an OmniDrive, which represents x, y and yaw only, so the odom is
# displaced within that plane. A z or roll offset would make these tests assert the
# drive's limits instead of the frame conversion.
_DISPLACED_ODOM = HomogeneousTransformationMatrix.from_xyz_rpy(
    0.5, 0.5, 0, yaw=np.pi / 2
)


def test_root_reaches_a_world_pose_through_a_displaced_odom():
    world, robot = _pr2_behind_an_odom(_DISPLACED_ODOM)
    target = Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root)

    robot.set_root_pose(target)

    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=1e-9
    )


def test_root_reaches_a_world_pose_through_an_undisplaced_odom():
    world, robot = _pr2_behind_an_odom(HomogeneousTransformationMatrix())
    target = Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root)

    robot.set_root_pose(target)

    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=1e-9
    )


# %% pose already expressed in the root connection's parent frame


def test_pose_in_the_root_connection_parent_frame_is_applied_unchanged():
    world, robot = _pr2_behind_an_odom(_DISPLACED_ODOM)
    connection = robot.root.parent_connection
    target = Pose.from_xyz_rpy(
        1.3, 2.0, 0.0, yaw=0.25, reference_frame=connection.parent
    )

    robot.set_root_pose(target)

    np.testing.assert_allclose(connection.origin.to_np(), target.to_np(), atol=1e-9)
