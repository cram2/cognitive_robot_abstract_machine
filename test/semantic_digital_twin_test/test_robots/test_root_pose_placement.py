import numpy as np
import pytest

from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix3Dimensional,
)
from krrood_spatial_math.spatial_types import Pose3Dimensional
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Body


def _pr2_behind_an_odom(
    world_T_odometry: HomogeneousTransformationMatrix3Dimensional,
) -> tuple[World, AbstractRobot]:
    """
    A world holding nothing but a PR2, reached through an odom frame at
    ``world_T_odometry``.

    The specification path is what puts the odom between the world root and the robot,
    so the placement is exercised against the same chain a specification builds.
    """
    world = World.create_with_root_body("root")
    try:
        robot = RobotSpecification(
            semantic_annotation_type=PR2, localization_pose=world_T_odometry
        ).spawn(world)
    except ParsingError as error:
        pytest.skip(f"PR2 URDF not available: {error}")
    return world, robot


# %% placing the root through a displaced odom

# The drive is an OmniDrive, which represents x, y and yaw only, so the odom is
# displaced within that plane. A z or roll offset would make these tests assert the
# drive's limits instead of the frame conversion.
_DISPLACED_ODOM = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
    0.5, 0.5, 0, yaw=np.pi / 2
)


def test_root_reaches_a_world_pose_through_a_displaced_odom():
    world, robot = _pr2_behind_an_odom(_DISPLACED_ODOM)
    target = Pose3Dimensional.from_xyz_rpy(
        1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root
    )

    robot.set_root_pose(target)

    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=1e-9
    )


def test_root_reaches_a_world_pose_through_an_undisplaced_odom():
    world, robot = _pr2_behind_an_odom(HomogeneousTransformationMatrix3Dimensional())
    target = Pose3Dimensional.from_xyz_rpy(
        1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root
    )

    robot.set_root_pose(target)

    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=1e-9
    )


# %% pose already expressed in the root connection's parent frame


def test_pose_in_the_root_connection_parent_frame_is_applied_unchanged():
    world, robot = _pr2_behind_an_odom(_DISPLACED_ODOM)
    connection = robot.root.parent_connection
    target = Pose3Dimensional.from_xyz_rpy(
        1.3, 2.0, 0.0, yaw=0.25, reference_frame=connection.parent
    )

    robot.set_root_pose(target)

    np.testing.assert_allclose(connection.origin.to_np(), target.to_np(), atol=1e-9)


# %% drive origin given in another frame than the drive's parent


def test_a_drive_origin_given_in_the_world_frame_is_reached_through_a_displaced_odom():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        odometry = Body(name=PrefixedName("odom"))
        base = Body(name=PrefixedName("base"))
        world.add_connection(
            FixedConnection(
                parent=root,
                child=odometry,
                parent_T_connection_expression=_DISPLACED_ODOM,
            )
        )
        drive = OmniDrive.create_with_dofs(world=world, parent=odometry, child=base)
        world.add_connection(drive)
    target = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        1.3, 2.0, 0.0, yaw=0.25, reference_frame=root
    )

    drive.origin = target

    np.testing.assert_allclose(base.global_pose.to_np(), target.to_np(), atol=1e-9)
