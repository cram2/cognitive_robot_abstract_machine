"""
Spatial types expressed in the frames of a :class:`semantic_digital_twin.world.World`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from krrood_spatial_math.spatial_types import (
    Point3Dimensional,
    Pose3Dimensional,
    Vector3Dimensional,
)
from krrood_spatial_math.spatial_types_2_dimensional import (
    EmbeddedPlaneProjection,
    Point2Dimensional,
    Pose2Dimensional,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures


@pytest.fixture
def rotated_mini_world(mini_world: World) -> World:
    """
    :func:`mini_world` with its tip turned a quarter turn about the root's z axis.
    """
    mini_world.connections[0].position = math.pi / 2
    return mini_world


@pytest.fixture
def tilted_world() -> World:
    """
    A world of two bodies whose tip is turned a quarter turn about the root's x axis, so
    the tip's z axis lies in the root's x-y plane.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        tip = Body(name=PrefixedName("tip"))
        connection = ScrewAxisConnection.create_with_dofs(
            world=world,
            parent=root,
            child=tip,
            screw_axis=ScrewAxisConnection.create_revolute_screw_axis(
                axis=Vector3Dimensional.X()
            ),
        )
        world.add_connection(connection)
    connection.position = math.pi / 2
    return world


# %% projecting into the plane of another frame


class TestPoint2DimensionalFromPoint3InWorld:
    def test_transforms_into_another_plane_frame(self, rotated_mini_world: World):
        root, tip = rotated_mini_world.root, rotated_mini_world.bodies[1]
        point = Point3Dimensional(1.0, 0.0, 2.0, reference_frame=tip)

        projected = Point2Dimensional.from_point_3_dimensional(point, plane_frame=root)

        assert projected.reference_frame is root
        np.testing.assert_allclose(
            projected.to_np(),
            EmbeddedPlaneProjection(plane_frame=root).project(point).to_np(),
            atol=1e-12,
        )


class TestPoint2DimensionalFromPoseInWorld:
    def test_transforms_into_another_plane_frame(self, rotated_mini_world: World):
        root, tip = rotated_mini_world.root, rotated_mini_world.bodies[1]
        pose = Pose3Dimensional.from_xyz_rpy(x=1.0, z=2.0, yaw=0.3, reference_frame=tip)

        projected = Point2Dimensional.from_pose(pose, plane_frame=root)

        assert projected.reference_frame is root
        np.testing.assert_allclose(
            projected.to_np(),
            EmbeddedPlaneProjection(plane_frame=root).project(pose.position).to_np(),
            atol=1e-12,
        )


class TestPose2DimensionalFromPoseInWorld:
    def test_from_pose_transforms_into_another_frame(self, mini_world: World):
        """
        Asking for another frame expresses the pose in that frame instead of attaching
        the frame to the pose's own coordinates.
        """
        mini_world.connections[0].position = math.pi / 2
        root, tip = mini_world.root, mini_world.bodies[1]
        tip_pose = Pose3Dimensional.from_xyz_rpy(
            1, 0, 2, 0, 0, 0.3, reference_frame=tip
        )

        p2 = Pose2Dimensional.from_pose(tip_pose, plane_frame=root)

        assert p2.reference_frame is root
        np.testing.assert_allclose(
            p2.to_np(),
            Pose2Dimensional.from_pose(mini_world.transform(tip_pose, root)).to_np(),
            atol=1e-12,
        )


class TestOrthogonalProjectionInWorld:
    def test_transforms_a_point_of_another_frame_into_the_plane_frame(
        self, tilted_world: World
    ):
        root, tip = tilted_world.root, tilted_world.bodies[1]
        tip_P_point = Point3Dimensional(1, 0, 2, reference_frame=tip)
        root_P_point = tilted_world.transform(tip_P_point, root)

        projected = EmbeddedPlaneProjection(plane_frame=root).project(tip_P_point)

        assert projected.reference_frame is root
        np.testing.assert_allclose(
            projected.to_np(), root_P_point.to_np()[:2], atol=1e-12
        )
