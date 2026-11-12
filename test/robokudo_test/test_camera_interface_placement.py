"""
Placement of a camera interface's camera in the world perception perceives into.
"""

from __future__ import annotations

# %% Imports
import numpy as np
import pytest
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.cas import CAS
from robokudo.exceptions import UnknownWorldFrameError

from ._perception_world import (
    FrameName,
    SharedWorld,
    camera_interface,
    camera_pose,
    place_camera_of,
    world_with_a_head_camera,
)


def bound_pose(
    world: World,
    world_T_camera: HomogeneousTransformationMatrix,
    located_through_tf: bool = False,
    world_frame: str = FrameName.WORLD,
) -> HomogeneousTransformationMatrix:
    """
    :return: ``world_T_camera``, measured against ``world_frame`` for the test camera,
        bound to the bodies of ``world``.
    """
    return camera_interface(located_through_tf=located_through_tf).bind_world_T_camera(
        world=world,
        world_frame=world_frame,
        camera_frame=FrameName.CAMERA,
        world_T_camera=world_T_camera,
    )


# %% world frame


class TestWorldFrame:
    """
    The world frame a camera pose is measured against.
    """

    def test_empty_world_gets_the_world_frame_as_its_root(self) -> None:
        """
        An empty world gets the measured world frame as its root.
        """
        world = World()
        world_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0, y=2.0, z=3.0
        )

        pose = bound_pose(world, world_T_camera)

        world_body = world.get_body_by_name(FrameName.WORLD)
        camera_body = world.get_body_by_name(FrameName.CAMERA)
        assert world.root is world_body
        assert pose.reference_frame is world_body
        assert pose.child_frame is camera_body
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())

    def test_unknown_world_frame_of_a_world_with_bodies_is_not_added(self) -> None:
        """
        A world with a root gets no body for a world frame it lacks.
        """
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName("root_of_another_world")))
        names_before = [body.name for body in world.bodies]

        with pytest.raises(UnknownWorldFrameError):
            bound_pose(world, camera_pose())

        assert [body.name for body in world.bodies] == names_before


# %% camera perception mounts


class TestCameraPerceptionMounts:
    """
    Cameras perception mounts below their world frame itself.
    """

    def test_adds_the_camera_below_a_known_world_frame_and_moves_it(self) -> None:
        """
        A camera the world lacks hangs below its world frame at the measured pose.
        """
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName(FrameName.WORLD)))
        bound_pose(world, camera_pose(x=1.0))
        world_T_camera = camera_pose(x=2.0)

        bound_pose(world, world_T_camera)

        [camera_body] = world.get_bodies_by_name(FrameName.CAMERA)
        assert camera_body.parent_connection.parent is world.get_body_by_name(
            FrameName.WORLD
        )
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())

    def test_camera_interfaces_of_one_world_share_its_camera(self) -> None:
        """
        A camera placed twice keeps one body, at the latest measured pose.
        """
        world = World()
        bound_pose(world, camera_pose(x=1.0))
        world_T_camera = camera_pose(x=2.0)

        bound_pose(world, world_T_camera)

        [camera_body] = world.get_bodies_by_name(FrameName.CAMERA)
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())


# %% camera of the world


class TestCameraOfTheWorld:
    """
    Cameras another process attached to the world.
    """

    def test_camera_of_the_world_is_located_by_the_world(self) -> None:
        """
        A robot's camera keeps its attachment and is located by the world.
        """
        world = world_with_a_head_camera()
        camera_body = world.get_body_by_name(FrameName.CAMERA)
        camera_pose_before = camera_body.global_transform.to_np()

        pose = bound_pose(world, camera_pose(x=4.0), located_through_tf=True)

        assert np.allclose(camera_body.global_transform.to_np(), camera_pose_before)
        assert pose.reference_frame is world.root
        assert pose.child_frame is camera_body
        assert np.allclose(pose.to_np(), camera_pose_before)

    def test_camera_of_the_world_needs_no_known_world_frame(self) -> None:
        """
        A robot's camera is located even against a world frame the world lacks.
        """
        world = world_with_a_head_camera()

        pose = bound_pose(
            world,
            camera_pose(),
            located_through_tf=True,
            world_frame="frame_unknown_to_the_world",
        )

        assert pose.child_frame is world.get_body_by_name(FrameName.CAMERA)


# %% camera of a shared world


class TestCameraOfASharedWorld:
    """
    Cameras of a world other processes share.
    """

    def test_shared_world_gets_nothing_for_a_camera_located_through_tf(
        self, *, shared_world: SharedWorld
    ) -> None:
        """
        A camera located through TF adds neither a body nor an annotation to a shared
        world.
        """
        world = shared_world.world
        body_ids_before = [body.id for body in world.bodies]

        place_camera_of(
            CAS(),
            world,
            camera_pose(x=1.0),
            camera_interface(located_through_tf=True),
        )

        assert [body.id for body in world.bodies] == body_ids_before
        assert world.get_semantic_annotations_by_type(Camera) == []

    def test_camera_located_through_tf_is_located_by_its_measured_pose(
        self, *, shared_world: SharedWorld
    ) -> None:
        """
        A camera located through TF that a shared world lacks keeps its measured pose.
        """
        cas = CAS()
        world_T_camera = camera_pose(x=1.0)

        place_camera_of(
            cas,
            shared_world.world,
            world_T_camera,
            camera_interface(located_through_tf=True),
        )

        pose = cas.camera_observation.world_T_camera
        assert pose.reference_frame is shared_world.world.get_body_by_name(
            FrameName.WORLD
        )
        assert pose.child_frame is cas.camera_observation.camera.root
        assert np.allclose(pose.to_np(), world_T_camera.to_np())

    def test_shared_world_gets_a_camera_not_located_through_tf(
        self, *, shared_world: SharedWorld
    ) -> None:
        """
        A camera not located through TF gets a body and an annotation in a shared world.
        """
        world = shared_world.world
        cas = CAS()
        world_T_camera = camera_pose(x=1.0)

        place_camera_of(
            cas, world, world_T_camera, camera_interface(located_through_tf=False)
        )

        camera_body = world.get_body_by_name(FrameName.CAMERA)
        assert camera_body.parent_connection.parent is world.get_body_by_name(
            FrameName.WORLD
        )
        assert world.get_semantic_annotations_by_type(Camera) == [
            cas.camera_observation.camera
        ]
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())

    def test_camera_of_a_robot_in_a_shared_world_is_located_by_the_world(
        self, *, shared_world_with_a_robot: SharedWorld
    ) -> None:
        """
        A robot's camera in a shared world is located by that world.
        """
        world = shared_world_with_a_robot.world
        camera_body = world.get_body_by_name(FrameName.CAMERA)

        pose = bound_pose(world, camera_pose(x=4.0), located_through_tf=True)

        assert pose.child_frame is camera_body
        assert np.allclose(pose.to_np(), camera_body.global_transform.to_np())
