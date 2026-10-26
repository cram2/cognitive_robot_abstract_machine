from __future__ import annotations

import numpy as np
import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.io.camera_placement import CameraPlacement, MeasuredCameraPose
from robokudo.cas import CAS
from robokudo.exceptions import UnknownWorldFrameError

from ._perception_world import (
    FrameName,
    SharedWorld,
    camera_pose,
    world_with_a_head_camera,
)


def measured_at(world_T_camera: HomogeneousTransformationMatrix) -> MeasuredCameraPose:
    """
    :param world_T_camera: The measured pose of the camera in the world frame.
    :return: The pose a camera interface measured against the test frames.
    """
    return MeasuredCameraPose(
        world_frame=FrameName.WORLD,
        camera_frame=FrameName.CAMERA,
        world_T_camera=world_T_camera,
    )


# %% world frame


class TestWorldFrame:
    def test_empty_world_gets_the_world_frame_as_its_root(self) -> None:
        world = World()
        cas = CAS()
        world_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0, y=2.0, z=3.0
        )

        CameraPlacement(world).place(cas, measured_at(world_T_camera))

        world_body = world.get_body_by_name(FrameName.WORLD)
        camera_body = world.get_body_by_name(FrameName.CAMERA)
        assert world.root is world_body
        assert cas.camera_to_world_transform.reference_frame is world_body
        assert cas.camera_to_world_transform.child_frame is camera_body
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())

    def test_unknown_world_frame_of_a_world_with_bodies_is_not_added(self) -> None:
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName("root_of_another_world")))
        names_before = [body.name for body in world.bodies]

        with pytest.raises(UnknownWorldFrameError):
            CameraPlacement(world).place(CAS(), measured_at(camera_pose()))

        assert [body.name for body in world.bodies] == names_before

    def test_names_the_frames_of_the_placed_camera_in_the_cas(self) -> None:
        cas = CAS()
        measured_pose = measured_at(camera_pose())

        CameraPlacement(World()).place(cas, measured_pose)

        assert (cas.world_frame, cas.camera_frame) == (
            measured_pose.world_frame,
            measured_pose.camera_frame,
        )


# %% camera perception adds


class TestCameraPerceptionAdds:
    def test_adds_the_camera_below_a_known_world_frame_and_moves_it(self) -> None:
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName(FrameName.WORLD)))
        camera_placement = CameraPlacement(world)
        camera_placement.place(CAS(), measured_at(camera_pose(x=1.0)))
        world_T_camera = camera_pose(x=2.0)

        camera_placement.place(CAS(), measured_at(world_T_camera))

        [camera_body] = world.get_bodies_by_name(FrameName.CAMERA)
        assert camera_body.parent_connection.parent is world.get_body_by_name(
            FrameName.WORLD
        )
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())

    def test_placements_of_one_world_share_its_camera(self) -> None:
        world = World()
        CameraPlacement(world).place(CAS(), measured_at(camera_pose(x=1.0)))
        world_T_camera = camera_pose(x=2.0)

        CameraPlacement(world).place(CAS(), measured_at(world_T_camera))

        [camera_body] = world.get_bodies_by_name(FrameName.CAMERA)
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())


# %% camera of the world


class TestCameraOfTheWorld:
    def test_camera_of_the_world_is_located_by_the_world(self) -> None:
        world = world_with_a_head_camera()
        camera_body = world.get_body_by_name(FrameName.CAMERA)
        camera_pose_before = camera_body.global_transform.to_np()
        cas = CAS()

        CameraPlacement(world).place(cas, measured_at(camera_pose(x=4.0)))

        assert np.allclose(camera_body.global_transform.to_np(), camera_pose_before)
        assert cas.camera_to_world_transform.reference_frame is world.root
        assert cas.camera_to_world_transform.child_frame is camera_body
        assert np.allclose(cas.camera_to_world_transform.to_np(), camera_pose_before)
        assert cas.world_frame == world.root.name.name

    def test_camera_of_the_world_needs_no_known_world_frame(self) -> None:
        world = world_with_a_head_camera()
        cas = CAS()

        CameraPlacement(world).place(
            cas,
            MeasuredCameraPose(
                world_frame="frame_unknown_to_the_world",
                camera_frame=FrameName.CAMERA,
                world_T_camera=camera_pose(),
            ),
        )

        assert cas.camera_to_world_transform.child_frame is world.get_body_by_name(
            FrameName.CAMERA
        )


# %% camera of a shared world


class TestCameraOfASharedWorld:
    def test_shared_world_gets_no_camera_body(self, shared_world: SharedWorld) -> None:
        body_ids_before = [body.id for body in shared_world.world.bodies]

        CameraPlacement(shared_world.world).place(
            CAS(), measured_at(camera_pose(x=1.0))
        )

        assert [body.id for body in shared_world.world.bodies] == body_ids_before

    def test_camera_of_a_shared_world_is_located_by_its_measured_pose(
        self, shared_world: SharedWorld
    ) -> None:
        cas = CAS()
        measured_pose = measured_at(camera_pose(x=1.0))

        CameraPlacement(shared_world.world).place(cas, measured_pose)

        assert (
            cas.camera_to_world_transform.reference_frame
            is shared_world.world.get_body_by_name(measured_pose.world_frame)
        )
        assert np.allclose(
            cas.camera_to_world_transform.to_np(),
            measured_pose.world_T_camera.to_np(),
        )
        assert cas.camera_frame == measured_pose.camera_frame

    def test_camera_of_a_robot_in_a_shared_world_is_located_by_the_world(
        self, shared_world_with_a_robot: SharedWorld
    ) -> None:
        world = shared_world_with_a_robot.world
        camera_body = world.get_body_by_name(FrameName.CAMERA)
        cas = CAS()

        CameraPlacement(world).place(cas, measured_at(camera_pose(x=4.0)))

        assert cas.camera_to_world_transform.child_frame is camera_body
        assert np.allclose(
            cas.camera_to_world_transform.to_np(), camera_body.global_transform.to_np()
        )
