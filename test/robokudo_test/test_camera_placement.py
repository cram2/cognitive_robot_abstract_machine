from __future__ import annotations

import numpy as np
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.camera_placement import CameraPlacement

from ._perception_world import (
    CAMERA_FRAME,
    WORLD_FRAME,
    camera_pose,
)

# %% camera placement


class TestCameraPlacement:
    def test_adds_missing_frames_and_moves_the_camera(self) -> None:
        world = World()
        world_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0, y=2.0, z=3.0
        )

        bound_world_T_camera = CameraPlacement(world).place(
            WORLD_FRAME, CAMERA_FRAME, world_T_camera
        )

        world_body = world.get_body_by_name(WORLD_FRAME)
        camera_body = world.get_body_by_name(CAMERA_FRAME)
        assert world.root is world_body
        assert bound_world_T_camera.reference_frame is world_body
        assert bound_world_T_camera.child_frame is camera_body
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())

    def test_moves_a_placed_camera_again(self) -> None:
        camera_placement = CameraPlacement(World())
        camera_placement.place(WORLD_FRAME, CAMERA_FRAME, camera_pose(x=1.0))
        world_T_camera = camera_pose(x=2.0)

        camera_placement.place(WORLD_FRAME, CAMERA_FRAME, world_T_camera)

        camera_body = camera_placement.world.get_body_by_name(CAMERA_FRAME)
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())

    def test_leaves_a_camera_located_by_another_kinematic_structure_in_place(
        self,
    ) -> None:
        world = World()
        world_body = Body(name=PrefixedName(WORLD_FRAME))
        head_body = Body(name=PrefixedName("head"))
        camera_body = Body(name=PrefixedName(CAMERA_FRAME))
        with world.modify_world():
            world.add_body(world_body)
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=world_body, child=head_body, world=world
                )
            )
            world.add_connection(FixedConnection(parent=head_body, child=camera_body))
        head_body.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                z=1.5, reference_frame=world_body
            )
        )
        camera_pose_before = camera_body.global_transform.to_np()
        world_T_camera = camera_pose(x=4.0)

        bound_world_T_camera = CameraPlacement(world).place(
            WORLD_FRAME, CAMERA_FRAME, world_T_camera
        )

        assert np.allclose(camera_body.global_transform.to_np(), camera_pose_before)
        assert bound_world_T_camera.child_frame is camera_body
        assert np.allclose(bound_world_T_camera.to_np(), world_T_camera.to_np())
