from __future__ import annotations

import numpy as np
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

from robokudo.cas import CAS
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    SemDTRayTracerCameraConfig,
)
from robokudo.io.semdt_raytracer_camera_interface import SemDTRayTracerCameraInterface

# %% camera placement


class TestRenderingCameraPlacement:
    def test_rendering_camera_is_placed_in_the_world_perception_perceives_into(
        self,
    ) -> None:
        camera_config = SemDTRayTracerCameraConfig(
            world_descriptor_name="world_semdt_raytracer_cylinders", resolution=16
        )
        world = World()
        cas = CAS()

        SemDTRayTracerCameraInterface(camera_config).set_data(cas, world)

        camera_to_world_transform = cas.camera_to_world_transform
        assert camera_to_world_transform.child_frame is world.get_body_by_name(
            camera_config.camera_frame
        )
        assert np.allclose(
            camera_to_world_transform.to_np(),
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=camera_config.camera_x,
                y=camera_config.camera_y,
                z=camera_config.camera_z,
                roll=camera_config.camera_roll,
                pitch=camera_config.camera_pitch,
                yaw=camera_config.camera_yaw,
            ).to_np(),
        )
