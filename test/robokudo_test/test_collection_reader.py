from __future__ import annotations

import numpy as np
import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.cas import CAS
from robokudo.descriptors.camera_configs.config_filereader_playback import (
    FileReaderCameraConfig,
)
from robokudo.exceptions import UnknownWorldFrameError
from robokudo.io.camera_interface import CameraInterface
from robokudo.pipeline import Pipeline

from ._perception_world import FrameName, camera_pose


class PlacesItsFixedCamera(CameraInterface):
    """
    A camera interface whose every frame was recorded from the camera pose its
    configuration fixes.
    """

    def has_new_data(self) -> bool:
        return True

    def set_data(self, cas: CAS, world: World) -> None:
        self.place_static_camera(cas, world)


def fixed_camera_at(world_T_camera: HomogeneousTransformationMatrix) -> CameraInterface:
    """
    :param world_T_camera: The pose of the camera in the world frame.
    :return: A camera interface whose configuration fixes its camera at
        ``world_T_camera`` against the test frames.
    """
    return PlacesItsFixedCamera(
        camera_config=FileReaderCameraConfig(
            static_camera_transform_enabled=True,
            static_world_frame=FrameName.WORLD,
            static_camera_frame=FrameName.CAMERA,
            static_world_T_camera=world_T_camera,
        )
    )


def pipeline_reading(camera_interface: CameraInterface) -> Pipeline:
    """
    :param camera_interface: The camera interface to read from.
    :return: A pipeline whose only annotator reads from ``camera_interface``.
    """
    pipeline = Pipeline("Pipeline")
    pipeline.add_child(
        CollectionReaderAnnotator(
            descriptor=CollectionReaderAnnotator.Descriptor(
                camera_config=camera_interface.camera_config,
                camera_interface=camera_interface,
            )
        )
    )
    return pipeline


# %% camera pose of a frame


class TestCameraPoseOfAFrame:
    def test_camera_is_placed_in_the_world_of_the_pipeline(self) -> None:
        camera_interface = fixed_camera_at(camera_pose(x=1.0))
        pipeline = pipeline_reading(camera_interface)
        [reader] = pipeline.children

        reader.update()

        camera_to_world_transform = pipeline.cas.camera_to_world_transform
        assert camera_to_world_transform.child_frame is pipeline.world.get_body_by_name(
            FrameName.CAMERA
        )
        assert np.allclose(
            camera_to_world_transform.to_np(),
            camera_interface.camera_config.static_world_T_camera.to_np(),
        )

    def test_camera_measured_against_a_frame_the_world_lacks_is_not_placed(
        self,
    ) -> None:
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName("root_of_another_world")))
        pipeline = pipeline_reading(fixed_camera_at(camera_pose()))
        pipeline.world = world
        [reader] = pipeline.children

        with pytest.raises(UnknownWorldFrameError):
            reader.update()


# %% camera without a fixed pose


class TestCameraWithoutAFixedPose:
    def test_camera_whose_configuration_fixes_no_pose_is_not_placed(self) -> None:
        camera_interface = PlacesItsFixedCamera(camera_config=FileReaderCameraConfig())
        world = World()
        cas = CAS()

        camera_interface.set_data(cas, world)

        assert cas.camera_to_world_transform is None
        assert world.is_empty()
