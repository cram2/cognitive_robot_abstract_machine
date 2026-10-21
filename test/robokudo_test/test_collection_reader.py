from __future__ import annotations

import numpy as np
import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.camera_placement import MeasuredCameraPose, SharedWorldCameraPlacement
from robokudo.cas import CAS
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.exceptions import UnknownWorldFrameError
from robokudo.io.camera_interface import CameraInterface
from robokudo.pipeline import Pipeline

from ._perception_world import FrameName, camera_pose


class MeasuresItsCameraPose(CameraInterface):
    """
    A camera interface whose every frame comes with the camera pose it measured.
    """

    def __init__(self, measured_pose: MeasuredCameraPose) -> None:
        super().__init__(camera_config=BaseCameraConfig(interface_type="test"))
        self.measured_pose = measured_pose
        """
        The camera pose reported with every frame.
        """

    def has_new_data(self) -> bool:
        return True

    def set_data(self, cas: CAS, world: World) -> MeasuredCameraPose:
        return self.measured_pose


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
    def test_camera_is_placed_where_its_interface_measured_it(self) -> None:
        measured_pose = MeasuredCameraPose(
            world_frame=FrameName.WORLD,
            camera_frame=FrameName.CAMERA,
            world_T_camera=camera_pose(x=1.0),
        )
        pipeline = pipeline_reading(MeasuresItsCameraPose(measured_pose))
        [reader] = pipeline.children

        reader.update()

        camera_to_world_transform = pipeline.cas.camera_to_world_transform
        assert camera_to_world_transform.child_frame is pipeline.world.get_body_by_name(
            FrameName.CAMERA
        )
        assert np.allclose(
            camera_to_world_transform.to_np(), measured_pose.world_T_camera.to_np()
        )

    def test_camera_is_placed_with_the_camera_placement_of_the_pipeline(
        self,
    ) -> None:
        world = World()
        with world.modify_world():
            world.add_body(Body(name=PrefixedName("root_of_the_shared_world")))
        pipeline = pipeline_reading(
            MeasuresItsCameraPose(
                MeasuredCameraPose(
                    world_frame=FrameName.WORLD,
                    camera_frame=FrameName.CAMERA,
                    world_T_camera=camera_pose(),
                )
            )
        )
        pipeline.camera_placement = SharedWorldCameraPlacement(world)
        [reader] = pipeline.children

        with pytest.raises(UnknownWorldFrameError):
            reader.update()
