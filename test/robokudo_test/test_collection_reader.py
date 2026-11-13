"""
Single-reader acquisition and CAS lifecycle.
"""

from __future__ import annotations

# %% Imports
import logging
from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np
import pytest
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from py_trees.common import Status
from semantic_digital_twin.datastructures.camera_model import (
    CameraModality,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.exceptions import InvalidCollectionReaderParent
from robokudo.cas import CAS
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.descriptors.camera_configs.config_filereader_playback import (
    FileReaderCameraConfig,
)
from robokudo.exceptions import UnknownWorldFrameError
from robokudo.io.camera_interface import CameraInterface
from robokudo.pipeline import Pipeline
from robokudo.types.core import Query

from ._perception_world import FrameName, camera_pose


# %% Acquisition fixture
@dataclass
class ReaderPipeline:
    """
    A pipeline with a controlled source of sensor data.
    """

    reader: CollectionReaderAnnotator
    """
    Annotator receiving the controlled camera data.
    """
    pipeline: Pipeline
    """
    Pipeline owning the CAS.
    """

    camera_interface: Mock
    """
    Camera interface with controlled readiness and acquisition.
    """


@pytest.fixture
def reader_pipeline() -> ReaderPipeline:
    """
    Construct a single-reader pipeline without external camera services.
    """
    camera_interface = Mock(spec=CameraInterface)
    descriptor = CollectionReaderAnnotator.Descriptor(
        camera_config=BaseCameraConfig(interface_type="test"),
        camera_interface=camera_interface,
    )
    reader = CollectionReaderAnnotator(descriptor)
    pipeline = Pipeline(children=[reader])
    return ReaderPipeline(reader, pipeline, camera_interface)


# %% Single-reader configuration


def test_collection_reader_descriptor_uses_configured_camera_interface(
    reader_pipeline: ReaderPipeline,
) -> None:
    """
    The reader parameters retain the configured camera interface.
    """
    parameters = reader_pipeline.reader.descriptor.parameters
    assert parameters.camera_interface is reader_pipeline.camera_interface


# %% Acquisition lifecycle


def test_collection_reader_waits_without_replacing_cas(
    reader_pipeline: ReaderPipeline,
) -> None:
    """
    Waiting for sensor data preserves the existing CAS.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = False
    original_cas = reader_pipeline.pipeline.cas
    assert reader_pipeline.reader.update() == Status.RUNNING
    assert reader_pipeline.pipeline.cas is original_cas
    reader_pipeline.camera_interface.has_new_data.assert_called_once_with()
    reader_pipeline.camera_interface.set_data.assert_not_called()


def test_collection_reader_populates_fresh_cas(reader_pipeline: ReaderPipeline) -> None:
    """
    Available sensor data is written once to a new pipeline CAS.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = True
    original_cas = reader_pipeline.pipeline.cas
    assert reader_pipeline.reader.update() == Status.SUCCESS
    assert reader_pipeline.pipeline.cas.cas_id == original_cas.cas_id + 1
    reader_pipeline.camera_interface.has_new_data.assert_called_once_with()
    reader_pipeline.camera_interface.set_data.assert_called_once_with(
        cas=reader_pipeline.pipeline.cas, world=reader_pipeline.pipeline.world
    )


def test_collection_reader_preserves_query(reader_pipeline: ReaderPipeline) -> None:
    """
    A fresh sensor CAS retains the existing query object.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = True
    query = Query()
    reader_pipeline.pipeline.cas.query = query
    assert reader_pipeline.reader.update() == Status.SUCCESS
    assert reader_pipeline.pipeline.cas.query is query


def test_collection_reader_reports_wait_threshold(
    reader_pipeline: ReaderPipeline,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Reaching the wait threshold emits a warning and resets the counter.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = False
    reader_pipeline.reader.iterations_since_last_data_warn_threshold = 1
    with caplog.at_level(logging.WARNING):
        assert reader_pipeline.reader.update() == Status.RUNNING
    assert reader_pipeline.reader.iterations_since_last_data == 0
    assert [record.levelno for record in caplog.records] == [
        logging.WARNING,
        logging.WARNING,
    ]


def test_collection_reader_requires_pipeline(reader_pipeline: ReaderPipeline) -> None:
    """
    Acquisition without an owning pipeline fails before writing sensor data.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = True
    reader_pipeline.pipeline.remove_child(reader_pipeline.reader)
    assert reader_pipeline.reader.update() == Status.FAILURE
    reader_pipeline.camera_interface.set_data.assert_not_called()


# %% Behavior-tree initialization


def test_collection_reader_tick_waits_for_data(reader_pipeline: ReaderPipeline) -> None:
    """
    Ticking the reader initializes its behavior-tree state and waits.
    """
    reader_pipeline.camera_interface.has_new_data.return_value = False
    reader_pipeline.reader.feedback_message = "previous feedback"
    reader_pipeline.reader.tick_once()
    assert reader_pipeline.reader.status == Status.RUNNING
    reader_pipeline.camera_interface.set_data.assert_not_called()


def test_collection_reader_initialization_requires_sequence(
    reader_pipeline: ReaderPipeline,
) -> None:
    """
    Invalid parent placement raises the specific initialization error.
    """
    reader_pipeline.pipeline.remove_child(reader_pipeline.reader)
    with pytest.raises(InvalidCollectionReaderParent):
        reader_pipeline.reader.initialise()


# %% camera pose of a frame


class PlacesItsFixedCamera(CameraInterface):
    """
    A camera interface whose every frame was recorded from the camera pose its
    configuration fixes.
    """

    def has_new_data(self) -> bool:
        return True

    def set_data(self, *, cas: CAS, world: World) -> None:
        self.store_camera_observation(
            cas=cas,
            world=world,
            camera_model=PinholeCameraModel.from_field_of_view(
                resolution=CameraResolution(), field_of_view=FieldOfView()
            ),
            camera_frame=FrameName.CAMERA,
            timestamp_nanoseconds=0,
            modalities=(CameraModality.COLOR,),
            world_T_camera=self.static_world_T_camera_if_configured(
                world=world, camera_frame=FrameName.CAMERA
            ),
        )


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


class TestCameraPoseOfAFrame:
    def test_camera_is_placed_in_the_world_of_the_pipeline(self) -> None:
        camera_interface = fixed_camera_at(camera_pose(x=1.0))
        pipeline = pipeline_reading(camera_interface)
        [reader] = pipeline.children

        reader.update()

        world_T_camera = pipeline.cas.camera_observation.world_T_camera
        assert world_T_camera.child_frame is pipeline.world.get_body_by_name(
            FrameName.CAMERA
        )
        assert np.allclose(
            world_T_camera.to_np(),
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


class TestCameraWithoutAFixedPose:
    def test_camera_whose_configuration_fixes_no_pose_has_no_pose(self) -> None:
        camera_interface = PlacesItsFixedCamera(camera_config=FileReaderCameraConfig())
        cas = CAS()

        camera_interface.set_data(cas=cas, world=World())

        assert cas.camera_observation.world_T_camera is None
