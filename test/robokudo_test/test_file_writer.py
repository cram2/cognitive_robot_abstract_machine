from __future__ import annotations

# %% Imports
import json
from collections.abc import Iterator

import numpy as np
import pytest
from py_trees.common import Status

from krrood.adapters.json_serializer import from_json
from robokudo.annotators.file_writer import FileWriter
from robokudo.types.cv import ColorToDepthRatio
from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_filereader_playback import (
    FileReaderCameraConfig,
)
from robokudo.io.file_reader_interface import RGBDFileReaderInterface
from robokudo.types.camera import CameraObservation
from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world import World

from .test_camera_interface_observation import runtime_world

# %% Camera observation recording


def test_file_writer_and_reader_roundtrip_recorded_camera_metadata(
    runtime_world: World, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The file roundtrip preserves camera metadata without persisting the pose.
    """
    camera_model = PinholeCameraModel(
        image_resolution=CameraResolution(width=4, height=3),
        focal_length_x=10.0,
        focal_length_y=11.0,
        principal_point=Point2(x=2.0, y=1.0),
    )
    camera_frame = Body(name=PrefixedName(name="camera_optical_frame"))
    camera = Camera(
        name=PrefixedName(name="camera"),
        root=camera_frame,
        camera_model=camera_model,
    )
    world_frame = Body(name=PrefixedName(name="map"))
    world_T_camera = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=1.0,
        pos_y=2.0,
        pos_z=3.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        quat_w=1.0,
        reference_frame=world_frame,
        child_frame=camera_frame,
    )
    timestamp_nanoseconds = 123_456_789_012
    cas = CAS(timestamp=timestamp_nanoseconds)
    cas.color_image = np.zeros((3, 4, 3), dtype=np.uint8)
    cas.depth_image = np.zeros((3, 4), dtype=np.uint16)
    cas.camera_observation = CameraObservation(
        camera=camera,
        effective_camera_model=camera_model,
        world_T_camera=world_T_camera,
        timestamp_nanoseconds=timestamp_nanoseconds,
    )
    descriptor = FileWriter.Descriptor()
    descriptor.parameters.target_dir = str(tmp_path)
    writer = FileWriter(descriptor=descriptor)
    monkeypatch.setattr(writer, "get_cas", lambda: cas)

    assert writer.update() is Status.SUCCESS

    observation_path = tmp_path / (
        f"{descriptor.parameters.filename_prefix}{cas.timestamp}_"
        f"{CASViews.CAMERA_OBSERVATION}.json"
    )
    with observation_path.open() as observation_file:
        recorded_observation = json.load(observation_file)
    assert set(recorded_observation) == {
        "camera_frame",
        "effective_camera_model",
        "timestamp_nanoseconds",
    }
    assert recorded_observation["camera_frame"] == camera_frame.name.name
    assert from_json(recorded_observation["effective_camera_model"]) == camera_model
    assert recorded_observation["timestamp_nanoseconds"] == timestamp_nanoseconds
    assert set(cas.views) == {
        CASViews.COLOR_IMAGE,
        CASViews.DEPTH_IMAGE,
        CASViews.CAMERA_OBSERVATION,
    }

    reader = RGBDFileReaderInterface(
        FileReaderCameraConfig(
            target_dir=str(tmp_path),
            loop=False,
            color_to_depth_ratio=ColorToDepthRatio(),
        )
    )
    replayed_cas = CAS()
    reader.set_data(replayed_cas)

    replayed_observation = replayed_cas.require_camera_observation()
    assert replayed_observation.camera.root.name.name == camera_frame.name.name
    assert replayed_observation.effective_camera_model == camera_model
    assert replayed_observation.timestamp_nanoseconds == timestamp_nanoseconds
    assert replayed_observation.world_T_camera is None
