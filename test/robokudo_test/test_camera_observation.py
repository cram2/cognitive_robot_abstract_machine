from __future__ import annotations

# %% Imports
from dataclasses import replace

from krrood.adapters.json_serializer import DataclassJSONSerializer
from robokudo.io.cas_view_codecs import (
    CameraObservationCodec,
    RecordedCameraObservation,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
import numpy as np
import pytest

from robokudo.cas import CAS
from robokudo.exceptions import (
    CameraObservationMissing,
    CameraPoseMissing,
    InvalidCameraObservation,
)
from robokudo.types.camera import CameraObservation
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% Camera behavior


def test_cas_exposes_camera_observation_without_copying_camera():
    world = World()
    root = Body(name=PrefixedName(name="world"))
    with world.modify_world():
        world.add_body(root)
    model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(width=640, height=480),
        field_of_view=FieldOfView(
            horizontal_angle=np.radians(90.0),
            vertical_angle=np.radians(60.0),
        ),
    )
    camera = Camera(
        name=PrefixedName(name="camera"),
        root=root,
        camera_model=model,
    )
    world_T_camera = HomogeneousTransformationMatrix(
        reference_frame=root,
        child_frame=root,
    )
    observation = CameraObservation(
        camera=camera,
        effective_camera_model=model,
        world_T_camera=world_T_camera,
        timestamp_nanoseconds=123,
    )
    cas = CAS()

    cas.camera_observation = observation

    assert cas.camera_observation is observation
    assert cas.require_camera_observation() is observation
    assert cas.camera_observation.camera is camera
    assert cas.camera_observation.resolution == model.resolution
    assert cas.camera_observation.field_of_view == model.field_of_view
    assert cas.camera_observation.world_T_camera_or_raise() is world_T_camera
    assert np.array_equal(
        cas.camera_observation.camera_T_world, world_T_camera.inverse()
    )


def test_cas_requires_camera_observation():
    with pytest.raises(CameraObservationMissing):
        CAS().require_camera_observation()


def test_camera_observation_requires_pose_for_spatial_access():
    world = World()
    root = Body(name=PrefixedName(name="world"))
    with world.modify_world():
        world.add_body(root)
    model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(width=640, height=480),
        field_of_view=FieldOfView(),
    )
    observation = CameraObservation(
        camera=Camera(
            name=PrefixedName(name="camera"),
            root=root,
            camera_model=model,
        ),
        effective_camera_model=model,
        world_T_camera=None,
        timestamp_nanoseconds=123,
    )

    with pytest.raises(CameraPoseMissing):
        observation.world_T_camera_or_raise()
    with pytest.raises(CameraPoseMissing):
        observation.camera_T_world


def test_camera_observation_returns_copy_with_pose():
    world = World()
    root = Body(name=PrefixedName(name="world"))
    with world.modify_world():
        world.add_body(root)
    model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(width=640, height=480),
        field_of_view=FieldOfView(),
    )
    observation = CameraObservation(
        camera=Camera(
            name=PrefixedName(name="camera"),
            root=root,
            camera_model=model,
        ),
        effective_camera_model=model,
        world_T_camera=None,
        timestamp_nanoseconds=123,
    )
    world_T_camera = HomogeneousTransformationMatrix(
        reference_frame=root,
        child_frame=root,
    )

    updated_observation = observation.with_world_T_camera(world_T_camera)

    assert observation.world_T_camera is None
    assert updated_observation.world_T_camera is world_T_camera
    assert updated_observation.camera is observation.camera
    assert updated_observation.effective_camera_model is model
    assert (
        updated_observation.timestamp_nanoseconds == observation.timestamp_nanoseconds
    )


# %% Observation frame contract


@pytest.fixture
def unposed_camera_observation() -> CameraObservation:
    """
    Provide a camera observation before a world pose has been acquired.
    """
    root = Body(name=PrefixedName(name="optical_frame"))
    model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(), field_of_view=FieldOfView()
    )
    return CameraObservation(
        camera=Camera(name=PrefixedName(name="camera"), root=root, camera_model=model),
        effective_camera_model=model,
        world_T_camera=None,
        timestamp_nanoseconds=0,
    )


def test_observation_rejects_pose_without_reference_frame(
    unposed_camera_observation: CameraObservation,
) -> None:
    """
    Pose replacement validates the reference frame before accepting the observation.
    """
    pose = HomogeneousTransformationMatrix(
        child_frame=unposed_camera_observation.camera.root
    )
    with pytest.raises(InvalidCameraObservation):
        unposed_camera_observation.with_world_T_camera(pose)


def test_observation_rejects_pose_for_another_camera(
    unposed_camera_observation: CameraObservation,
) -> None:
    """
    A sampled child must identify the observation's own optical camera.
    """
    pose = HomogeneousTransformationMatrix(
        reference_frame=Body(name=PrefixedName(name="world")),
        child_frame=Body(name=PrefixedName(name="another_camera")),
    )
    with pytest.raises(InvalidCameraObservation):
        unposed_camera_observation.with_world_T_camera(pose)


@pytest.mark.parametrize("with_pose", [False, True])
def test_observation_resolves_unspecified_child_to_camera_root(
    unposed_camera_observation: CameraObservation,
    with_pose: bool,
) -> None:
    """
    An unspecified sampled child denotes the camera's optical root.
    """
    observation = unposed_camera_observation
    if with_pose:
        observation = observation.with_world_T_camera(
            HomogeneousTransformationMatrix(
                reference_frame=Body(name=PrefixedName(name="world"))
            )
        )
    assert observation.pose_child is observation.camera.root


def test_observation_accepts_ros_frame_alias(
    unposed_camera_observation: CameraObservation,
) -> None:
    """
    A leading ROS slash can distinguish body names for the same optical frame.
    """
    root = unposed_camera_observation.camera.root
    alias = Body(name=PrefixedName(name="/" + root.name.name, prefix=root.name.prefix))
    pose = HomogeneousTransformationMatrix(
        reference_frame=Body(name=PrefixedName(name="world")), child_frame=alias
    )
    observation = unposed_camera_observation.with_world_T_camera(pose)
    assert observation.pose_child is alias


def test_observation_rejects_alias_from_another_robot(
    unposed_camera_observation: CameraObservation,
) -> None:
    """
    Matching ROS frame names do not alias optical bodies from different prefixes.
    """
    root = unposed_camera_observation.camera.root
    alias = Body(name=PrefixedName(name="/" + root.name.name, prefix="another_robot"))
    pose = HomogeneousTransformationMatrix(
        reference_frame=Body(name=PrefixedName(name="world")), child_frame=alias
    )
    with pytest.raises(InvalidCameraObservation):
        unposed_camera_observation.with_world_T_camera(pose)


# %% Recorded frame validation


@pytest.mark.parametrize("missing_reference", [False, True])
def test_decoded_observation_rejects_invalid_pose_frames(
    unposed_camera_observation: CameraObservation,
    missing_reference: bool,
) -> None:
    """
    Recorded data passes the same frame validation as directly constructed observations.
    """
    observation = unposed_camera_observation
    world = World()
    reference = Body(name=PrefixedName(name="world"))
    other_camera = Body(name=PrefixedName(name="another_camera"))
    with world.modify_world():
        for body in (reference, observation.camera.root, other_camera):
            world.add_body(body)
        for child in (observation.camera.root, other_camera):
            world.add_connection(FixedConnection(parent=reference, child=child))
        world.add_semantic_annotation(observation.camera)
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    codec = CameraObservationCodec()
    payload = codec.encode(observation)
    recorded = RecordedCameraObservation(
        camera_id=str(observation.camera.id),
        effective_camera_model=observation.effective_camera_model,
        world_T_camera=HomogeneousTransformationMatrix(
            reference_frame=None if missing_reference else reference,
            child_frame=observation.camera.root if missing_reference else other_camera,
        ),
        timestamp_nanoseconds=observation.timestamp_nanoseconds,
    )
    invalid_payload = replace(
        payload, payload=DataclassJSONSerializer.to_json(recorded)
    )
    with pytest.raises(InvalidCameraObservation):
        codec.decode_with_tracker(invalid_payload, tracker)
