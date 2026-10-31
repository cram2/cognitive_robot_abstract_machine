# %% Imports
import numpy as np
import pytest
from std_msgs.msg import String

from robokudo.types.camera import CameraObservation

from robokudo.cas import CASViews
from robokudo.io.cas_view_codecs import CASViewCodecRegistry
from robokudo.types.cv import ColorToDepthRatio
from robokudo.types.tf import StampedTransform
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.camera_model import (
    PinholeCameraModel,
    RationalPolynomialCameraDistortion,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% Camera behavior


def test_bytes_codec_roundtrip():
    registry = CASViewCodecRegistry()
    input_payload = b"\x00\x01\x02\x03"

    encoded = registry.encode_view("blob", input_payload)
    assert encoded is not None
    assert encoded["serializer_id"] == "bytes_v1"

    decoded_name, decoded_payload = registry.decode_view(encoded)
    assert decoded_name == "blob"
    assert decoded_payload == input_payload


def test_ros_message_codec_roundtrip():
    registry = CASViewCodecRegistry()
    msg = String(data="hello")

    encoded = registry.encode_view("msg", msg)
    assert encoded is not None
    assert encoded["serializer_id"] == "ros_message_v1"

    decoded_name, decoded_msg = registry.decode_view(encoded)
    assert decoded_name == "msg"
    assert isinstance(decoded_msg, String)
    assert decoded_msg.data == "hello"


def test_stamped_transform_codec_roundtrip():
    registry = CASViewCodecRegistry()
    transform = StampedTransform()
    transform.rotation = [0.0, 0.0, 0.0, 1.0]
    transform.translation = [1.0, 2.0, 3.0]
    transform.frame = "map"
    transform.child_frame = "camera"
    transform.timestamp.sec = 123
    transform.timestamp.nanosec = 456

    encoded = registry.encode_view("tf", transform)
    assert encoded is not None
    assert encoded["serializer_id"] == "robokudo_stamped_transform_v1"

    decoded_name, decoded_transform = registry.decode_view(encoded)
    assert decoded_name == "tf"
    assert isinstance(decoded_transform, StampedTransform)
    assert decoded_transform.rotation == transform.rotation
    assert decoded_transform.translation == transform.translation
    assert decoded_transform.frame == transform.frame
    assert decoded_transform.child_frame == transform.child_frame
    assert decoded_transform.timestamp.sec == transform.timestamp.sec
    assert decoded_transform.timestamp.nanosec == transform.timestamp.nanosec


@pytest.mark.parametrize("has_world_pose", [True, False])
def test_camera_observation_codec_resolves_camera_and_preserves_pose_state(
    monkeypatch, has_world_pose
):
    world = World()
    root = Body(name=PrefixedName(name="world"))
    with world.modify_world():
        world.add_body(root)
    camera_model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(width=640, height=480),
        field_of_view=FieldOfView(),
        distortion=RationalPolynomialCameraDistortion(
            radial_coefficient_1=0.1,
            tangential_coefficient_1=0.01,
        ),
    )
    camera = Camera(
        name=PrefixedName(name="camera"),
        root=root,
        camera_model=camera_model,
    )
    with world.modify_world():
        world.add_semantic_annotation(camera)
    tracker = WorldEntityWithIDKwargsTracker.from_world(world)
    monkeypatch.setattr(
        "robokudo.io.cas_view_codecs.world.get_world_entity_tracker",
        lambda: tracker,
    )
    observation = CameraObservation(
        camera=camera,
        effective_camera_model=camera_model,
        world_T_camera=root.global_transform if has_world_pose else None,
        timestamp_nanoseconds=123,
    )
    registry = CASViewCodecRegistry()

    encoded = registry.encode_view("camera_observation", observation)
    decoded_name, decoded_observation = registry.decode_view(encoded)

    assert decoded_name == "camera_observation"
    assert decoded_observation.camera is camera
    assert decoded_observation.effective_camera_model == camera_model
    if has_world_pose:
        np.testing.assert_allclose(
            decoded_observation.world_T_camera.to_np(),
            observation.world_T_camera.to_np(),
        )
    else:
        assert decoded_observation.world_T_camera is None
    assert (
        decoded_observation.timestamp_nanoseconds == observation.timestamp_nanoseconds
    )


def test_color_to_depth_ratio_roundtrip():
    registry = CASViewCodecRegistry()
    ratio = ColorToDepthRatio(horizontal=0.5, vertical=0.25)

    encoded = registry.encode_view(CASViews.COLOR_TO_DEPTH_RATIO, ratio)

    assert registry.decode_view(encoded) == (CASViews.COLOR_TO_DEPTH_RATIO, ratio)


def test_decode_view_unknown_serializer_raises():
    registry = CASViewCodecRegistry()
    with pytest.raises(ValueError):
        registry.decode_view(
            {
                "view_name": "unsupported",
                "serializer_id": "unknown_codec_v1",
                "type_name": "builtins.int",
                "payload": 1,
                "metadata": {},
            }
        )


def test_decode_view_missing_required_field_raises():
    registry = CASViewCodecRegistry()
    with pytest.raises(KeyError):
        registry.decode_view(
            {
                "serializer_id": "bytes_v1",
                "type_name": "builtins.bytes",
                "payload": "AA==",
                "metadata": {},
            }
        )
