import pytest
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from std_msgs.msg import String

from robokudo.cas import CASViews
from robokudo.io.cas_view_codecs import CASViewCodecRegistry, KrroodCodec
from robokudo.types.cv import ColorToDepthRatio
from robokudo.types.tf import StampedTransform


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


# %% world entities


def test_krrood_codec_resolves_world_entities_against_the_given_world() -> None:
    world = World()
    world_body = Body(name=PrefixedName("map"))
    with world.modify_world():
        world.add_body(world_body)
    transform = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, reference_frame=world_body
    )
    codec = KrroodCodec()

    decoded = codec.decode(
        codec.encode(transform),
        **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
    )

    assert decoded.reference_frame is world_body
