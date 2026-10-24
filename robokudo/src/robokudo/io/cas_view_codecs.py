"""
Codec registry and serializers for CAS view persistence.
"""

from __future__ import annotations

from robokudo.io.exceptions import CASViewCodecMissing, UnknownViewSerializer

import array
import base64
import importlib
import io
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from krrood.adapters.json_serializer import DataclassJSONSerializer
from krrood.ormatic.utils import classproperty

import numpy as np
from typing_extensions import Any, Iterable
from robokudo import world
from robokudo.exceptions import InvalidCameraObservation
from robokudo.io.cas_annotation_codecs import krrood_to_json, krrood_from_json
from robokudo.types.camera import CameraObservation
from robokudo.io.open3d_codec_utils import (
    is_open3d_point_cloud,
    encode_open3d_point_cloud_to_base64_pcd,
    decode_open3d_point_cloud_from_base64_pcd,
)
from robokudo.types.tf import StampedTransform
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


def _recursive_convert(value: Any) -> Any:
    """
    Recursively convert non-JSON-friendly array-like values into plain containers.
    """
    if isinstance(value, array.array):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_recursive_convert(item) for item in value]
    if isinstance(value, dict):
        return {k: _recursive_convert(v) for k, v in value.items()}
    return value


def _ros_message_to_dict(msg: Any) -> dict[str, Any]:
    """
    Convert a ROS message object into plain dictionary data.
    """
    result: dict[str, Any] = {}
    for field_name, _ in msg.get_fields_and_field_types().items():
        value = getattr(msg, field_name)
        if hasattr(value, "get_fields_and_field_types"):
            result[field_name] = _ros_message_to_dict(value)
        elif isinstance(value, list):
            result[field_name] = [_recursive_convert(item) for item in value]
        else:
            result[field_name] = _recursive_convert(value)
    return result


def _dict_to_ros_message(message_type: type[Any], data_dict: dict[str, Any]) -> Any:
    """
    Populate a ROS message instance from plain dictionary data.
    """
    msg = message_type()
    for field_name, _ in msg.get_fields_and_field_types().items():
        if field_name not in data_dict:
            continue
        value = data_dict[field_name]
        current_field = getattr(msg, field_name)
        if hasattr(current_field, "get_fields_and_field_types"):
            setattr(msg, field_name, _dict_to_ros_message(type(current_field), value))
        elif isinstance(current_field, list):
            setattr(msg, field_name, value)
        else:
            setattr(msg, field_name, value)
    return msg


def _is_json_primitive(value: Any) -> bool:
    """
    Return ``True`` when a value is a JSON primitive.
    """
    return value is None or isinstance(value, (bool, int, float, str))


def _is_json_like(value: Any) -> bool:
    """
    Return ``True`` when a value can be represented as plain JSON data.
    """
    if _is_json_primitive(value):
        return True
    if isinstance(value, list):
        return all(_is_json_like(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_like(v) for k, v in value.items())
    return False


def _full_type_name(value: Any) -> str:
    """
    Return the fully qualified type name for a value.
    """
    clazz = value.__class__
    return f"{clazz.__module__}.{clazz.__name__}"


def _load_type(type_name: str) -> type[Any]:
    """
    Load a Python type from its fully qualified name.
    """
    module_name, class_name = type_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@dataclass
class ViewPayload:
    """
    Persistence envelope for a single CAS view.
    """

    serializer_id: str
    payload: Any
    type_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_document(self, view_name: str) -> dict[str, Any]:
        """
        Build a storage document for one encoded view.
        """
        return {
            "view_name": view_name,
            "serializer_id": self.serializer_id,
            "type_name": self.type_name,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_document(document: dict[str, Any]) -> "ViewPayload":
        """
        Create a payload object from a storage document.
        """
        return ViewPayload(
            serializer_id=document["serializer_id"],
            payload=document["payload"],
            type_name=document["type_name"],
            metadata=document.get("metadata", {}),
        )


class ViewCodec:
    """
    Interface for encoding and decoding CAS view values.
    """

    serializer_id: str

    def can_encode(self, value: Any) -> bool:
        """
        Return ``True`` when this codec can encode ``value``.
        """
        raise NotImplementedError

    def encode(self, value: Any) -> ViewPayload:
        """
        Encode a runtime value into a :class:`ViewPayload`.
        """
        raise NotImplementedError

    def decode(self, payload: ViewPayload) -> Any:
        """
        Decode a persisted payload back into a runtime value.
        """
        raise NotImplementedError


class JsonLikeCodec(ViewCodec):
    """
    Codec for plain JSON-like Python values.
    """

    serializer_id: str = "json_like_v1"

    def can_encode(self, value: Any) -> bool:
        """
        Check whether the value is already JSON-compatible.
        """
        return _is_json_like(value)

    def encode(self, value: Any) -> ViewPayload:
        """
        Wrap a JSON-like value without further transformation.
        """
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=value,
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> Any:
        """
        Return the raw payload value.
        """
        return payload.payload


class TupleCodec(ViewCodec):
    """
    Codec for tuple values.
    """

    serializer_id: str = "tuple_v1"

    def can_encode(self, value: Any) -> bool:
        """
        Check whether the value is a tuple.
        """
        return isinstance(value, tuple)

    def encode(self, value: tuple[Any, ...]) -> ViewPayload:
        """
        Encode a tuple as a list payload.
        """
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=list(value),
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> tuple[Any, ...]:
        """
        Decode list payload data back to a tuple.
        """
        return tuple(payload.payload)


class BytesCodec(ViewCodec):
    """
    Codec for raw bytes data.
    """

    serializer_id: str = "bytes_v1"

    def can_encode(self, value: Any) -> bool:
        """
        Check whether the value is ``bytes`` or ``bytearray``.
        """
        return isinstance(value, (bytes, bytearray))

    def encode(self, value: bytes | bytearray) -> ViewPayload:
        """
        Encode bytes as a base64 text payload.
        """
        as_bytes = bytes(value)
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=base64.b64encode(as_bytes).decode("ascii"),
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> bytes:
        """
        Decode base64 text payload into raw bytes.
        """
        return base64.b64decode(payload.payload.encode("ascii"))


class NumpyCodec(ViewCodec):
    """
    Codec for NumPy arrays.
    """

    serializer_id: str = "numpy_npy_base64_v1"

    def can_encode(self, value: Any) -> bool:
        """
        Check whether the value is a NumPy array.
        """
        return isinstance(value, np.ndarray)

    def encode(self, value: np.ndarray) -> ViewPayload:
        """
        Encode a NumPy array as base64 encoded ``.npy`` data.
        """
        mem_file = io.BytesIO()
        np.save(mem_file, value)
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=base64.b64encode(mem_file.getvalue()).decode("ascii"),
            type_name=_full_type_name(value),
            metadata={"dtype": str(value.dtype), "shape": list(value.shape)},
        )

    def decode(self, payload: ViewPayload) -> np.ndarray:
        """
        Decode base64 encoded ``.npy`` payload data.
        """
        mem_file = io.BytesIO(base64.b64decode(payload.payload.encode("ascii")))
        mem_file.seek(0)
        return np.load(mem_file)


class RosMessageCodec(ViewCodec):
    """
    Codec for ROS message instances.
    """

    serializer_id: str = "ros_message_v1"

    def can_encode(self, value: Any) -> bool:
        """
        Check whether the value looks like a ROS message.
        """
        return hasattr(value, "get_fields_and_field_types")

    def encode(self, value: Any) -> ViewPayload:
        """
        Encode a ROS message to dictionary payload data.
        """
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=_ros_message_to_dict(value),
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> Any:
        """
        Decode dictionary payload data to a ROS message object.
        """
        message_type = _load_type(payload.type_name)
        return _dict_to_ros_message(message_type, payload.payload)


class Open3DPointCloudCodec(ViewCodec):
    """
    Codec for Open3D point cloud values.
    """

    serializer_id: str = "open3d_pointcloud_pcd_base64_v1"

    def can_encode(self, value: Any) -> bool:
        """
        Check whether the value is an Open3D point cloud.
        """
        return is_open3d_point_cloud(value)

    def encode(self, value: Any) -> ViewPayload:
        """
        Encode a point cloud as base64-encoded PCD data.
        """
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=encode_open3d_point_cloud_to_base64_pcd(value),
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> Any:
        """
        Decode base64-encoded PCD data to an Open3D point cloud.
        """
        return decode_open3d_point_cloud_from_base64_pcd(payload.payload)


class StampedTransformCodec(ViewCodec):
    """
    Codec for RoboKudo ``StampedTransform`` values.
    """

    serializer_id: str = "robokudo_stamped_transform_v1"

    def can_encode(self, value: Any) -> bool:
        """
        Check whether the value is a ``StampedTransform``.
        """
        return isinstance(value, StampedTransform)

    def encode(self, value: StampedTransform) -> ViewPayload:
        """
        Encode a ``StampedTransform`` into plain dictionary data.
        """
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload={
                "rotation": list(value.rotation),
                "translation": list(value.translation),
                "frame": value.frame,
                "child_frame": value.child_frame,
                "timestamp": {
                    "secs": int(value.timestamp.sec),
                    "nsecs": int(value.timestamp.nanosec),
                },
            },
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> StampedTransform:
        """
        Decode plain dictionary data to a ``StampedTransform``.
        """
        data = payload.payload
        result = StampedTransform()
        result.rotation = data["rotation"]
        result.translation = data["translation"]
        result.frame = data["frame"]
        result.child_frame = data["child_frame"]
        result.timestamp.sec = int(data["timestamp"]["secs"])
        result.timestamp.nanosec = int(data["timestamp"]["nsecs"])
        return result


class HomogeneousTransformationMatrixCodec(ViewCodec):
    """
    Codec for SemDT homogeneous transformation matrices.
    """

    serializer_id: str = "semdt_homogeneous_transform_v1"

    def can_encode(self, value: Any) -> bool:
        """
        Check whether the value is a homogeneous transformation matrix.
        """
        return isinstance(value, HomogeneousTransformationMatrix)

    def encode(self, value: HomogeneousTransformationMatrix) -> ViewPayload:
        """
        Encode a transformation matrix with its native JSON representation.
        """
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=value.to_json(),
            type_name=_full_type_name(value),
            metadata={
                "reference_frame_name": (
                    str(value.reference_frame.name)
                    if value.reference_frame is not None
                    else None
                ),
                "child_frame_name": (
                    str(value.child_frame.name)
                    if value.child_frame is not None
                    else None
                ),
            },
        )

    def decode(self, payload: ViewPayload) -> HomogeneousTransformationMatrix:
        """
        Decode matrix payload data using the active world entity tracker.
        """
        tracker = world.get_world_entity_tracker()
        kwargs = tracker.create_kwargs() if tracker is not None else {}
        return HomogeneousTransformationMatrix.from_json(payload.payload, **kwargs)


# %% Recorded camera observations
@dataclass
class RecordedCameraObservation:
    """
    Image calibration and optical pose linked to a recorded camera.
    """

    camera_id: str
    """
    Identifier of the semantic camera in the recorded world.
    """

    effective_camera_model: PinholeCameraModel
    """
    Projection model matching the recorded image.
    """

    world_T_camera: HomogeneousTransformationMatrix | None
    """
    Sampled optical-frame pose in world coordinates.
    """

    timestamp_nanoseconds: int
    """
    Frame acquisition time in nanoseconds since the Unix epoch.
    """


@dataclass
class CameraObservationCodec(ViewCodec):
    """
    Persist optical image observations linked to recorded cameras.
    """

    @classproperty
    def serializer_id(cls: type[CameraObservationCodec]) -> str:
        """
        Identifier of the recorded camera observation format.

        :return: Serialization identifier used by the codec registry.
        """
        return "robokudo_camera_observation_v3"

    def can_encode(self, value: Any) -> bool:
        return isinstance(value, CameraObservation)

    def encode(self, value: CameraObservation) -> ViewPayload:
        """
        Record the optical pose and image calibration linked to the semantic camera.

        :param value: Camera observation to persist.
        :return: Serialized observation linked to its semantic camera.
        """
        recorded = RecordedCameraObservation(
            camera_id=str(value.camera.id),
            effective_camera_model=value.effective_camera_model,
            world_T_camera=value.world_T_camera,
            timestamp_nanoseconds=value.timestamp_nanoseconds,
        )
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=DataclassJSONSerializer.to_json(recorded),
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> CameraObservation:
        return self.decode_with_tracker(payload, world.get_world_entity_tracker())

    def decode_with_tracker(
        self, payload: ViewPayload, tracker: WorldEntityWithIDKwargsTracker
    ) -> CameraObservation:
        """
        Restore the observation against a recorded world's entity tracker.

        :param payload: Serialized optical image observation.
        :param tracker: Entity tracker resolving the recorded camera and pose frames.
        :return: Observation referencing the tracked semantic camera.
        :raises InvalidCameraObservation: If the recorded camera or model is invalid.
        """
        recorded = DataclassJSONSerializer.from_json(
            payload.payload, RecordedCameraObservation, **tracker.create_kwargs()
        )
        camera = tracker.get(UUID(recorded.camera_id))
        if not isinstance(camera, Camera):
            raise InvalidCameraObservation(reason="recorded entity is not a camera")
        if not isinstance(recorded.effective_camera_model, PinholeCameraModel):
            raise InvalidCameraObservation(
                reason="the payload does not contain a pinhole camera model"
            )
        return CameraObservation(
            camera=camera,
            effective_camera_model=recorded.effective_camera_model,
            world_T_camera=recorded.world_T_camera,
            timestamp_nanoseconds=recorded.timestamp_nanoseconds,
        )


class KrroodCodec(ViewCodec):
    """
    Codec that delegates serialization to KRROOD.
    """

    serializer_id: str = "krrood_json_v1"

    def can_encode(self, value: Any) -> bool:
        """
        Probe whether KRROOD can serialize the given value.
        """
        try:
            krrood_to_json(value)
            return True
        except Exception:
            return False

    def encode(self, value: Any) -> ViewPayload:
        """
        Encode a value using KRROOD JSON serialization.
        """
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=krrood_to_json(value),
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> Any:
        """
        Decode a value using KRROOD JSON deserialization.
        """
        return krrood_from_json(payload.payload)


@dataclass
class CASViewCodecRegistry:
    """
    Registry that routes CAS views to matching codecs.
    """

    codecs: list[ViewCodec] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Initialize a default codec order when no codecs are provided.
        """
        if self.codecs:
            return
        self.codecs.extend(
            [
                TupleCodec(),
                BytesCodec(),
                NumpyCodec(),
                Open3DPointCloudCodec(),
                StampedTransformCodec(),
                CameraObservationCodec(),
                HomogeneousTransformationMatrixCodec(),
                RosMessageCodec(),
                KrroodCodec(),
                JsonLikeCodec(),
            ]
        )

    def _find_encoder(self, value: Any) -> ViewCodec | None:
        """
        Return the first registered codec that can encode ``value``.
        """
        for codec in self.codecs:
            if codec.can_encode(value):
                return codec
        return None

    def _find_decoder(self, serializer_id: str) -> ViewCodec:
        """
        Return the codec registered for a serializer identifier.
        """
        for codec in self.codecs:
            if codec.serializer_id == serializer_id:
                return codec
        raise UnknownViewSerializer(serializer_id=serializer_id)

    def encode_view(
        self, view_name: str, value: Any, strict: bool = True
    ) -> dict[str, Any] | None:
        """
        Encode a single CAS view into a storage document.
        """
        codec = self._find_encoder(value)
        if codec is None:
            if strict:
                raise CASViewCodecMissing(view_name=view_name, value_type=type(value))
            return None
        return codec.encode(value).to_document(view_name=view_name)

    def decode_view(self, document: dict[str, Any]) -> tuple[str, Any]:
        """
        Decode a single storage document into one CAS view.
        """
        view_name = document["view_name"]
        payload = ViewPayload.from_document(document)
        codec = self._find_decoder(payload.serializer_id)
        return view_name, codec.decode(payload)

    def encode_views(
        self, views: dict[str, Any], strict: bool = True
    ) -> list[dict[str, Any]]:
        """
        Encode multiple CAS views into storage documents.
        """
        documents: list[dict[str, Any]] = []
        for view_name, value in views.items():
            document = self.encode_view(view_name, value, strict=strict)
            if document is not None:
                documents.append(document)
        return documents

    def decode_views(self, documents: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """
        Decode storage documents into CAS view values.
        """
        result: dict[str, Any] = {}
        for document in documents:
            view_name, view_value = self.decode_view(document)
            result[view_name] = view_value
        return result
