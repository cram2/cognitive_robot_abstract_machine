"""Codec registry and serializers for CAS view persistence."""

from __future__ import annotations

import array
import base64
import importlib
import io
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Any, Dict, Iterable, List, Optional
from robokudo import world
from robokudo.io.cas_annotation_codecs import krrood_to_json, krrood_from_json
from robokudo.io.open3d_codec_utils import (
    o3d,
    is_open3d_point_cloud,
    encode_open3d_point_cloud_to_base64_pcd,
    decode_open3d_point_cloud_from_base64_pcd,
)
from robokudo.types.tf import StampedTransform
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


def _recursive_convert(value: Any) -> Any:
    """Recursively convert non-JSON-friendly array-like values into plain containers."""
    if isinstance(value, array.array):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_recursive_convert(item) for item in value]
    if isinstance(value, dict):
        return {k: _recursive_convert(v) for k, v in value.items()}
    return value


def _ros_message_to_dict(msg: Any) -> Dict[str, Any]:
    """Convert a ROS message object into plain dictionary data."""
    result: Dict[str, Any] = {}
    for field_name, _ in msg.get_fields_and_field_types().items():
        value = getattr(msg, field_name)
        if hasattr(value, "get_fields_and_field_types"):
            result[field_name] = _ros_message_to_dict(value)
        elif isinstance(value, list):
            result[field_name] = [_recursive_convert(item) for item in value]
        else:
            result[field_name] = _recursive_convert(value)
    return result


def _dict_to_ros_message(message_type: type[Any], data_dict: Dict[str, Any]) -> Any:
    """Populate a ROS message instance from plain dictionary data."""
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
    """Return ``True`` when a value is a JSON primitive."""
    return value is None or isinstance(value, (bool, int, float, str))


def _is_json_like(value: Any) -> bool:
    """Return ``True`` when a value can be represented as plain JSON data."""
    if _is_json_primitive(value):
        return True
    if isinstance(value, list):
        return all(_is_json_like(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_like(v) for k, v in value.items())
    return False


def _full_type_name(value: Any) -> str:
    """Return the fully qualified type name for a value."""
    clazz = value.__class__
    return f"{clazz.__module__}.{clazz.__name__}"


def _load_type(type_name: str) -> type[Any]:
    """Load a Python type from its fully qualified name."""
    module_name, class_name = type_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _is_open3d_pinhole_camera_intrinsic(value: Any) -> bool:
    """Check whether a value is an Open3D pinhole camera intrinsic object."""
    value_type = value.__class__
    return (
        value_type.__name__ == "PinholeCameraIntrinsic"
        and value_type.__module__.startswith("open3d.")
        and hasattr(value, "intrinsic_matrix")
        and hasattr(value, "width")
        and hasattr(value, "height")
    )


@dataclass
class ViewPayload:
    """Persistence envelope for a single CAS view."""

    serializer_id: str
    payload: Any
    type_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_document(self, view_name: str) -> Dict[str, Any]:
        """Build a storage document for one encoded view."""
        return {
            "view_name": view_name,
            "serializer_id": self.serializer_id,
            "type_name": self.type_name,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_document(document: Dict[str, Any]) -> "ViewPayload":
        """Create a payload object from a storage document."""
        return ViewPayload(
            serializer_id=document["serializer_id"],
            payload=document["payload"],
            type_name=document["type_name"],
            metadata=document.get("metadata", {}),
        )


class ViewCodec:
    """Interface for encoding and decoding CAS view values."""

    serializer_id: str

    def can_encode(self, value: Any) -> bool:
        """Return ``True`` when this codec can encode ``value``."""
        raise NotImplementedError

    def encode(self, value: Any) -> ViewPayload:
        """Encode a runtime value into a :class:`ViewPayload`."""
        raise NotImplementedError

    def decode(self, payload: ViewPayload) -> Any:
        """Decode a persisted payload back into a runtime value."""
        raise NotImplementedError


class JsonLikeCodec(ViewCodec):
    """Codec for plain JSON-like Python values."""

    serializer_id: str = "json_like_v1"

    def can_encode(self, value: Any) -> bool:
        """Check whether the value is already JSON-compatible."""
        return _is_json_like(value)

    def encode(self, value: Any) -> ViewPayload:
        """Wrap a JSON-like value without further transformation."""
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=value,
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> Any:
        """Return the raw payload value."""
        return payload.payload


class TupleCodec(ViewCodec):
    """Codec for tuple values."""

    serializer_id: str = "tuple_v1"

    def can_encode(self, value: Any) -> bool:
        """Check whether the value is a tuple."""
        return isinstance(value, tuple)

    def encode(self, value: tuple[Any, ...]) -> ViewPayload:
        """Encode a tuple as a list payload."""
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=list(value),
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> tuple[Any, ...]:
        """Decode list payload data back to a tuple."""
        return tuple(payload.payload)


class BytesCodec(ViewCodec):
    """Codec for raw bytes data."""

    serializer_id: str = "bytes_v1"

    def can_encode(self, value: Any) -> bool:
        """Check whether the value is ``bytes`` or ``bytearray``."""
        return isinstance(value, (bytes, bytearray))

    def encode(self, value: bytes | bytearray) -> ViewPayload:
        """Encode bytes as a base64 text payload."""
        as_bytes = bytes(value)
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=base64.b64encode(as_bytes).decode("ascii"),
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> bytes:
        """Decode base64 text payload into raw bytes."""
        return base64.b64decode(payload.payload.encode("ascii"))


class NumpyCodec(ViewCodec):
    """Codec for NumPy arrays."""

    serializer_id: str = "numpy_npy_base64_v1"

    def can_encode(self, value: Any) -> bool:
        """Check whether the value is a NumPy array."""
        return isinstance(value, np.ndarray)

    def encode(self, value: np.ndarray) -> ViewPayload:
        """Encode a NumPy array as base64 encoded ``.npy`` data."""
        mem_file = io.BytesIO()
        np.save(mem_file, value)
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=base64.b64encode(mem_file.getvalue()).decode("ascii"),
            type_name=_full_type_name(value),
            metadata={"dtype": str(value.dtype), "shape": list(value.shape)},
        )

    def decode(self, payload: ViewPayload) -> np.ndarray:
        """Decode base64 encoded ``.npy`` payload data."""
        mem_file = io.BytesIO(base64.b64decode(payload.payload.encode("ascii")))
        mem_file.seek(0)
        return np.load(mem_file)


class RosMessageCodec(ViewCodec):
    """Codec for ROS message instances."""

    serializer_id: str = "ros_message_v1"

    def can_encode(self, value: Any) -> bool:
        """Check whether the value looks like a ROS message."""
        return hasattr(value, "get_fields_and_field_types")

    def encode(self, value: Any) -> ViewPayload:
        """Encode a ROS message to dictionary payload data."""
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=_ros_message_to_dict(value),
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> Any:
        """Decode dictionary payload data to a ROS message object."""
        message_type = _load_type(payload.type_name)
        return _dict_to_ros_message(message_type, payload.payload)


class Open3DPointCloudCodec(ViewCodec):
    """Codec for Open3D point cloud values."""

    serializer_id: str = "open3d_pointcloud_pcd_base64_v1"

    def can_encode(self, value: Any) -> bool:
        """Check whether the value is an Open3D point cloud."""
        return is_open3d_point_cloud(value)

    def encode(self, value: Any) -> ViewPayload:
        """Encode a point cloud as base64-encoded PCD data."""
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=encode_open3d_point_cloud_to_base64_pcd(value),
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> Any:
        """Decode base64-encoded PCD data to an Open3D point cloud."""
        return decode_open3d_point_cloud_from_base64_pcd(payload.payload)


class Open3DPinholeCameraIntrinsicCodec(ViewCodec):
    """Codec for Open3D pinhole camera intrinsic values."""

    serializer_id: str = "open3d_pinhole_camera_intrinsic_v1"

    def can_encode(self, value: Any) -> bool:
        """Check whether the value is an Open3D pinhole camera intrinsic."""
        return _is_open3d_pinhole_camera_intrinsic(value)

    def encode(self, value: Any) -> ViewPayload:
        """Encode an Open3D pinhole intrinsic into JSON-compatible payload data."""
        intrinsic_matrix = np.asarray(value.intrinsic_matrix, dtype=float)
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError(
                "Open3D pinhole camera intrinsic matrix must have shape (3, 3)."
            )
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload={
                "width": int(value.width),
                "height": int(value.height),
                "intrinsic_matrix": intrinsic_matrix.tolist(),
            },
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> Any:
        """Decode payload data to an Open3D pinhole intrinsic object."""
        if o3d is None:  # pragma: no cover - guarded by optional dependency
            raise RuntimeError(
                "Open3D is not available but Open3D camera intrinsic payload was provided."
            )

        payload_data: Dict[str, Any] = payload.payload
        width = int(payload_data["width"])
        height = int(payload_data["height"])
        intrinsic_matrix = np.asarray(payload_data["intrinsic_matrix"], dtype=float)
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError(
                "Serialized Open3D camera intrinsic matrix must have shape (3, 3)."
            )

        fx = float(intrinsic_matrix[0, 0])
        fy = float(intrinsic_matrix[1, 1])
        cx = float(intrinsic_matrix[0, 2])
        cy = float(intrinsic_matrix[1, 2])

        try:
            camera_intrinsic_type = _load_type(payload.type_name)
        except Exception:  # pragma: no cover - fallback for cpu/cuda module differences
            camera_intrinsic_type = o3d.camera.PinholeCameraIntrinsic

        try:
            return camera_intrinsic_type(width, height, fx, fy, cx, cy)
        except Exception:
            camera_intrinsic = camera_intrinsic_type()
            camera_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
            return camera_intrinsic


class StampedTransformCodec(ViewCodec):
    """Codec for RoboKudo ``StampedTransform`` values."""

    serializer_id: str = "robokudo_stamped_transform_v1"

    def can_encode(self, value: Any) -> bool:
        """Check whether the value is a ``StampedTransform``."""
        return isinstance(value, StampedTransform)

    def encode(self, value: StampedTransform) -> ViewPayload:
        """Encode a ``StampedTransform`` into plain dictionary data."""
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
        """Decode plain dictionary data to a ``StampedTransform``."""
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
    """Codec for SemDT homogeneous transformation matrices."""

    serializer_id: str = "semdt_homogeneous_transform_v1"

    def can_encode(self, value: Any) -> bool:
        """Check whether the value is a homogeneous transformation matrix."""
        return isinstance(value, HomogeneousTransformationMatrix)

    def encode(self, value: HomogeneousTransformationMatrix) -> ViewPayload:
        """Encode a transformation matrix with its native JSON representation."""
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=value.to_json(),
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> HomogeneousTransformationMatrix:
        """Decode matrix payload data using the active world entity tracker."""
        tracker = world.get_world_entity_tracker()
        kwargs = tracker.create_kwargs() if tracker is not None else {}
        return HomogeneousTransformationMatrix.from_json(payload.payload, **kwargs)


class KrroodCodec(ViewCodec):
    """Codec that delegates serialization to KRROOD."""

    serializer_id: str = "krrood_json_v1"

    def can_encode(self, value: Any) -> bool:
        """Probe whether KRROOD can serialize the given value."""
        try:
            krrood_to_json(value)
            return True
        except Exception:
            return False

    def encode(self, value: Any) -> ViewPayload:
        """Encode a value using KRROOD JSON serialization."""
        return ViewPayload(
            serializer_id=self.serializer_id,
            payload=krrood_to_json(value),
            type_name=_full_type_name(value),
        )

    def decode(self, payload: ViewPayload) -> Any:
        """Decode a value using KRROOD JSON deserialization."""
        return krrood_from_json(payload.payload)


@dataclass
class CASViewCodecRegistry:
    """Registry that routes CAS views to matching codecs."""

    codecs: List[ViewCodec] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize a default codec order when no codecs are provided."""
        if self.codecs:
            return
        self.codecs.extend(
            [
                TupleCodec(),
                BytesCodec(),
                NumpyCodec(),
                Open3DPointCloudCodec(),
                Open3DPinholeCameraIntrinsicCodec(),
                StampedTransformCodec(),
                HomogeneousTransformationMatrixCodec(),
                RosMessageCodec(),
                KrroodCodec(),
                JsonLikeCodec(),
            ]
        )

    def _find_encoder(self, value: Any) -> Optional[ViewCodec]:
        """Return the first registered codec that can encode ``value``."""
        for codec in self.codecs:
            if codec.can_encode(value):
                return codec
        return None

    def _find_decoder(self, serializer_id: str) -> ViewCodec:
        """Return the codec registered for a serializer identifier."""
        for codec in self.codecs:
            if codec.serializer_id == serializer_id:
                return codec
        raise ValueError(f"Unknown CAS view serializer '{serializer_id}'.")

    def encode_view(
        self, view_name: str, value: Any, strict: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Encode a single CAS view into a storage document."""
        codec = self._find_encoder(value)
        if codec is None:
            if strict:
                raise TypeError(
                    f"No codec registered for CAS view '{view_name}' with type '{type(value)}'."
                )
            return None
        return codec.encode(value).to_document(view_name=view_name)

    def decode_view(self, document: Dict[str, Any]) -> tuple[str, Any]:
        """Decode a single storage document into one CAS view."""
        view_name = document["view_name"]
        payload = ViewPayload.from_document(document)
        codec = self._find_decoder(payload.serializer_id)
        return view_name, codec.decode(payload)

    def encode_views(
        self, views: Dict[str, Any], strict: bool = True
    ) -> List[Dict[str, Any]]:
        """Encode multiple CAS views into storage documents."""
        documents: List[Dict[str, Any]] = []
        for view_name, value in views.items():
            document = self.encode_view(view_name, value, strict=strict)
            if document is not None:
                documents.append(document)
        return documents

    def decode_views(self, documents: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Decode storage documents into CAS view values."""
        result: Dict[str, Any] = {}
        for document in documents:
            view_name, view_value = self.decode_view(document)
            result[view_name] = view_value
        return result
