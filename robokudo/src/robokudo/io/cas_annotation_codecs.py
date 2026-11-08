"""
KRROOD-backed codecs for CAS annotation serialization.
"""

from __future__ import annotations

from typing_extensions import Any

import numpy as np

from krrood.adapters.json_serializer import (
    ExternalClassJSONSerializer,
    from_json,
    to_json,
)
from krrood.adapters.json_field import JSONField
from krrood.utils import get_full_class_name
from robokudo.io.open3d_codec_utils import (
    o3d,
    encode_open3d_point_cloud_to_base64_pcd,
    decode_open3d_point_cloud_from_base64_pcd,
)

try:
    # Side effect import: registers ROS message serializers from SemDT.
    import semantic_digital_twin.adapters.ros  # noqa: F401
except (
    ImportError
):  # pragma: no cover - only needed in environments with ROS message fields
    pass


class Open3DPointCloudJSONSerializer(
    ExternalClassJSONSerializer[o3d.geometry.PointCloud]
):
    """
    Serialize Open3D point clouds using base64-encoded PCD payloads.
    """

    @classmethod
    def to_json(cls, instance: o3d.geometry.PointCloud, **kwargs) -> dict[str, Any]:
        """
        Convert an Open3D point cloud into a JSON-compatible payload.
        """
        return {
            JSONField.TYPE: get_full_class_name(type(instance)),
            "payload": encode_open3d_point_cloud_to_base64_pcd(instance),
        }

    @classmethod
    def from_json(
        cls,
        data: dict[str, Any],
        clazz: type[o3d.geometry.PointCloud],
        **kwargs: Any,
    ) -> o3d.geometry.PointCloud:
        """
        Restore an Open3D point cloud from a JSON payload.
        """
        return decode_open3d_point_cloud_from_base64_pcd(data["payload"])


class NumpyScalarJSONSerializer(ExternalClassJSONSerializer[np.generic]):
    """
    Serialize NumPy scalar values with dtype-preserving metadata.
    """

    @classmethod
    def to_json(cls, instance: np.generic, **kwargs) -> dict[str, Any]:
        """
        Convert a NumPy scalar value to JSON-compatible data.
        """
        return {
            JSONField.TYPE: get_full_class_name(type(instance)),
            "dtype": str(instance.dtype),
            "value": instance.item(),
        }

    @classmethod
    def from_json(
        cls,
        data: dict[str, Any],
        clazz: type[np.generic],
        **kwargs: Any,
    ) -> Any:
        """
        Recreate a NumPy scalar value from serialized data.
        """
        dtype = np.dtype(data["dtype"])
        return np.array(data["value"], dtype=dtype).item()


def serialize_annotations(annotations: list[Any]) -> list[dict[str, Any]]:
    """
    Serialize annotation objects with the KRROOD serializer.
    """
    return [to_json(annotation) for annotation in annotations]


def deserialize_annotations(data: Any, **kwargs: Any) -> list[Any]:
    """
    Deserialize annotation payloads with optional constructor keyword arguments.
    """
    if data is None:
        return []

    # Avoid KRROOD's top-level list helper because it currently drops kwargs.
    if isinstance(data, list):
        return [from_json(item, **kwargs) for item in data]

    return [from_json(data, **kwargs)]


def krrood_to_json(value: Any) -> Any:
    """
    Serialize an arbitrary value with KRROOD.
    """
    return to_json(value)


def krrood_from_json(value: Any, **kwargs: Any) -> Any:
    """
    Deserialize an arbitrary value with KRROOD.
    """
    return from_json(value, **kwargs)
