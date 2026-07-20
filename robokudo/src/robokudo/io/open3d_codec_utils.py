"""
Shared Open3D serialization helpers for CAS codecs.
"""

from __future__ import annotations

import base64
import tempfile
from typing_extensions import Any

import open3d as o3d


def is_open3d_point_cloud(value: Any) -> bool:
    """
    Return ``True`` when ``value`` is an Open3D point cloud.
    """
    return isinstance(value, o3d.geometry.PointCloud)


def encode_open3d_point_cloud_to_base64_pcd(point_cloud: Any) -> str:
    """
    Encode an Open3D point cloud to a base64-encoded PCD payload.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".pcd")
    o3d.io.write_point_cloud(temp_file.name, point_cloud)
    temp_file.seek(0)
    payload = base64.b64encode(temp_file.read()).decode("ascii")
    temp_file.close()
    return payload


def decode_open3d_point_cloud_from_base64_pcd(payload: str) -> Any:
    """
    Decode a base64-encoded PCD payload into an Open3D point cloud.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".pcd")
    temp_file.write(base64.b64decode(payload.encode("ascii")))
    temp_file.flush()
    point_cloud = o3d.io.read_point_cloud(temp_file.name)
    temp_file.close()
    return point_cloud
