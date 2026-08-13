"""
SDT Region helpers for RoboKudo.

This module provides utilities to:
- Normalize region naming
- Compute region poses and OBBs
- Convert regions to annotations/markers
"""

from __future__ import annotations

from typing_extensions import Iterable, Optional, Tuple, Union

import numpy as np
import open3d as o3d

import robokudo.types.annotation
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Box
from semantic_digital_twin.world_description.world_entity import Region


def _box_obb_from_shape(
    box: Box, world_T_region: HomogeneousTransformationMatrix
) -> o3d.geometry.OrientedBoundingBox:
    """
    Build an OBB for a single SDT Box shape.
    """
    # Box origin is in the region frame; combine with region global pose
    box_origin: HomogeneousTransformationMatrix = box.origin

    # world_T_box = world_T_region * region_T_box
    world_T_box = world_T_region @ box_origin

    # Build OBB from center, rotation, scale
    center_np = world_T_box.to_position().to_np()
    if center_np.shape[0] == 4:
        center_np = center_np[:3]
    center = center_np.reshape(3, 1).astype(float)
    R = world_T_box.to_rotation_matrix().to_np().astype(float)
    if R.shape == (4, 4):
        R = R[:3, :3]
    extent = (
        np.array([box.scale.x, box.scale.y, box.scale.z], dtype=float)
        .reshape(3, 1)
        .astype(float)
    )
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    return obb


def region_obb(
    region: Region, world: Optional[World] = None
) -> o3d.geometry.OrientedBoundingBox:
    """
    Return an OBB for the region in world coordinates.

    Strategy:
    - If a single Box shape exists, build directly from its scale and pose.
    - Otherwise, fall back to combined mesh or bounding boxes.
    """
    # Get world_T_region as homogeneous matrix (global_pose is Pose in current SDT API)
    world_T_region = region.global_transform

    # Fast path: single Box
    if len(region.area) == 1 and isinstance(region.area[0], Box):
        return _box_obb_from_shape(region.area[0], world_T_region)

    # General path: approximate via combined mesh bounding box
    # This assumes region.area.combined_mesh is in region frame; transform to world
    mesh = region.area.combined_mesh
    if mesh is None:
        raise ValueError("Region has no shapes; cannot compute OBB")

    mesh = mesh.copy()
    mesh.apply_transform(world_T_region.to_np())

    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(mesh.vertices)
    )
    return obb


def region_obb_in_cam_coordinates(
    world: World,
    region: Region,
    world_T_cam: Union[np.ndarray, HomogeneousTransformationMatrix],
) -> o3d.geometry.OrientedBoundingBox:
    """
    Transform region OBB to camera frame using world_to_cam matrix.
    """
    obb = region_obb(region, world=world)
    if isinstance(world_T_cam, HomogeneousTransformationMatrix):
        R = world_T_cam.to_rotation_matrix().to_np()
        t = world_T_cam.to_position().to_np()
    else:
        R = world_T_cam[:3, :3]
        t = world_T_cam[:3, 3]
    obb.rotate(R, center=(0, 0, 0))
    obb.translate(t)
    return obb


def region_pose_annotation(region: Region) -> robokudo.types.annotation.PoseAnnotation:
    """
    Create a PoseAnnotation from the region's global pose.
    """
    pose = robokudo.types.annotation.PoseAnnotation()
    T = region.global_pose
    pose.translation = T.to_position().to_np().tolist()
    pose.rotation = T.to_quaternion().to_np().tolist()
    return pose


def region_marker_array(
    regions: Iterable[Region],
    highlighted: Optional[Iterable[str]] = None,
    highlight_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    default_color: Tuple[float, float, float] = (0.0, 0.0, 0.7),
):
    """
    Optional helper for visualization markers.
    This can replace BaseSemanticMap.publish_visualization_markers.
    """
    # Outline only; implement in actual migration if needed.
    raise NotImplementedError("Marker creation sketch placeholder")
