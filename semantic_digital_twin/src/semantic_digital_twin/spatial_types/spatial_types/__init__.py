"""
The spatial types, which live in :mod:`krrood_spatial_math.spatial_types`.

This module keeps the names importable here, and keeps JSON documents that name a
spatial type by this module readable.
"""

from krrood_spatial_math.spatial_types import (
    AxisAngle,
    GenericHomogeneousSpatialType,
    GenericRotatableSpatialType,
    GenericSpatialType,
    HomogeneousTransformationMatrix,
    Line3,
    OrthogonalProjection,
    Point,
    Point2,
    Point3,
    Pose,
    Pose2D,
    Quaternion,
    RotationMatrix,
    SpatialFrameKey,
    SpatialType,
    Vector3,
    rotation_matrix_to_quaternion,
)
