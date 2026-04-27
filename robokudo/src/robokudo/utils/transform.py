"""Transformation utilities for RoboKudo.

This module provides conversion methods for common transformation calculations.
It operates on ROS types and NumPy data structures, independent of RK Annotation Types.
For RK Annotation type conversions, see :mod:`robokudo.utils.type_conversion`.

The module supports:

* Converting between ROS poses and transformation matrices
* Working with rotation matrices, quaternions and Euler angles
* Handling coordinate frame transformations
* Constructing transformations from geometric primitives

Dependencies:

* NumPy for numerical operations
* TF for transformation calculations
* ROS geometry_msgs for pose data types
"""

from __future__ import annotations

import math
from typing_extensions import Iterable

import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation
from typing_extensions import TYPE_CHECKING, Tuple, List, Union

if TYPE_CHECKING:
    import numpy.typing as npt


def get_pose_from_transform_matrix(transform: npt.NDArray) -> Pose:
    """Get a ROS geometry_msgs Pose from a 4x4 transformation matrix.

    :param transform: 4x4 transformation matrix
    :return: Equivalent pose
    """
    pose = Pose()

    translation = transform[:3, 3]
    rotation = transform[:3, :3]
    quaternion = Rotation.from_matrix(rotation).as_quat(True)

    pose.position.x = translation[0]
    pose.position.y = translation[1]
    pose.position.z = translation[2]

    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]

    return pose


def get_transform_matrix_from_pose(pose: Pose) -> npt.NDArray:
    """Get a 4x4 transformation matrix from a ROS geometry_msgs Pose.

    :param pose: Equivalent pose
    :return: 4x4 transformation matrix
    """
    orientation = np.asarray(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
    )

    translation = np.asarray(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ]
    )

    r = Rotation.from_quat(orientation).as_matrix()

    pose_t = get_transform_matrix(r, translation)

    return pose_t


def get_transform_matrix(rotation: Iterable, translation: Iterable) -> npt.NDArray:
    """Create a 4x4 transformation matrix from rotation and translation.

    :param rotation: 3x3 rotation matrix
    :param translation: 3D translation vector
    :return: 4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def get_transform_matrix_from_translation(translation: npt.NDArray) -> npt.NDArray:
    """Create a 4x4 transformation matrix from translation only.

    :param translation: 3D translation vector
    :return: 4x4 homogeneous transformation matrix with identity rotation
    """
    T = np.eye(4)
    T[:3, 3] = translation
    return T


def quaternion_about_axis(angle: float, axis: Tuple[int, int, int]) -> npt.NDArray:
    """Create a 4x4 transformation matrix for rotation around an axis.

    :param angle: Rotation angle in radians
    :param axis: 3D vector defining rotation axis
    :return: 4x4 homogeneous transformation matrix
    """
    axis_normalized = np.array(axis) / np.linalg.norm(axis)
    rotation_vector = angle * axis_normalized
    return Rotation.from_rotvec(rotation_vector).as_quat(True)


def get_transform_matrix_for_rotation_around_axis(
    angle: float, axis: Tuple[int, int, int]
) -> npt.NDArray:
    """Create 4x4 transformation matrix for rotation around arbitrary axis.

    :param angle: Rotation angle in radians
    :param axis: 3D vector defining rotation axis
    :return: 4x4 homogeneous transformation matrix
    """
    quat = quaternion_about_axis(angle, axis)
    return get_transform_matrix_from_q(quat, np.array([0, 0, 0]))


def get_transform_matrix_from_q(
    quaternion: Union[npt.NDArray, List[float], Tuple[float, float, float, float]],
    translation: Union[npt.NDArray, List[float], Tuple[float, float, float]],
) -> npt.NDArray:
    """Create a 4x4 transformation matrix from quaternion and translation.

    :param quaternion: Rotation quaternion [x,y,z,w]
    :param translation: 3D translation vector
    :return: 4x4 homogeneous transformation matrix
    """
    r = Rotation.from_quat(quaternion)

    T = np.eye(4)  # 4x4 rotation + translation matrix
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = translation
    return T


def get_rotation_matrix_from_euler_angles(x: float, y: float, z: float) -> npt.NDArray:
    """Create a 3x3 rotation matrix from Euler angles.

    :param x: Rotation around X axis in radians
    :param y: Rotation around Y axis in radians
    :param z: Rotation around Z axis in radians
    :return: 3x3 rotation matrix

    .. note::
       Applies rotations in X,Y,Z order.
    """
    return Rotation.from_euler("xyz", [x, y, z]).as_matrix()


def get_rotation_matrix_from_q(quaternion: npt.NDArray) -> npt.NDArray:
    """Create a 3x3 rotation matrix from quaternion.

    :param quaternion: Rotation quaternion [x,y,z,w]
    :return: 3x3 rotation matrix
    """
    return Rotation.from_quat(quaternion).as_matrix()


def get_quaternion_from_rotation_matrix(rotation_matrix: npt.NDArray) -> npt.NDArray:
    """Convert a 3x3 rotation matrix to quaternion.

    :param rotation_matrix: 3x3 rotation matrix
    :return: Rotation quaternion [x,y,z,w]
    """

    # the input of quaternion_from_matrix is 3x3
    T = np.eye(3)
    T[:3, :3] = rotation_matrix

    return Rotation.from_matrix(T).as_quat(True)


def get_quaternion_from_transform_matrix(transform_matrix: npt.NDArray) -> npt.NDArray:
    """Extract quaternion from 4x4 transformation matrix.

    :param transform_matrix: 4x4 homogeneous transformation matrix
    :return: Rotation quaternion [x,y,z,w]
    """
    return Rotation.from_matrix(transform_matrix[:3, :3]).as_quat(True)


def get_translation_from_transform_matrix(transform_matrix: npt.NDArray) -> npt.NDArray:
    """Extract translation from 4x4 transformation matrix.

    :param transform_matrix: 4x4 homogeneous transformation matrix
    :return: 3D translation vector [x,y,z]
    """
    return transform_matrix[:3, 3]


def get_rotation_from_transform_matrix(transform_matrix: npt.NDArray) -> npt.NDArray:
    """Extract rotation matrix from 4x4 transformation matrix.

    :param transform_matrix: 4x4 homogeneous transformation matrix
    :return: 3x3 rotation matrix
    """
    return transform_matrix[:3, :3]


def get_transform_from_plane_equation(plane_equation: npt.NDArray) -> npt.NDArray:
    """Create transformation matrix from plane equation.

    Creates a transformation that aligns with the plane defined by ax + by + cz + d = 0.
    The transformation's Z axis aligns with the plane normal.

    :param plane_equation: Plane coefficients [a,b,c,d]
    :return: 4x4 homogeneous transformation matrix
    """

    # Build transform from plane equation
    # 1. Build translation
    [a, b, c, d] = plane_equation
    normal = np.array([a, b, c])
    normal_length = np.linalg.norm(normal)
    unit_normal = normal / normal_length
    plane_translation = (-d / normal_length) * unit_normal

    # 2. Build rotation
    un_x = unit_normal[0]
    un_y = unit_normal[1]
    un_z = unit_normal[2]
    un_xy_sqrt = math.sqrt(un_x**2 + un_y**2)

    if un_xy_sqrt < 1e-6:
        plane_rotation = np.eye(3)
    else:
        plane_rotation = np.array(
            [
                [un_y / un_xy_sqrt, -un_x / un_xy_sqrt, 0],
                [(un_x * un_z) / un_xy_sqrt, (un_y * un_z) / un_xy_sqrt, -un_xy_sqrt],
                [un_x, un_y, un_z],
            ]
        )

    transform = get_transform_matrix(np.linalg.inv(plane_rotation), plane_translation)

    return transform


def construct_rotation_matrix(
    pose_orientation: npt.NDArray, axis_order: Tuple[int, int, int]
) -> npt.NDArray:
    """Construct rotation matrix with reordered axes.

    Creates a rotation matrix by reordering the axes from an input orientation.
    Useful for switching coordinate axes, e.g. exchanging Z with X axis.

    :param pose_orientation: Input 3x3 rotation matrix
    :param axis_order: Tuple [a,b,c] specifying axis order (0=X, 1=Y, 2=Z)
    :return: Reordered 3x3 rotation matrix
    :raises ValueError: If axis order contains invalid elements (not 0, 1, 2) or does not have exactly 3 elements.

    .. note::
       The third axis is computed as cross product of first two to ensure orthogonality.
    """
    if len(axis_order) != 3:
        raise ValueError("Axis order must contain exactly 3 elements.")
    if not set(axis_order).issubset({0, 1, 2}):
        raise ValueError("Axis order must contain only elements 0, 1, 2.")

    # Initialize a matrix filled with zeros
    new_rotation = np.zeros((3, 3))

    # Assign the specified axes to the new rotation matrix
    for i, axis_index in enumerate(axis_order):
        if i < 2:  # For the first two axes, we directly use the pose orientation
            new_rotation[:, i] = pose_orientation[:, axis_index]
        else:  # For the third axis, we compute the cross product
            cross_product_axis = np.cross(
                pose_orientation[:, axis_order[0]], pose_orientation[:, axis_order[1]]
            )
            new_rotation[:, i] = cross_product_axis

    return new_rotation


def get_rotation_matrix_from_direction_vector(
    direction: npt.NDArray, up_hint: npt.NDArray = np.array([0, 0, 1])
) -> npt.NDArray:
    """
    Construct a rotation matrix whose x-axis points along the given direction.
    The y and z axes are constructed using an up vector hint.

    :param direction: A 3D vector representing the direction.
    :param up_hint: A 3D vector hinting the up direction, default is [0, 0, 1].
    :return: A 3x3 rotation matrix.
    """
    # Normalize the direction vector
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Direction vector cannot have a length of 0.")
    x_axis = direction / length

    # Avoid degeneracy if the direction and up vector are nearly aligned
    if abs(np.dot(x_axis, up_hint)) > 0.99:
        up_hint = np.array([0, 1, 0])

    # Calculate the z-axis as the cross product of the x-axis and the up hint
    z_axis = np.cross(x_axis, up_hint)
    z_axis /= np.linalg.norm(z_axis)

    # Calculate the y-axis as the cross product of the z-axis and the x-axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Compose 3x3 rotation matrix
    rot = np.column_stack((x_axis, y_axis, z_axis))
    return rot
