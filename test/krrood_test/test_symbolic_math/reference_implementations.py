from typing_extensions import Tuple, Union, List

import numpy as np

from semantic_digital_twin.datastructures.types import AnyMatrix4x4, NpMatrix4x4
from semantic_digital_twin.spatial_types.derivatives import Derivatives


def quaternion_multiply(quaternion1: np.ndarray, quaternion0: np.ndarray) -> np.ndarray:
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float64,
    )


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array(
        (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
        dtype=np.float64,
    )


def qv_mult(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Transforms a vector by a quaternion
    :param quaternion: Quaternion
    :param vector: vector
    :return: transformed vector
    """
    q = quaternion
    v = [vector[0], vector[1], vector[2], 0]
    return quaternion_multiply(quaternion_multiply(q, v), quaternion_conjugate(q))[:-1]


def quaternion_from_axis_angle(
    axis: Union[List[float], Tuple[float, float, float], np.ndarray], angle: float
) -> np.ndarray:
    half_angle = angle / 2
    return np.array(
        [
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
            np.cos(half_angle),
        ],
        dtype=np.float64,
    )


_EPS = np.finfo(float).eps * 4.0


def rpy_from_matrix(rotation_matrix: NpMatrix4x4) -> Tuple[float, float, float]:
    """
    :param rotation_matrix: 4x4 Matrix
    :type rotation_matrix: Matrix
    :return: roll, pitch, yaw
    :rtype: (Union[float, Symbol], Union[float, Symbol], Union[float, Symbol])
    """
    i = 0
    j = 1
    k = 2

    cy = np.sqrt(
        rotation_matrix[i, i] * rotation_matrix[i, i]
        + rotation_matrix[j, i] * rotation_matrix[j, i]
    )
    if cy - _EPS > 0:
        roll = np.arctan2(rotation_matrix[k, j], rotation_matrix[k, k])
        pitch = np.arctan2(-rotation_matrix[k, i], cy)
        yaw = np.arctan2(rotation_matrix[j, i], rotation_matrix[i, i])
    else:
        roll = np.arctan2(-rotation_matrix[j, k], rotation_matrix[j, j])
        pitch = np.arctan2(-rotation_matrix[k, i], cy)
        yaw = 0
    return roll, pitch, yaw


def rpy_from_quaternion(
    qx: float, qy: float, qz: float, qw: float
) -> Tuple[float, float, float]:
    return rpy_from_matrix(rotation_matrix_from_quaternion(qx, qy, qz, qw))


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> NpMatrix4x4:
    """
    Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
    https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    :return: 4x4 Matrix
    """
    rx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(roll), -np.sin(roll), 0],
            [0, np.sin(roll), np.cos(roll), 0],
            [0, 0, 0, 1],
        ]
    )
    ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch), 0],
            [0, 1, 0, 0],
            [-np.sin(pitch), 0, np.cos(pitch), 0],
            [0, 0, 0, 1],
        ]
    )
    rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, 0],
            [np.sin(yaw), np.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return np.dot(np.dot(rz, ry), rx)


def rotation_matrix_from_quaternion(
    x: float, y: float, z: float, w: float
) -> NpMatrix4x4:
    """
    Unit quaternion to 4x4 rotation matrix according to:
    https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
    :return: 4x4 Matrix
    """
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    return np.array(
        [
            [w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0],
            [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x, 0],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def rotation_matrix_from_axis_angle(
    axis: Union[List[float], Tuple[float, float, float], np.ndarray], angle: float
) -> NpMatrix4x4:
    return rotation_matrix_from_quaternion(*quaternion_from_axis_angle(axis, angle))


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return quaternion_from_rotation_matrix(rotation_matrix_from_rpy(roll, pitch, yaw))


def quaternion_from_rotation_matrix(matrix: AnyMatrix4x4) -> np.ndarray:
    """
    :param matrix: 4x4 Matrix
    :return: array length 4
    """
    q = np.empty((4,), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / np.sqrt(t * M[3, 3])
    return q


def axis_angle_from_quaternion(
    x: float, y: float, z: float, w: float
) -> Tuple[np.ndarray, float]:
    l = np.linalg.norm(np.array([x, y, z, w]))
    x, y, z, w = x / l, y / l, z / l, w / l
    w2 = np.sqrt(1 - w**2)
    if w2 == 0:
        m = 1
    else:
        m = w2
    if w2 == 0:
        angle = 0
    else:
        angle = 2 * np.arccos(min(max(-1, w), 1))
    if w2 == 0:
        x = 0
        y = 0
        z = 1
    else:
        x = x / m
        y = y / m
        z = z / m
    return np.array([x, y, z]), angle


def axis_angle_from_rotation_matrix(m: NpMatrix4x4) -> Tuple[np.ndarray, float]:
    return axis_angle_from_quaternion(*quaternion_from_rotation_matrix(m))


def axis_angle_from_rpy(
    roll: float, pitch: float, yaw: float
) -> Tuple[np.ndarray, float]:
    return axis_angle_from_quaternion(*quaternion_from_rpy(roll, pitch, yaw))


def gauss(n: float) -> float:
    return (n**2 + n) / 2


def max_velocity_from_horizon_and_jerk(
    prediction_horizon: int,
    vel_limit: float,
    acc_limit: float,
    jerk_limit: float,
    sample_period: float,
    max_derivative: Derivatives,
):
    n2 = int((prediction_horizon) / 2)
    vel_jerk_limited = (gauss(n2) + gauss(n2 - 1)) * jerk_limit * sample_period**2
    vel_acc_limits = acc_limit * sample_period * prediction_horizon
    if max_derivative == Derivatives.jerk:
        return min(vel_limit, vel_acc_limits, vel_jerk_limited)
    if max_derivative == Derivatives.acceleration:
        return min(vel_limit, vel_acc_limits)
    return vel_limit


def limit(a: float, lower_limit: float, upper_limit: float) -> float:
    return max(lower_limit, min(upper_limit, a))


def angle_between_vector(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    :param v1: vector length 3
    :param v2: vector length 3
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def my_cross(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    :param v1: vector length 3 or 4
    :param v2: vector length 3 or 4
    :return: cross product vector length 3
    """
    return np.cross(v1[:-1], v2[:-1])


def point_to_single_caster_angle(px, py, caster_v, forward_velocity):
    max_angular_velocity = 1
    z = np.array([0, 0, 1, 0])
    x = np.array([1, 0, 0, 0])
    center_P_p = np.array([px, py, 0, 1])
    c_V_p = center_P_p - caster_v
    c_V_goal = my_cross(c_V_p, z)
    angle = angle_between_vector(x[:-1], c_V_goal)
    radius = np.linalg.norm(c_V_p)
    if radius > 0.01:
        circumference = 2 * np.pi * radius
        number_of_revolutions = forward_velocity / circumference
        angular_velocity = number_of_revolutions / (2 * np.pi)
        angular_velocity = min(
            max(angular_velocity, -max_angular_velocity), max_angular_velocity
        )
    else:
        angular_velocity = max_angular_velocity
    return angle, c_V_goal


def shortest_angular_distance(from_angle, to_angle):
    """Given 2 angles, this returns the shortest angular
    difference.  The inputs and ouputs are of course radians.

    The result would always be -pi <= result <= pi. Adding the result
    to "from" will always get you an equivelent angle to "to".
    """
    return normalize_angle(to_angle - from_angle)


def normalize_angle(angle):
    """Normalizes the angle to be -pi to +pi
    It takes and returns radians."""
    a = normalize_angle_positive(angle)
    if a > np.pi:
        a -= 2.0 * np.pi
    return a


def normalize_angle_positive(angle):
    """Normalizes the angle to be 0 to 2*pi
    It takes and returns radians."""
    return angle % (2.0 * np.pi)


def quaternion_slerp(q1, q2, t):
    """
    spherical linear interpolation that takes into account that q == -q
    :param q1: 4x1 Matrix
    :param q2: 4x1 Matrix
    :param t: float, 0-1
    :return: 4x1 Matrix; Return spherical linear interpolation between two quaternions.
    """
    cos_half_theta = q1.dot(q2)

    if cos_half_theta < 0:
        q2 = -q2
        cos_half_theta = -cos_half_theta

    if abs(cos_half_theta) > 1:
        return q1
    # enforce acos(x) with -1 < x < 1
    cos_half_theta = min(1, cos_half_theta)
    cos_half_theta = max(-1, cos_half_theta)

    half_theta = np.arccos(cos_half_theta)

    sin_half_theta = np.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratio_a = np.sin((1.0 - t) * half_theta) / sin_half_theta
    ratio_b = np.sin(t * half_theta) / sin_half_theta

    if 0.001 > abs(sin_half_theta):
        return 0.5 * q1 + 0.5 * q2
    return ratio_a * q1 + ratio_b * q2
