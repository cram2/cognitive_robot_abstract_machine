"""Implementation of helper functions and classes for internal usage only.

Functions:
_block -- wrap multiple statements into a single block.

Classes:
GeneratorList -- implementation of generator list wrappers.
"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from inspect import isgeneratorfunction
from typing import Union, Iterator

import numpy as np
from typing_extensions import (
    Tuple,
    Callable,
    List,
    Dict,
    TYPE_CHECKING,
    Sequence,
    Any,
    Iterable,
)

from semantic_digital_twin.collision_checking.collision_detector import ClosestPoints
from semantic_digital_twin.collision_checking.collision_rules import AllowSelfCollisions
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import (
    Pose,
    Quaternion,
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from pycram.tf_transformations import (
    quaternion_about_axis,
    quaternion_multiply,
    quaternion_matrix,
)

if TYPE_CHECKING:
    from pycram.view_manager import CameraDescription


def collision_check(robot: AbstractRobot, world: World) -> List[ClosestPoints]:
    """
    This method checks if a given robot collides with any object within the world
    which it is not allowed to collide with.
    This is done checking iterating over every object within the world and checking
    if the robot collides with it. Careful the floor will be ignored.
    If there is a collision with an object that was not within the allowed collision
    list the function will raise a RobotInCollision exception.

    :param robot: The robot object in the (Bullet)World where it should be checked if it collides with something
    :param allowed_collision: dict of objects with which the robot is allowed to collide each object correlates to a list of links of which this object consists
    :param world: The world in which collision should be checked
    :raises: RobotInCollision if the robot collides with an object it is not allowed to collide with.
    """
    world.collision_manager.clear_temporary_rules()
    world.collision_manager.add_temporary_rule(AllowSelfCollisions(robot=robot))
    world.collision_manager.update_collision_matrix(buffer=0.0)
    return [
        contact
        for contact in world.collision_manager.compute_collisions().contacts
        if contact.distance <= 0.0
    ]


def link_pose_for_joint_config(obj: Body, joint_config: Dict[str, float]) -> Pose:
    """
    Get the pose a link would be in if the given joint configuration would be applied to the object.
    This is done by using the respective object in the prospection world and applying the joint configuration
    to this one. After applying the joint configuration the link position is taken from there.

    :param obj: The body for which the pose should be calculated
    :param joint_config: Dict with the goal joint configuration
    :return: The pose of the link after applying the joint configuration
    """
    reasoning_world = deepcopy(obj._world)
    for joint_name, joint_pose in joint_config.items():
        reasoning_world.state[
            reasoning_world.get_degree_of_freedom_by_name(joint_name).id
        ].position = joint_pose
    reasoning_world.notify_state_change()
    return reasoning_world.get_body_by_name(obj.name).global_pose


def get_rays_from_min_max(
    min_bound: Sequence[float],
    max_bound: Sequence[float],
    step_size_in_meters: float = 0.01,
) -> np.ndarray:
    """
    Get rays from min and max bounds as an array of start and end 3D points.
    Note: The rays are not steped in the x direction as the rays are cast parallel to the x-axis.

    Example:
    >>> min_bound = [0, 0, 0]
    >>> max_bound = [1, 2, 3]
    >>> rays = get_rays_from_min_max(min_bound, max_bound, 1)
    >>> rays.shape
    (6, 3, 2)
    >>> rays
    array([
    [[0. , 1. ],
     [0. , 0. ],
     [0. , 0. ]],
    [[0. , 1. ],
     [0. , 0. ],
     [1.5, 1.5]],
    [[0. , 1. ],
     [0. , 0. ],
     [3. , 3. ]],
    [[0. , 1. ],
     [2. , 2. ],
     [0. , 0. ]],
    [[0. , 1. ],
     [2. , 2. ],
     [1.5, 1.5]],
    [[0. , 1. ],
     [2. , 2. ],
     [3. , 3. ]]
     ])

    :param min_bound: The minimum bound of the rays, a sequence of 3 floats.
    :param max_bound: The maximum bound of the rays, a sequence of 3 floats.
    :param step_size_in_meters: The step size in meters between the rays.
    :return: The rays as an array of shape (n, 3, 2) where n is number of rays, 3 is because each point has x, y, and z,
    and 2 is for the start and end points of the rays.
    """
    min_bound = np.array(min_bound)
    max_bound = np.array(max_bound)
    n_steps = np.ceil(
        np.abs(max_bound[1:] - min_bound[1:]) / step_size_in_meters
    ).astype(int)
    rays_start_x = np.ones((n_steps[0], n_steps[1])) * min_bound[0]
    rays_end_x = np.ones((n_steps[0], n_steps[1])) * max_bound[0]
    y_values = np.linspace(min_bound[1], max_bound[1], n_steps[0])
    z_values = np.linspace(min_bound[2], max_bound[2], n_steps[1])
    rays_start_y = np.tile(y_values, (n_steps[1], 1)).T
    rays_end_y = rays_start_y
    rays_start_z = np.tile(z_values, (n_steps[0], 1))
    rays_end_z = rays_start_z
    rays_start = np.stack((rays_start_x, rays_start_y, rays_start_z), axis=-1)
    rays_end = np.stack((rays_end_x, rays_end_y, rays_end_z), axis=-1)
    rays_start = rays_start.reshape(-1, 3)
    rays_end = rays_end.reshape(-1, 3)
    # The shape of rays is (num_rays, 3, 2), while num_rays = n_steps[0] (num y step) * n_steps[1] (num z step)
    return np.stack((rays_start, rays_end), axis=-1)


def chunks(lst: Union[List, np.ndarray], n: int) -> Iterator[List]:
    """
    Yield successive n-sized chunks from lst.

    :param lst: The list from which chunks should be yielded
    :param n: Size of the chunks
    :return: A list of size n from lst
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class bcolors:
    """
    Color codes which can be used to highlight Text in the Terminal. For example,
    for warnings.
    Usage:
    Firstly import the class into the file.
    print(f'{bcolors.WARNING} Some Text {bcolors.ENDC}')
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def axis_angle_to_quaternion(axis: List, angle: float) -> Tuple:
    """
    Convert axis-angle to quaternion.

    :param axis: (x, y, z) tuple representing rotation axis.
    :param angle: rotation angle in degree
    :return: The quaternion representing the axis angle
    """
    angle = math.radians(angle)
    axis_length = math.sqrt(sum([i**2 for i in axis]))
    normalized_axis = tuple(i / axis_length for i in axis)

    x = normalized_axis[0] * math.sin(angle / 2)
    y = normalized_axis[1] * math.sin(angle / 2)
    z = normalized_axis[2] * math.sin(angle / 2)
    w = math.cos(angle / 2)

    return tuple((x, y, z, w))


def adjust_camera_pose_based_on_target(
    cam_pose: Pose,
    target_pose: Pose,
    camera_description: CameraDescription,
) -> Pose:
    """
    Adjust the given cam_pose orientation such that it is facing the target_pose, which partly depends on the
     front_facing_axis of the that is defined in the camera_description.

    :param cam_pose: The camera pose.
    :param target_pose: The target pose.
    :param camera_description: The camera description.
    :return: The adjusted camera pose.
    """
    quaternion = get_quaternion_between_camera_and_target(
        cam_pose, target_pose, camera_description
    )
    # apply the rotation to the camera pose using quaternion multiplication
    return apply_quaternion_to_pose(cam_pose, quaternion)


def get_quaternion_between_camera_and_target(
    cam_pose: Pose,
    target_pose: Pose,
    camera_description: "CameraDescription",
) -> np.ndarray:
    """
    Get the quaternion between the camera and the target.

    :param cam_pose: The camera pose.
    :param target_pose: The target pose.
    :param camera_description: The camera description.
    :return: The quaternion between the camera and the target.
    """
    # Get the front facing axis of the camera in the world frame
    front_facing_axis = transform_vector_using_pose(
        camera_description.front_facing_axis, cam_pose
    )
    front_facing_axis = front_facing_axis - np.array(cam_pose.position.to_list())

    # Get the vector from the camera to the target
    camera_to_target = cam_pose.get_vector_to_pose(target_pose)

    # Get the quaternion between the camera and target
    return get_quaternion_between_two_vectors(front_facing_axis, camera_to_target)


def transform_vector_using_pose(vector: Sequence, pose) -> np.ndarray:
    """
    Transform a vector using a pose.

    :param vector: The vector.
    :param pose: The pose.
    :return: The transformed vector.
    """
    vector = np.array(vector).reshape(1, 3)
    return (
        pose.to_transform_stamped("pose")
        .apply_transform_to_array_of_points(vector)
        .flatten()
    )


def apply_quaternion_to_pose(pose: Pose, quaternion: np.ndarray) -> Pose:
    """
    Apply a quaternion to a pose.

    :param pose: The pose.
    :param quaternion: The quaternion.
    :return: The new pose.
    """
    pose_quaternion = np.array(pose.orientation.to_list())
    new_quaternion = quaternion_multiply(quaternion, pose_quaternion)
    return Pose(pose.position.to_list(), new_quaternion.tolist())


def get_quaternion_between_two_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Get the quaternion between two vectors.

    :param v1: The first vector.
    :param v2: The second vector.
    :return: The quaternion between the two vectors.
    """
    axis, angle = get_axis_angle_between_two_vectors(v1, v2)
    return quaternion_about_axis(angle, axis)


def get_axis_angle_between_two_vectors(
    v1: np.ndarray, v2: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Get the axis and angle between two vectors.

    :param v1: The first vector.
    :param v2: The second vector.
    :return: The axis and angle between the two vectors.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2) - 1e-9)  # to avoid numerical errors
    return axis, angle


def wxyz_to_xyzw(wxyz: List[float]) -> List[float]:
    """
    Convert a quaternion from WXYZ to XYZW format.
    """
    return [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]


def xyzw_to_wxyz(xyzw: List[float]) -> List[float]:
    """
    Convert a quaternion from XYZW to WXYZ format.

    :param xyzw: The quaternion in XYZW format.
    """
    return [xyzw[3], *xyzw[:3]]


def wxyz_to_xyzw_arr(wxyz: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion from WXYZ to XYZW format.

    :param wxyz: The quaternion in WXYZ format.
    """
    xyzw = np.zeros(4)
    xyzw[:3] = wxyz[1:]
    xyzw[3] = wxyz[0]
    return xyzw


def xyzw_to_wxyz_arr(xyzw: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion from XYZW to WXYZ format.

    :param xyzw: The quaternion in XYZW format.
    """
    wxyz = np.zeros(4)
    wxyz[0] = xyzw[3]
    wxyz[1:] = xyzw[:3]
    return wxyz


def translate_pose_along_local_axis(
    pose: Pose, axis: Union[List, np.ndarray], distance: float
) -> Pose:
    """
    Translate a pose along a given 3d vector (axis) by a given distance. The axis is given in the local coordinate
    frame of the pose. The axis is normalized and then scaled by the distance.

    :param pose: The pose that should be translated
    :param axis: The local axis along which the translation should be performed
    :param distance: The distance by which the pose should be translated

    :return: The translated pose
    """
    normalized_translation_vector = np.array(axis) / np.linalg.norm(axis)

    rot_matrix = pose.to_rotation_matrix().to_np()[:3, :3]
    translation_in_world = rot_matrix @ normalized_translation_vector
    scaled_translation_vector = (
        np.array(pose.to_position().to_list()[:3]) + translation_in_world * distance
    )

    return Pose(
        Point3.from_iterable(scaled_translation_vector),
        pose.to_quaternion(),
        reference_frame=pose.reference_frame,
    )
