"""3D geometry math utilities for RoboKudo.

This module provides utilities for 3D geometric calculations.

:module: math_helper
:synopsis: 3D geometric calculations and utilities
:moduleauthor: RoboKudo Team
"""

from robokudo.utils.transform import get_rotation_matrix_from_q
from typing_extensions import Iterable, Optional, Tuple, List
import numpy as np


def intersection_point(
    point1: Tuple[float, float, float], point2: Tuple[float, float, float], t: float
) -> Tuple[float, float, float]:
    """Calculate point along line segment using parameter t.

    :param point1: Start point (x,y,z)
    :param point2: End point (x,y,z)
    :param t: Interpolation parameter [0,1]
    :return: Interpolated point (x,y,z)

    :Example:

    .. code-block:: python

        P1 = (0, 0, 0)
        P2 = (1, 1, 1)
        mid = intersection_point(P1, P2, 0.5)  # Returns (0.5, 0.5, 0.5)

    .. note::
        Uses linear interpolation: P = P1 + t*(P2 - P1)
    """
    if t < 0 or t > 1:
        raise ValueError("t must be between 0 and 1")

    x1, y1, z1 = point1
    x2, y2, z2 = point2
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    z = z1 + t * (z2 - z1)
    return (x, y, z)


def distance(
    point1: Tuple[float, float, float], point2: Tuple[float, float, float]
) -> float:
    """Calculate Euclidean distance between two 3D points.

    :param point1: First point (x,y,z)
    :param point2: Second point (x,y,z)
    :return: Euclidean distance

    :Example:

    .. code-block:: python

        p1 = (0, 0, 0)
        p2 = (1, 1, 1)
        d = distance(p1, p2)  # Returns sqrt(3)
    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5


def intersecting_spheres(
    point1: Tuple[float, float, float],
    point2: Tuple[float, float, float],
    spheres: Iterable[Tuple[str, Tuple[float, float, float], float]],
) -> List[
    Tuple[float, str, Tuple[float, float, float], float, Tuple[float, float, float]]
]:
    """
    Check where the first intersection between the line segment described by P1 and P2
    and the spheres in our parameters happens.

    Spheres must be a list of triples (name, center, radius)

    :param point1: Start point of the line segment (x, y, z).
    :param point2: End point of the line segment (x, y, z).
    :param spheres: Iterable of tuples (name, center, radius) where center is a tuple (x, y, z).
    :return: List of tuples (dist to point1, name, center, radius, first_intersection_from_point1).

    :Example:

    .. code-block:: python

        P1 = (0, 0, 0)
        P2 = (1, 1, 1)
        spheres = [
            ("sphere1", (0.5, 0.5, 0.5), 0.1),
            ("sphere2", (0.7, 0.7, 0.7), 0.1)
        ]
        intersections = intersecting_spheres(P1, P2, spheres)

    .. note::
        Returns intersections sorted by distance from point1
    """
    intersections = []

    for name, C, r in spheres:
        t = compute_line_intersection_point(point1, point2, C, r)
        t = min(t) if t is not None else t

        if t is not None:
            intersection = intersection_point(point1, point2, t)
            dist = distance(point1, intersection)
            intersections.append((dist, name, C, r, intersection))

    # Sort by distance
    intersections.sort(key=lambda x: x[0])

    return intersections


def compute_line_intersection_point(
    point1: Tuple[float, float, float],
    point2: Tuple[float, float, float],
    sphere_center: Tuple[float, float, float],
    sphere_radius: float,
) -> Optional[List[float]]:
    """Compute the intersection points between a sphere and a line segment.

    :param point1: First point of the line (x, y, z)
    :param point2: Second point of the line (x, y, z)
    :param sphere_center: Center of the sphere
    :param sphere_radius: Radius of the sphere
    :return: List of intersection points on the line or None if line does not intersect
    """
    if sphere_radius < 0:
        raise ValueError("Sphere radius must be positive")

    d1 = distance(point1, sphere_center)
    if point1 == point2:
        if sphere_center == point1 or d1 < sphere_radius:
            return [0.0]  # The point is inside the sphere
        return None

    d2 = distance(point2, sphere_center)

    # Check if both points are inside the sphere
    if d1 < sphere_radius and d2 < sphere_radius:
        return [0.0, 1.0]  # The entire segment is inside the sphere

    # Extract coordinates
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    cx, cy, cz = sphere_center
    r = sphere_radius

    # Coefficients for the quadratic equation
    a = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    b = 2 * ((x2 - x1) * (x1 - cx) + (y2 - y1) * (y1 - cy) + (z2 - z1) * (z1 - cz))
    c = (
        cx**2
        + cy**2
        + cz**2
        + x1**2
        + y1**2
        + z1**2
        - 2 * (cx * x1 + cy * y1 + cz * z1)
        - r**2
    )

    # Discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None  # No intersection

    sqrt_discriminant = discriminant**0.5
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)

    # Check if t1 and t2 are within [0, 1]
    t1_valid = 0 <= t1 <= 1
    t2_valid = 0 <= t2 <= 1

    if t1_valid and t2_valid:
        return [t1, t2]
    elif t1_valid:
        return [t1]
    elif t2_valid:
        return [t2]
    else:
        return None


def does_line_intersect_sphere(
    point1: Tuple[float, float, float],
    point2: Tuple[float, float, float],
    sphere_center: Tuple[float, float, float],
    sphere_radius: float,
) -> bool:
    """Test if line segment intersects sphere.

    :param point1: Line segment start point (x,y,z)
    :param point2: Line segment end point (x,y,z)
    :param sphere_center: Sphere center point (x,y,z)
    :param sphere_radius: Sphere radius
    :return: True if line segment intersects sphere

    :Example:

    .. code-block:: python

        point1 = (0, 0, 0)
        point2 = (1, 1, 1)
        sphere_center = (0.5, 0.5, 0.5)
        sphere_radius = 0.1
        intersects = does_line_intersect_sphere(point1, point2, sphere_center, sphere_radius)

    .. note::
        Uses quadratic equation to find intersection points
    """
    if sphere_radius < 0:
        raise ValueError("Sphere radius must be positive")
    if point1 == point2:
        if sphere_center == point1:
            return True
        return distance(point1, point2) < sphere_radius

    t = compute_line_intersection_point(point1, point2, sphere_center, sphere_radius)
    # If there is a valid t value, the segment intersects the sphere
    return t is not None


def compute_direction_vector_angle(
    direction_vector: np.ndarray, floor: np.ndarray = np.array([0, 1, 0], dtype=float)
) -> float:
    """
    Computes the angle of a direction vector in degrees relative to the floor.
    The angle is computed as the arctangent of the y and x components of the vector.
    :param direction_vector: A 3D vector (numpy array) representing the direction.
    :param floor: A 3D vector (numpy array) representing the floor normal. Default is (0, 1, 0).
    :return: Angle in degrees.
    """

    if np.allclose(direction_vector, 0):
        raise ValueError("Direction vector is zero.")
    normalized_floor = floor / np.linalg.norm(floor)

    vertical = np.dot(direction_vector, normalized_floor)
    horizontal = np.linalg.norm(direction_vector - vertical * normalized_floor)

    ang = np.degrees(np.arctan2(vertical, horizontal))  # (-90..90)
    return ang


def intersecting_cuboids(
    point1: Tuple[float, float, float],
    point2: Tuple[float, float, float],
    cuboids: List[
        Tuple[
            str,
            Tuple[float, float, float],
            Tuple[float, float, float, float],
            Tuple[float, float, float],
        ]
    ],
) -> List[Tuple[float, str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Check where the first intersection between the line segment described by P1 and P2
    and the cuboids in our parameters happens. The intersection problem is solved using the slab method (see
    https://en.wikipedia.org/wiki/Slab_method).

    Cuboids must be a list of tuples (name, position, orientation, size)

    :param point1: Start point of the line segment (x, y, z).
    :param point2: End point of the line segment (x, y, z).
    :param cuboids: List of tuples (name, position, orientation, size) where position is a tuple (x, y, z),
                     orientation is a tuple (x, y, z, w) representing a quaternion, and size is a tuple (width, height, depth).

    :return: List of tuples (dist to P1, name, cuboid_min, cuboid_max, first_intersection_from_P1).
    """
    intersections = []
    point1 = np.array(point1, dtype=float)
    point2 = np.array(point2, dtype=float)

    for name, cuboid_position, cuboid_orientation, cuboid_size in cuboids:
        if cuboid_size[0] < 0 or cuboid_size[1] < 0 or cuboid_size[2] < 0:
            raise ValueError("Cuboid size cannot be negative")
        cuboid_position = np.array(cuboid_position, dtype=float)
        cuboid_size = np.array(cuboid_size, dtype=float)
        half_size = cuboid_size / 2

        # Convert quaternion to rotation matrix
        quat = np.array(cuboid_orientation, dtype=float)
        rot = get_rotation_matrix_from_q(quat)  # x, y, z, w
        # inverse of a rotation matrix = transpose
        rot_inv = rot.T

        # Transform P1 and P2 into cuboid local space
        # This helps, because we dont need to transform the cuboid based on its orientation, but only the line segment
        P1_local = rot_inv @ (point1 - cuboid_position)
        P2_local = rot_inv @ (point2 - cuboid_position)
        direction_local = P2_local - P1_local

        # avoid division by zero
        dir_eps = np.where(direction_local == 0, 1e-12, direction_local)
        # Calculate t values for intersection with the slabs defined by the cuboid
        t_min = (-half_size - P1_local) / dir_eps
        t_max = (half_size - P1_local) / dir_eps

        # Ensure t_min and t_max are ordered correctly. t1 and t2 are the entry and exit points of the line segment
        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)

        t_enter = np.max(t1)
        t_exit = np.min(t2)

        if t_enter <= t_exit:
            # Compute world-space AABB bounds for output
            # Useful for output annotator struct visualization
            corners_local = np.array([[-1, -1, -1], [1, 1, 1]]) * half_size
            corners_world = (rot @ corners_local.T).T + cuboid_position
            cuboid_min = np.min(corners_world, axis=0)
            cuboid_max = np.max(corners_world, axis=0)

            if t_enter < 0 and t_exit > 1:
                # Segment is entirely inside the cube
                intersection_world = point1  # P1 is inside the cube
                dist = 0.0
            elif t_enter < 0 and 0 <= t_exit <= 1:
                # Segment starts inside and exits outside
                intersection_local = intersection_point(P1_local, P2_local, t_exit)
                intersection_world = rot @ intersection_local + cuboid_position
                dist = distance(point1, intersection_world)
            elif 0 <= t_enter <= 1:
                # Segment starts outside and intersects the cube (use t_enter)
                intersection_local = intersection_point(P1_local, P2_local, t_enter)
                intersection_world = rot @ intersection_local + cuboid_position
                dist = distance(point1, intersection_world)
            else:
                # t_enter > 1: Intersection is beyond P2, ignore
                continue

            intersections.append(
                (dist, name, cuboid_min, cuboid_max, intersection_world)
            )

    intersections.sort(key=lambda x: x[0])
    return intersections
