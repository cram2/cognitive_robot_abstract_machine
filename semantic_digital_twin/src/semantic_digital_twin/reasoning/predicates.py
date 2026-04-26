from __future__ import annotations
from abc import ABC
from dataclasses import dataclass

import numpy as np
import math
import trimesh.boolean
from trimesh.collision import CollisionManager

from krrood.entity_query_language.predicate import (
    Predicate,
    Symbol,
    symbolic_function,
)
from random_events.interval import Interval
from typing_extensions import List, TYPE_CHECKING, Iterable, Type

from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.spatial_computations.ik_solver import (
    MaxIterationsException,
    UnreachableException,
)
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.spatial_types import Vector3, Point3
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import (
    Body,
    Region,
    KinematicStructureEntity,
)

if TYPE_CHECKING:
    from semantic_digital_twin.robots.abstract_robot import (
        Camera,
    )


@symbolic_function
def stable(obj: Body) -> bool:
    """
    Checks if an object is stable in the world. Stable meaning that its position will not change after simulating
    physics in the World. This will be done by simulating the world for 10 seconds and comparing
    the previous coordinates with the coordinates after the simulation.

    :param obj: The object which should be checked
    :return: True if the given object is stable in the world False else
    """
    raise NotImplementedError("Needs multiverse")


@symbolic_function
def contact(
    body1: Body,
    body2: Body,
    threshold: float = 0.001,
) -> bool:
    """
    Checks if two objects are in contact or not.

    :param body1: The first object
    :param body2: The second object
    :param threshold: The threshold for contact detection
    :return: True if the two objects are in contact False else
    """
    tcd = body1._world.collision_manager.collision_detector
    result = tcd.check_collision_between_bodies(body1, body2)

    if result is None:
        return False
    return result.distance < threshold


@symbolic_function
def get_visible_bodies(camera: Camera) -> List[KinematicStructureEntity]:
    """
    Get all bodies and regions that are visible from the given camera using a segmentation mask.

    :param camera: The camera for which the visible objects should be returned
    :return: A list of bodies/regions that are visible from the camera
    """
    rt = RayTracer(camera._world)
    rt.update_scene()

    # This ignores the camera orientation and sets it to identity
    cam_pose = np.eye(4, dtype=float)
    cam_pose[:3, 3] = camera.root.global_transform.to_np()[:3, 3]

    seg = rt.create_segmentation_mask(
        HomogeneousTransformationMatrix(cam_pose, reference_frame=camera._world.root),
        resolution=256,
        min_distance=0.2,
    )
    indices = np.unique(seg)
    indices = indices[indices > -1]
    bodies = [camera._world.kinematic_structure[i] for i in indices]

    return bodies


@symbolic_function
def visible(camera: Camera, obj: KinematicStructureEntity) -> bool:
    """
    Checks if a body/region is visible by the given camera.
    """
    return obj in get_visible_bodies(camera)


@symbolic_function
def occluding_bodies(camera: Camera, body: Body) -> List[Body]:
    """
    Determines the bodies that occlude a given body in the scene as seen from a specified camera.

    This function uses a ray-tracing approach to check occlusion. Every body that hides anything from the target body
    is an occluding body.

    :param camera: The camera for which the occluding bodies should be returned
    :param body: The body for which the occluding bodies should be returned
    :return: A list of bodies that are occluding the given body.
    """

    # get camera pose
    camera_pose = np.eye(4, dtype=float)
    camera_pose[:3, 3] = camera.root.global_transform.to_np()[:3, 3]
    camera_pose = HomogeneousTransformationMatrix(
        camera_pose, reference_frame=camera._world.root
    )

    # create a world only containing the target body
    world_without_occlusion = World()
    root = Body(name=PrefixedName("root"))
    with world_without_occlusion.modify_world():
        world_without_occlusion.add_body(root)
        copied_body = Body.from_json(body.to_json())
        root_T_body = body.global_transform
        root_T_body.reference_frame = root
        root_to_copied_body = FixedConnection(
            parent=root,
            child=copied_body,
            parent_T_connection_expression=root_T_body,
        )
        world_without_occlusion.add_connection(root_to_copied_body)

    # get segmentation mask without occlusion
    ray_tracer_without_occlusion = RayTracer(world_without_occlusion)
    ray_tracer_without_occlusion.update_scene()
    segmentation_mask_without_occlusion = (
        ray_tracer_without_occlusion.create_segmentation_mask(
            camera_pose, resolution=256, min_distance=0.1
        )
    )

    # get segmentation mask with occlusion
    ray_tracer_with_occlusion = RayTracer(camera._world)
    ray_tracer_with_occlusion.update_scene()
    segmentation_mask_with_occlusion = (
        ray_tracer_with_occlusion.create_segmentation_mask(
            camera_pose, resolution=256, min_distance=0.1
        )
    )

    mask_without_occluders = segmentation_mask_without_occlusion[
        segmentation_mask_without_occlusion == copied_body.index
    ].nonzero()

    mask_with_occluders = segmentation_mask_with_occlusion[
        mask_without_occluders != body.index
    ]
    indices = np.unique(mask_with_occluders)
    indices = indices[indices > -1]
    bodies = [camera._world.kinematic_structure[i] for i in indices]
    return bodies


@symbolic_function
def reachable(pose: HomogeneousTransformationMatrix, root: Body, tip: Body) -> bool:
    """
    Checks if a manipulator can reach a given position.
    This is determined by inverse kinematics.

    :param pose: The pose to reach
    :param root: The root of the kinematic chain.
    :param tip: The threshold between the end effector and the position.
    :return: True if the end effector is closer than the threshold to the target position, False in every other case
    """
    try:
        root._world.compute_inverse_kinematics(
            root=root, tip=tip, target=pose, max_iterations=1000
        )
    except MaxIterationsException as e:
        return False
    except UnreachableException as e:
        return False
    return True


@symbolic_function
def compute_euclidean_planar_distance(body1: Body, body2: Body, ignore_dimension: Vector3):
    """
    Computes the Euclidean distance between two bodies in 2D space, ignoring a specific dimension
    specified by the user. The ignored dimension is set to zero before the distance calculation. This
    function can be used to handle scenarios where computations are restricted to certain spatial
    planes.

    :param body1: The first body to compute the distance from. It uses the global pose of the body
                      to extract the position.
    :param body2: The second body to compute the distance to. It also utilizes the global pose of
                      the body to extract the position.
    :param ignore_dimension: Specifies which dimension (x, y, or z) should be ignored in the
                                     computation. The ignored dimension is set to zero for both
                                     positions prior to calculating the distance.

    :return: The Euclidean distance between the two bodies in the 2D plane after ignoring the
               specified dimension.
    """
    body1_position = body1.global_pose.to_position()
    body2_position = body2.global_pose.to_position()

    if np.allclose(ignore_dimension, Vector3.X()):
        body1_position.x = 0.0
        body2_position.x = 0.0
    elif np.allclose(ignore_dimension, Vector3.Y()):
        body1_position.y = 0.0
        body2_position.y = 0.0
    elif np.allclose(ignore_dimension, Vector3.Z()):
        body1_position.z = 0.0
        body2_position.z = 0.0


    return body1_position.euclidean_distance(
        body2_position
    )


@symbolic_function
def is_supported_by(
    supported_body: Body, supporting_body: Body, max_intersection_height: float = 0.1
) -> bool:
    """
    Checks if one object is supporting another object.

    :param supported_body: Object that is supported
    :param supporting_body: Object that potentially supports the first object
    :param max_intersection_height: Maximum height of the intersection between the two objects.
    If the intersection is higher than this value, the check returns False due to unhandled clipping.
    :return: True if the second object is supported by the first object, False otherwise
    """
    if Below(
        supported_body.center_of_mass,
        supporting_body.center_of_mass,
        supported_body.global_transform,
    )():
        return False
    bounding_box_supported_body = (
        supported_body.collision.as_bounding_box_collection_at_origin(
            HomogeneousTransformationMatrix(reference_frame=supported_body)
        ).event
    )
    bounding_box_supporting_body = (
        supporting_body.collision.as_bounding_box_collection_at_origin(
            HomogeneousTransformationMatrix(reference_frame=supported_body)
        ).event
    )

    intersection = (
        bounding_box_supported_body & bounding_box_supporting_body
    ).bounding_box()

    if intersection.is_empty():
        return False

    z_intersection: Interval = intersection[SpatialVariables.z.value]
    size = sum([si.upper - si.lower for si in z_intersection.simple_sets])
    return size < max_intersection_height

@symbolic_function
def is_supporting(supporting_body: Body, max_intersection_height: float = 0.1) -> bool:
    """
    Determine if any body in the world is supported by a given supporting body.

    This function iterates over all bodies in the provided world and checks
    if any of them are supported by the specified supporting body. The
    support determination is performed using the helper function `is_supported_by`.
    Bodies for which the computation fails are skipped.

    :param supporting_body: The body that is being checked to determine if it is supporting other bodies in the world.
    :param max_intersection_height: The maximum allowable intersection
    height for a body to be considered supported. Defaults to 0.1.

    :return: True if any body in the world is supported by the supporting_body,
    False otherwise.
    """
    for candidate in supporting_body._world.bodies_with_collision:
        if candidate is supporting_body:
            continue
        if is_supported_by(candidate, supporting_body, max_intersection_height):
            return True

    return False


@symbolic_function
def is_body_in_region(body: Body, region: Region) -> float:
    """
    Check if the body is in the region by computing the fraction of the body's
    collision volume that lies inside the region's area volume.

    Implementation detail: both the body and region meshes are defined in their
    respective local frames; we must transform them into a common (world) frame
    using their global poses before computing the boolean intersection.

    :param body: The body for which the check should be done.
    :param region: The region to check if the body is in.
    :return: The percentage (0.0..1.0) of the body's volume that lies in the region.
    """
    # Retrieve meshes in local frames
    body_mesh_local = body.collision.combined_mesh
    region_mesh_local = region.area.combined_mesh

    # Transform copies of the meshes into the world frame
    body_mesh = body_mesh_local.copy().apply_transform(body.global_transform.to_np())
    region_mesh = region_mesh_local.copy().apply_transform(
        region.global_transform.to_np()
    )
    intersection = trimesh.boolean.intersection([body_mesh, region_mesh])

    # no body volume -> zero fraction
    body_volume = body_mesh.volume
    if body_volume <= 1e-12:
        return 0.0

    return intersection.volume / body_volume


@dataclass
class KinematicStructureEntitySpatialRelation(Symbol, ABC):
    """
    Base class for spatial relations between two KinematicStructureEntity instances.
    Implementations typically compare the centers of mass computed from the KSE's collision geometry.
    """

    body: KinematicStructureEntity
    """
    The KSE for which the check should be done.
    """

    other: KinematicStructureEntity
    """
    The other KSE.
    """


@dataclass
class PointSpatialRelation(Symbol, ABC):
    """
    Check if the point is spatially related to the other point.
    """

    point: Point3
    """
    The point for which the check should be done.
    """

    other: Point3
    """
    The other point.
    """


@dataclass
class ViewDependentSpatialRelation(PointSpatialRelation, ABC):

    point_of_semantic_annotation: HomogeneousTransformationMatrix
    """
    The reference spot from where to look at the bodies.
    """

    eps: float = 1e-12
    """
    A small value to avoid division by zero.
    """

    spatial_relation_result: bool = False

    def _signed_distance_along_direction(self, index: int) -> float:
        """
        Calculate the spatial relation between self.point and self.other with respect to a given
        reference point (self.point_of_semantic_annotation) and a specified axis index. This function computes the
        signed distance along a specified direction derived from the reference point
        to compare the positions.

        :param index: The index of the axis in the transformation matrix along which
            the spatial relation is computed.
        :return: The signed distance between the first and the second points along the given direction.
        """
        ref_np = self.point_of_semantic_annotation.to_np()
        front_world = ref_np[:3, index]
        front_norm = front_world / (np.linalg.norm(front_world) + self.eps)
        front_norm = Vector3(
            x=front_norm[0],
            y=front_norm[1],
            z=front_norm[2],
            reference_frame=self.point_of_semantic_annotation.reference_frame,
        )

        s_body = front_norm.dot(self.point.to_vector3())
        s_other = front_norm.dot(self.other.to_vector3())
        return (s_body - s_other).compile()()


@dataclass
class LeftOf(ViewDependentSpatialRelation):
    """
    The "left" direction is taken as the -Y axis of the given point of semantic_annotation.
    """

    def __call__(self) -> bool:
        self.spatial_relation_result = self._signed_distance_along_direction(1) > 0.0
        return self.spatial_relation_result


@dataclass
class RightOf(ViewDependentSpatialRelation):
    """
    The "right" direction is taken as the +Y axis of the given point of semantic_annotation.
    """

    def __call__(self) -> bool:
        self.spatial_relation_result = self._signed_distance_along_direction(1) < 0.0
        return self.spatial_relation_result


@dataclass
class Above(ViewDependentSpatialRelation):
    """
    The "above" direction is taken as the +Z axis of the given point of semantic_annotation.
    """

    def __call__(self) -> bool:
        self.spatial_relation_result = self._signed_distance_along_direction(2) > 0.0
        return self.spatial_relation_result


@dataclass
class Below(ViewDependentSpatialRelation):
    """
    The "below" direction is taken as the -Z axis of the given point of semantic_annotation.
    """

    def __call__(self) -> bool:
        self.spatial_relation_result = self._signed_distance_along_direction(2) < 0.0
        return self.spatial_relation_result


@dataclass
class Behind(ViewDependentSpatialRelation):
    """
    The "behind" direction is defined as the -X axis of the given point of semantic annotation.
    """

    def __call__(self) -> bool:
        self.spatial_relation_result = self._signed_distance_along_direction(0) < 0.0
        return self.spatial_relation_result


@dataclass
class InFrontOf(ViewDependentSpatialRelation):
    """
    The "in front of" direction is defined as the +X axis of the given point of semantic annotation.
    """

    def __call__(self) -> bool:
        self.result = self._signed_distance_along_direction(0) > 0.0
        return self.result


@dataclass
class InsideOf(KinematicStructureEntitySpatialRelation):
    """
    The "inside of" relation is defined as the fraction of the volume of self.body
    that lies within the bounding box of self.other.

    Readily, `InsideOf(a,b) = 1.` means that `a` is completely inside `b`.
    """

    containment_ratio: float = 0.0

    def __call__(self) -> float:
        self.containment_ratio = self.compute_containment_ratio()
        return self.containment_ratio

    def compute_containment_ratio(self) -> float:
        """
        Compute the containment ratio of self.body inside self.other.
        """
        if self.other.combined_mesh is None:
            return 0.0

        # Get meshes in their local (body) frames
        mesh_a_local = self.body.combined_mesh
        mesh_b_local = self.other.combined_mesh

        # Check if either mesh is empty
        if (
            mesh_a_local is None
            or mesh_a_local.is_empty
            or mesh_b_local is None
            or mesh_b_local.is_empty
        ):
            return 0.0

        # Transform meshes from body frame to world frame
        mesh_a = mesh_a_local.copy()
        mesh_a.apply_transform(self.body.global_transform.to_np())

        mesh_b = mesh_b_local.copy()
        mesh_b.apply_transform(self.other.global_transform.to_np())

        # Use bounding box of mesh_b to check if mesh_a is inside mesh_b
        mesh_b_bbox = mesh_b.bounding_box

        if not mesh_b_bbox.is_watertight:
            return 0.0

        inside = mesh_b_bbox.contains(mesh_a.vertices)
        if len(inside) == 0:
            return 0.0
        return sum(inside) / len(inside)


@dataclass
class ContainsType(Predicate):
    """
    Predicate that checks if any object in the iterable is of the given type.
    """

    iterable: Iterable
    """
    Iterable to check for objects of the given type.
    """

    obj_type: Type
    """
    Object type to check for.
    """

    def __call__(self) -> bool:
        return any(isinstance(obj, self.obj_type) for obj in self.iterable)


@symbolic_function
def is_place_occupied(
    box: BoundingBox, world: World, allowed_bodies: List[Body] = None
) -> bool:
    """
    Checks if the given region (as a box at its pose) intersects with any collidable
    object in the world, excluding `allowed_bodies`.

    The region is converted to a box mesh at the region pose and tested against
    each body's world-aligned collision mesh using trimesh's collision manager.

    :param box: The region (axis-aligned box in its own local frame with pose in `region.origin`).
    :param world: The world providing bodies with enabled collisions.
    :param allowed_bodies: Bodies to ignore during the check.
    :return: True if any collision is found, False otherwise.
    """
    allowed_bodies = set(allowed_bodies or [])

    # Build a mesh for the region box at its current pose
    region_box_shape = box.as_shape()  # returns a Box centered at the region
    region_mesh = region_box_shape.mesh.copy()
    # region_mesh.apply_transform(region_box_shape.origin.to_np())
    region_mesh.apply_transform(
        world.transform(region_box_shape.origin, world.root).to_np()
    )

    # Prepare collision manager with the region mesh
    cm = CollisionManager()
    cm.add_object("region", region_mesh)

    # Iterate over collidable bodies and test collision
    for body in world.bodies_with_collision:
        if body in allowed_bodies:
            continue

        mesh_local = getattr(body.collision, "combined_mesh", None)
        if mesh_local is None or getattr(mesh_local, "is_empty", False):
            continue

        # Transform body mesh into world frame
        body_mesh = mesh_local.copy()
        body_mesh.apply_transform(body.global_pose.to_np())

        # Early exit on first collision
        if cm.in_collision_single(body_mesh):
            return True

    return False


@symbolic_function
def allclose(array1: np.ndarray, array2: np.ndarray, atol=1e-3) -> bool:
    """
    Symbolic wrapper around `np.allclose`.
    """
    return np.allclose(array1, array2, atol=atol)
