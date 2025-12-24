from __future__ import annotations
from abc import ABC
from dataclasses import dataclass

import numpy as np
import trimesh.boolean
from krrood.entity_query_language.predicate import (
    Predicate,
    Symbol,
)
from random_events.interval import Interval
from typing_extensions import List, TYPE_CHECKING, Iterable, Type

from ..collision_checking.trimesh_collision_detector import TrimeshCollisionDetector
from ..datastructures.prefixed_name import PrefixedName
from ..datastructures.variables import SpatialVariables
from ..spatial_computations.ik_solver import (
    MaxIterationsException,
    UnreachableException,
)
from ..spatial_computations.raytracer import RayTracer
from ..spatial_types import Vector3
from ..spatial_types.spatial_types import HomogeneousTransformationMatrix
from ..world import World
from ..world_description.connections import FixedConnection
from ..world_description.geometry import TriangleMesh
from ..world_description.world_entity import Body, Region, KinematicStructureEntity

if TYPE_CHECKING:
    from ..robots.abstract_robot import (
        Camera,
    )


def stable(obj: Body) -> bool:
    """
    Checks if an object is stable in the world. Stable meaning that its position will not change after simulating
    physics in the World. This will be done by simulating the world for 10 seconds and comparing
    the previous coordinates with the coordinates after the simulation.

    :param obj: The object which should be checked
    :return: True if the given object is stable in the world False else
    """
    raise NotImplementedError("Needs multiverse")


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
    tcd = TrimeshCollisionDetector(body1._world)
    result = tcd.check_collision_between_bodies(body1, body2)

    if result is None:
        return False
    return result.contact_distance < threshold


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
    cam_pose[:3, 3] = camera.root.global_pose.to_np()[:3, 3]

    seg = rt.create_segmentation_mask(
        HomogeneousTransformationMatrix(cam_pose, reference_frame=camera._world.root),
        resolution=256,
        min_dist=0.2,
    )
    indices = np.unique(seg)
    indices = indices[indices > -1]
    bodies = [camera._world.kinematic_structure[i] for i in indices]

    return bodies


def visible(camera: Camera, obj: KinematicStructureEntity) -> bool:
    """
    Checks if a body/region is visible by the given camera.
    """
    return obj in get_visible_bodies(camera)


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
    camera_pose[:3, 3] = camera.root.global_pose.to_np()[:3, 3]
    camera_pose = HomogeneousTransformationMatrix(
        camera_pose, reference_frame=camera._world.root
    )

    # create a world only containing the target body
    world_without_occlusion = World()
    root = Body(name=PrefixedName("root"))
    with world_without_occlusion.modify_world():
        world_without_occlusion.add_body(root)
        copied_body = Body.from_json(body.to_json())
        root_T_body = body.global_pose
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
            camera_pose, resolution=256, min_dist=0.1
        )
    )

    # get segmentation mask with occlusion
    ray_tracer_with_occlusion = RayTracer(camera._world)
    ray_tracer_with_occlusion.update_scene()
    segmentation_mask_with_occlusion = (
        ray_tracer_with_occlusion.create_segmentation_mask(
            camera_pose, resolution=256, min_dist=0.1
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
    if Below(supported_body, supporting_body, supported_body.global_pose)():
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
    body_mesh = body_mesh_local.copy().apply_transform(body.global_pose.to_np())
    region_mesh = region_mesh_local.copy().apply_transform(region.global_pose.to_np())
    intersection = trimesh.boolean.intersection([body_mesh, region_mesh])

    # no body volume -> zero fraction
    body_volume = body_mesh.volume
    if body_volume <= 1e-12:
        return 0.0

    return intersection.volume / body_volume


@dataclass
class SpatialRelation(Symbol, ABC):
    """
    Check if the KSE is spatially related to the other KSE if you are looking from the point of semantic annotation.
    The comparison is done using the centers of mass computed from the KSE's collision geometry.
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
class ViewDependentSpatialRelation(SpatialRelation, ABC):

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
        Calculate the spatial relation between self.body and self.other with respect to a given
        reference point (self.point_of_semantic_annotation) and a specified axis index. This function computes the
        signed distance along a specified direction derived from the reference point
        to compare the positions of the centers of mass of the two bodies.

        :param index: The index of the axis in the transformation matrix along which
            the spatial relation is computed.
        :return: The signed distance between the first and the second body's centers
            of mass along the given direction.
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

        s_body = front_norm.dot(self.body.center_of_mass.to_vector3())
        s_other = front_norm.dot(self.other.center_of_mass.to_vector3())
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
class InsideOf(SpatialRelation):
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
        mesh_a.apply_transform(self.body.global_pose.to_np())

        mesh_b = mesh_b_local.copy()
        mesh_b.apply_transform(self.other.global_pose.to_np())

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
