from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import trimesh.boolean
from trimesh.collision import CollisionManager
from typing_extensions import List, TYPE_CHECKING, Iterable, Type

from krrood.entity_query_language.predicate import (
    Predicate,
    RenderedFields,
    Symbol,
    SymbolicFunction,
    symbolic_function,
    Triple,
)
from krrood.entity_query_language.utils import camel_case_to_words
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Prepositions
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Adjective,
    clause,
    Copula,
    Noun,
    Verb,
)
from krrood.inheritance_path_length import inheritance_path_length
from semantic_digital_twin.spatial_computations.ik_solver import (
    MaxIterationsException,
    UnreachableException,
)
from semantic_digital_twin.spatial_types import Vector3, Point3, math
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.world_entity import (
    Body,
    Region,
    KinematicStructureEntity,
    TBody,
    TRegion,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass(eq=False)
class InContactWith(Triple[TBody, TBody]):
    """
    Whether two bodies are touching, by how close their collision geometry comes.

    Touching is a judgement about a distance rather than a distance, so the distance is
    what :meth:`compute_distance` answers and :attr:`maximum_distance` is where the
    judgement is stated.
    """

    body1: TBody
    """
    The first body.
    """

    body2: TBody
    """
    The other body.
    """

    maximum_distance: float = 0.001
    """
    How close the two have to come before they count as touching, in metres.
    """

    @property
    def subject(self) -> TBody:
        return self.body1

    @property
    def object(self) -> TBody:
        return self.body2

    def __call__(self) -> bool:
        distance = self.compute_distance()
        return distance is not None and distance < self.maximum_distance

    def compute_distance(self) -> Optional[float]:
        """
        :return: How far apart the two bodies' collision geometry is, or ``None`` when
            the collision detector reports no result for the pair at all.
        """
        detector = self.body1._world.collision_manager.collision_detector
        result = detector.check_collision_between_bodies(
            self.body1, self.body2, distance=self.maximum_distance
        )
        if result is None:
            return None
        return result.distance

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        Reads as *"the body is in contact with the other body"*.

        :param fields: The rendered fragment for each field, keyed by field name.
        """
        return clause(
            Noun(fields["body1"]),
            Copula(),
            Prepositions.IN,
            Noun.bare("contact"),
            Prepositions.WITH,
            Noun(fields["body2"]),
        )


@dataclass(eq=False)
class Reachable(Predicate):
    """
    Whether a kinematic chain can put its tip at a pose, answered by inverse kinematics.
    """

    pose: HomogeneousTransformationMatrix
    """
    The pose to reach.
    """

    root: Body
    """
    The root of the kinematic chain.
    """

    tip: Body
    """
    The end of the kinematic chain that has to arrive at the pose.
    """

    maximum_iterations: int = 1000
    """
    How long the solver may search before the pose counts as out of reach.
    """

    def __call__(self) -> bool:
        try:
            self.root._world.compute_inverse_kinematics(
                root=self.root,
                tip=self.tip,
                target=self.pose,
                max_iterations=self.maximum_iterations,
            )
        except (MaxIterationsException, UnreachableException):
            return False
        return True

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        Reads as *"the pose is reachable by the tip"*.

        :param fields: The rendered fragment for each field, keyed by field name.
        """
        return clause(
            Noun(fields["pose"]),
            Copula(),
            Adjective("reachable"),
            Prepositions.BY,
            Noun(fields["tip"]),
        )


@symbolic_function
def compute_euclidean_planar_distance(
    body1: Body, body2: Body, ignore_dimension: Vector3
):
    """
    Computes the Euclidean distance between two bodies in 2D space, ignoring a specific
    dimension specified by the user. The ignored dimension is set to zero before the
    distance calculation. This function can be used to handle scenarios where
    computations are restricted to certain spatial planes.

    :param body1: The first body to compute the distance from. It uses the global pose
        of the body to extract the position.
    :param body2: The second body to compute the distance to. It also utilizes the
        global pose of the body to extract the position.
    :param ignore_dimension: Specifies which dimension (x, y, or z) should be ignored in
        the computation. The ignored dimension is set to zero for both positions prior
        to calculating the distance.
    :return: The Euclidean distance between the two bodies in the 2D plane after
        ignoring the specified dimension.
    """
    body1_position = body1.global_pose.position
    body2_position = body2.global_pose.position

    if np.allclose(ignore_dimension, Vector3.X()):
        body1_position.x = 0.0
        body2_position.x = 0.0
    elif np.allclose(ignore_dimension, Vector3.Y()):
        body1_position.y = 0.0
        body2_position.y = 0.0
    elif np.allclose(ignore_dimension, Vector3.Z()):
        body1_position.z = 0.0
        body2_position.z = 0.0

    return body1_position.euclidean_distance(body2_position)


@dataclass(eq=False)
class SupportedBy(Triple[TBody, TBody]):
    """
    Whether one body rests on another.

    An object rests on what touches it and pushes it up, which is read off how the two
    meet rather than from where their middles lie: a container carries its own middle
    above what stands on its floor, a wall's bounding box reaches far past the wall, and
    what touches an object only from the side does not hold it up. Up is the world's up.
    """

    supported: TBody
    """
    The body that may be resting.
    """

    supporting: TBody
    """
    The body that may be holding it up.
    """

    maximum_intersection_height: float = 0.1
    """
    How deep, in metres, the two may sink into each other. Sunk deeper, they are a
    clipping the simulation did not resolve, and the reading is refused.
    """

    contact_tolerance: float = 0.005
    """
    How far apart, in metres, the two may be and still count as touching, and so how
    far above the supporting body the supported body may stand and still rest on it.

    A body is set down by a motion that stops where it can rather than exactly on the
    surface, so a support read from overlapping volume alone would hold for almost no
    placement at all. Measured on a robot stacking boxes, a placement missed the surface
    it was aimed at by 1.9 mm; the default leaves room for that while staying far below
    the centimetres by which a body that is genuinely in the air clears a surface.
    """

    maximum_slope: float = np.radians(30.0)
    """
    How steeply, in radians, the supporting body may slope where the two meet and still
    hold the supported body up.
    """

    @property
    def subject(self) -> TBody:
        return self.supported

    @property
    def object(self) -> TBody:
        return self.supporting

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        Reads as *"the supported body is supported by the supporting body"*.

        :attr:`maximum_intersection_height` is a tolerance of the reading rather than
        part of the claim, so it is left unspoken.

        :param fields: The rendered fragment for each field, keyed by field name.
        """
        return clause(
            Noun(fields["supported"]),
            Copula(),
            Adjective("supported"),
            Prepositions.BY,
            Noun(fields["supporting"]),
        )

    def __call__(self) -> bool:
        if self.supported is self.supporting:
            return False

        collision_detector = self.supported._world.collision_manager.collision_detector
        touch = collision_detector.check_collision_between_bodies(
            self.supported, self.supporting, distance=self.contact_tolerance
        )
        if touch is None or touch.distance >= self.contact_tolerance:
            return False

        if touch.body_a is not self.supported:
            touch = touch.reverse()
        root_V_push = touch.root_V_contact_normal_from_b_to_a[:3]
        if root_V_push[2] < np.cos(self.maximum_slope) * np.linalg.norm(root_V_push):
            return False

        return -touch.distance < self.maximum_intersection_height


@dataclass(eq=False)
class Supports(Predicate):
    """
    Whether anything in the world rests on a body.
    """

    supporting_body: Body
    """
    The body that may be holding something up.
    """

    maximum_intersection_height: float = 0.1
    """
    How far two bodies may overlap vertically, in metres, before the reading is refused
    as unhandled clipping.
    """

    def __call__(self) -> bool:
        for candidate in self.supporting_body._world.bodies_with_collision:
            if candidate is self.supporting_body:
                continue
            if SupportedBy(
                candidate, self.supporting_body, self.maximum_intersection_height
            )():
                return True
        return False

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        Reads as *"the body is supporting a body"*, naming what is held up, which the
        class name leaves implicit.

        :param fields: The rendered fragment for each field, keyed by field name.
        """
        return clause(
            Noun(fields["supporting_body"]),
            Copula(),
            Adjective("supporting"),
            Noun("body"),
        )


@dataclass(eq=False)
class InsideRegion(Triple[TBody, TRegion]):
    """
    Whether a body lies in a region, by what fraction of its collision volume falls
    inside the region's area.

    How much counts as inside is a judgement rather than a measurement, so the fraction
    is what :meth:`compute_contained_fraction` answers and
    :attr:`minimum_contained_fraction` is where the judgement is stated.
    """

    body: TBody
    """
    The body that may be in the region.
    """

    region: TRegion
    """
    The region it may be in.
    """

    minimum_contained_fraction: float = 0.5
    """
    How much of the body has to lie inside the region before it counts as being in it.
    """

    @property
    def subject(self) -> TBody:
        return self.body

    @property
    def object(self) -> TRegion:
        return self.region

    def __call__(self) -> bool:
        return self.compute_contained_fraction() >= self.minimum_contained_fraction

    def compute_contained_fraction(self) -> float:
        """
        :return: The fraction (0.0..1.0) of the body's volume lying in the region.
        """
        # Retrieve meshes in local frames
        local_body_mesh = self.body.collision.combined_mesh
        local_region_mesh = self.region.area.combined_mesh

        # Transform copies of the meshes into the world frame
        body_mesh = local_body_mesh.copy().apply_transform(
            self.body.global_transform.to_np()
        )
        region_mesh = local_region_mesh.copy().apply_transform(
            self.region.global_transform.to_np()
        )
        intersection = trimesh.boolean.intersection([body_mesh, region_mesh])

        # no body volume -> zero fraction
        body_volume = body_mesh.volume
        if body_volume <= 1e-12:
            return 0.0

        return intersection.volume / body_volume

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        Reads as *"the body is inside the region"*.

        :param fields: The rendered fragment for each field, keyed by field name.
        """
        return clause(
            Noun(fields["body"]),
            Copula(),
            Prepositions.INSIDE,
            Noun(fields["region"]),
        )


@dataclass(eq=False)
class KinematicStructureEntitySpatialRelation(Predicate, ABC):
    """
    Base class for spatial relations between two KinematicStructureEntity instances.

    Implementations typically compare the centers of mass computed from the KSE's
    collision geometry.
    """

    body: KinematicStructureEntity
    """
    The KSE for which the check should be done.
    """

    other: KinematicStructureEntity
    """
    The other KSE.
    """


@dataclass(eq=False)
class PointSpatialRelation(Predicate, ABC):
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


@dataclass(eq=False)
class ViewDependentSpatialRelation(PointSpatialRelation, ABC):
    """
    A spatial relation between two points, read from somewhere in particular.

    Which way is left, above or in front depends on where it is being seen from, so the
    relation carries that spot as an operand of its own.
    """

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        Reads as *"the point is left of the other point, seen from the point of view"*,
        with the direction taken from the relation's own name.

        :param fields: The rendered fragment for each field, keyed by field name.
        """
        direction = camel_case_to_words(cls.__name__).lower()
        return clause(
            Noun(fields["point"]),
            Copula(),
            Adjective(direction),
            Noun(fields["other"]),
            Prepositions.FROM,
            Noun(fields["point_of_view"]),
        )

    point_of_view: HomogeneousTransformationMatrix
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
        Calculate the spatial relation between self.point and self.other with respect to
        a given reference point (self.point_of_semantic_annotation) and a specified axis
        index. This function computes the signed distance along a specified direction
        derived from the reference point to compare the positions.

        :param index: The index of the axis in the transformation matrix along which the
            spatial relation is computed.
        :return: The signed distance between the first and the second points along the
            given direction.
        """
        ref_np = self.point_of_view.to_np()
        front_world = ref_np[:3, index]
        front_norm = front_world / (np.linalg.norm(front_world) + self.eps)
        front_norm = Vector3(
            x=front_norm[0],
            y=front_norm[1],
            z=front_norm[2],
            reference_frame=self.point_of_view.reference_frame,
        )

        s_body = front_norm.dot(self.point.vector3)
        s_other = front_norm.dot(self.other.vector3)
        return (s_body - s_other).compile()()


@dataclass(eq=False)
class LeftOf(ViewDependentSpatialRelation):
    """
    The "left" direction is taken as the -Y axis of the given point of
    semantic_annotation.
    """

    def __call__(self) -> bool:
        self.spatial_relation_result = self._signed_distance_along_direction(1) > 0.0
        return self.spatial_relation_result


@dataclass(eq=False)
class RightOf(ViewDependentSpatialRelation):
    """
    The "right" direction is taken as the +Y axis of the given point of
    semantic_annotation.
    """

    def __call__(self) -> bool:
        self.spatial_relation_result = self._signed_distance_along_direction(1) < 0.0
        return self.spatial_relation_result


@dataclass(eq=False)
class Above(ViewDependentSpatialRelation):
    """
    The "above" direction is taken as the +Z axis of the given point of
    semantic_annotation.
    """

    def __call__(self) -> bool:
        self.spatial_relation_result = self._signed_distance_along_direction(2) > 0.0
        return self.spatial_relation_result


@dataclass(eq=False)
class Below(ViewDependentSpatialRelation):
    """
    The "below" direction is taken as the -Z axis of the given point of
    semantic_annotation.
    """

    def __call__(self) -> bool:
        self.spatial_relation_result = self._signed_distance_along_direction(2) < 0.0
        return self.spatial_relation_result


@dataclass(eq=False)
class Behind(ViewDependentSpatialRelation):
    """
    The "behind" direction is defined as the -X axis of the given point of semantic
    annotation.
    """

    def __call__(self) -> bool:
        self.spatial_relation_result = self._signed_distance_along_direction(0) < 0.0
        return self.spatial_relation_result


@dataclass(eq=False)
class InFrontOf(ViewDependentSpatialRelation):
    """
    The "in front of" direction is defined as the +X axis of the given point of semantic
    annotation.
    """

    def __call__(self) -> bool:
        self.result = self._signed_distance_along_direction(0) > 0.0
        return self.result


@dataclass(eq=False)
class InsideOf(KinematicStructureEntitySpatialRelation):
    """
    Whether one thing lies inside another, by what fraction of its volume falls within
    the other's bounding box.

    How much counts as inside is a judgement rather than a measurement, so the fraction
    is what :meth:`compute_containment_ratio` answers and
    :attr:`minimum_containment_ratio` is where the judgement is stated.
    """

    minimum_containment_ratio: float = 0.5
    """
    How much of the body has to lie inside the other before it counts as being in it.

    Half by default: a thing more than half swallowed is in, and one less than half
    swallowed is merely overlapping. Callers that want the fraction itself rather than a
    verdict read :meth:`compute_containment_ratio`.
    """

    containment_ratio: float = 0.0
    """
    What the last call measured, kept so a caller can read it off the relation it just
    evaluated.
    """

    def __call__(self) -> bool:
        self.containment_ratio = self.compute_containment_ratio()
        return self.containment_ratio >= self.minimum_containment_ratio

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

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        Reads as *"the body is inside the other"*.

        :param fields: The rendered fragment for each field, keyed by field name.
        """
        return clause(
            Noun(fields["body"]),
            Copula(),
            Prepositions.INSIDE,
            Noun(fields["other"]),
        )


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

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(
            Noun(fields["iterable"]),
            Verb("contain"),
            Noun("instance"),
            Prepositions.OF,
            Noun(fields["obj_type"]),
        )


@dataclass(eq=False)
class PlaceIsOccupied(Predicate):
    """
    Whether anything already stands in a stretch of the world.

    The stretch is a box at a pose, tested against every collidable body's own collision
    mesh.
    """

    box: VolumetricBoundingBox
    """
    The stretch of space asked about, in its own local frame.
    """

    pose: Pose
    """
    Where that box stands.
    """

    world: World
    """
    The world whose collidable bodies are tested against it.
    """

    allowed_bodies: Optional[List[Body]] = None
    """
    Bodies that may stand there without the place counting as occupied.
    """

    def __call__(self) -> bool:
        ignored = set(self.allowed_bodies or [])

        # Build a mesh for the region box at its current pose
        region_box_shape = self.box.as_shape()  # returns a Box centered at the region
        region_mesh = region_box_shape.mesh.copy()
        region_mesh.apply_transform(
            self.world.transform(self.pose, self.world.root).to_np()
        )

        # Prepare collision manager with the region mesh
        cm = CollisionManager()
        cm.add_object("region", region_mesh)

        # Iterate over collidable bodies and test collision
        for body in self.world.bodies_with_collision:
            if body in ignored:
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

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        Reads as *"the place is occupied"*.

        :param fields: The rendered fragment for each field, keyed by field name.
        """
        return clause(Noun(fields["box"]), Copula(), Adjective("occupied"))


@symbolic_function
def allclose(array1: np.ndarray, array2: np.ndarray, atol=1e-3) -> bool:
    """
    Symbolic wrapper around `np.allclose`.
    """
    return np.allclose(array1, array2, atol=atol)
