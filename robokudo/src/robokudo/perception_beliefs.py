"""
What perception believes about the objects a pipeline pass found, and which perceivable
objects of the semantic digital twin those beliefs are about.

A belief matched to an object of the world moves it; any other belief becomes a new body
with the most specific semantic annotation its classification names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
from krrood.patterns.classproperty import classproperty
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Point3
from scipy.optimize import linear_sum_assignment
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    IsPerceivable,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    UnknownObject,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description import geometry
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

import robokudo.types.annotation
from robokudo.annotators.cluster_color import ClusterColorAnnotator, Color
from robokudo.defs import PACKAGE_NAME
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    Classification,
    SemanticColor,
)
from robokudo.types.scene import ObjectHypothesis

# %% beliefs


@dataclass(kw_only=True, eq=False)
class PerceptionBelief:
    """
    An object a finished pipeline pass found, as the semantic annotation perception
    believes it is, located in the world.
    """

    hypothesis: ObjectHypothesis
    """
    The hypothesis the object was found as.
    """

    body: Body
    """
    The body, of no world yet, that shows and collides with the shape the object was
    seen with.
    """

    pose: HomogeneousTransformationMatrix
    """
    The pose of the geometry the object was seen with, relative to the world frame of
    the pass; the pose :attr:`body` is placed at.
    """

    scale: geometry.Scale
    """
    The lengths of the geometry the object was seen with along the axes of its pose.
    """

    believed_class: type[IsPerceivable]
    """
    The semantic annotation the object is believed to be.
    """

    @classproperty
    def minimum_box_point_count(cls) -> int:
        """
        The fewest points a box in three dimensions can be fitted to.
        """
        return 4

    @classmethod
    def from_hypothesis(
        cls,
        *,
        hypothesis: ObjectHypothesis,
        world_T_camera: HomogeneousTransformationMatrix,
        minimum_class_confidence: float,
    ) -> PerceptionBelief | None:
        """
        :param hypothesis: The hypothesis a finished pass found.
        :param world_T_camera: The pose of the camera of that pass in the world.
        :param minimum_class_confidence: How confident a classification has to be for
            the object to be believed of the class it names.
        :return: The belief about ``hypothesis``, seen with the latest shape fitted to
            it or else the box :meth:`_compute_belief_bounding_box` computes; None, with
            a warning logged, if it has neither.
        """
        shape_annotation = hypothesis.latest_annotation_of_type(
            annotation_type=robokudo.types.annotation.Shape
        )
        seen_geometry = (
            shape_annotation.geometry
            if shape_annotation is not None
            else cls._compute_belief_bounding_box(hypothesis=hypothesis)
        )
        if seen_geometry is None:
            logging.getLogger(PACKAGE_NAME).warning(
                f"Leaving out an object hypothesis classified as "
                f"{hypothesis.class_label}: it has neither a fitted shape, a 3D bounding "
                f"box nor enough points to fit one to."
            )
            return None

        shape = replace(seen_geometry, origin=HomogeneousTransformationMatrix())
        color = cls._compute_dominant_belief_color(hypothesis=hypothesis)
        if color is not None:
            shape.color = color
        visual = ShapeCollection(shapes=[shape])
        return cls(
            hypothesis=hypothesis,
            body=Body(visual=visual, collision=visual.copy_without_reference_frame()),
            pose=world_T_camera @ seen_geometry.origin,
            scale=seen_geometry.local_frame_bounding_box.scale,
            believed_class=hypothesis.believed_semantic_class(
                minimum_class_confidence=minimum_class_confidence
            ),
        )

    @classmethod
    def _compute_belief_bounding_box(
        cls, *, hypothesis: ObjectHypothesis
    ) -> geometry.Box | None:
        """
        :param hypothesis: The hypothesis.
        :return: The box ``hypothesis`` was seen in, with its origin relative to the
            camera: its latest 3D bounding box, or else the box enclosing its points;
            None if it has neither a 3D bounding box nor
            :attr:`minimum_box_point_count` points.
        """
        bounding_box = hypothesis.latest_annotation_of_type(
            annotation_type=BoundingBox3DAnnotation
        )
        if bounding_box is not None:
            return geometry.Box(
                origin=HomogeneousTransformationMatrix.from_xyz_quaternion(
                    *bounding_box.pose.translation, *bounding_box.pose.rotation
                ),
                scale=geometry.Scale(
                    x=bounding_box.x_length,
                    y=bounding_box.y_length,
                    z=bounding_box.z_length,
                ),
            )
        if (
            hypothesis.points is None
            or len(hypothesis.points.points) < cls.minimum_box_point_count
        ):
            return None

        oriented_box = hypothesis.points.get_oriented_bounding_box(robust=True)
        points_T_box = np.eye(4)
        points_T_box[:3, :3] = oriented_box.R
        points_T_box[:3, 3] = oriented_box.center
        x_length, y_length, z_length = oriented_box.extent

        return geometry.Box(
            origin=HomogeneousTransformationMatrix(data=points_T_box),
            scale=geometry.Scale(x=x_length, y=y_length, z=z_length),
        )

    @staticmethod
    def _compute_dominant_belief_color(
        *, hypothesis: ObjectHypothesis
    ) -> geometry.Color | None:
        """
        :param hypothesis: The hypothesis.
        :return: The color covering the largest part of ``hypothesis``, or None if no
            color was annotated.
        """
        semantic_colors = [
            annotation
            for annotation in hypothesis.annotations
            if isinstance(annotation, SemanticColor)
        ]
        if not semantic_colors:
            return None
        dominant_color = max(semantic_colors, key=lambda color: color.ratio)
        blue, green, red = ClusterColorAnnotator.color_name_to_bgr_values[
            Color[dominant_color.color.upper()]
        ]
        return geometry.Color.from_rgba(
            red=red / 255, green=green / 255, blue=blue / 255
        )

    @property
    def center(self) -> Point3:
        """
        The position of the center of the object, relative to the root of the world.
        """
        return (self.pose.reference_frame.global_transform @ self.pose).position

    # %% changing the world

    def add_to_world(self, *, world: World) -> None:
        """
        Add :attr:`body` to ``world`` at the box the object was seen in, below the world
        frame of its pass, as an annotation of :attr:`believed_class`.

        :param world: The world the belief is about.
        """
        with world.modify_world():
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=world, parent=self.pose.reference_frame, child=self.body
                )
            )
            world.add_semantic_annotation(
                self.believed_class(
                    root=self.body, class_label=self.hypothesis.class_label
                )
            )
            self.body.parent_connection.origin = self.pose

    def update_object_in_world(
        self, *, object_in_world: IsPerceivable, world: World
    ) -> None:
        """
        Move ``object_in_world`` to where the object was seen, keeping its orientation
        and shape. An :class:`UnknownObject` becomes an annotation of
        :attr:`believed_class`; any other semantic annotation keeps its class.

        :param object_in_world: The object of ``world`` the belief was matched to.
        :param world: The world ``object_in_world`` belongs to.
        """
        self._move(body=object_in_world.root, world=world)
        if isinstance(object_in_world, UnknownObject):
            self._replace_unknown_object(unknown_object=object_in_world, world=world)
            return
        if self.believed_class is not UnknownObject and not isinstance(
            object_in_world, self.believed_class
        ):
            logging.getLogger(PACKAGE_NAME).info(
                f"Keeping {type(object_in_world).__name__} {object_in_world.root.name} "
                f"although it was classified as {self.believed_class.__name__}."
            )

    def _move(self, *, body: Body, world: World) -> None:
        """
        Move ``body`` so that its center lies where the object was seen, keeping its
        orientation. A body that cannot move relative to its parent stays where it is.

        :param body: The body of the object the belief was matched to.
        :param world: The world ``body`` belongs to.
        """
        if not isinstance(body.parent_connection, Connection6DoF):
            return
        root_T_body = body.global_transform
        body.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=root_T_body.position
                + (self.center - ObjectMatcher.visual_center(body=body)),
                rotation_matrix=root_T_body.rotation_matrix,
                reference_frame=world.root,
            )
        )

    def _replace_unknown_object(
        self, *, unknown_object: UnknownObject, world: World
    ) -> None:
        """
        Replace ``unknown_object`` by an annotation of :attr:`believed_class` rooted at
        its body, unless the object is believed to be an :class:`UnknownObject`.

        :param unknown_object: The object of ``world`` the belief was matched to.
        :param world: The world ``unknown_object`` belongs to.
        """
        if self.believed_class is UnknownObject:
            return
        with world.modify_world():
            world.remove_semantic_annotation(unknown_object)
            world.add_semantic_annotation(
                self.believed_class(
                    root=unknown_object.root, class_label=self.hypothesis.class_label
                )
            )


# %% matching


@dataclass(kw_only=True)
class ObjectMatcher:
    """
    Decides which perceivable object of the world each belief of a pass is about.

    A belief can only be about an object whose center lies within
    :attr:`maximum_match_distance` of its own and whose size is similar. Of all
    assignments of beliefs to such objects, the one taking the most beliefs wins, and
    among those the one with the shortest total distance, where an object of another
    class counts as :attr:`class_mismatch_penalty` farther away.
    """

    maximum_match_distance: float = 0.2
    """
    How far, in meters, the center of a belief may lie from the center of an object for
    the belief to be about that object.
    """

    maximum_match_size_ratio: float = 2.0
    """
    How many times longer the longest edge of a belief may be than that of an object, or
    the other way round, for the belief to be about that object.
    """

    class_mismatch_penalty: float = 0.2
    """
    How much farther, in meters, an object of another class than a belief counts as than
    it is.
    """

    def match(
        self, *, beliefs: list[PerceptionBelief], world: World
    ) -> dict[PerceptionBelief, IsPerceivable]:
        """
        :param beliefs: The beliefs of a pass.
        :param world: The world whose perceivable objects the beliefs can be about.
        :return: The object each matched belief is about. No object is taken by more
            than one belief.
        """
        objects_in_world = self._perceivable_objects(world=world)
        if not beliefs or not objects_in_world:
            return {}

        centers = [
            self.visual_center(body=object_in_world.root)
            for object_in_world in objects_in_world
        ]
        distances = np.array(
            [
                [float(belief.center.euclidean_distance(center)) for center in centers]
                for belief in beliefs
            ]
        )
        admissible = (distances <= self.maximum_match_distance) & np.array(
            [
                [
                    self._agree_in_size(belief=belief, body=object_in_world.root)
                    for object_in_world in objects_in_world
                ]
                for belief in beliefs
            ]
        )
        class_penalties = np.array(
            [
                [
                    (
                        self.class_mismatch_penalty
                        if self._differ_in_class(
                            belief=belief, object_in_world=object_in_world
                        )
                        else 0.0
                    )
                    for object_in_world in objects_in_world
                ]
                for belief in beliefs
            ]
        )
        # Costlier than all admissible pairs together, so that the assignment takes as
        # many admissible pairs as it can before it shortens their distances.
        inadmissible_cost = (
            self.maximum_match_distance + self.class_mismatch_penalty
        ) * (min(distances.shape) + 1)
        belief_indices, object_in_world_indices = linear_sum_assignment(
            np.where(admissible, distances + class_penalties, inadmissible_cost)
        )
        return {
            beliefs[belief_index]: objects_in_world[object_in_world_index]
            for belief_index, object_in_world_index in zip(
                belief_indices, object_in_world_indices
            )
            if admissible[belief_index, object_in_world_index]
        }

    @staticmethod
    def visual_center(*, body: Body) -> Point3:
        """
        :param body: The body of an object.
        :return: The position of the center of the visual shapes of ``body``, or of
            ``body`` itself if it has none, relative to the root of the world.
        """
        root_T_body = body.global_transform
        if not body.visual.shapes:
            return root_T_body.position
        bounding_box = body.visual.as_bounding_box_collection_in_frame(
            body
        ).bounding_box()
        return root_T_body @ bounding_box.center

    @staticmethod
    def _perceivable_objects(*, world: World) -> list[IsPerceivable]:
        """
        :param world: The world.
        :return: The perceivable objects of ``world`` a belief can be about, one per
            body.
        """
        perceivable_objects = {}
        for annotation in world.get_semantic_annotations_by_type(IsPerceivable):
            if not isinstance(annotation, HasRootBody):
                continue
            perceivable_objects.setdefault(annotation.root.id, annotation)
        return list(perceivable_objects.values())

    def _agree_in_size(self, *, belief: PerceptionBelief, body: Body) -> bool:
        """
        :param belief: The belief.
        :param body: The body of an object.
        :return: Whether the longest edges of the box ``belief`` was seen in and the
            visual shapes of ``body`` are similar; True if ``body`` has no visual
            shapes.
        """
        if not body.visual.shapes:
            return True
        body_scale = body.visual.scale
        belief_edge = max(belief.scale.x, belief.scale.y, belief.scale.z)
        body_edge = max(body_scale.x, body_scale.y, body_scale.z)
        return max(belief_edge, body_edge) <= self.maximum_match_size_ratio * min(
            belief_edge, body_edge
        )

    @staticmethod
    def _differ_in_class(
        *, belief: PerceptionBelief, object_in_world: IsPerceivable
    ) -> bool:
        """
        :param belief: The belief.
        :param object_in_world: The object.
        :return: Whether the object of ``belief`` was classified as something
            ``object_in_world`` is not: another semantic annotation, or, for a
            :class:`UnknownObject`, another class label.
        """
        class_label = belief.hypothesis.class_label
        if class_label is None:
            return False
        if isinstance(object_in_world, UnknownObject):
            return (
                object_in_world.class_label is not None
                and object_in_world.class_label != class_label
            )
        semantic_class = belief.hypothesis.semantic_class
        return semantic_class is None or not isinstance(object_in_world, semantic_class)
