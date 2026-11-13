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
from krrood.ormatic.utils import classproperty
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from scipy.optimize import linear_sum_assignment
from semantic_digital_twin.reasoning.queries import annotation_class_by_label
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    IsPerceivable,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Human,
    PerceivedObject,
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
from robokudo.types.scene import HumanHypothesis, ObjectHypothesis

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

    annotation: IsPerceivable
    """
    The semantic annotation the object is believed to be, rooted at a body of no world
    yet that shows and collides with the shape the object was seen with.
    """

    pose: HomogeneousTransformationMatrix
    """
    The pose of the geometry the object was seen with, relative to the world frame of
    the pass; the pose its body is placed at.
    """

    scale: geometry.Scale
    """
    The lengths of the geometry the object was seen with along the axes of its pose.
    """

    semantic_class: type[IsPerceivable] | None
    """
    The semantic annotation the object is named as, however confident its
    classification is: :class:`Human` for a human, else the one its class label names;
    None if the label names none or the object is unclassified.
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
        A human becomes a :class:`Human`; any other object becomes the semantic
        annotation its classification names if that classification is at least
        ``minimum_class_confidence`` confident, else a :class:`PerceivedObject`.

        :param hypothesis: The hypothesis a finished pass found.
        :param world_T_camera: The pose of the camera of that pass in the world.
        :param minimum_class_confidence: How confident a classification has to be.
        :return: The belief about ``hypothesis``, seen with the latest shape fitted to
            it or else the box :meth:`_compute_belief_bounding_box` computes; None, with
            a warning logged, if it has neither.
        """
        shape_annotation = hypothesis.latest_annotation_of_type(
            robokudo.types.annotation.Shape
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
        body = Body(visual=visual, collision=visual.copy_without_reference_frame())

        if isinstance(hypothesis, HumanHypothesis):
            semantic_class = Human
            believed_class = Human
        else:
            semantic_class = (
                annotation_class_by_label(hypothesis.class_label)
                if hypothesis.class_label is not None
                else None
            )
            classification = hypothesis.latest_annotation_of_type(Classification)
            is_confidently_classified = (
                classification is not None
                and classification.confidence >= minimum_class_confidence
            )
            believed_class = (
                semantic_class
                if semantic_class is not None and is_confidently_classified
                else PerceivedObject
            )
        return cls(
            hypothesis=hypothesis,
            annotation=believed_class(root=body, class_label=hypothesis.class_label),
            pose=world_T_camera @ seen_geometry.origin,
            scale=seen_geometry.local_frame_bounding_box.scale,
            semantic_class=semantic_class,
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
        bounding_box = hypothesis.latest_annotation_of_type(BoundingBox3DAnnotation)
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
    def _compute_dominant_belief_color(*, hypothesis: ObjectHypothesis) -> geometry.Color | None:
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
    def center(self) -> np.ndarray:
        """
        The position of the center of the object, relative to the root of the world.
        """
        root_T_object = (
            self.pose.reference_frame.global_transform.to_np() @ self.pose.to_np()
        )
        return root_T_object[:3, 3]

    # %% changing the world

    def add_to_world(self, *, world: World) -> None:
        """
        Add :attr:`annotation` and its body to ``world``, the body at the box the object
        was seen in, below the world frame of its pass.

        :param world: The world the belief is about.
        """
        body = self.annotation.root
        with world.modify_world():
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=world, parent=self.pose.reference_frame, child=body
                )
            )
            world.add_semantic_annotation(self.annotation)
        body.parent_connection.origin = self.pose

    def update_known_object(self, *, known_object: IsPerceivable, world: World) -> None:
        """
        Move ``known_object`` to where the object was seen, keeping its orientation and
        shape, and turn it into the semantic annotation of :attr:`annotation` if it is a
        :class:`PerceivedObject`. Any other semantic annotation keeps its class.

        :param known_object: The object of ``world`` the belief was matched to.
        :param world: The world ``known_object`` belongs to.
        """
        self._move(body=known_object.root, world=world)
        self._classify(known_object=known_object, world=world)

    def _move(self, *, body: Body, world: World) -> None:
        """
        Move ``body`` so that its center lies where the object was seen, keeping its
        orientation. A body that cannot move relative to its parent stays where it is.

        :param body: The body of the object the belief was matched to.
        :param world: The world ``body`` belongs to.
        """
        if not isinstance(body.parent_connection, Connection6DoF):
            return
        root_T_body = body.global_transform.to_np().copy()
        root_T_body[:3, 3] += self.center - ObjectMatcher.visual_center(body=body)
        body.parent_connection.origin = HomogeneousTransformationMatrix(
            data=root_T_body, reference_frame=world.root
        )

    def _classify(self, *, known_object: IsPerceivable, world: World) -> None:
        """
        Replace ``known_object`` by an annotation of the class of :attr:`annotation`
        rooted at its body, if ``known_object`` is a :class:`PerceivedObject` and
        :attr:`annotation` is not.

        :param known_object: The object of ``world`` the belief was matched to.
        :param world: The world ``known_object`` belongs to.
        """
        believed_class = type(self.annotation)
        if believed_class is PerceivedObject or isinstance(
            known_object, believed_class
        ):
            return
        if not isinstance(known_object, PerceivedObject):
            logging.getLogger(PACKAGE_NAME).info(
                f"Keeping {type(known_object).__name__} {known_object.root.name} "
                f"although it was classified as {believed_class.__name__}."
            )
            return
        with world.modify_world():
            world.remove_semantic_annotation(known_object)
            world.add_semantic_annotation(
                believed_class(
                    root=known_object.root, class_label=self.annotation.class_label
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
        known_objects = self._perceivable_objects(world=world)
        if not beliefs or not known_objects:
            return {}

        centers = [self.visual_center(body=known.root) for known in known_objects]
        distances = np.array(
            [
                [np.linalg.norm(belief.center - center) for center in centers]
                for belief in beliefs
            ]
        )
        admissible = (distances <= self.maximum_match_distance) & np.array(
            [
                [
                    self._agree_in_size(belief=belief, body=known.root)
                    for known in known_objects
                ]
                for belief in beliefs
            ]
        )
        class_penalties = np.array(
            [
                [
                    (
                        self.class_mismatch_penalty
                        if self._differ_in_class(belief=belief, known_object=known)
                        else 0.0
                    )
                    for known in known_objects
                ]
                for belief in beliefs
            ]
        )
        # Costlier than all admissible pairs together, so that the assignment takes as
        # many admissible pairs as it can before it shortens their distances.
        inadmissible_cost = (
            self.maximum_match_distance + self.class_mismatch_penalty
        ) * (min(distances.shape) + 1)
        belief_indices, known_object_indices = linear_sum_assignment(
            np.where(admissible, distances + class_penalties, inadmissible_cost)
        )
        return {
            beliefs[belief_index]: known_objects[known_object_index]
            for belief_index, known_object_index in zip(
                belief_indices, known_object_indices
            )
            if admissible[belief_index, known_object_index]
        }

    @staticmethod
    def visual_center(*, body: Body) -> np.ndarray:
        """
        :param body: The body of an object.
        :return: The position of the center of the visual shapes of ``body``, or of
            ``body`` itself if it has none, relative to the root of the world.
        """
        root_T_body = body.global_transform.to_np()
        if not body.visual.shapes:
            return root_T_body[:3, 3]
        bounding_box = body.visual.as_bounding_box_collection_in_frame(
            body
        ).bounding_box()
        return (root_T_body @ bounding_box.center.to_np())[:3]

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
        known_scale = body.visual.scale
        belief_edge = max(belief.scale.x, belief.scale.y, belief.scale.z)
        known_edge = max(known_scale.x, known_scale.y, known_scale.z)
        return max(belief_edge, known_edge) <= self.maximum_match_size_ratio * min(
            belief_edge, known_edge
        )

    @staticmethod
    def _differ_in_class(
        *, belief: PerceptionBelief, known_object: IsPerceivable
    ) -> bool:
        """
        :param belief: The belief.
        :param known_object: The object.
        :return: Whether the object of ``belief`` was classified as something
            ``known_object`` is not: another semantic annotation, or, for a
            :class:`PerceivedObject`, another class label.
        """
        class_label = belief.annotation.class_label
        if class_label is None:
            return False
        if isinstance(known_object, PerceivedObject):
            return (
                known_object.class_label is not None
                and known_object.class_label != class_label
            )
        return belief.semantic_class is None or not isinstance(
            known_object, belief.semantic_class
        )
