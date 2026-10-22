"""
The objects perception found, kept as semantic annotations in the semantic digital twin.

After a pipeline pass, each object it found is matched to a perceivable object of the
world. A matched object moves to where it was seen; any other object becomes a new body
with the most specific semantic annotation its classification names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from uuid import UUID

import numpy as np
from scipy.optimize import linear_sum_assignment
from semantic_digital_twin.reasoning.queries import annotation_class_by_label
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    IsPerceivable,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    PerceivedObject,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.geometry import Color as SemDTColor
from semantic_digital_twin.world_description.geometry import Shape as SemDTShape
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.annotators.cluster_color import ClusterColorAnnotator, Color
from robokudo.cas import CAS, CASViews
from robokudo.defs import PACKAGE_NAME
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    Classification,
    PoseAnnotation,
    SemanticColor,
    Shape,
    StampedPoseAnnotation,
)
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import (
    get_camera_to_world_transform_matrix,
    transform_pose_from_camera_to_world,
)
from robokudo.utils.transform import get_transform_matrix_from_q

# %% detections


@dataclass(eq=False)
class Detection:
    """
    An object a finished pipeline pass found, located in the world.
    """

    hypothesis: ObjectHypothesis
    """
    The hypothesis the object was found as.
    """

    pose: HomogeneousTransformationMatrix
    """
    The pose of the 3D bounding box of the object, relative to the world frame of the
    pass.
    """

    extent: Scale
    """
    The lengths of the 3D bounding box of the object along its axes.
    """

    shape: SemDTShape
    """
    The shape the object was seen with, relative to its 3D bounding box.
    """

    semantic_class: type[IsPerceivable] | None
    """
    The semantic annotation the class label of the object names, or None if it names
    none or the object is unclassified.
    """

    @property
    def center(self) -> np.ndarray:
        """
        The position of the center of the object, relative to the root of the world.
        """
        root_T_object = (
            self.pose.reference_frame.global_transform.to_np() @ self.pose.to_np()
        )
        return root_T_object[:3, 3]

    @property
    def class_label(self) -> str | None:
        """
        The class the object was classified as, or None if it is unclassified.
        """
        return self.hypothesis.class_label

    @property
    def class_confidence(self) -> float:
        """
        How confident the latest classification of the object is; 0 if it is
        unclassified.
        """
        classification = self.hypothesis.latest_annotation_of_type(Classification)
        if classification is None:
            return 0.0
        return classification.confidence


# %% matching


@dataclass
class ObjectMatcher:
    """
    Decides which perceivable object of the world each detection of a pass is.

    A detection can only be taken as an object whose center lies within
    :attr:`maximum_match_distance` of its own and whose size is similar. Of all
    assignments of detections to such objects, the one taking the most detections wins,
    and among those the one with the shortest total distance, where an object of another
    class counts as :attr:`class_mismatch_penalty` farther away.
    """

    maximum_match_distance: float = 0.2
    """
    How far, in meters, the center of a detection may lie from the center of an object
    to be taken as that object.
    """

    maximum_match_size_ratio: float = 2.0
    """
    How many times longer the longest edge of a detection may be than that of an object,
    or the other way round, to be taken as that object.
    """

    class_mismatch_penalty: float = 0.2
    """
    How much farther, in meters, an object of another class than a detection counts as
    than it is.
    """

    def match(
        self, detections: list[Detection], known_objects: list[IsPerceivable]
    ) -> dict[Detection, IsPerceivable]:
        """
        :param detections: The detections of a pass.
        :param known_objects: The perceivable objects the detections can be taken as,
            each rooted at a body.
        :return: The object each matched detection is taken as. No object is taken by
            more than one detection.
        """
        if not detections or not known_objects:
            return {}

        centers = [self.center_of(known_object.root) for known_object in known_objects]
        distances = np.array(
            [
                [np.linalg.norm(detection.center - center) for center in centers]
                for detection in detections
            ]
        )
        admissible = (distances <= self.maximum_match_distance) & np.array(
            [
                [
                    self._agree_in_size(detection, known_object.root)
                    for known_object in known_objects
                ]
                for detection in detections
            ]
        )
        class_penalties = np.array(
            [
                [
                    (
                        self.class_mismatch_penalty
                        if self._differ_in_class(detection, known_object)
                        else 0.0
                    )
                    for known_object in known_objects
                ]
                for detection in detections
            ]
        )
        # Costlier than all admissible pairs together, so that the assignment takes as
        # many admissible pairs as it can before it shortens their distances.
        inadmissible_cost = (
            self.maximum_match_distance + self.class_mismatch_penalty
        ) * (min(distances.shape) + 1)
        detection_indices, known_object_indices = linear_sum_assignment(
            np.where(admissible, distances + class_penalties, inadmissible_cost)
        )
        return {
            detections[detection_index]: known_objects[known_object_index]
            for detection_index, known_object_index in zip(
                detection_indices, known_object_indices
            )
            if admissible[detection_index, known_object_index]
        }

    @staticmethod
    def center_of(body: Body) -> np.ndarray:
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

    def _agree_in_size(self, detection: Detection, body: Body) -> bool:
        """
        :param detection: The detection.
        :param body: The body of an object.
        :return: Whether the longest edges of ``detection`` and the visual shapes of
            ``body`` are similar; True if ``body`` has no visual shapes.
        """
        if not body.visual.shapes:
            return True
        known_extent = body.visual.scale
        detection_edge = max(detection.extent.x, detection.extent.y, detection.extent.z)
        known_edge = max(known_extent.x, known_extent.y, known_extent.z)
        return max(detection_edge, known_edge) <= self.maximum_match_size_ratio * min(
            detection_edge, known_edge
        )

    @staticmethod
    def _differ_in_class(detection: Detection, known_object: IsPerceivable) -> bool:
        """
        :param detection: The detection.
        :param known_object: The object.
        :return: Whether ``detection`` was classified as something ``known_object`` is
            not: another semantic annotation, or, for a :class:`PerceivedObject`,
            another class label.
        """
        if detection.class_label is None:
            return False
        if isinstance(known_object, PerceivedObject):
            return (
                known_object.class_label is not None
                and known_object.class_label != detection.class_label
            )
        return detection.semantic_class is None or not isinstance(
            known_object, detection.semantic_class
        )


# %% perceived objects


@dataclass
class PerceivedObjects:
    """
    Puts the objects perception finds into :attr:`world`, each as a body with a
    perceivable semantic annotation.
    """

    world: World
    """
    The world the objects are kept in.
    """

    matcher: ObjectMatcher = field(default_factory=ObjectMatcher)
    """
    Decides which perceivable object of :attr:`world` each found object is.
    """

    minimum_class_confidence: float = 0.5
    """
    How confident a classification has to be for an object to get the semantic
    annotation it names.
    """

    def apply(self, cas: CAS) -> list[UUID | None]:
        """
        Put the objects a finished pipeline pass found into :attr:`world`.

        Each object is annotated with its pose in the world frame and matched to a
        perceivable object of :attr:`world`. A matched object moves to where it was
        seen, keeping its orientation and shape, and a :class:`PerceivedObject` gets the
        semantic annotation a confident classification names. Any other object becomes a
        new body with the shape and color it was seen with, annotated with the semantic
        annotation a confident classification names, or as a :class:`PerceivedObject`.
        An object that cannot be located in :attr:`world`, because ``cas`` holds no
        camera pose in it or the object has no 3D bounding box, gets no body.

        :param cas: The CAS of the finished pass.
        :return: The ids of the bodies of the found objects, in the order of the objects
            in ``cas``; None for an object that got no body.
        """
        hypotheses = cas.filter_annotations_by_type(ObjectHypothesis)
        self._annotate_world_poses(hypotheses, cas)
        detections = [self._locate(hypothesis, cas) for hypothesis in hypotheses]
        matches = self.matcher.match(
            [detection for detection in detections if detection is not None],
            self._perceivable_objects(),
        )

        body_ids = []
        for detection in detections:
            if detection is None:
                body_ids.append(None)
                continue
            known_object = matches.get(detection)
            if known_object is None:
                body_ids.append(self._add_object(detection).root.id)
                continue
            self._move(known_object.root, detection)
            self._classify(known_object, detection)
            body_ids.append(known_object.root.id)
        return body_ids

    # %% locating

    def _annotate_world_poses(
        self, hypotheses: list[ObjectHypothesis], cas: CAS
    ) -> None:
        """
        Annotate each hypothesis with its latest pose, in the world frame.

        :param hypotheses: The hypotheses to annotate.
        :param cas: The CAS whose camera pose locates the hypotheses.
        """
        if cas.camera_to_world_transform is None:
            return

        for hypothesis in hypotheses:
            pose = hypothesis.latest_annotation_of_type(PoseAnnotation)
            if pose is None:
                continue
            world_pose = transform_pose_from_camera_to_world(cas, pose)
            stamped_pose = StampedPoseAnnotation()
            stamped_pose.source = type(self).__name__
            stamped_pose.translation = world_pose.translation
            stamped_pose.rotation = world_pose.rotation
            if cas.world_frame is not None:
                stamped_pose.frame = cas.world_frame
            if cas.contains(CASViews.CAMERA_INFO):
                stamped_pose.timestamp = cas.get(CASViews.CAMERA_INFO).header.stamp
            hypothesis.annotations.append(stamped_pose)

    def _locate(self, hypothesis: ObjectHypothesis, cas: CAS) -> Detection | None:
        """
        :param hypothesis: The hypothesis to locate.
        :param cas: The CAS whose camera pose locates the hypothesis.
        :return: ``hypothesis`` located at its latest 3D bounding box in the world frame
            of ``cas``, or None if ``cas`` holds no camera pose relative to a body of
            :attr:`world`, or ``hypothesis`` has no 3D bounding box.
        """
        bounding_box = hypothesis.latest_annotation_of_type(BoundingBox3DAnnotation)
        camera_to_world_transform = cas.camera_to_world_transform
        if camera_to_world_transform is None or bounding_box is None:
            return None
        world_body = camera_to_world_transform.reference_frame
        if world_body not in self.world.kinematic_structure_entities:
            return None

        camera_T_bounding_box = get_transform_matrix_from_q(
            bounding_box.pose.rotation, bounding_box.pose.translation
        )
        extent = Scale(
            x=bounding_box.x_length,
            y=bounding_box.y_length,
            z=bounding_box.z_length,
        )
        return Detection(
            hypothesis=hypothesis,
            pose=HomogeneousTransformationMatrix(
                data=get_camera_to_world_transform_matrix(cas) @ camera_T_bounding_box,
                reference_frame=world_body,
            ),
            extent=extent,
            shape=self._shape_of(hypothesis, camera_T_bounding_box, extent),
            semantic_class=(
                annotation_class_by_label(hypothesis.class_label)
                if hypothesis.class_label is not None
                else None
            ),
        )

    def _shape_of(
        self,
        hypothesis: ObjectHypothesis,
        camera_T_bounding_box: np.ndarray,
        extent: Scale,
    ) -> SemDTShape:
        """
        :param hypothesis: The hypothesis.
        :param camera_T_bounding_box: The pose of the 3D bounding box of ``hypothesis``
            relative to the camera.
        :param extent: The lengths of the 3D bounding box of ``hypothesis``.
        :return: The latest shape fitted to ``hypothesis``, or its 3D bounding box if
            none was, relative to the bounding box and in the dominant color of
            ``hypothesis``, if it has one.
        """
        fitted_shape = hypothesis.latest_annotation_of_type(Shape)
        if fitted_shape is None:
            shape = Box(scale=extent)
        else:
            shape = replace(
                fitted_shape.geometry,
                origin=HomogeneousTransformationMatrix(
                    data=np.linalg.inv(camera_T_bounding_box)
                    @ fitted_shape.geometry.origin.to_np()
                ),
            )
        color = self._dominant_color_of(hypothesis)
        if color is not None:
            shape.color = color
        return shape

    @staticmethod
    def _dominant_color_of(hypothesis: ObjectHypothesis) -> SemDTColor | None:
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
        blue_green_red = ClusterColorAnnotator.color_name_to_bgr_values[
            Color[dominant_color.color.upper()]
        ]
        return SemDTColor.from_rgb([value / 255 for value in reversed(blue_green_red)])

    # %% objects of the world

    def _perceivable_objects(self) -> list[IsPerceivable]:
        """
        :return: The perceivable objects of :attr:`world` a found object can be taken
            as, one per body.
        """
        perceivable_objects = {}
        for annotation in self.world.get_semantic_annotations_by_type(IsPerceivable):
            if not isinstance(annotation, HasRootBody):
                continue
            perceivable_objects.setdefault(annotation.root.id, annotation)
        return list(perceivable_objects.values())

    def _move(self, body: Body, detection: Detection) -> None:
        """
        Move ``body`` so that its center lies where ``detection`` saw it, keeping its
        orientation. A body that cannot move relative to its parent stays where it is.

        :param body: The body of the object ``detection`` was taken as.
        :param detection: The detection.
        """
        if not isinstance(body.parent_connection, Connection6DoF):
            return
        root_T_body = body.global_transform.to_np().copy()
        root_T_body[:3, 3] += detection.center - self.matcher.center_of(body)
        body.parent_connection.origin = HomogeneousTransformationMatrix(
            data=root_T_body, reference_frame=self.world.root
        )

    def _classify(self, known_object: IsPerceivable, detection: Detection) -> None:
        """
        Replace ``known_object`` by the semantic annotation a confident classification
        of ``detection`` names, if ``known_object`` is a :class:`PerceivedObject`. Any
        other semantic annotation keeps its class.

        :param known_object: The object ``detection`` was taken as.
        :param detection: The detection.
        """
        semantic_class = self._confident_semantic_class(detection)
        if semantic_class is None or isinstance(known_object, semantic_class):
            return
        if not isinstance(known_object, PerceivedObject):
            logging.getLogger(PACKAGE_NAME).info(
                f"Keeping {type(known_object).__name__} {known_object.root.name} "
                f"although it was classified as {semantic_class.__name__}."
            )
            return
        with self.world.modify_world():
            self.world.remove_semantic_annotation(known_object)
            self.world.add_semantic_annotation(
                semantic_class(
                    root=known_object.root, class_label=detection.class_label
                )
            )

    def _add_object(self, detection: Detection) -> IsPerceivable:
        """
        Add the object ``detection`` saw to :attr:`world`, as a body at its 3D bounding
        box below the world frame of its pass that shows and collides with the shape it
        was seen with, annotated with the semantic annotation a confident classification
        names, or as a :class:`PerceivedObject`.

        :param detection: The detection.
        :return: The annotation of the new object.
        """
        visual = ShapeCollection(shapes=[detection.shape])
        body = Body(visual=visual, collision=visual.copy_without_reference_frame())
        semantic_class = self._confident_semantic_class(detection) or PerceivedObject
        annotation = semantic_class(root=body, class_label=detection.class_label)
        with self.world.modify_world():
            self.world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=self.world,
                    parent=detection.pose.reference_frame,
                    child=body,
                )
            )
            self.world.add_semantic_annotation(annotation)
        body.parent_connection.origin = detection.pose
        return annotation

    def _confident_semantic_class(
        self, detection: Detection
    ) -> type[IsPerceivable] | None:
        """
        :param detection: The detection.
        :return: The semantic annotation the classification of ``detection`` names, or
            None if it names none or is less confident than
            :attr:`minimum_class_confidence`.
        """
        if detection.class_confidence < self.minimum_class_confidence:
            return None
        return detection.semantic_class
