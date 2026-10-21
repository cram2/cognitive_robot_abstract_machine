"""
The objects perception found, kept as bodies in the semantic digital twin.

After a pipeline pass, each object it found is matched to a known object: one perception
already believes in, or a perceivable object of the world. A matched object moves to
where it was seen; any other object becomes a new body.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from scipy.optimize import linear_sum_assignment
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    IsPerceivable,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.cas import CAS, CASViews
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    PoseAnnotation,
    StampedPoseAnnotation,
)
from robokudo.types.belief_state import ObjectBeliefState
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

    @property
    def center(self) -> np.ndarray:
        """
        The position of the center of the object, relative to the root of the world.
        """
        root_T_object = (
            self.pose.reference_frame.global_transform.to_np() @ self.pose.to_np()
        )
        return root_T_object[:3, 3]


# %% known objects


@dataclass
class KnownObject(ABC):
    """
    An object of the world a detection can be taken as.
    """

    world: World
    """
    The world the object is kept in.
    """

    @property
    @abstractmethod
    def body(self) -> Body:
        """
        The body of the object.
        """

    @property
    @abstractmethod
    def class_label(self) -> str | None:
        """
        The class the object is known as, or None if its class is unknown.
        """

    @abstractmethod
    def update(self, detection: Detection) -> None:
        """
        Update the object to ``detection``, the object seen again.

        :param detection: The detection taken as this object.
        """

    @property
    def center(self) -> np.ndarray:
        """
        The position of the center of the visual shapes of the object, or of its body if
        it has none, relative to the root of the world.
        """
        root_T_body = self.body.global_transform.to_np()
        if not self.body.visual.shapes:
            return root_T_body[:3, 3]
        bounding_box = self.body.visual.as_bounding_box_collection_in_frame(
            self.body
        ).bounding_box()
        return (root_T_body @ bounding_box.center.to_np())[:3]

    @property
    def extent(self) -> Scale | None:
        """
        The lengths of the visual shapes of the object along the axes of its body, or
        None if it has none.
        """
        if not self.body.visual.shapes:
            return None
        return self.body.visual.scale


@dataclass
class BelievedObject(KnownObject):
    """
    An object perception believes in, kept as a body perception created.
    """

    belief: ObjectBeliefState
    """
    The belief in the object.
    """

    @property
    def body(self) -> Body:
        return self.belief.body

    @property
    def class_label(self) -> str | None:
        return self.belief.latest_hypothesis.class_label

    def update(self, detection: Detection) -> None:
        """
        :meth:`KnownObject.update` by moving and reshaping the body to the 3D bounding
        box of ``detection``. A body that cannot move relative to its parent, such as
        one another process attached rigidly, keeps its pose and shape.

        .. note:: Only the move reaches the processes :attr:`world` is shared with;
            they keep the shape the object was first believed in with.
        """
        self.belief.add_hypothesis(detection.hypothesis)
        if not isinstance(self.body.parent_connection, Connection6DoF):
            return
        with self.world.modify_world(publish_changes=False):
            self.body.visual.shapes.clear()
            self.body.visual.append(Box(scale=detection.extent))
        self.body.parent_connection.origin = detection.pose


@dataclass
class PerceivableObject(KnownObject):
    """
    An object of the world annotated as perceivable, kept as a body perception did not
    create.
    """

    annotation: IsPerceivable
    """
    The annotation that marks the object as perceivable, rooted at its body.
    """

    @property
    def body(self) -> Body:
        return self.annotation.root

    @property
    def class_label(self) -> str | None:
        return self.annotation.class_label

    def update(self, detection: Detection) -> None:
        """
        :meth:`KnownObject.update` by moving the body so that its center lies where
        ``detection`` saw it, keeping its orientation and shape.

        A body that cannot move relative to its parent stays where it is.
        """
        if not isinstance(self.body.parent_connection, Connection6DoF):
            return
        root_T_body = self.body.global_transform.to_np().copy()
        root_T_body[:3, 3] += detection.center - self.center
        self.body.parent_connection.origin = HomogeneousTransformationMatrix(
            data=root_T_body, reference_frame=self.world.root
        )


# %% matching


@dataclass
class ObjectMatcher:
    """
    Decides which known object each detection of a pass is.

    A detection can only be taken as a known object whose center lies within
    :attr:`maximum_match_distance` of its own, whose class label does not differ from
    its own, and whose size is similar. Of all assignments of detections to such known
    objects, the one taking the most detections wins, and among those the one with the
    shortest total distance.
    """

    maximum_match_distance: float = 0.2
    """
    How far, in meters, the center of a detection may lie from the center of a known
    object to be taken as that object.
    """

    maximum_match_size_ratio: float = 2.0
    """
    How many times longer the longest edge of a detection may be than that of a known
    object, or the other way round, to be taken as that object.
    """

    def match(
        self, detections: list[Detection], known_objects: list[KnownObject]
    ) -> dict[Detection, KnownObject]:
        """
        :param detections: The detections of a pass.
        :param known_objects: The objects the detections can be taken as.
        :return: The known object each matched detection is taken as. No known object is
            taken by more than one detection.
        """
        if not detections or not known_objects:
            return {}

        distances = np.array(
            [
                [
                    np.linalg.norm(detection.center - known_object.center)
                    for known_object in known_objects
                ]
                for detection in detections
            ]
        )
        admissible = (distances <= self.maximum_match_distance) & np.array(
            [
                [self._agree(detection, known_object) for known_object in known_objects]
                for detection in detections
            ]
        )
        # Costlier than all admissible pairs together, so that the assignment takes as
        # many admissible pairs as it can before it shortens their distances.
        inadmissible_cost = self.maximum_match_distance * (min(distances.shape) + 1)
        detection_indices, known_object_indices = linear_sum_assignment(
            np.where(admissible, distances, inadmissible_cost)
        )
        return {
            detections[detection_index]: known_objects[known_object_index]
            for detection_index, known_object_index in zip(
                detection_indices, known_object_indices
            )
            if admissible[detection_index, known_object_index]
        }

    def _agree(self, detection: Detection, known_object: KnownObject) -> bool:
        """
        :param detection: The detection.
        :param known_object: The known object.
        :return: Whether ``detection`` and ``known_object`` agree in class, where both
            have one, and in size, where the known object has one.
        """
        detection_label = detection.hypothesis.class_label
        known_label = known_object.class_label
        if (
            detection_label is not None
            and known_label is not None
            and detection_label != known_label
        ):
            return False

        known_extent = known_object.extent
        if known_extent is None:
            return True
        detection_edge = max(detection.extent.x, detection.extent.y, detection.extent.z)
        known_edge = max(known_extent.x, known_extent.y, known_extent.z)
        return max(detection_edge, known_edge) <= self.maximum_match_size_ratio * min(
            detection_edge, known_edge
        )


# %% perceived objects


@dataclass
class PerceivedObjects:
    """
    The objects perception believes in, each kept as a body in :attr:`world`.
    """

    world: World
    """
    The world the bodies of the objects are kept in.
    """

    matcher: ObjectMatcher = field(default_factory=ObjectMatcher)
    """
    Decides which known object each found object is.
    """

    beliefs: dict[UUID, ObjectBeliefState] = field(default_factory=dict)
    """
    The objects perception believes in, by the id of their body.
    """

    def apply(self, cas: CAS) -> list[UUID | None]:
        """
        Put the objects a finished pipeline pass found into :attr:`world`.

        Each object is annotated with its pose in the world frame and matched to a known
        object: one perception believes in, or a perceivable object of :attr:`world`. A
        matched object moves to where it was seen, and one perception believes in also
        takes the new shape. Any other object becomes a new body. An object that cannot
        be located in :attr:`world`, because ``cas`` holds no camera pose in it or the
        object has no 3D bounding box, gets no body.

        :param cas: The CAS of the finished pass.
        :return: The ids of the bodies of the found objects, in the order of the
            objects in ``cas``; None for an object that got no body.
        """
        hypotheses = cas.filter_annotations_by_type(ObjectHypothesis)
        self._annotate_world_poses(hypotheses, cas)
        detections = [self._locate(hypothesis, cas) for hypothesis in hypotheses]
        matches = self.matcher.match(
            [detection for detection in detections if detection is not None],
            self._known_objects(),
        )

        body_ids = []
        for hypothesis, detection in zip(hypotheses, detections):
            if detection is None:
                body_ids.append(None)
                continue
            known_object = matches.get(detection)
            if known_object is None:
                body_ids.append(self._believe_in_new_object(detection).uuid)
                continue
            known_object.update(detection)
            body_ids.append(known_object.body.id)
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
        return Detection(
            hypothesis=hypothesis,
            pose=HomogeneousTransformationMatrix(
                data=get_camera_to_world_transform_matrix(cas) @ camera_T_bounding_box,
                reference_frame=world_body,
            ),
            extent=Scale(
                x=bounding_box.x_length,
                y=bounding_box.y_length,
                z=bounding_box.z_length,
            ),
        )

    # %% known objects

    def _known_objects(self) -> list[KnownObject]:
        """
        :return: The objects of :attr:`world` a found object can be taken as: the ones
            perception believes in, and the perceivable ones it did not create.
        """
        believed_objects = [
            BelievedObject(world=self.world, belief=belief)
            for belief in self.beliefs.values()
            if belief.body in self.world.kinematic_structure_entities
        ]
        perceivable_objects = {}
        for annotation in self.world.get_semantic_annotations_by_type(IsPerceivable):
            if not isinstance(annotation, HasRootBody):
                continue
            if annotation.root.id in self.beliefs:
                continue
            perceivable_objects.setdefault(
                annotation.root.id,
                PerceivableObject(world=self.world, annotation=annotation),
            )
        return believed_objects + list(perceivable_objects.values())

    # %% beliefs

    def _believe_in_new_object(self, detection: Detection) -> ObjectBeliefState:
        """
        Believe in a new object seen as ``detection``, as a box at its 3D bounding box
        below the world frame of its pass.

        :param detection: The detection the object was seen as.
        :return: The new belief.
        """
        belief = ObjectBeliefState(
            body=Body(visual=ShapeCollection(shapes=[Box(scale=detection.extent)]))
        ).add_hypothesis(detection.hypothesis)
        self.beliefs[belief.uuid] = belief
        with self.world.modify_world():
            self.world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=self.world,
                    parent=detection.pose.reference_frame,
                    child=belief.body,
                )
            )
        belief.body.parent_connection.origin = detection.pose
        return belief
