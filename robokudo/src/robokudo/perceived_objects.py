"""
The objects perception found, kept as bodies in the semantic digital twin.

After a pipeline pass, the objects it found are matched to the objects perception
already believes in. A matched object moves its body to where it was seen; any other
object becomes a new body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
from scipy.optimize import linear_sum_assignment
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Dict, List, Optional

from robokudo.cas import CAS, CASViews
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    PoseAnnotation,
    StampedPoseAnnotation,
)
from robokudo.types.belief_state import ObjectBeliefState
from robokudo.types.cv import ImageROI
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import (
    get_camera_to_world_transform_matrix,
    transform_pose_from_camera_to_world,
)
from robokudo.utils.hypothesis_comparators import ObjectHypothesisComparator
from robokudo.utils.transform import get_transform_matrix_from_q

DEFAULT_MATCH_THRESHOLD = 0.05
"""
How similar a found object has to be to a believed object to be taken as that object.
"""

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

    match_threshold: float = DEFAULT_MATCH_THRESHOLD
    """
    How similar, between 0 and 1, a found object has to be to a believed object to be
    taken as that object.
    """

    beliefs: Dict[UUID, ObjectBeliefState] = field(default_factory=dict)
    """
    The objects perception believes in, by the id of their body.
    """

    def apply(self, cas: CAS) -> List[UUID]:
        """
        Put the objects a finished pipeline pass found into :attr:`world`.

        Each object is annotated with its pose in the world frame, then matched to a
        believed object: a match moves that object's body to where it was seen and
        reshapes it, any other object becomes a new body. Without a camera pose in
        ``cas`` an object cannot be located, so its belief is kept outside the world.

        :param cas: The CAS of the finished pass.
        :return: The ids of the bodies of the found objects, in the order of the
            objects in ``cas``.
        """
        hypotheses = cas.filter_annotations_by_type(ObjectHypothesis)
        self._annotate_world_poses(hypotheses, cas)
        matches = self._match(hypotheses, cas)

        body_ids = []
        for index, hypothesis in enumerate(hypotheses):
            belief = matches.get(index)
            if belief is None:
                belief = self._believe_in_new_object(hypothesis, cas)
            else:
                self._update_belief(belief, hypothesis, cas)
            body_ids.append(belief.uuid)
        return body_ids

    # %% matching

    def _annotate_world_poses(
        self, hypotheses: List[ObjectHypothesis], cas: CAS
    ) -> None:
        """
        Annotate each hypothesis with its latest pose, in the world frame.

        :param hypotheses: The hypotheses to annotate.
        :param cas: The CAS whose camera pose locates the hypotheses.
        """
        if cas.camera_to_world_transform is None:
            return

        for hypothesis in hypotheses:
            pose = self._latest_pose_of(hypothesis)
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

    @staticmethod
    def _latest_pose_of(hypothesis: ObjectHypothesis) -> Optional[PoseAnnotation]:
        """
        :param hypothesis: The hypothesis.
        :return: The latest pose annotation of ``hypothesis`` in camera coordinates.
        """
        for annotation in reversed(hypothesis.annotations):
            if type(annotation) is PoseAnnotation:
                return annotation
        return None

    def _match(
        self, hypotheses: List[ObjectHypothesis], cas: CAS
    ) -> Dict[int, ObjectBeliefState]:
        """
        Pair the hypotheses with the believed objects they most likely are.

        :param hypotheses: The hypotheses to match.
        :param cas: The CAS of the hypotheses.
        :return: The believed object of each matched hypothesis, by the index of the
            hypothesis.
        """
        beliefs = list(self.beliefs.values())
        if not hypotheses or not beliefs:
            return {}

        comparator = (
            ObjectHypothesisComparator(lambda: cas)
            .with_comparator_for(ImageROI, weight=0.2)
            .with_comparator_for(StampedPoseAnnotation, weight=0.4)
            .with_comparator_for(BoundingBox3DAnnotation, weight=0.4)
        )
        similarities = np.array(
            [
                [
                    comparator.compute_similarity(hypothesis, belief.latest_hypothesis)
                    for belief in beliefs
                ]
                for hypothesis in hypotheses
            ]
        )
        hypothesis_indices, belief_indices = linear_sum_assignment(-similarities)
        return {
            int(hypothesis_index): beliefs[belief_index]
            for hypothesis_index, belief_index in zip(hypothesis_indices, belief_indices)
            if similarities[hypothesis_index, belief_index] > self.match_threshold
        }

    # %% bodies

    def _believe_in_new_object(
        self, hypothesis: ObjectHypothesis, cas: CAS
    ) -> ObjectBeliefState:
        """
        Believe in a new object seen as ``hypothesis``, as a box at its latest 3D
        bounding box below the world frame of ``cas``.

        :param hypothesis: The hypothesis the object was seen as.
        :param cas: The CAS whose camera pose locates the hypothesis.
        :return: The new belief.
        """
        belief = ObjectBeliefState.create_with_new_body().add_hypothesis(hypothesis)
        self.beliefs[belief.uuid] = belief

        world_body = self._world_body_of(cas)
        if world_body is None:
            return belief

        bounding_box = belief.latest_bbox_3d
        if bounding_box is not None:
            belief.body.visual.append(Box(scale=self._scale_of(bounding_box)))
        with self.world.modify_world():
            self.world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=self.world, parent=world_body, child=belief.body
                )
            )
        if bounding_box is not None:
            belief.body.parent_connection.origin = self._world_T_bounding_box(
                bounding_box, cas, world_body, belief.body
            )
        return belief

    def _update_belief(
        self, belief: ObjectBeliefState, hypothesis: ObjectHypothesis, cas: CAS
    ) -> None:
        """
        Move and reshape the body of ``belief`` to the latest 3D bounding box of
        ``hypothesis``, the object seen again.

        .. note:: Only the move reaches the processes :attr:`world` is shared with;
            they keep the shape the object was first believed in with.

        :param belief: The belief in the object seen again.
        :param hypothesis: The hypothesis it was seen as.
        :param cas: The CAS whose camera pose locates the hypothesis.
        """
        belief.add_hypothesis(hypothesis)

        world_body = self._world_body_of(cas)
        bounding_box = belief.latest_bbox_3d
        if world_body is None or bounding_box is None:
            return

        with self.world.modify_world(publish_changes=False):
            belief.body.visual.shapes.clear()
            belief.body.visual.append(Box(scale=self._scale_of(bounding_box)))
        belief.body.parent_connection.origin = self._world_T_bounding_box(
            bounding_box, cas, world_body, belief.body
        )

    def _world_body_of(self, cas: CAS) -> Optional[Body]:
        """
        :param cas: The CAS of the current pass.
        :return: The body of the world frame ``cas`` locates its camera in, or None if
            the CAS names no world frame.
        """
        if cas.world_frame is None:
            return None
        return self.world.get_body_by_name(PrefixedName(name=cas.world_frame))

    @staticmethod
    def _scale_of(bounding_box: BoundingBox3DAnnotation) -> Scale:
        """
        :param bounding_box: The bounding box.
        :return: The extent of the box along its axes.
        """
        return Scale(
            x=bounding_box.x_length, y=bounding_box.y_length, z=bounding_box.z_length
        )

    @staticmethod
    def _world_T_bounding_box(
        bounding_box: BoundingBox3DAnnotation,
        cas: CAS,
        world_body: Body,
        object_body: Body,
    ) -> HomogeneousTransformationMatrix:
        """
        :param bounding_box: The bounding box in camera coordinates.
        :param cas: The CAS holding the camera pose.
        :param world_body: The world frame body.
        :param object_body: The body the bounding box belongs to.
        :return: The pose of the bounding box in the world frame.
        """
        camera_T_bounding_box = get_transform_matrix_from_q(
            bounding_box.pose.rotation, bounding_box.pose.translation
        )
        return HomogeneousTransformationMatrix(
            data=np.matmul(
                get_camera_to_world_transform_matrix(cas), camera_T_bounding_box
            ),
            reference_frame=world_body,
            child_frame=object_body,
        )
