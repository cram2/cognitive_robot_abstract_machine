from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from rclpy.node import Node
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.reasoning.predicates import visible
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import IsPerceivable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
)
from typing_extensions import Any, Dict, Optional, Self, Type, List, TYPE_CHECKING

from coraplex.datastructures.enums import ExecutionType
from coraplex.exceptions import (
    AmbiguousDetection,
    NothingDetected,
    PerceivedObjectNotInWorld,
    PerceptionSourceUnavailable,
    UnidentifiedDetections,
    UnknownExecutionType,
)
from coraplex.ros import create_action_client

if TYPE_CHECKING:
    from robokudo_msgs.msg import ObjectDesignator

ROBOKUDO_QUERY_ACTION_NAME = "robokudo/query"
"""
Name of the action RoboKudo answers perception queries on.
"""

# %% queries


@dataclass
class PerceptionQuery(SubclassJSONSerializer):
    """
    What to look for and where.

    Travels to whichever process answers it, so the world and the robot are carried by
    reference and resolved against the world on the receiving side.
    """

    semantic_annotation: Type[SemanticAnnotation]
    """
    The semantic annotation for which to perceive.
    """

    region: BoundingBox
    """
    The region in which the object should be detected.
    """

    robot: AbstractRobot
    """
    ' Robot annotation of the robot that should perceive the object.
    """

    world: World
    """
    The world in which the object should be detected.
    """

    trust_detected_orientation: bool = True
    """
    Whether to also apply the perception source's detected orientation.

    When False, only the detected position is applied to the world; the object's
    existing orientation is kept. See :meth:`Detection.apply_to`.
    """

    def from_world(self) -> List[Body]:
        """
        Answer this query from the world model alone.

        :return: The bodies of the queried annotation that lie inside the region and are
            visible to the robot's camera.
        """
        bodies = [
            body
            for sem_instance in self.world.get_semantic_annotations_by_type(
                self.semantic_annotation
            )
            for body in sem_instance.bodies
        ]
        region_bodies = [
            body
            for body in bodies
            if self.region.contains(body.global_transform.to_position())
        ]

        robot_camera = self.robot.get_default_camera()
        return [body for body in region_bodies if visible(robot_camera, body)]

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["semantic_annotation"] = to_json(self.semantic_annotation)
        result["region"] = to_json(self.region)
        result["robot_id"] = to_json(self.robot.id)
        result["trust_detected_orientation"] = self.trust_detected_orientation
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        return cls(
            semantic_annotation=from_json(data["semantic_annotation"], **kwargs),
            region=from_json(data["region"], **kwargs),
            robot=tracker.get_world_entity_with_id(id=from_json(data["robot_id"])),
            world=kwargs["world"],
            trust_detected_orientation=data.get("trust_detected_orientation", True),
        )


# %% detections


@dataclass
class Detection:
    """
    One object reported by a perception source.
    """

    semantic_annotation: Type[SemanticAnnotation]
    """
    The annotation the perception source identified the object as.
    """

    pose: Pose
    """
    Where the perception source saw the object.
    """

    def apply_to(
        self, world: World, trust_orientation: bool = True
    ) -> List[IsPerceivable]:
        """
        Write this detection into ``world``, moving the perceived body to where it was
        seen.

        :param world: The world holding the annotations this detection refers to.
        :param trust_orientation: Whether to also apply the detected orientation. When
            False, only the detected position is applied and the body keeps its existing
            orientation; useful while a perception source's orientation estimate is not
            yet reliable enough for grasp planning.
        :return: The annotations that were updated.
        :raises PerceivedObjectNotInWorld: If the world holds no such annotation.
        :raises AmbiguousDetection: If the annotation describes more than one body.
        """
        matching_annotations = self.resolve_annotations(world)
        matching_annotation = matching_annotations[0]
        body = matching_annotation.root
        robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]

        # Foundation Pose may return a upside down pose. this detects and fixes it by rotation around x by 180 degree
        detected_global_pose = world.transform(
            self.pose, world.root
        ).to_homogeneous_matrix()
        detected_global_pose = self._with_z_up_and_x_towards(
            root_T_frame=detected_global_pose,
            root_P_target=robot.root.global_transform.to_position(),
        )

        if trust_orientation:
            body.parent_connection.origin = detected_global_pose
        else:
            parent_T_object = world.transform(
                detected_global_pose, body.parent_connection.parent
            )
            parent_origin = body.parent_connection.origin
            body.parent_connection.origin = (
                HomogeneousTransformationMatrix.from_point_rotation_matrix(
                    point=parent_T_object.to_position(),
                    rotation_matrix=parent_origin.to_rotation_matrix(),
                    reference_frame=body.parent_connection.parent,
                )
            )
        return matching_annotations

    @staticmethod
    def _with_z_up_and_x_towards(
        root_T_frame: HomogeneousTransformationMatrix, root_P_target: Point3
    ) -> HomogeneousTransformationMatrix:
        """
        Return ``root_T_frame`` rotated by 180° around its own axes so that its z axis
        points along the world z axis and its x axis points towards ``root_P_target`` as
        closely as those rotations allow.

        The position is left untouched, and every axis stays on the axis it started on.
        :param root_T_frame:``HomogeneousTransformationMatrix`` which will be rotated
        :param root_P_target:``Point3`` towards which root_T_frame x axis will point
        :return: rotated transformation matrix
        """
        root_V_target = root_P_target - root_T_frame.to_position()

        flip_z = root_T_frame[2, 2] < 0
        flip_x = root_T_frame.to_rotation_matrix().x_vector().dot(root_V_target) > 0

        return root_T_frame @ HomogeneousTransformationMatrix.from_xyz_rpy(
            roll=np.pi if flip_z and not flip_x else 0,
            pitch=np.pi if flip_z and flip_x else 0,
            yaw=np.pi if flip_x and not flip_z else 0,
        )

    def resolve_annotations(self, world: World) -> List[IsPerceivable]:
        """
        Find the annotations in ``world`` that this detection identified.

        Several annotations may describe the same body, which is not ambiguous; only an
        annotation describing more than one body is.

        :param world: The world to search.
        :return: The matching annotations, all sharing one root body.
        :raises PerceivedObjectNotInWorld: If the world holds no instance of the
            annotation.
        :raises AmbiguousDetection: If the matching annotations sit on different bodies.
        """
        candidates = world.get_semantic_annotations_by_type(self.semantic_annotation)
        if not candidates:
            raise PerceivedObjectNotInWorld(self.semantic_annotation)
        bodies = {annotation.root for annotation in candidates}
        if len(bodies) > 1:
            raise AmbiguousDetection(self.semantic_annotation, len(bodies))
        return candidates


# %% perception sources


class PerceptionInterface(ABC):
    """
    A source of detections.
    """

    @abstractmethod
    def detect(
        self, query: PerceptionQuery, accept_first_if_multiple: bool = False
    ) -> List[Detection]:
        """
        Answer a perception query.

        :param query: What to look for and where.
        :param accept_first_if_multiple: Whether if there are multiple results of the
            same type returned, accept the first one
        :return: The objects this source saw.
        """

    @staticmethod
    def narrow_to_single_detection(
        detections: List[Detection],
        query: PerceptionQuery,
        accept_first_if_multiple: bool,
    ) -> List[Detection]:
        """
        Reduce what a source saw to the one detection a query is answered with.

        :param detections: What the source reported for the query.
        :param query: The query that was answered.
        :param accept_first_if_multiple: Whether several candidates may be resolved by
            taking the first one.
        :return: The single detection the query is answered with.
        :raises NothingDetected: If the source saw nothing.
        :raises UnidentifiedDetections: If several candidates were seen and none of them
            may be picked.
        """
        if not detections:
            raise NothingDetected(query.semantic_annotation)
        if len(detections) > 1 and not accept_first_if_multiple:
            raise UnidentifiedDetections(query.semantic_annotation, len(detections))
        return detections[:1]

    @staticmethod
    def for_execution_type(
        execution_type: Optional[ExecutionType], ros_node: Optional[Node] = None
    ) -> PerceptionInterface:
        """
        Pick the source that matches how the plan is being executed.

        :param execution_type: Whether the plan drives the real robot or a simulated
            one; None when nothing is executing the plan.
        :param ros_node: Node a real source reaches its perception pipeline through.
        :return: The source to answer queries with.
        :raises UnknownExecutionType: If the execution type has no source.
        """
        if execution_type in (ExecutionType.SIMULATED, ExecutionType.NO_EXECUTION):
            return WorldPerception()
        if execution_type == ExecutionType.REAL:
            return RoboKudoPerception(ros_node=ros_node)
        raise UnknownExecutionType(execution_type)


@dataclass
class WorldPerception(PerceptionInterface):
    """
    Source that reads the objects straight out of the world model.

    Reports what a perfect sensor would see: every body of the queried annotation that
    lies inside the region and is visible to the robot's camera, at the pose the world
    already holds.
    """

    def detect(
        self, query: PerceptionQuery, accept_first_if_multiple: bool = False
    ) -> List[Detection]:
        annotation_by_body = {}
        for annotation in query.world.get_semantic_annotations_by_type(
            query.semantic_annotation
        ):
            annotation_by_body.setdefault(annotation.root, type(annotation))

        detections = [
            Detection(
                semantic_annotation=annotation_by_body[body], pose=body.global_pose
            )
            for body in query.from_world()
            if body in annotation_by_body
        ]

        return self.narrow_to_single_detection(
            detections, query, accept_first_if_multiple
        )


@dataclass
class RoboKudoPerception(PerceptionInterface):
    """
    Source that asks a running RoboKudo pipeline over its query action.
    """

    ros_node: Node
    """
    Node the action client is created on.
    """

    action_name: str = ROBOKUDO_QUERY_ACTION_NAME
    """
    Action RoboKudo answers queries on.
    """

    server_timeout: timedelta = field(default=timedelta(seconds=10))
    """
    How long to wait for the action server before giving up.
    """

    def detect(
        self, query: PerceptionQuery, accept_first_if_multiple: bool = False
    ) -> List[Detection]:
        # RoboKudo's messages only exist where its pipeline is installed, so they are
        # imported here rather than at module level: reading the world in simulation must
        # not depend on them.
        from robokudo_msgs.action import Query
        from robokudo_msgs.msg import ObjectDesignator

        client = create_action_client(self.action_name, Query, self.ros_node)
        if not client.wait_for_server(timeout_sec=self.server_timeout.total_seconds()):
            raise PerceptionSourceUnavailable(self.action_name)

        goal = Query.Goal(
            obj=ObjectDesignator(type=query.semantic_annotation.__name__.lower())
        )
        result = client.send_goal(goal).result
        detections = [
            self._to_detection(designator, query)
            for designator in result.res
            if designator.pose and self._can_be_requested_object(designator, query)
        ]

        return self.narrow_to_single_detection(
            detections, query, accept_first_if_multiple
        )

    @staticmethod
    def _can_be_requested_object(
        designator: ObjectDesignator, query: PerceptionQuery
    ) -> bool:
        """
        Whether a reported designator could be the object the query asked for.

        A pipeline that only localizes reports no class label, and every object it found
        is then a candidate. One that classifies lets the objects that were not asked
        for be discarded here.

        :param designator: The designator the pipeline reported.
        :param query: The query it was answering.
        :return: True if the designator is a candidate for the requested object.
        """
        if not designator.type:
            return True
        return (
            designator.type.strip().lower()
            == query.semantic_annotation.__name__.lower()
        )

    def _to_detection(
        self, designator: ObjectDesignator, query: PerceptionQuery
    ) -> Detection:
        """
        Convert one RoboKudo object designator into a detection.

        The detection is identified as the queried annotation: a pipeline that classifies
        has already been narrowed to it by :meth:`_can_be_requested_object`, and one that
        only localizes answered where the queried object is.

        :param designator: The designator RoboKudo reported.
        :param query: The query it was answering, supplying the annotation and the frame.
        :return: The detection carrying the queried annotation and the reported pose.
        """
        world = query.world
        pose_stamped = designator.pose[0]
        return Detection(
            semantic_annotation=query.semantic_annotation,
            pose=Pose.from_xyz_quaternion(
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z,
                pose_stamped.pose.orientation.x,
                pose_stamped.pose.orientation.y,
                pose_stamped.pose.orientation.z,
                pose_stamped.pose.orientation.w,
                reference_frame=world.get_kinematic_structure_entity_by_name(
                    pose_stamped.header.frame_id
                ),
            ),
        )
