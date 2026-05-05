import logging
from dataclasses import dataclass

from rclpy.action import ActionClient
from robokudo_msgs.action import Query

from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.adapters.ros import PoseStampedToSemDTConverter
from semantic_digital_twin.reasoning.predicates import visible
from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Region,
    KinematicStructureEntity,
    Body,
)
from typing_extensions import Type, List

logger = logging.getLogger("pycram")


@dataclass
class PerceptionQuery:
    semantic_annotation: Type[SemanticAnnotation]
    """
    The semantic annotation for which to perceive
    """

    region: BoundingBox
    """
    The region in which the object should be detected
    """

    robot: AbstractRobot
    """'
    Robot annotation of the robot that should perceive the object.
    """

    world: World
    """
    The world in which the object should be detected.
    """

    context: Context
    """
    The context of the plan
    """

    def from_world(self) -> List[Body]:
        result = []
        sem_instances = self.world.get_semantic_annotations_by_type(
            self.semantic_annotation
        )
        bodies = []
        for sem_instance in sem_instances:
            bodies.extend(sem_instance.bodies)

        region_bodies = list(
            filter(
                None,
                [
                    (
                        body
                        if self.region.contains(body.global_transform.to_position())
                        else None
                    )
                    for body in bodies
                ],
            )
        )

        robot_camera = list(
            filter(
                None,
                [
                    cam if isinstance(cam, Camera) else None
                    for cam in self.robot.sensors
                ],
            )
        )[0]
        for body in region_bodies:
            if visible(robot_camera, body):
                result.append(body)
        return result

    def from_robokudo(self):
        from robokudo_msgs.msg import ObjectDesignator

        self._client = ActionClient(self.context.ros_node, Query, "/robokudo/query")
        logger.info("Waiting for action server /robokudo/query")
        self._client.wait_for_server()

        future = self._client.send_goal_async(Query.Goal())
        future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        goal_handle = future.result()
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.robokudo_callback)

    def robokudo_callback(self, future):
        result = future.result().result
        with self.world.modify_world():
            for obj in result.res:
                if obj.type == "cracker_cheezit_box_original":
                    cheezits = self.world.get_body_by_name("cheeze_it.obj")
                    original_parent = cheezits.parent_kinematic_structure_entity
                    new_pose = self.world.transform(
                        PoseStampedToSemDTConverter.convert(obj.pose[0], self.world),
                        cheezits.parent_kinematic_structure_entity,
                    )
                    new_pose = Pose.from_xyz_quaternion(
                        new_pose.x,
                        new_pose.y,
                        new_pose.z,
                        *cheezits.parent_connection.origin.to_quaternion().to_np(),
                    )
                    self.world.remove_connection(cheezits.parent_connection)
                    self.world.add_connection(
                        FixedConnection(
                            original_parent, cheezits, new_pose.to_homogeneous_matrix()
                        )
                    )
        ...
