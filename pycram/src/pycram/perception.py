import logging
from dataclasses import dataclass

from rclpy.action import ActionClient
from robokudo_msgs.action import Query

from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.reasoning.predicates import visible
from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Camera
from semantic_digital_twin.world import World
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
        future.add_done_callback(self.robokudo_callback)

    def robokudo_callback(self, future):
        result = future.result().result
        for obj in result.res:
            print(obj)
