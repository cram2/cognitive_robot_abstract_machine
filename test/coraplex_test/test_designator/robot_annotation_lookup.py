from __future__ import annotations

from typing_extensions import Type, TypeVar

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

RobotAnnotation = TypeVar("RobotAnnotation", bound=AbstractRobot)


def get_or_create_robot_annotation(
    world: World, annotation_type: Type[RobotAnnotation]
) -> RobotAnnotation:
    """
    Return the world's existing semantic annotation of the given robot type, or create
    it from the world when none exists yet.

    :param world: The world to look up the annotation in.
    :param annotation_type: The robot annotation type to look up or create.
    :return: The existing or newly created robot annotation.
    """
    annotations = world.get_semantic_annotations_by_type(annotation_type)
    if annotations:
        return annotations[0]
    return annotation_type.from_world(world)
