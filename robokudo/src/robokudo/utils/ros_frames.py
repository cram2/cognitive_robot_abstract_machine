"""
Normalize ROS frame identifiers and resolve their world bodies.
"""

from __future__ import annotations

# %% Imports
from typing_extensions import TYPE_CHECKING

from robokudo.exceptions import InvalidROSFrameId

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body

# %% Frame identifiers


def normalize_ros_frame_id(frame_id: str) -> str:
    """
    Return a frame identifier without ROS's optional leading slash.

    :param frame_id: ROS frame identifier to normalize.
    :return: Frame identifier with leading slashes removed.
    :raises InvalidROSFrameId: If normalization leaves an empty identifier.
    """
    normalized = frame_id.lstrip("/")
    if not normalized:
        raise InvalidROSFrameId(frame_id=frame_id)
    return normalized


# %% World bodies


def find_bodies_for_ros_frame(world: World, frame_id: str) -> list[Body]:
    """
    Find the bodies a ROS frame identifies: those whose full name, prefix included, is
    the frame, or else those whose name without prefix is, as a robot driver publishes
    the frames of a robot whose bodies carry its name as prefix.

    :param world: World containing the bodies to search.
    :param frame_id: ROS frame identifier, with or without leading slashes.
    :return: All bodies matching the normalized frame identifier.
    :raises InvalidROSFrameId: If the frame identifier is empty.
    """
    normalized = normalize_ros_frame_id(frame_id)
    bodies_of_full_name = [
        body
        for body in world.bodies
        if normalize_ros_frame_id(str(body.name)) == normalized
    ]
    if bodies_of_full_name:
        return bodies_of_full_name
    return [body for body in world.bodies if body.name.name == normalized]
