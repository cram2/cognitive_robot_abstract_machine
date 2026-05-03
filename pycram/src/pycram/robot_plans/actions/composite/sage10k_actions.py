from dataclasses import dataclass

from pycram.robot_plans.actions.base import ActionDescription
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    navigation_map_at_target,
)


@dataclass
class Sage10kOpenDoor(ActionDescription):
    """
    Open a door.

    This action creates a Graph of Convex Sets (GCS) navigation map at the door handle.
    Using this GCS, an underspecified move to reach plan is mounted as subplan followed up by an
    opening action is executed.
    """

    door: Door
