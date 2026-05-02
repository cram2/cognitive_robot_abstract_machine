from dataclasses import dataclass

from pycram.robot_plans.actions.base import ActionDescription
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    navigation_map_at_target,
)


@dataclass
class Sage10kOpenDoor(ActionDescription):
    door: Door

    def position_to_stand(self):
        gcs = navigation_map_at_target(self.door.handle.root)
