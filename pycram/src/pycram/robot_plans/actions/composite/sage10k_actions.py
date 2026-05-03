from dataclasses import dataclass

from krrood.entity_query_language.factories import underspecified, variable
from pycram.datastructures.enums import Arms
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.core.container import OpenAction
from pycram.robot_plans.actions.core.misc import MoveToReach
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from semantic_digital_twin.spatial_types.spatial_types import Pose
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

    def execute(self) -> None:
        """
        Execute the action by mounting subplans for reaching and opening the door.

        This method creates a navigation map around the door handle and then
        performs a sequential plan of reaching the handle and opening the door.
        """
        navigation_map_at_target(target=self.door.handle.root)

        arm = Arms.LEFT

        reach_action = underspecified(MoveToReach)(
            target_pose=Pose.from_xyz_rpy(x=0.25, reference_frame=self.door.handle.root),
            robot_x=0.8,
            robot_y=0.0,
            hip_rotation=0.0,
            grasp_description=underspecified(GraspDescription)(
                approach_direction=...,
                vertical_alignment=...,
                manipulator=variable(Manipulator, self.world.semantic_annotations),
            ),
        )

        open_action = OpenAction(object_designator=self.door.handle.root, arm=arm)

        self.add_subplan(sequential([reach_action, open_action])).perform()
