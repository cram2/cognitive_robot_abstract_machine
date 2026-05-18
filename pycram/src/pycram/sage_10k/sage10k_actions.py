from dataclasses import dataclass

import rustworkx

from krrood.entity_query_language.factories import underspecified, variable
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.core.container import OpenAction
from pycram.robot_plans.actions.core.misc import MoveToReach
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.semantic_annotations.semantic_annotations import Door
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    navigation_map_at_target,
    translate_free_space_to_where_condition,
)
from semantic_digital_twin.exceptions import PointOccupiedError


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
        gcs = navigation_map_at_target(target=self.door.handle.root)

        arm = Arms.LEFT
        pre_grasp_pose = self.door.handle.pre_grasp_pose()

        # Find a node in free space that is near the pre-grasp pose.
        target_node = gcs.node_of_point(self.door.handle.pre_grasp_pose().position)
        if target_node is None:
            raise PointOccupiedError(
                self.world.transform(
                    self.door.handle.pre_grasp_pose(), self.world.root
                ).position
            )

        gcs = gcs.create_subgraph(
            list(
                rustworkx.node_connected_component(
                    gcs.graph, gcs.box_to_index_map[target_node]
                )
            )
        )

        reach_query = underspecified(MoveToReach)(
            target_pose=pre_grasp_pose,
            robot_x=...,
            robot_y=...,
            hip_rotation=0.0,
            grasp_description=underspecified(GraspDescription)(
                approach_direction=ApproachDirection.FRONT,
                vertical_alignment=VerticalAlignment.NoAlignment,
                manipulator=variable(Manipulator, self.world.semantic_annotations),
                rotate_gripper=False,
            ),
        )

        where_condition = translate_free_space_to_where_condition(
            gcs.free_space_event,
            reach_query.expression,
            x_variable_name="MoveToReach.robot_x",
            y_variable_name="MoveToReach.robot_y",
        )

        reach_action = reach_query.where(where_condition)

        open_action = OpenAction(object_designator=self.door.handle.root, arm=arm)

        self.add_subplan(sequential([reach_action, open_action])).perform()
