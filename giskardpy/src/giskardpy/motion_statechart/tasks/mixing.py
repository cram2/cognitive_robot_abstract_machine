from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    CartesianOrientation,
)
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class SpiralMixing(Task):

    end_time: float
    object_name: Body
    tool_height: float
    tip_link: Body
    root_link: Body
    radial_increment: float
    angle_increment: float
    upward_increment: float
    weight: float = DefaultWeights.WEIGHT_BELOW_CA

    def __post_init__(self):

        root_T_tip = context.world._forward_kinematic_manager.compose_expression(
            root_link=self.root_link, tip_link=self.tip_link
        )
        t = context.time_symbol

        r = self.radial_increment * t
        a = self.angle_increment * t
        h = self.upward_increment * t

        object_T_goal = HomogeneousTransformationMatrix()
        x = r * sm.cos(a)
        y = r * sm.sin(a)
        z = h

        object_T_goal.x = x
        object_T_goal.y = y
        object_T_goal.z = z

        root_T_object = context.world._forward_kinematic_manager.compose_expression(
            root_link=self.root_link, tip_link=self.object_name
        )
        root_T_goal = root_T_object.dot(object_T_goal)
        root_T_goal.z += self.tool_height + 0.05

        self.add_point_goal_constraints(
            frame_P_current=root_T_tip.to_position(),
            frame_P_goal=root_T_goal.to_position(),
            reference_velocity=CartesianPosition.default_reference_velocity,
            weight=self.weight,
        )
        context.add_debug_expression("root_T_goal", root_T_goal)
        self.add_rotation_goal_constraints(
            frame_R_current=root_T_tip.to_rotation_matrix(),
            frame_R_goal=root_T_goal.to_rotation_matrix(),
            reference_velocity=CartesianOrientation.default_reference_velocity,
            weight=self.weight,
        )

        self.observation_expression = t > self.end_time
