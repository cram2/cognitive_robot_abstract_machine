from __future__ import division

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Goal
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class InsertCylinder(Goal):
    cylinder_name: Body = field(kw_only=True)
    hole_point: Point3 = field(kw_only=True)
    cylinder_height: Optional[float] = None
    up: Optional[Vector3] = None
    pre_grasp_height: float = 0.1
    tilt: float = np.pi / 10
    get_straight_after: float = 0.02

    def __post_init__(self):
        self.root = context.world.root
        self.tip = self.cylinder_name
        if self.cylinder_height is None:
            self.cylinder_height = context.world.links[self.tip].collisions[0].height
        else:
            self.cylinder_height = self.cylinder_height
        self.root_P_hole = context.world.transform(
            target_frame=self.root, spatial_object=self.hole_point
        )
        if self.up is None:
            self.up = Vector3.Z()
            self.up.reference_frame = self.root
        self.root_V_up = context.world.transform(
            target_frame=self.root, spatial_object=self.up
        )

        self.weight = DefaultWeights.WEIGHT_ABOVE_CA

        root_P_hole = self.root_P_hole
        root_V_up = self.root_V_up
        root_T_tip = context.world._forward_kinematic_manager.compose_expression(
            self.root, self.tip
        )
        root_P_tip = root_T_tip.to_position()
        tip_P_cylinder_bottom = Vector3.Z() * self.cylinder_height / 2
        root_P_cylinder_bottom = root_T_tip @ tip_P_cylinder_bottom
        root_P_tip = root_P_tip + root_P_cylinder_bottom
        root_V_cylinder_z = root_T_tip @ -Vector3.Z()

        # %% straight line goal
        root_P_top = root_P_hole + root_V_up * self.pre_grasp_height
        distance_to_top = root_P_tip.euclidean_distance(root_P_top)

        distance_to_line, root_P_on_line = root_P_tip.distance_to_line_segment(
            root_P_hole, root_P_top
        )
        distance_to_hole = (root_P_hole - root_P_tip).norm()

        reach_top = Task(name="Reach Top")
        self.add_task(reach_top)
        reach_top.add_point_goal_constraints(
            frame_P_current=root_P_tip,
            frame_P_goal=root_P_top,
            reference_velocity=0.1,
            weight=self.weight,
        )
        reach_top.observation_expression = distance_to_top < 0.01

        # %% tilted orientation goal
        tilt_error = root_V_cylinder_z.angle_between(root_V_up)
        tilt_task = Task(name="Slightly Tilted")
        self.add_task(tilt_task)
        tilt_task.add_position_constraint(
            expr_current=tilt_error,
            expr_goal=self.tilt,
            reference_velocity=0.1,
            weight=self.weight,
        )
        root_V_cylinder_z.vis_frame = self.tip
        tilt_task.observation_expression = sm.abs(tilt_error - self.tilt) <= 0.01

        init_done = f"{reach_top} and {tilt_task}"

        reach_top.end_condition = init_done

        # %% move down
        stay_on_line = Task(name="Stay on Straight Line")
        self.add_task(stay_on_line)
        stay_on_line.add_point_goal_constraints(
            frame_P_current=root_P_tip,
            frame_P_goal=root_P_on_line,
            reference_velocity=0.1,
            weight=self.weight,
            name="pregrasp",
        )
        stay_on_line.observation_expression = distance_to_line < 0.01

        insert_task = Task(name="Insert")
        self.add_task(insert_task)
        insert_task.add_point_goal_constraints(
            frame_P_current=root_P_tip,
            frame_P_goal=root_P_hole,
            reference_velocity=0.1,
            weight=self.weight,
            name="insertion",
        )
        insert_task.start_condition = init_done
        insert_task.observation_expression = distance_to_hole < 0.01

        bottom_reached = f"{insert_task} and {stay_on_line}"

        tilt_task.end_condition = bottom_reached
        # %% tilt straight
        # tilt_monitor.observation_expression = cas.less(tilt_error, 0.01)

        tilt_straight_task = Task(name="Tilt Straight")
        self.add_task(tilt_straight_task)
        tilt_straight_task.add_vector_goal_constraints(
            frame_V_current=root_V_cylinder_z,
            frame_V_goal=root_V_up,
            reference_velocity=0.1,
            weight=self.weight,
        )
        tilt_straight_task.start_condition = bottom_reached
        # tilt_straight_task.end_condition = tilt_monitor.observation_state
        tilt_straight_task.observation_expression = tilt_error <= 0.01

        self.observation_expression = tilt_straight_task.observation_expression
