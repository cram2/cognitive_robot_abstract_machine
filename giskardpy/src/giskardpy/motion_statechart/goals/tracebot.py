from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Goal, Task, NodeArtifacts
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.feature_functions import AngleGoal, ReachPoint
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False, repr=False)
class StayOnLineTask(Task):
    """
    Keeps the tool point on the line between two points.
    """

    tip_P_tool: Point3 = field(kw_only=True)
    """
    Controlled point, expressed in the tip frame.
    """

    root_link: KinematicStructureEntity = field(kw_only=True)
    """
    Root link of the kinematic chain.
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """
    Body that is controlled.
    """

    root_P_start: Point3 = field(kw_only=True)
    """
    Start of the line segment, expressed in the root frame.
    """

    root_P_end: Point3 = field(kw_only=True)
    """
    End of the line segment, expressed in the root frame.
    """

    reference_velocity: float = field(default=0.1, kw_only=True)
    """
    Normalization velocity of the constraint in m/s.
    """

    threshold: float = field(default=0.01, kw_only=True)
    """
    Distance in m at or below which the tool counts as on the line.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_tool = root_T_tip @ self.tip_P_tool
        distance_to_line, root_P_on_line = root_P_tool.distance_to_line_segment(
            self.root_P_start, self.root_P_end
        )

        artifacts = NodeArtifacts()
        artifacts.geometry.add_point_goal_constraints(
            frame_P_current=root_P_tool,
            frame_P_goal=root_P_on_line,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )
        artifacts.observation = distance_to_line < self.threshold
        return artifacts


@dataclass(eq=False, repr=False)
class InsertCylinder(Goal):
    """
    Inserts a grasped cylinder into a hole.

    1. Reach a pre-insertion point above the hole while slightly tilted.
    2. Move the cylinder down the line into the hole, staying on the line.
    3. Straighten the cylinder so it is aligned with the hole axis.
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """
    Controlled tip of the kinematic chain; the grasped body to insert.
    """

    tip_P_tool: Point3 = field(kw_only=True)
    """
    Leading insertion point, e.g. the object's tip.
    """

    hole_point: Point3 = field(kw_only=True)
    """
    Position of the hole to insert into.
    """

    tip_V_axis: Vector3 = field(kw_only=True)
    """
    Insertion axis of the object, in the tip frame.
    """

    up_axis: Vector3 = field(kw_only=True)
    """
    Axis pointing out of the hole.
    """

    pre_grasp_height: float = 0.2
    """
    Distance above the hole along the up axis at which the insertion starts.
    """

    tilt: float = np.pi / 10
    """
    Angle in rad by which the cylinder is tilted during the approach.
    """

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """
    Task priority relative to other tasks.
    """

    reach_top: ReachPoint = field(init=False)
    tilt_task: AngleGoal = field(init=False)
    stay_on_line: StayOnLineTask = field(init=False)
    insert_task: ReachPoint = field(init=False)
    tilt_straight_task: AlignPlanes = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        root = context.world.root

        root_V_up = context.world.transform(self.up_axis, root)
        root_P_hole = context.world.transform(self.hole_point, root)
        root_P_top = root_P_hole + root_V_up * self.pre_grasp_height

        self.reach_top = ReachPoint(
            name="Reach Top",
            root_link=root,
            tip_link=self.tip_link,
            tip_point=self.tip_P_tool,
            reference_point=root_P_top,
            maximum_velocity=0.1,
            weight=self.weight,
        )
        self.tilt_task = AngleGoal(
            name="Slightly Tilted",
            root_link=root,
            tip_link=self.tip_link,
            tip_vector=self.tip_V_axis,
            reference_vector=root_V_up,
            lower_angle=self.tilt,
            upper_angle=self.tilt,
            threshold=0.01,
            weight=self.weight,
        )
        self.stay_on_line = StayOnLineTask(
            name="Stay on Straight Line",
            root_link=root,
            tip_link=self.tip_link,
            tip_P_tool=self.tip_P_tool,
            root_P_start=root_P_hole,
            root_P_end=root_P_top,
            weight=self.weight,
        )
        self.insert_task = ReachPoint(
            name="Insert",
            root_link=root,
            tip_link=self.tip_link,
            tip_point=self.tip_P_tool,
            reference_point=root_P_hole,
            maximum_velocity=0.05,
            weight=self.weight,
        )
        self.tilt_straight_task = AlignPlanes(
            name="Tilt Straight",
            root_link=root,
            tip_link=self.tip_link,
            tip_normal=self.tip_V_axis,
            goal_normal=root_V_up,
            reference_velocity=0.025,
            weight=self.weight,
        )

        self.add_nodes(
            [
                self.reach_top,
                self.tilt_task,
                self.stay_on_line,
                self.insert_task,
                self.tilt_straight_task,
            ]
        )

        init_done = sm.trinary_logic_and(
            self.reach_top.observation_variable,
            self.tilt_task.observation_variable,
        )
        bottom_reached = sm.trinary_logic_and(
            self.insert_task.observation_variable,
            self.stay_on_line.observation_variable,
        )

        self.reach_top.end_condition = init_done
        self.tilt_task.end_condition = bottom_reached
        self.insert_task.start_condition = init_done
        self.tilt_straight_task.start_condition = bottom_reached

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self.tilt_straight_task.observation_variable)
