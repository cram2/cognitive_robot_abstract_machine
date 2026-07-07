from __future__ import division

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Goal, Task, NodeArtifacts
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False, repr=False)
class ReachPointTask(Task):
    """Moves an offset relative to the tool point to a goal point."""

    tip_P_tool: Point3 = field(kw_only=True)
    """Controlled point, expressed in the tip frame."""

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Root link of the kinematic chain."""

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Body that is controlled."""

    root_P_goal: Point3 = field(kw_only=True)
    """Goal position of the tool point, expressed in the root frame."""

    reference_velocity: float = field(default=0.1, kw_only=True)
    """Reference velocity in m/s used to normalize the constraint."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_tool = root_T_tip @ self.tip_P_tool
        distance_to_goal = root_P_tool.euclidean_distance(self.root_P_goal)

        artifacts = NodeArtifacts()
        artifacts.constraints.add_point_goal_constraints(
            frame_P_current=root_P_tool,
            frame_P_goal=self.root_P_goal,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )
        artifacts.observation = distance_to_goal < 0.01
        return artifacts


@dataclass(eq=False, repr=False)
class SlightlyTiltedTask(Task):
    """Tilts the tip axis by a fixed angle relative to a reference axis."""

    tip_V_axis: Vector3 = field(kw_only=True)
    """Axis of interest, expressed in the tip frame."""

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Root link of the kinematic chain."""

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Body that is controlled."""

    root_V_reference: Vector3 = field(kw_only=True)
    """Reference axis to tilt against, expressed in the root frame."""

    tilt: float = field(kw_only=True)
    """Target angle in rad between the tip axis and the reference axis."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_V_axis = root_T_tip @ self.tip_V_axis
        cos_tilt = root_V_axis.dot(self.root_V_reference)
        cos_goal = float(np.cos(self.tilt))

        artifacts = NodeArtifacts()
        artifacts.constraints.add_position_constraint(
            expr_current=cos_tilt,
            expr_goal=cos_goal,
            reference_velocity=0.1,
            quadratic_weight=self.weight,
        )
        artifacts.observation = sm.abs(cos_tilt - cos_goal) <= 0.01
        return artifacts


@dataclass(eq=False, repr=False)
class StayOnLineTask(Task):
    """Keeps the tool point on the line between two points."""

    tip_P_tool: Point3 = field(kw_only=True)
    """Controlled point, expressed in the tip frame."""

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Root link of the kinematic chain."""

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Body that is controlled."""

    root_P_start: Point3 = field(kw_only=True)
    """Start of the line segment, expressed in the root frame."""

    root_P_end: Point3 = field(kw_only=True)
    """End of the line segment, expressed in the root frame."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_tool = root_T_tip @ self.tip_P_tool
        distance_to_line, root_P_on_line = root_P_tool.distance_to_line_segment(
            self.root_P_start, self.root_P_end
        )

        artifacts = NodeArtifacts()
        artifacts.constraints.add_point_goal_constraints(
            frame_P_current=root_P_tool,
            frame_P_goal=root_P_on_line,
            reference_velocity=0.1,
            quadratic_weight=self.weight,
        )
        artifacts.observation = distance_to_line < 0.01
        return artifacts


@dataclass(eq=False, repr=False)
class TiltStraightTask(Task):
    """Aligns the tip axis with a reference axis."""

    tip_V_axis: Vector3 = field(kw_only=True)
    """Axis of interest, expressed in the tip frame."""

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Root link of the kinematic chain."""

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Body that is controlled."""

    root_V_reference: Vector3 = field(kw_only=True)
    """Reference axis to align with, expressed in the root frame."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_V_axis = root_T_tip @ self.tip_V_axis
        tilt_error = root_V_axis.angle_between(self.root_V_reference)

        artifacts = NodeArtifacts()
        artifacts.constraints.add_vector_goal_constraints(
            frame_V_current=root_V_axis,
            frame_V_goal=self.root_V_reference,
            reference_velocity=0.025,
            quadratic_weight=self.weight,
        )
        artifacts.observation = tilt_error <= 0.01
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
    """Cylinder body to insert. controlled tip of the kinematic chain."""

    tip_P_tool: Point3 = field(kw_only=True)
    """Leading insertion point (e.g. the object's tip), expressed in the tip frame."""

    hole_point: Point3 = field(kw_only=True)
    """Position of the hole to insert the cylinder into."""

    tip_V_axis: Optional[Vector3] = None
    """Insertion axis of the object in the tip frame. If None, tip-frame Z is used."""

    up_axis: Optional[Vector3] = None
    """Axis pointing out of the hole. If None, the world's z-axis is used."""

    pre_grasp_height: float = 0.2
    """Distance above the hole along the up axis at which the insertion starts."""

    tilt: float = np.pi / 10
    """Angle in rad by which the cylinder is tilted during the approach."""

    weight: float = DefaultWeights.WEIGHT_ABOVE_CA
    """Task priority relative to other tasks."""

    reach_top: ReachPointTask = field(init=False)
    tilt_task: SlightlyTiltedTask = field(init=False)
    stay_on_line: StayOnLineTask = field(init=False)
    insert_task: ReachPointTask = field(init=False)
    tilt_straight_task: TiltStraightTask = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        root = context.world.root

        tip_V_axis = self.tip_V_axis if self.tip_V_axis is not None else Vector3.Z()
        if self.up_axis is None:
            root_V_up = Vector3.Z(root)
        else:
            root_V_up = context.world.transform(self.up_axis, root)
        root_P_hole = context.world.transform(self.hole_point, root)
        root_P_top = root_P_hole + root_V_up * self.pre_grasp_height

        self.reach_top = ReachPointTask(
            name="Reach Top",
            root_link=root,
            tip_link=self.tip_link,
            tip_P_tool=self.tip_P_tool,
            root_P_goal=root_P_top,
            reference_velocity=0.1,
            weight=self.weight,
        )
        self.tilt_task = SlightlyTiltedTask(
            name="Slightly Tilted",
            root_link=root,
            tip_link=self.tip_link,
            tip_V_axis=tip_V_axis,
            root_V_reference=root_V_up,
            tilt=self.tilt,
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
        self.insert_task = ReachPointTask(
            name="Insert",
            root_link=root,
            tip_link=self.tip_link,
            tip_P_tool=self.tip_P_tool,
            root_P_goal=root_P_hole,
            reference_velocity=0.05,
            weight=self.weight,
        )
        self.tilt_straight_task = TiltStraightTask(
            name="Tilt Straight",
            root_link=root,
            tip_link=self.tip_link,
            tip_V_axis=tip_V_axis,
            root_V_reference=root_V_up,
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
