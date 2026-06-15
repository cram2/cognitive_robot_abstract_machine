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
    Body,
    KinematicStructureEntity,
)


@dataclass(eq=False, repr=False)
class ReachTopTask(Task):
    """Moves the body's bottom to the pre-grasp point above the hole."""

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Root link of the kin chain."""

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Body that is controlled."""

    root_P_hole: Point3 = field(kw_only=True)
    """Position of the hole, expressed in the root frame."""

    root_V_up: Vector3 = field(kw_only=True)
    """Axis pointing out of the hole, expressed in the root frame."""

    pre_grasp_height: float = field(kw_only=True)
    """Distance above the hole along the up axis at which the body is placed."""

    cylinder_height: float = field(kw_only=True)
    """Height of the body, used to locate its bottom."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        tip_P_cylinder_bottom = -Vector3.Z() * self.cylinder_height / 2
        root_P_tip = root_T_tip.to_position() + root_T_tip @ tip_P_cylinder_bottom
        root_P_top = self.root_P_hole + self.root_V_up * self.pre_grasp_height
        distance_to_top = root_P_tip.euclidean_distance(root_P_top)

        artifacts = NodeArtifacts()
        artifacts.constraints.add_point_goal_constraints(
            frame_P_current=root_P_tip,
            frame_P_goal=root_P_top,
            reference_velocity=0.1,
            quadratic_weight=self.weight,
        )
        artifacts.observation = distance_to_top < 0.01
        return artifacts


@dataclass(eq=False, repr=False)
class SlightlyTiltedTask(Task):
    """Tilts the body by a fixed angle relative to the up axis."""

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Root link of the kin chain."""

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Body that is controlled."""

    root_V_up: Vector3 = field(kw_only=True)
    """Axis pointing out of the hole, expressed in the root frame."""

    tilt: float = field(kw_only=True)
    """Target angle in rad between the body's axis and the up axis."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_V_cylinder_z = root_T_tip @ Vector3.Z()
        tilt_error = root_V_cylinder_z.angle_between(self.root_V_up)

        artifacts = NodeArtifacts()
        artifacts.constraints.add_position_constraint(
            expr_current=tilt_error,
            expr_goal=self.tilt,
            reference_velocity=0.1,
            quadratic_weight=self.weight,
        )
        artifacts.observation = sm.abs(tilt_error - self.tilt) <= 0.01
        return artifacts


@dataclass(eq=False, repr=False)
class StayOnLineTask(Task):
    """Keeps the body's bottom on the line between the hole and the pre-grasp point."""

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Root link of the kin chain."""

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Body that is controlled."""

    root_P_hole: Point3 = field(kw_only=True)
    """Position of the hole, expressed in the root frame."""

    root_V_up: Vector3 = field(kw_only=True)
    """Axis pointing out of the hole, expressed in the root frame."""

    pre_grasp_height: float = field(kw_only=True)
    """Distance above the hole along the up axis at which the line ends."""

    cylinder_height: float = field(kw_only=True)
    """Height of the body, used to locate its bottom."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        tip_P_cylinder_bottom = -Vector3.Z() * self.cylinder_height / 2
        root_P_tip = root_T_tip.to_position() + root_T_tip @ tip_P_cylinder_bottom
        root_P_top = self.root_P_hole + self.root_V_up * self.pre_grasp_height
        distance_to_line, root_P_on_line = root_P_tip.distance_to_line_segment(
            self.root_P_hole, root_P_top
        )

        artifacts = NodeArtifacts()
        artifacts.constraints.add_point_goal_constraints(
            frame_P_current=root_P_tip,
            frame_P_goal=root_P_on_line,
            reference_velocity=0.1,
            quadratic_weight=self.weight,
            name="pregrasp",
        )
        artifacts.observation = distance_to_line < 0.01
        return artifacts


@dataclass(eq=False, repr=False)
class InsertTask(Task):
    """Moves the body's bottom down into the hole."""

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Root link of the kin chain."""

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Body that is controlled."""

    root_P_hole: Point3 = field(kw_only=True)
    """Position of the hole, expressed in the root frame."""

    cylinder_height: float = field(kw_only=True)
    """Height of the body, used to locate its bottom."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        tip_P_cylinder_bottom = -Vector3.Z() * self.cylinder_height / 2
        root_P_tip = root_T_tip.to_position() + root_T_tip @ tip_P_cylinder_bottom
        distance_to_hole = root_P_tip.euclidean_distance(self.root_P_hole)

        artifacts = NodeArtifacts()
        artifacts.constraints.add_point_goal_constraints(
            frame_P_current=root_P_tip,
            frame_P_goal=self.root_P_hole,
            reference_velocity=0.01,
            quadratic_weight=self.weight,
            name="insertion",
        )
        artifacts.observation = distance_to_hole < 0.01
        return artifacts


@dataclass(eq=False, repr=False)
class TiltStraightTask(Task):
    """Aligns the body's axis with the hole axis."""

    root_link: KinematicStructureEntity = field(kw_only=True)
    """Root link of the kin chain."""

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """Body that is controlled."""

    root_V_up: Vector3 = field(kw_only=True)
    """Axis pointing out of the hole, expressed in the root frame."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_V_cylinder_z = root_T_tip @ Vector3.Z()
        tilt_error = root_V_cylinder_z.angle_between(self.root_V_up)

        artifacts = NodeArtifacts()
        artifacts.constraints.add_vector_goal_constraints(
            frame_V_current=root_V_cylinder_z,
            frame_V_goal=self.root_V_up,
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

    cylinder_name: Body = field(kw_only=True)
    """Cylinder body to insert. Used as the tip link of the kin chain."""

    hole_point: Point3 = field(kw_only=True)
    """Position of the hole to insert the cylinder into."""

    cylinder_height: Optional[float] = None
    """Height of the cylinder. If None, it is taken from the cylinder's collision shape."""

    up: Optional[Vector3] = None
    """Axis pointing out of the hole. If None, the world's z-axis is used."""

    pre_grasp_height: float = 0.2
    """Distance above the hole along the up axis at which the insertion starts."""

    tilt: float = np.pi / 10
    """Angle in rad by which the cylinder is tilted during the approach."""

    weight: float = DefaultWeights.WEIGHT_ABOVE_CA
    """Task priority relative to other tasks."""

    reach_top: ReachTopTask = field(init=False)
    tilt_task: SlightlyTiltedTask = field(init=False)
    stay_on_line: StayOnLineTask = field(init=False)
    insert_task: InsertTask = field(init=False)
    tilt_straight_task: TiltStraightTask = field(init=False)

    def expand(self, context: MotionStatechartContext) -> None:
        root = context.world.root

        if self.cylinder_height is None:
            self.cylinder_height = self.cylinder_name.collision.shapes[0].height

        if self.up is None:
            up = Vector3.Z(root)
        else:
            up = self.up
        root_P_hole = context.world.transform(self.hole_point, root)
        root_V_up = context.world.transform(up, root)

        self.reach_top = ReachTopTask(
            name="Reach Top",
            root_link=root,
            tip_link=self.cylinder_name,
            root_P_hole=root_P_hole,
            root_V_up=root_V_up,
            pre_grasp_height=self.pre_grasp_height,
            cylinder_height=self.cylinder_height,
            weight=self.weight,
        )
        self.tilt_task = SlightlyTiltedTask(
            name="Slightly Tilted",
            root_link=root,
            tip_link=self.cylinder_name,
            root_V_up=root_V_up,
            tilt=self.tilt,
            weight=self.weight,
        )
        self.stay_on_line = StayOnLineTask(
            name="Stay on Straight Line",
            root_link=root,
            tip_link=self.cylinder_name,
            root_P_hole=root_P_hole,
            root_V_up=root_V_up,
            pre_grasp_height=self.pre_grasp_height,
            cylinder_height=self.cylinder_height,
            weight=self.weight,
        )
        self.insert_task = InsertTask(
            name="Insert",
            root_link=root,
            tip_link=self.cylinder_name,
            root_P_hole=root_P_hole,
            cylinder_height=self.cylinder_height,
            weight=self.weight,
        )
        self.tilt_straight_task = TiltStraightTask(
            name="Tilt Straight",
            root_link=root,
            tip_link=self.cylinder_name,
            root_V_up=root_V_up,
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