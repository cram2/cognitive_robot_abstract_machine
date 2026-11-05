from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cramph.node import (
    CompositeNode,
    EndedByOwner,
    NodeArtifacts,
    expanded_child_field,
)
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.graph_node import ConvergingTask, MotionNodeArtifacts
from giskardpy.motion_statechart.mixins import (
    HasReferenceVelocityParameter,
    HasRootLinkParameter,
    HasTipLinkParameter,
    HasWeightParameter,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.feature_functions import AngleGoal, ReachPoint
from krrood.symbolic_math.symbolic_math import logic_and
from krrood_spatial_math.spatial_types import Point3, Vector3

# %% staying on a line


@dataclass(eq=False, repr=False, kw_only=True)
class StayOnLineTask(
    ConvergingTask,
    HasRootLinkParameter,
    HasTipLinkParameter,
    HasReferenceVelocityParameter,
):
    """
    Keeps a point on the tip on the line segment between two points, with an error in
    meters.
    """

    tip_P_tool: Point3
    """
    The controlled point.
    """

    root_P_start: Point3
    """
    Start of the line segment.
    """

    root_P_end: Point3
    """
    End of the line segment.
    """

    reference_velocity: float = 0.1
    """
    The velocity the constraints of this task are normalized with, in m/s.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build motion constraints that pull the controlled point onto the line segment.

        :return: The artifacts of this task, whose error is the distance between the
            controlled point and the line segment.
        """
        root_T_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_tool = root_T_tip @ self.context.world.transform(
            target_frame=self.tip_link, spatial_object=self.tip_P_tool
        )
        distance_to_line, root_P_on_line = root_P_tool.distance_to_line_segment(
            self.context.world.transform(
                target_frame=self.root_link, spatial_object=self.root_P_start
            ),
            self.context.world.transform(
                target_frame=self.root_link, spatial_object=self.root_P_end
            ),
        )

        artifacts = MotionNodeArtifacts()
        artifacts.geometry.add_point_goal_constraints(
            frame_P_current=root_P_tool,
            frame_P_goal=root_P_on_line,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )
        artifacts.error = distance_to_line
        return artifacts


# %% inserting a cylinder


@dataclass(eq=False, repr=False, kw_only=True)
class InsertCylinder(
    EndedByOwner, CompositeNode, HasTipLinkParameter, HasWeightParameter
):
    """
    Inserts a grasped cylinder into a hole.

    1. Reach a point above the hole while slightly tilted.
    2. Move the cylinder down into the hole, staying on the line through the hole.
    3. Straighten the cylinder so it is aligned with the hole axis.

    Observes True once the cylinder is straightened.
    """

    tip_P_tool: Point3
    """
    The point of the cylinder that leads the insertion.
    """

    hole_point: Point3
    """
    Position of the hole to insert into.
    """

    tip_V_axis: Vector3
    """
    Insertion axis of the cylinder.
    """

    hole_normal: Vector3
    """
    Axis pointing out of the hole.
    """

    pre_grasp_height: float = 0.2
    """
    Distance above the hole along :attr:`hole_normal` at which the insertion starts, in
    m.
    """

    tilt: float = np.pi / 10
    """
    Angle in rad by which the cylinder is tilted during the approach.
    """

    tilt_threshold: float = 0.01
    """
    How far from :attr:`tilt` the cylinder still counts as tilted, in rad.
    """

    approach_velocity: float = 0.1
    """
    Reference velocity while reaching the point above the hole, in m/s.
    """

    insertion_velocity: float = 0.05
    """
    Reference velocity while moving down into the hole, in m/s.
    """

    straightening_velocity: float = 0.025
    """
    Reference angular velocity while straightening the cylinder, in rad/s.
    """

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """
    Above collision avoidance by default.
    """

    reach_top: ReachPoint = expanded_child_field()
    """
    Moves the leading point above the hole.
    """

    tilt_task: AngleGoal = expanded_child_field()
    """
    Holds the cylinder tilted until it reached the hole.
    """

    stay_on_line: StayOnLineTask = expanded_child_field()
    """
    Keeps the leading point on the line between the hole and the point above it.
    """

    insert_task: ReachPoint = expanded_child_field()
    """
    Moves the leading point into the hole.
    """

    tilt_straight_task: AlignPlanes = expanded_child_field()
    """
    Aligns the cylinder with the hole axis.
    """

    def expand(self) -> None:
        root = self.context.world.root
        root_V_up = self.context.world.transform(
            target_frame=root, spatial_object=self.hole_normal
        )
        root_P_hole = self.context.world.transform(
            target_frame=root, spatial_object=self.hole_point
        )
        root_P_top = root_P_hole + root_V_up * self.pre_grasp_height

        self.reach_top = ReachPoint(
            name=f"{self.name}/reach top",
            root_link=root,
            tip_link=self.tip_link,
            controlled_feature=self.tip_P_tool,
            reference_feature=root_P_top,
            reference_velocity=self.approach_velocity,
            weight=self.weight,
        )
        self.tilt_task = AngleGoal(
            name=f"{self.name}/slightly tilted",
            root_link=root,
            tip_link=self.tip_link,
            controlled_feature=self.tip_V_axis,
            reference_feature=root_V_up,
            lower_angle=self.tilt,
            upper_angle=self.tilt,
            threshold=self.tilt_threshold,
            weight=self.weight,
        )
        self.stay_on_line = StayOnLineTask(
            name=f"{self.name}/stay on straight line",
            root_link=root,
            tip_link=self.tip_link,
            tip_P_tool=self.tip_P_tool,
            root_P_start=root_P_hole,
            root_P_end=root_P_top,
            weight=self.weight,
        )
        self.insert_task = ReachPoint(
            name=f"{self.name}/insert",
            root_link=root,
            tip_link=self.tip_link,
            controlled_feature=self.tip_P_tool,
            reference_feature=root_P_hole,
            reference_velocity=self.insertion_velocity,
            weight=self.weight,
        )
        self.tilt_straight_task = AlignPlanes(
            name=f"{self.name}/tilt straight",
            root_link=root,
            tip_link=self.tip_link,
            controlled_feature=self.tip_V_axis,
            reference_feature=root_V_up,
            reference_velocity=self.straightening_velocity,
            weight=self.weight,
        )
        self._add_children_to_statechart(
            [
                self.reach_top,
                self.tilt_task,
                self.stay_on_line,
                self.insert_task,
                self.tilt_straight_task,
            ]
        )

        above_hole_and_tilted = logic_and(
            self.reach_top.observes_true, self.tilt_task.observes_true
        )
        bottom_reached = logic_and(
            self.insert_task.observes_true, self.stay_on_line.observes_true
        )
        self.reach_top.success_condition = above_hole_and_tilted
        self.tilt_task.success_condition = bottom_reached
        self.insert_task.start_condition = above_hole_and_tilted
        self.tilt_straight_task.start_condition = bottom_reached

    def build_artifacts(self) -> NodeArtifacts:
        """
        Observe what straightening the cylinder observes.
        """
        return NodeArtifacts(observation=self.tilt_straight_task.observation_variable)
