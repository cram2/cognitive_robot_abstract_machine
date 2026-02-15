from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, unique

import krrood.symbolic_math.symbolic_math as sm
import numpy as np

from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import (
    Goal,
    Task,
    NodeArtifacts,
    DebugExpression,
    MotionStatechartNode,
)
from semantic_digital_twin.spatial_types import (
    Point3,
    RotationMatrix,
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@unique
class MoveAroundHingeAlign(int, Enum):
    """Defines when to align the gripper with the hinge rotation axis."""

    LAST = 0  # Align only at the last waypoint
    ALL = 1  # Align at all waypoints


@dataclass(eq=False, repr=False)
class _PointReachedMonitor(MotionStatechartNode):
    """Internal monitor node that becomes true when tip reaches a target point."""

    root_P_tip: Point3 = field(kw_only=True)
    root_P_goal: Point3 = field(kw_only=True)
    threshold: float = field(default=0.01, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        distance = self.root_P_tip.euclidean_distance(self.root_P_goal)
        return NodeArtifacts(observation=distance < sm.Scalar(self.threshold))


@dataclass(eq=False, repr=False)
class _WaypointTask(Task):
    """
    Internal task node that drives the tip toward a waypoint position,
    optionally aligning a gripper axis with a rotation axis.
    """

    root_link: KinematicStructureEntity = field(kw_only=True)
    tip_link: KinematicStructureEntity = field(kw_only=True)
    root_P_goal: Point3 = field(kw_only=True)
    reference_linear_velocity: float = field(kw_only=True)
    weight: float = field(kw_only=True)
    root_V_tip_grasp_axis: Optional[Vector3] = field(default=None, kw_only=True)
    root_V_object_rotation_axis: Optional[Vector3] = field(default=None, kw_only=True)
    reference_angular_velocity: float = field(default=0.5, kw_only=True)
    align: bool = field(default=False, kw_only=True)

    def build(self, context: BuildContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()

        # FIX (issue #3 note): this recomputes FK independently from expand(), which is
        # intentional — _WaypointTask.build() may be called in a different context scope.
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_tip = root_T_tip.to_position()

        artifacts.constraints.add_point_goal_constraints(
            frame_P_current=root_P_tip,
            frame_P_goal=self.root_P_goal,
            reference_velocity=self.reference_linear_velocity,
            weight=self.weight,
        )

        if (
            self.align
            and self.root_V_tip_grasp_axis is not None
            and self.root_V_object_rotation_axis is not None
        ):
            artifacts.constraints.add_vector_goal_constraints(
                frame_V_current=self.root_V_tip_grasp_axis,
                frame_V_goal=self.root_V_object_rotation_axis,
                reference_velocity=self.reference_angular_velocity,
                weight=self.weight,
                name=self.name,
            )

        return artifacts


@dataclass
class MoveAroundHinge(Goal):
    """
    Moves the end-effector around a hinge (like a door handle) following a curved trajectory.

    Creates multiple waypoints that trace an arc around the hinge axis, optionally aligning
    the gripper orientation with the rotation axis for better pushing capability.
    """

    handle_name: Body
    root_link: Body
    tip_link: Body
    tip_gripper_axis: Optional[Vector3] = None
    reference_linear_velocity: float = 0.1
    reference_angular_velocity: float = 0.5
    weight: float = DefaultWeights.WEIGHT_BELOW_CA
    goal_angle: Optional[float] = None
    multipliers: Optional[List[Tuple[float, float, str]]] = None
    offset: Optional[Vector3] = None
    align_gripper: MoveAroundHingeAlign = MoveAroundHingeAlign.LAST
    _waypoint_debug_expressions: List[DebugExpression] = field(
        default_factory=list, init=False, repr=False
    )

    def expand(self, context: BuildContext) -> None:
        """Create and add child tasks and monitors for each waypoint."""

        if self.multipliers is None:
            self.multipliers = [
                (11 / 10, -0.7, "down_short"),
                (7 / 5, -0.3, "down_long"),
                (7 / 5, 0.4, "up_long"),
            ]

        handle_connection = self.handle_name.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        door_body: KinematicStructureEntity = handle_connection.parent
        hinge_connection = door_body.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        root_P_tip = root_T_tip.to_position()
        object_joint_angle = context.world.state[hinge_connection.dof.id].position

        # axis is a Vector3 — convert to numpy for from_iterable
        object_V_object_rotation_axis = Vector3.from_iterable(
            hinge_connection.axis.to_np()
        )
        root_T_door_expr = context.world.compose_forward_kinematics_expression(
            self.root_link, door_body
        )

        # FIX (issue #2): offset is a Vector3 in an arbitrary frame. To apply it as
        # a translation on root_T_door_expr, we must first express it in the root frame.
        # HTM @ Vector3 applies only the rotation (w=0), giving us the rotated direction.
        # We then build a pure-translation HTM from that root-frame vector and premultiply,
        # matching the original: root_T_offset.dot(root_T_door_expr).
        if self.offset is not None:
            root_V_offset = root_T_door_expr @ self.offset
            root_T_offset = HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=Point3(x=root_V_offset.x, y=root_V_offset.y, z=root_V_offset.z),
                rotation_matrix=None,
            )
            root_T_door_expr = root_T_offset @ root_T_door_expr

        root_V_tip_grasp_axis = None
        root_V_object_rotation_axis = None
        if self.tip_gripper_axis is not None:
            self.tip_gripper_axis.scale(1)
            root_V_tip_grasp_axis = root_T_tip @ self.tip_gripper_axis
            root_V_object_rotation_axis = (
                root_T_door_expr @ object_V_object_rotation_axis
            )

        waypoints = self._compute_waypoints(
            context,
            door_body,
            object_V_object_rotation_axis,
            object_joint_angle,
            root_T_door_expr,
        )

        self._create_waypoint_tasks(
            waypoints, root_P_tip, root_V_tip_grasp_axis, root_V_object_rotation_axis
        )

    def _compute_waypoints(
        self,
        context: BuildContext,
        door_body: KinematicStructureEntity,
        object_V_object_rotation_axis: Vector3,
        object_joint_angle: float,
        root_T_door_expr: HomogeneousTransformationMatrix,
    ) -> List[Tuple[Point3, str]]:
        """
        Compute waypoints for moving around the hinge.

        :return: List of (waypoint_position, waypoint_name) tuples
        """
        door_P_handle = context.world.compute_forward_kinematics(
            door_body, self.handle_name
        )

        # FIX (issue #4): call .to_np() on the full Point3 (returns shape (4,1)) and
        # slice+flatten, rather than calling .to_np() on individual sm.Scalar components
        # (which may return 0-d or shape-(1,) arrays and produce an ambiguous stack shape).
        temp_point = door_P_handle.to_np()[:3].flatten()

        direction_axis = int(np.argmax(np.abs(temp_point)))

        waypoints = []

        for axis_multi, angle_multi, goal_name in self.multipliers:
            door_P_intermediate_point = np.zeros(3)
            door_P_intermediate_point[direction_axis] = (
                temp_point[direction_axis] * axis_multi
            )
            door_P_intermediate_point = Point3(
                x=door_P_intermediate_point[0],
                y=door_P_intermediate_point[1],
                z=door_P_intermediate_point[2],
            )

            if self.goal_angle is None:
                desired_angle = object_joint_angle * angle_multi
            else:
                desired_angle = self.goal_angle * angle_multi

            door_R_door_rotated = RotationMatrix.from_axis_angle(
                axis=object_V_object_rotation_axis, angle=desired_angle
            )
            door_T_door_rotated = (
                HomogeneousTransformationMatrix.from_point_rotation_matrix(
                    point=None, rotation_matrix=door_R_door_rotated
                )
            )

            door_rotated_P_top = (
                door_T_door_rotated.inverse() @ door_P_intermediate_point
            )

            # FIX (issue #1): apply the full root_T_door_expr (translation + rotation),
            # not just its rotation component. The previous code stripped the translation
            # by extracting only .to_rotation_matrix(), placing all waypoints at the origin.
            root_P_top = root_T_door_expr @ door_rotated_P_top

            waypoints.append((root_P_top, goal_name))

            self._waypoint_debug_expressions.append(
                DebugExpression(
                    name=f"goal_point_{goal_name}",
                    expression=root_P_top,
                    color=Color(0, 0.5, 0.5, 1),
                )
            )

        return waypoints

    def _create_waypoint_tasks(
        self,
        waypoints: List[Tuple[Point3, str]],
        root_P_tip: Point3,
        root_V_tip_grasp_axis: Optional[Vector3],
        root_V_object_rotation_axis: Optional[Vector3],
    ):
        """
        Create tasks and monitors for each waypoint.

        :param waypoints: List of (waypoint_position, waypoint_name) tuples
        :param root_P_tip: Current tip position expression
        :param root_V_tip_grasp_axis: Gripper axis in root frame (or None)
        :param root_V_object_rotation_axis: Hinge rotation axis in root frame (or None)
        """
        num_waypoints = len(waypoints)
        old_position_monitor = None

        for i, (root_P_top, goal_name) in enumerate(waypoints):
            is_first = i == 0
            is_last = i == num_waypoints - 1

            should_align = (
                (is_last and self.align_gripper == MoveAroundHingeAlign.LAST)
                or (self.align_gripper == MoveAroundHingeAlign.ALL)
            ) and self.tip_gripper_axis is not None

            task = _WaypointTask(
                name=goal_name,
                root_link=self.root_link,
                tip_link=self.tip_link,
                root_P_goal=root_P_top,
                reference_linear_velocity=self.reference_linear_velocity,
                weight=self.weight,
                root_V_tip_grasp_axis=root_V_tip_grasp_axis,
                root_V_object_rotation_axis=root_V_object_rotation_axis,
                reference_angular_velocity=self.reference_angular_velocity,
                align=should_align,
            )
            self.add_node(task)

            position_monitor = _PointReachedMonitor(
                name=f"{goal_name}_pos_monitor",
                root_P_tip=root_P_tip,
                root_P_goal=root_P_top,
            )
            if old_position_monitor is not None:
                position_monitor.start_condition = (
                    old_position_monitor.observation_variable
                )
            self.add_node(position_monitor)

            if is_first:
                task.start_condition = self.start_condition
                task.end_condition = position_monitor.observation_variable
            elif is_last:
                task.start_condition = old_position_monitor.observation_variable
                task.end_condition = (
                    self.end_condition | position_monitor.observation_variable
                )
            else:
                task.start_condition = old_position_monitor.observation_variable
                task.end_condition = position_monitor.observation_variable

            old_position_monitor = position_monitor

        self.observation_expression = old_position_monitor.observation_variable

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(debug_expressions=self._waypoint_debug_expressions)
