from __future__ import division

from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.binding_policy import (
    GoalBindingPolicy,
    ForwardKinematicsBinding,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class GraspBar(Task):
    """
    Grasps a bar-like object with specified alignment and positioning constraints.

    The tip_link can be positioned at any point along the bar_axis within bar_center ± bar_length.
    The tip_grasp_axis will be aligned with bar_axis, but rotation around it is allowed.

    This is similar to CartesianPose but with more freedom - allowing translation along the bar
    and rotation around the grasp axis.
    """

    root_link: Body = field(kw_only=True)
    """
    Root link of the kinematic chain.
    """

    tip_link: Body = field(kw_only=True)
    """
    Tip link of the kinematic chain.
    """

    tip_grasp_axis: Vector3 = field(kw_only=True)
    """
    Axis of tip_link that will be aligned with bar_axis.
    """

    bar_center: Point3 = field(kw_only=True)
    """
    Center of the bar to be grasped.
    """

    bar_axis: Vector3 = field(kw_only=True)
    """
    Alignment of the bar to be grasped.
    """

    bar_length: float = field(kw_only=True)
    """
    Length of the bar to be grasped.
    """

    threshold: float = field(default=0.01, kw_only=True)
    """
    Distance threshold for goal achievement in meters.
    """

    reference_linear_velocity: float = field(default=0.1, kw_only=True)
    """
    Reference velocity for position constraints in m/s.
    """

    reference_angular_velocity: float = field(default=0.5, kw_only=True)
    """
    Reference velocity for orientation constraints in rad/s.
    """

    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    """
    Task priority relative to other tasks.
    """

    binding_policy: GoalBindingPolicy = field(
        default=GoalBindingPolicy.Bind_on_start, kw_only=True
    )
    """
    Describes when the goal is computed. See GoalBindingPolicy for more information.
    """

    _fk_binding_bar_center: ForwardKinematicsBinding = field(
        kw_only=True, init=False, repr=False
    )
    """
    Forward kinematics binding for the bar center reference frame.
    """

    _fk_binding_bar_axis: ForwardKinematicsBinding = field(
        kw_only=True, init=False, repr=False
    )
    """
    Forward kinematics binding for the bar axis reference frame.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build motion constraints for grasping the bar.

        :param context: Provides access to world model and kinematic expressions.
        :return: NodeArtifacts containing constraints and observation conditions.
        """
        artifacts = NodeArtifacts()

        self._fk_binding_bar_center = ForwardKinematicsBinding(
            name=PrefixedName("root_T_bar_center_ref", str(self.name)),
            root=self.root_link,
            tip=self.bar_center.reference_frame,
            build_context=context,
        )

        self._fk_binding_bar_axis = ForwardKinematicsBinding(
            name=PrefixedName("root_T_bar_axis_ref", str(self.name)),
            root=self.root_link,
            tip=self.bar_axis.reference_frame,
            build_context=context,
        )

        root_P_bar_center = self._fk_binding_bar_center.root_T_tip @ self.bar_center
        """
        The center point of the bar expressed in the root link frame.
        """

        root_V_bar_axis = self._fk_binding_bar_axis.root_T_tip @ self.bar_axis
        root_V_bar_axis = root_V_bar_axis / root_V_bar_axis.norm()
        """
        The normalized direction vector of the bar expressed in the root link frame.
        """

        tip_V_tip_grasp_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.tip_grasp_axis
        )
        tip_V_tip_grasp_axis = tip_V_tip_grasp_axis / tip_V_tip_grasp_axis.norm()
        """
        The normalized grasp axis of the tip link expressed in the tip link frame.
        """

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        """
        The transformation from root link to tip link.
        """

        root_V_tip_normal = root_T_tip @ tip_V_tip_grasp_axis
        """
        The tip's grasp axis expressed in the root link frame.
        """

        artifacts.constraints.add_vector_goal_constraints(
            name=f"{self.name}/align_grasp_axis",
            frame_V_current=root_V_tip_normal,
            frame_V_goal=root_V_bar_axis,
            reference_velocity=self.reference_angular_velocity,
            weight=self.weight,
        )

        root_P_tip = root_T_tip.to_position()
        """
        The position of the tip link expressed in the root link frame.
        """

        root_P_line_start = root_P_bar_center + root_V_bar_axis * self.bar_length / 2
        """
        The start point of the bar line segment (bar_center + bar_length/2 along bar_axis).
        """

        root_P_line_end = root_P_bar_center - root_V_bar_axis * self.bar_length / 2
        """
        The end point of the bar line segment (bar_center - bar_length/2 along bar_axis).
        """

        dist, nearest = root_P_tip.distance_to_line_segment(
            root_P_line_start, root_P_line_end
        )
        """
        Distance from tip to the nearest point on the bar line segment, and the nearest point itself.
        """

        artifacts.constraints.add_point_goal_constraints(
            name=f"{self.name}/position_on_bar",
            frame_P_current=root_P_tip,
            frame_P_goal=nearest,
            reference_velocity=self.reference_linear_velocity,
            weight=self.weight,
        )

        artifacts.observation = dist < sm.Scalar(self.threshold)

        return artifacts

    def on_start(self, context: MotionStatechartContext):
        if self.binding_policy == GoalBindingPolicy.Bind_on_start:
            self._fk_binding_bar_center.bind(context)
            self._fk_binding_bar_axis.bind(context)


@dataclass
class GraspBarOffset(Task):
    """
    Grasps a bar-like object with an offset from the bar center.
    """

    root_link: Body = field(kw_only=True)
    """
    Root link of the kinematic chain.
    """

    tip_link: Body = field(kw_only=True)
    """
    Tip link of the kinematic chain.
    """

    tip_grasp_axis: Vector3 = field(kw_only=True)
    """
    Axis of tip_link that will be aligned with bar_axis.
    """

    bar_center: Point3 = field(kw_only=True)
    """
    Center of the bar to be grasped.
    """

    bar_axis: Vector3 = field(kw_only=True)
    """
    Alignment of the bar to be grasped.
    """

    bar_length: float = field(kw_only=True)
    """
    Length of the bar to be grasped.
    """

    grasp_axis_offset: Vector3 = field(kw_only=True)
    """
    Offset of the tip_link to the bar_center along the grasp axis.
    """

    handle_link: Body = field(kw_only=True)
    """
    Link where the handle/bar is attached.
    """

    threshold: float = field(default=0.01, kw_only=True)
    """
    Distance threshold for goal achievement in meters.
    """

    reference_linear_velocity: float = field(default=0.1, kw_only=True)
    """
    Reference velocity for position constraints in m/s.
    """

    reference_angular_velocity: float = field(default=0.5, kw_only=True)
    """
    Reference velocity for orientation constraints in rad/s.
    """

    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    """
    Task priority relative to other tasks.
    """

    binding_policy: GoalBindingPolicy = field(
        default=GoalBindingPolicy.Bind_on_start, kw_only=True
    )
    """
    Describes when the goal is computed. See GoalBindingPolicy for more information.
    """

    _fk_binding_handle: ForwardKinematicsBinding = field(
        kw_only=True, init=False, repr=False
    )
    """
    Forward kinematics binding for the handle link.
    """

    _fk_binding_bar_axis: ForwardKinematicsBinding = field(
        kw_only=True, init=False, repr=False
    )
    """
    Forward kinematics binding for the bar axis reference frame.
    """

    _fk_binding_grasp_offset: ForwardKinematicsBinding = field(
        kw_only=True, init=False, repr=False
    )
    """
    Forward kinematics binding for the grasp axis offset reference frame.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build motion constraints for grasping the bar with offset.

        :param context: Provides access to world model and kinematic expressions.
        :return: NodeArtifacts containing constraints and observation conditions.
        """
        artifacts = NodeArtifacts()

        self._fk_binding_handle = ForwardKinematicsBinding(
            name=PrefixedName("root_T_handle", str(self.name)),
            root=self.root_link,
            tip=self.handle_link,
            build_context=context,
        )

        handle_P_bar_center = context.world.transform(
            target_frame=self.handle_link, spatial_object=self.bar_center
        )
        """
        The bar center position expressed in the handle link frame.
        """

        root_P_bar_center = self._fk_binding_handle.root_T_tip @ handle_P_bar_center
        """
        The bar center position expressed in the root link frame.
        """

        self._fk_binding_bar_axis = ForwardKinematicsBinding(
            name=PrefixedName("root_T_bar_axis_ref", str(self.name)),
            root=self.root_link,
            tip=self.bar_axis.reference_frame,
            build_context=context,
        )

        self._fk_binding_grasp_offset = ForwardKinematicsBinding(
            name=PrefixedName("root_T_grasp_offset_ref", str(self.name)),
            root=self.root_link,
            tip=self.grasp_axis_offset.reference_frame,
            build_context=context,
        )

        root_V_grasp_offset = (
            self._fk_binding_grasp_offset.root_T_tip @ self.grasp_axis_offset
        )
        """
        The grasp axis offset vector expressed in the root link frame.
        """

        tip_V_tip_grasp_axis = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.tip_grasp_axis
        )
        tip_V_tip_grasp_axis = tip_V_tip_grasp_axis / tip_V_tip_grasp_axis.norm()
        """
        The normalized grasp axis of the tip link expressed in the tip link frame.
        """

        root_V_bar_axis = self._fk_binding_bar_axis.root_T_tip @ self.bar_axis
        root_V_bar_axis = root_V_bar_axis / root_V_bar_axis.norm()
        """
        The normalized direction vector of the bar expressed in the root link frame.
        """

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        """
        The transformation from root link to tip link.
        """

        root_V_tip_normal = root_T_tip @ tip_V_tip_grasp_axis
        """
        The tip's grasp axis expressed in the root link frame.
        """

        artifacts.constraints.add_vector_goal_constraints(
            name=f"{self.name}/align_grasp_axis",
            frame_V_current=root_V_tip_normal,
            frame_V_goal=root_V_bar_axis,
            reference_velocity=self.reference_angular_velocity,
            weight=self.weight,
        )

        root_P_tip = root_T_tip.to_position()
        """
        The position of the tip link expressed in the root link frame.
        """

        root_P_bar_center_offset = root_P_bar_center + root_V_grasp_offset
        """
        The offset bar center position (bar_center + grasp_axis_offset) expressed in the root link frame.
        """

        root_P_line_start = (
            root_P_bar_center_offset + root_V_bar_axis * self.bar_length / 2
        )
        """
        The start point of the bar line segment (offset_bar_center + bar_length/2 along bar_axis).
        """

        root_P_line_end = (
            root_P_bar_center_offset - root_V_bar_axis * self.bar_length / 2
        )
        """
        The end point of the bar line segment (offset_bar_center - bar_length/2 along bar_axis).
        """

        dist, nearest = root_P_tip.distance_to_line_segment(
            root_P_line_start, root_P_line_end
        )
        """
        Distance from tip to the nearest point on the bar line segment, and the nearest point itself.
        """

        artifacts.constraints.add_point_goal_constraints(
            name=f"{self.name}/position_on_bar",
            frame_P_current=root_P_tip,
            frame_P_goal=nearest,
            reference_velocity=self.reference_linear_velocity,
            weight=self.weight,
        )

        artifacts.observation = dist < sm.Scalar(self.threshold)

        return artifacts

    def on_start(self, context: MotionStatechartContext):
        if self.binding_policy == GoalBindingPolicy.Bind_on_start:
            self._fk_binding_handle.bind(context)
            self._fk_binding_bar_axis.bind(context)
            self._fk_binding_grasp_offset.bind(context)
