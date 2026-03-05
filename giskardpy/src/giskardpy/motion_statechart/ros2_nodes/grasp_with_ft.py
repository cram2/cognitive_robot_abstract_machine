from dataclasses import dataclass, field
from typing import Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import (
    Goal,
    NodeArtifacts,
    CancelMotion,
)
from giskardpy.motion_statechart.monitors.payload_monitors import CountSeconds
from giskardpy.motion_statechart.ros2_nodes.payload_force_torque import (
    PayloadForceTorque,
    ForceTorqueThresholds,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.grasp_bar import GraspBarOffset
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.ros2_nodes.handle_offset_correction import (
    HandleOffsetCorrection,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
)
from semantic_digital_twin.world_description.world_entity import Body

import krrood.symbolic_math.symbolic_math as sm


@dataclass(eq=False, repr=False)
class GraspWithForceTorqueGoal(Goal):
    root_link: Body = field(kw_only=True)
    tip_link: Body = field(kw_only=True)
    handle_name: Body = field(kw_only=True)
    tip_grasp_axis: Vector3 = field(kw_only=True)
    bar_axis: Vector3 = field(kw_only=True)
    tip_retract: Point3 = field(kw_only=True)
    handle_align_axis: Vector3 = field(kw_only=True)
    tip_align_axis: Vector3 = field(kw_only=True)
    grasp_axis_offset: Vector3 = field(kw_only=True)
    pre_grasp_axis_offset: Vector3 = field(kw_only=True)
    hinge_joint: ActiveConnection1DOF = field(kw_only=True)
    bar_length: float = field(default=0.01, kw_only=True)
    timeout: float = field(default=10.0, kw_only=True)
    ft_grasp_ref_speed: float = field(default=1.0, kw_only=True)
    tip_push: Optional[Point3] = field(default=None, kw_only=True)
    camera_link: Optional[Body] = field(default=None, kw_only=True)
    handle_correction_offset: Optional[Vector3] = field(default=None, kw_only=True)
    door_move_connection: Optional[Connection6DoF] = field(default=None, kw_only=True)
    ft_topic: str = field(default="/hsrb/wrist_wrench/compensated", kw_only=True)

    def expand(self, context: MotionStatechartContext) -> None:
        reference_linear_velocity = 0.1 * self.ft_grasp_ref_speed
        reference_angular_velocity = 0.5 * self.ft_grasp_ref_speed

        bar_center = Point3(reference_frame=self.handle_name)

        # Lock the hinge joint while grasping
        jpl_hinge_lock = JointPositionList(
            goal_state=JointState.from_mapping({self.hinge_joint: 0.0}),
            name="Lock Hinge while grasp",
        )
        jpl_hinge_lock.start_condition = self.start_condition
        self.add_node(jpl_hinge_lock)

        # Pre-grasp: position tip along bar with offset
        pre_grasp = GraspBarOffset(
            name="pre grasp",
            root_link=self.root_link,
            tip_link=self.tip_link,
            tip_grasp_axis=self.tip_grasp_axis,
            bar_center=bar_center,
            bar_axis=self.bar_axis,
            bar_length=self.bar_length,
            grasp_axis_offset=self.grasp_axis_offset,
            handle_link=self.handle_name,
            reference_linear_velocity=reference_linear_velocity,
            reference_angular_velocity=reference_angular_velocity,
        )
        pre_grasp.start_condition = self.start_condition
        self.add_node(pre_grasp)

        # Wire HandleOffsetCorrection if camera_link and related args are provided
        if (
            self.camera_link is not None
            and self.handle_correction_offset is not None
            and self.door_move_connection is not None
        ):
            handle_offset_correction = HandleOffsetCorrection(
                root_link=self.root_link,
                tip_link=self.camera_link,
                door_move_connection=self.door_move_connection,
                goal_vector=self.handle_correction_offset,
                name="handle offset correction",
            )
            handle_offset_correction.start_condition = self.start_condition
            self.add_node(handle_offset_correction)
            next_condition = handle_offset_correction.observation_variable
        else:
            next_condition = pre_grasp.observation_variable

        # Align tip plane with handle plane
        ap_pre_grasp = AlignPlanes(
            name="grasp align",
            root_link=self.root_link,
            tip_link=self.tip_link,
            tip_normal=self.tip_align_axis,
            goal_normal=self.handle_align_axis,
        )
        ap_pre_grasp.start_condition = self.start_condition
        self.add_node(ap_pre_grasp)

        # Align tip grasp axis with bar axis
        ap_tip_grasp = AlignPlanes(
            name="tip grasp align",
            root_link=self.root_link,
            tip_link=self.tip_link,
            tip_normal=self.tip_grasp_axis,
            goal_normal=self.bar_axis,
        )
        ap_tip_grasp.start_condition = self.start_condition
        self.add_node(ap_tip_grasp)

        # Timeout monitor — replaces Sleep
        sleep_cancel = CountSeconds(
            seconds=self.timeout,
            name="ft sleep cancel",
        )
        sleep_cancel.start_condition = next_condition
        self.add_node(sleep_cancel)

        ft_monitor = PayloadForceTorque(
            threshold_enum=ForceTorqueThresholds.DOOR.value,
            topic_name=self.ft_topic,
            name="grasp ft monitor",
        )
        ft_monitor.start_condition = next_condition
        self.add_node(ft_monitor)

        # FT grasp motion: push tip towards bar until ft_monitor triggers
        if self.tip_push is not None:
            ft_grasp = CartesianPosition(
                root_link=self.root_link,
                tip_link=self.tip_link,
                goal_point=self.tip_push,
                name="ft grasp",
                reference_velocity=reference_linear_velocity,
                threshold=0.001,
            )
        else:
            ft_grasp = GraspBarOffset(
                name="ft grasp",
                root_link=self.root_link,
                tip_link=self.tip_link,
                tip_grasp_axis=self.tip_grasp_axis,
                bar_center=bar_center,
                bar_axis=self.bar_axis,
                bar_length=self.bar_length,
                grasp_axis_offset=self.pre_grasp_axis_offset,
                handle_link=self.handle_name,
                reference_linear_velocity=reference_linear_velocity,
                reference_angular_velocity=reference_angular_velocity,
            )
        ft_grasp.start_condition = next_condition
        self.add_node(ft_grasp)  # ← register first
        ft_grasp.end_condition = ft_monitor.observation_variable

        # Retract after ft threshold reached
        retract = CartesianPosition(
            root_link=self.root_link,
            tip_link=self.tip_link,
            goal_point=self.tip_retract,
            name="retract after ft",
            reference_velocity=reference_linear_velocity,
            threshold=0.001,
        )
        retract.start_condition = ft_monitor.observation_variable
        self.add_node(retract)  # ← register first
        retract.end_condition = retract.observation_variable

        # Cancel if timeout fires but ft_monitor never triggered
        ft_cancel = CancelMotion(
            exception=Exception("Door not touched!"),
            name="FT CancelMotion",
        )
        ft_cancel.start_condition = sm.trinary_logic_and(
            sm.trinary_logic_not(ft_monitor.observation_variable),
            sleep_cancel.observation_variable,
        )
        self.add_node(ft_cancel)

        self._retract = retract

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self._retract.observation_variable)
