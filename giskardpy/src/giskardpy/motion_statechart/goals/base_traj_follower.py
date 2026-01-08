from __future__ import division

from dataclasses import field, dataclass

from line_profiler import profile

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Goal
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import Connection


@dataclass
class BaseTrajFollower(Goal):
    connection: Connection = field(kw_only=True)
    track_only_velocity: bool = False
    weight: float = DefaultWeights.WEIGHT_ABOVE_CA

    def __post_init__(self):
        self.joint: OmniDrive = self.connection
        self.odom_link = self.joint.parent
        self.base_footprint_link = self.joint.child
        self.task = Task(name="base")
        self.add_task(self.task)
        trajectory = trajectory  # fixme god_map?
        self.trajectory_length = len(trajectory)
        self.add_trans_constraints()
        self.add_rot_constraints()

    @profile
    def x_symbol(
        self,
        t: int,
        free_variable_name: PrefixedName,
        derivative: Derivatives = Derivatives.position,
    ) -> sm.FloatVariable:
        expr = (
            f"god_map.trajectory.get_exact({t})['{free_variable_name}'][{derivative}]"
        )
        return symbol_manager.get_symbol(expr)

    @profile
    def current_traj_point(
        self,
        free_variable_name: PrefixedName,
        start_t: float,
        derivative: Derivatives = Derivatives.position,
    ) -> sm.Scalar:
        time = context.time_symbol
        b_result_cases = []
        for t in range(self.trajectory_length):
            b = t * context.qp_controller.mpc_dt
            eq_result = self.x_symbol(t, free_variable_name, derivative)
            b_result_cases.append((b, eq_result))
            # FIXME if less eq cases behavior changed
        return sm.if_less_eq_cases(
            a=time + start_t,
            b_result_cases=b_result_cases,
            else_result=self.x_symbol(
                self.trajectory_length - 1, free_variable_name, derivative
            ),
        )

    @profile
    def make_odom_T_base_footprint_goal(
        self, t_in_s: float, derivative: Derivatives = Derivatives.position
    ):
        x = self.current_traj_point(self.joint.x.name, t_in_s, derivative)
        if isinstance(self.joint, OmniDrive) or derivative == 0:
            y = self.current_traj_point(self.joint.y.name, t_in_s, derivative)
        else:
            y = 0
        rot = self.current_traj_point(self.joint.yaw.name, t_in_s, derivative)
        odom_T_base_footprint_goal = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=x, y=y, yaw=rot
        )
        return odom_T_base_footprint_goal

    @profile
    def make_map_T_base_footprint_goal(
        self, t_in_s: float, derivative: Derivatives = Derivatives.position
    ):
        odom_T_base_footprint_goal = self.make_odom_T_base_footprint_goal(
            t_in_s, derivative
        )
        map_T_odom = context.world._forward_kinematic_manager.compose_expression(
            context.world.root_link_name, self.odom_link
        )
        return map_T_odom @ odom_T_base_footprint_goal

    @profile
    def trans_error_at(self, t_in_s: float):
        odom_T_base_footprint_goal = self.make_odom_T_base_footprint_goal(t_in_s)
        map_T_odom = context.world.compute_forward_kinematics(
            context.world.root_link_name, self.odom_link
        )
        map_T_base_footprint_goal = map_T_odom @ odom_T_base_footprint_goal
        map_T_base_footprint_current = (
            context.world._forward_kinematic_manager.compose_expression(
                context.world.root_link_name, self.base_footprint_link
            )
        )

        frame_P_goal = map_T_base_footprint_goal.to_position()
        frame_P_current = map_T_base_footprint_current.to_position()
        error = (frame_P_goal - frame_P_current) / context.qp_controller.mpc_dt
        return error[0], error[1]

    @profile
    def add_trans_constraints(self):
        errors_x = []
        errors_y = []
        map_T_base_footprint = (
            context.world._forward_kinematic_manager.compose_expression(
                context.world.root_link_name, self.base_footprint_link
            )
        )
        for t in range(context.qp_controller.prediction_horizon):
            x = self.current_traj_point(
                self.joint.x_velocity.name,
                t * context.qp_controller.mpc_dt,
                Derivatives.velocity,
            )
            if isinstance(self.joint, OmniDrive):
                y = self.current_traj_point(
                    self.joint.y_velocity.name,
                    t * context.qp_controller.mpc_dt,
                    Derivatives.velocity,
                )
            else:
                y = 0
            base_footprint_P_vel = Vector3((x, y, 0))
            map_P_vel = map_T_base_footprint @ base_footprint_P_vel
            if t == 0 and not self.track_only_velocity:
                actual_error_x, actual_error_y = self.trans_error_at(0)
                errors_x.append(map_P_vel[0] + actual_error_x)
                errors_y.append(map_P_vel[1] + actual_error_y)
            else:
                errors_x.append(map_P_vel[0])
                errors_y.append(map_P_vel[1])
        weight_vel = DefaultWeights.WEIGHT_ABOVE_CA
        lba_x = errors_x
        uba_x = errors_x
        lba_y = errors_y
        uba_y = errors_y

        self.task.add_velocity_constraint(
            lower_velocity_limit=lba_x,
            upper_velocity_limit=uba_x,
            weight=weight_vel,
            task_expression=map_T_base_footprint.to_position().x,
            velocity_limit=0.5,
            name="/vel x",
        )
        if isinstance(self.joint, OmniDrive):
            self.task.add_velocity_constraint(
                lower_velocity_limit=lba_y,
                upper_velocity_limit=uba_y,
                weight=weight_vel,
                task_expression=map_T_base_footprint.to_position().y,
                velocity_limit=0.5,
                name="/vel y",
            )

    @profile
    def rot_error_at(self, t_in_s: int):
        rotation_goal = self.current_traj_point(self.joint.yaw.name, t_in_s)
        rotation_current = self.joint.yaw.variables.position
        error = (
            sm.shortest_angular_distance(rotation_current, rotation_goal)
            / context.qp_controller.mpc_dt
        )
        return error

    @profile
    def add_rot_constraints(self):
        errors = []
        for t in range(context.qp_controller.prediction_horizon):
            errors.append(
                self.current_traj_point(
                    self.joint.yaw.name,
                    t * context.qp_controller.mpc_dt,
                    Derivatives.velocity,
                )
            )
            if t == 0 and not self.track_only_velocity:
                errors[-1] += self.rot_error_at(t)
        self.task.add_velocity_constraint(
            lower_velocity_limit=errors,
            upper_velocity_limit=errors,
            weight=DefaultWeights.WEIGHT_BELOW_CA,
            task_expression=self.joint.yaw.variables.position,
            velocity_limit=0.5,
            name="/rot",
        )
