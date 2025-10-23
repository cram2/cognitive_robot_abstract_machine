from dataclasses import field
from typing import Optional, Dict, List, Tuple, Union

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.monitors.joint_monitors import JointGoalReached
from giskardpy.motion_statechart.tasks.task import Task, WEIGHT_BELOW_CA
from giskardpy.utils.decorators import validated_dataclass
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection,
    PrismaticConnection,
    ActiveConnection1DOF,
)


@validated_dataclass
class JointPositionList(Task):
    goal_state: Dict[ActiveConnection1DOF, Union[int, float]]
    threshold: float = 0.01
    weight: float = WEIGHT_BELOW_CA
    max_velocity: float = 1.0

    def __post_init__(self):
        self.current_positions = []
        self.goal_positions = []
        self.velocity_limits = []
        self.connections = []
        if len(self.goal_state) == 0:
            raise GoalInitalizationException(f"Can't initialize {self} with no joints.")

        for connection, goal_position in self.goal_state.items():
            self.connections.append(connection)
            if not isinstance(connection, ActiveConnection1DOF):
                raise GoalInitalizationException(
                    f"Connection {connection.name} must be of type ActiveConnection1DOF"
                )

            ul_pos = connection.dof.upper_limits.position
            ll_pos = connection.dof.lower_limits.position
            if ll_pos is not None:
                goal_position = cas.limit(goal_position, ll_pos, ul_pos)

            ul_vel = connection.dof.upper_limits.velocity
            ll_vel = connection.dof.lower_limits.velocity
            velocity_limit = cas.limit(self.max_velocity, ll_vel, ul_vel)

            self.current_positions.append(connection.dof.symbols.position)
            self.goal_positions.append(goal_position)
            self.velocity_limits.append(velocity_limit)

        for connection, current, goal, velocity_limit in zip(
            self.connections,
            self.current_positions,
            self.goal_positions,
            self.velocity_limits,
        ):
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                error = cas.shortest_angular_distance(current, goal)
            else:
                error = goal - current

            self.add_equality_constraint(
                name=f"{self.name}/{connection.name}",
                reference_velocity=velocity_limit,
                equality_bound=error,
                weight=self.weight,
                task_expression=current,
            )
        joint_monitor = JointGoalReached(
            goal_state=self.goal_state,
            threshold=self.threshold,
            name=f"{self.name}_monitor",
        )
        self.observation_expression = joint_monitor.observation_expression


@validated_dataclass
class MirrorJointPosition(Task):
    mapping: Dict[Union[PrefixedName, str], str] = field(default_factory=lambda: dict)
    threshold: float = 0.01
    weight: Optional[float] = None
    max_velocity: Optional[float] = None

    def __post_init__(self):
        if self.weight is None:
            self.weight = WEIGHT_BELOW_CA
        if self.max_velocity is None:
            self.max_velocity = 1.0
        self.current_positions = []
        self.goal_positions = []
        self.velocity_limits = []
        self.connections = []
        goal_state = {}
        for joint_name, target_joint_name in self.mapping.items():
            connection = god_map.world.get_connection_by_name(joint_name)
            self.connections.append(connection)
            target_connection = god_map.world.get_connection_by_name(target_joint_name)

            ll_vel = connection.dof.lower_limits
            ul_vel = connection.dof.upper_limits
            velocity_limit = cas.limit(self.max_velocity, ll_vel, ul_vel)
            self.current_positions.append(connection.position)
            self.goal_positions.append(target_connection.position)
            self.velocity_limits.append(velocity_limit)
            goal_state[joint_name.name] = self.goal_positions[-1]

        for connection, current, goal, velocity_limit in zip(
            self.connections,
            self.current_positions,
            self.goal_positions,
            self.velocity_limits,
        ):
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                error = cas.shortest_angular_distance(current, goal)
            else:
                error = goal - current

            self.add_equality_constraint(
                name=f"{self.name}/{connection.name}",
                reference_velocity=velocity_limit,
                equality_bound=0,
                weight=self.weight,
                task_expression=error,
            )
        joint_monitor = JointGoalReached(
            goal_state=goal_state, threshold=self.threshold
        )
        self.observation_expression = joint_monitor.observation_expression


@validated_dataclass
class JointPositionLimitList(Task):
    lower_upper_limits: Dict[Union[PrefixedName, str], Tuple[float, float]]
    weight: float = WEIGHT_BELOW_CA
    max_velocity: float = 1

    def __post_init__(self):
        """
        Calls JointPosition for a list of joints.
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param max_velocity: will be applied to all joints, you should group joint types, e.g., prismatic joints
        :param hard: turns this into a hard constraint.
        """
        self.current_positions = []
        self.lower_limits = []
        self.upper_limits = []
        self.velocity_limits = []
        self.connections = []
        self.joint_names = list(sorted(self.lower_upper_limits.keys()))
        if len(self.lower_upper_limits) == 0:
            raise GoalInitalizationException(f"Can't initialize {self} with no joints.")

        for joint_name, (lower_limit, upper_limit) in self.lower_upper_limits.items():
            connection: ActiveConnection1DOF = god_map.world.get_connection_by_name(
                joint_name
            )
            self.connections.append(connection)

            ll_pos = connection.dof.lower_limits.position
            ul_pos = connection.dof.upper_limits.position

            if ll_pos is not None:
                lower_limit = min(ul_pos, max(ll_pos, lower_limit))
                upper_limit = min(ul_pos, max(ll_pos, upper_limit))

            ll_vel = connection.dof.lower_limits.velocity
            ul_vel = connection.dof.upper_limits.velocity

            velocity_limit = min(ul_vel, max(ll_vel, self.max_velocity))

            self.current_positions.append(connection.position)
            self.lower_limits.append(lower_limit)
            self.upper_limits.append(upper_limit)
            self.velocity_limits.append(velocity_limit)

        for connection, current, lower_limit, upper_limit, velocity_limit in zip(
            self.connections,
            self.current_positions,
            self.lower_limits,
            self.upper_limits,
            self.velocity_limits,
        ):
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                lower_error = cas.shortest_angular_distance(current, lower_limit)
                upper_error = cas.shortest_angular_distance(current, upper_limit)
            else:
                lower_error = lower_limit - current
                upper_error = upper_limit - current

            self.add_inequality_constraint(
                name=f"{self.name}/{connection.name}",
                reference_velocity=velocity_limit,
                lower_error=lower_error,
                upper_error=upper_error,
                weight=self.weight,
                task_expression=current,
            )


@validated_dataclass
class JustinTorsoLimit(Task):
    connection: ActiveConnection
    lower_limit: Optional[float] = None
    upper_limit: Optional[float] = None
    weight: float = WEIGHT_BELOW_CA
    max_velocity: float = 1

    def __post_init__(self):
        joint = self.connection

        current = joint.q3

        if isinstance(self.connection, RevoluteConnection) or isinstance(
            self.connection, PrismaticConnection
        ):
            lower_error = cas.shortest_angular_distance(current, self.lower_limit)
            upper_error = cas.shortest_angular_distance(current, self.upper_limit)
        else:
            lower_error = self.lower_limit - current
            upper_error = self.upper_limit - current

        god_map.debug_expression_manager.add_debug_expression("torso 4 joint", current)
        god_map.debug_expression_manager.add_debug_expression(
            "torso 2 joint", joint.q1.get_symbol(Derivatives.position)
        )
        god_map.debug_expression_manager.add_debug_expression(
            "torso 3 joint", joint.q2.get_symbol(Derivatives.position)
        )
        god_map.debug_expression_manager.add_debug_expression(
            "lower_limit", self.lower_limit
        )
        god_map.debug_expression_manager.add_debug_expression(
            "upper_limit", self.upper_limit
        )

        self.add_inequality_constraint(
            name=self.name,
            reference_velocity=1,
            lower_error=lower_error,
            upper_error=upper_error,
            weight=self.weight,
            task_expression=current,
        )


@validated_dataclass
class JointVelocityLimit(Task):
    joints: List[ActiveConnection1DOF]
    weight: float = WEIGHT_BELOW_CA
    max_velocity: float = 1
    hard: bool = False

    def __post_init__(self):
        """
        Limits the joint velocity of a revolute joint.
        :param joint_name:
        :param group_name: if joint_name is not unique, will search in this group for matches.
        :param weight:
        :param max_velocity: rad/s
        :param hard: turn this into a hard constraint.
        """
        for joint in self.joints:
            current_joint = joint.joint_position_expression
            try:
                limit_expr = joint.dof.upper_limits.velocity
                max_velocity = cas.min(self.max_velocity, limit_expr)
            except IndexError:
                max_velocity = self.max_velocity
            if self.hard:
                self.add_velocity_constraint(
                    lower_velocity_limit=-max_velocity,
                    upper_velocity_limit=max_velocity,
                    weight=self.weight,
                    task_expression=current_joint,
                    velocity_limit=max_velocity,
                    lower_slack_limit=0,
                    upper_slack_limit=0,
                )
            else:
                self.add_velocity_constraint(
                    lower_velocity_limit=-max_velocity,
                    upper_velocity_limit=max_velocity,
                    weight=self.weight,
                    task_expression=current_joint,
                    velocity_limit=max_velocity,
                    name=joint.name.name,
                )


@validated_dataclass
class JointVelocity(Task):
    connections: List[ActiveConnection1DOF]
    vel_goal: float
    weight: float = WEIGHT_BELOW_CA
    max_velocity: float = 1
    hard: bool = False

    def __post_init__(self):
        """
        Limits the joint velocity of a revolute joint.
        :param connection:
        :param group_name: if connection is not unique, will search in this group for matches.
        :param weight:
        :param max_velocity: rad/s
        :param hard: turn this into a hard constraint.
        """
        for connection in self.connections:
            current_joint = connection.dof.symbols.position
            try:
                limit_expr = connection.dof.upper_limits.velocity
                max_velocity = cas.min(self.max_velocity, limit_expr)
            except IndexError:
                max_velocity = self.max_velocity
            self.add_velocity_eq_constraint(
                velocity_goal=self.vel_goal,
                weight=self.weight,
                task_expression=current_joint,
                velocity_limit=max_velocity,
                name=str(connection.name),
            )


@validated_dataclass
class UnlimitedJointGoal(Task):
    connection: ActiveConnection1DOF
    goal_position: float

    def __post_init__(self):
        connection_symbol = self.connection.dof.symbols.position
        self.add_position_constraint(
            expr_current=connection_symbol,
            expr_goal=self.goal_position,
            reference_velocity=2,
            weight=WEIGHT_BELOW_CA,
        )


@validated_dataclass
class AvoidJointLimits(Task):
    """
    Calls AvoidSingleJointLimits for each joint in joint_list
    """

    percentage: float = 15

    connection_list: Optional[List[ActiveConnection1DOF]] = None
    """list of joints for which AvoidSingleJointLimits will be called"""

    weight: float = WEIGHT_BELOW_CA

    def __post_init__(self):
        if self.connection_list is None:
            self.connection_list = god_map.world.controlled_connections
        for connection in self.connection_list:
            if isinstance(connection, (RevoluteConnection, PrismaticConnection)):
                if not connection.dof.has_position_limits():
                    continue
                weight = self.weight
                connection_symbol = connection.dof.symbols.position
                percentage = self.percentage / 100.0
                lower_limit = connection.dof.lower_limits.position
                upper_limit = connection.dof.upper_limits.position
                max_velocity = cas.min(100, connection.dof.upper_limits.velocity)

                joint_range = upper_limit - lower_limit
                center = (upper_limit + lower_limit) / 2.0

                max_error = joint_range / 2.0 * percentage

                upper_goal = center + joint_range / 2.0 * (1 - percentage)
                lower_goal = center - joint_range / 2.0 * (1 - percentage)

                upper_err = upper_goal - connection_symbol
                lower_err = lower_goal - connection_symbol

                error = cas.max(
                    cas.abs(cas.min(upper_err, 0)), cas.abs(cas.max(lower_err, 0))
                )
                weight = weight * (error / max_error)

                self.add_inequality_constraint(
                    reference_velocity=max_velocity,
                    name=str(connection.name),
                    lower_error=lower_err,
                    upper_error=upper_err,
                    weight=weight,
                    task_expression=connection_symbol,
                )
