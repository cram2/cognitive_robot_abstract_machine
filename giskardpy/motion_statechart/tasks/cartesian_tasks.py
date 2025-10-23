from dataclasses import field
from typing import Optional

import numpy as np

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.tasks.task import Task, WEIGHT_ABOVE_CA
from giskardpy.utils.decorators import validated_dataclass
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body


@validated_dataclass
class CartesianPosition(Task):
    default_reference_velocity = 0.2
    root_link: Body
    tip_link: Body
    goal_point: cas.Point3
    threshold: float = 0.01
    reference_velocity: Optional[float] = None
    weight: float = WEIGHT_ABOVE_CA
    absolute: bool = False

    def __post_init__(self):
        """
        See CartesianPose.
        """
        if self.reference_velocity is None:
            self.reference_velocity = self.default_reference_velocity
        if self.absolute:
            root_P_goal = god_map.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_point
            )
        else:
            root_T_x = god_map.world.compose_forward_kinematics_expression(
                self.root_link, self.goal_point.reference_frame
            )
            root_P_goal = root_T_x.dot(self.goal_point)
            root_P_goal = self.update_expression_on_starting(root_P_goal)

        r_P_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        self.add_point_goal_constraints(
            frame_P_goal=root_P_goal,
            frame_P_current=r_P_c,
            reference_velocity=self.reference_velocity,
            weight=self.weight,
        )
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.name}/target",
            root_P_goal.y,
            color=Color(0.0, 0.0, 1.0, 1.0),
            derivative=Derivatives.position,
            derivatives_to_plot=[Derivatives.position],
        )

        cap = (
            self.reference_velocity
            * god_map.qp_controller.config.mpc_dt
            * (god_map.qp_controller.config.prediction_horizon - 2)
        )
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.name}/upper_cap",
            root_P_goal.y + cap,
            derivatives_to_plot=[
                Derivatives.position,
            ],
        )
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.name}/lower_cap",
            root_P_goal.y - cap,
            derivatives_to_plot=[
                Derivatives.position,
            ],
        )
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.name}/current",
            r_P_c.y,
            color=Color(1.0, 0.0, 0.0, 1.0),
            derivative=Derivatives.position,
            derivatives_to_plot=Derivatives.range(
                Derivatives.position, Derivatives.jerk
            ),
        )

        distance_to_goal = root_P_goal.euclidean_distance(r_P_c)
        self.observation_expression = distance_to_goal < self.threshold


@validated_dataclass
class CartesianPositionStraight(Task):
    root_link: Body
    tip_link: Body
    goal_point: cas.Point3
    threshold: float = 0.01
    reference_velocity: Optional[float] = CartesianPosition.default_reference_velocity
    absolute: bool = False
    weight: float = WEIGHT_ABOVE_CA

    def __post_init__(self):
        """
        Same as CartesianPosition, but tries to move the tip_link in a straight line to the goal_point.
        """
        if self.absolute:
            root_P_goal = god_map.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_point
            )
        else:
            root_T_x = god_map.world.compose_forward_kinematics_expression(
                self.root_link, self.goal_point.reference_frame
            )
            root_P_goal = root_T_x.dot(self.goal_point)
            root_P_goal = self.update_expression_on_starting(root_P_goal)

        root_P_tip = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        t_T_r = god_map.world.compose_forward_kinematics_expression(
            self.tip_link, self.root_link
        )
        tip_P_goal = t_T_r.dot(root_P_goal)

        # Create rotation matrix, which rotates the tip link frame
        # such that its x-axis shows towards the goal position.
        # The goal frame is called 'a'.
        # Thus, the rotation matrix is called t_R_a.
        tip_V_error = cas.Vector3.from_iterable(tip_P_goal)
        trans_error = tip_V_error.norm()
        # x-axis
        tip_V_intermediate_error = tip_V_error.safe_division(trans_error)
        # y- and z-axis
        tip_V_intermediate_y = cas.Vector3.from_iterable(np.random.random((3,)))
        tip_V_intermediate_y.scale(1)
        y = tip_V_intermediate_error.cross(tip_V_intermediate_y)
        z = tip_V_intermediate_error.cross(y)
        t_R_a = cas.RotationMatrix.from_vectors(x=tip_V_intermediate_error, y=-z, z=y)

        # Apply rotation matrix on the fk of the tip link
        tip_T_root = god_map.world.compute_forward_kinematics(
            self.tip_link, self.root_link
        )
        root_T_tip = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        a_T_t = t_R_a.inverse() @ tip_T_root @ root_T_tip

        expr_p = a_T_t.to_position()
        dist = (root_P_goal - root_P_tip).norm()

        self.add_equality_constraint_vector(
            reference_velocities=[self.reference_velocity] * 3,
            equality_bounds=[dist, 0, 0],
            weights=[WEIGHT_ABOVE_CA, WEIGHT_ABOVE_CA * 2, WEIGHT_ABOVE_CA * 2],
            task_expression=expr_p[:3],
            names=["line/x", "line/y", "line/z"],
        )
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.name}/current_point", root_P_tip, color=Color(1, 0, 0, 1)
        )
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.name}/goal_point", root_P_goal, color=Color(0, 0, 1, 1)
        )
        self.observation_expression = dist < self.threshold


@validated_dataclass
class CartesianOrientation(Task):
    default_reference_velocity = 0.2
    root_link: Body
    tip_link: Body
    goal_orientation: cas.RotationMatrix
    threshold: float = 0.01
    reference_velocity: Optional[float] = None
    weight: float = WEIGHT_ABOVE_CA
    absolute: bool = False
    point_of_debug_matrix: Optional[cas.Point3] = None

    def __post_init__(self):
        """
        See CartesianPose.
        """
        if self.reference_velocity is None:
            self.reference_velocity = self.default_reference_velocity

        if self.absolute:
            root_R_goal = god_map.world.transform(
                target_frame=self.root_link, spatial_object=self.goal_orientation
            )
        else:
            root_T_x = god_map.world.compose_forward_kinematics_expression(
                self.root_link, self.goal_orientation.reference_frame
            )
            root_R_goal = root_T_x.dot(self.goal_orientation)
            root_R_goal = self.update_expression_on_starting(root_R_goal)

        r_T_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        r_R_c = r_T_c.to_rotation_matrix()

        self.add_rotation_goal_constraints(
            frame_R_current=r_R_c,
            frame_R_goal=root_R_goal,
            reference_velocity=self.reference_velocity,
            weight=self.weight,
        )
        if self.point_of_debug_matrix is None:
            point = r_T_c.to_position()
        else:
            if self.absolute:
                point = self.point_of_debug_matrix
            else:
                root_T_x = god_map.world.compose_forward_kinematics_expression(
                    self.root_link, self.point_of_debug_matrix.reference_frame
                )
                point = root_T_x.dot(self.point_of_debug_matrix)
                point = self.update_expression_on_starting(point)
        debug_trans_matrix = cas.TransformationMatrix.from_point_rotation_matrix(
            point=point, rotation_matrix=root_R_goal
        )
        debug_current_trans_matrix = (
            cas.TransformationMatrix.from_point_rotation_matrix(
                point=r_T_c.to_position(), rotation_matrix=r_R_c
            )
        )
        # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/goal_orientation', debug_trans_matrix)
        # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/current_orientation',
        #                                                       debug_current_trans_matrix)

        rotation_error = r_R_c.rotational_error(root_R_goal)
        self.observation_expression = cas.abs(rotation_error) < self.threshold


@validated_dataclass
class CartesianPose(Task):
    root_link: Body
    tip_link: Body
    goal_pose: cas.TransformationMatrix
    reference_linear_velocity: float = field(
        default=CartesianPosition.default_reference_velocity
    )
    reference_angular_velocity: float = field(
        default=CartesianOrientation.default_reference_velocity
    )
    threshold: float = field(default=0.01)
    absolute: bool = False
    weight: float = field(default=WEIGHT_ABOVE_CA)

    def __post_init__(self):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param absolute: if False, the goal is updated when start_condition turns True.
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        self.goal_ref = self.goal_pose.reference_frame
        goal_orientation = self.goal_pose.to_rotation_matrix()
        goal_point = self.goal_pose.to_position()

        if self.absolute:
            root_T_goal_ref_np = god_map.world.compute_forward_kinematics_np(
                self.root_link, self.goal_ref
            )
            root_T_goal_ref = cas.TransformationMatrix(root_T_goal_ref_np)
            root_P_goal = root_T_goal_ref @ goal_point
            root_R_goal = root_T_goal_ref @ goal_orientation
        else:
            root_T_x = god_map.world.compose_forward_kinematics_expression(
                self.root_link, self.goal_ref
            )
            root_P_goal = root_T_x @ goal_point
            root_P_goal = self.update_expression_on_starting(root_P_goal)
            root_R_goal = root_T_x @ goal_orientation
            root_R_goal = self.update_expression_on_starting(root_R_goal)

        r_P_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        self.add_point_goal_constraints(
            frame_P_goal=root_P_goal,
            frame_P_current=r_P_c,
            reference_velocity=self.reference_linear_velocity,
            weight=self.weight,
        )

        distance_to_goal = root_P_goal.euclidean_distance(r_P_c)

        r_T_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        r_R_c = r_T_c.to_rotation_matrix()

        self.add_rotation_goal_constraints(
            frame_R_current=r_R_c,
            frame_R_goal=root_R_goal,
            reference_velocity=self.reference_angular_velocity,
            weight=self.weight,
        )
        # debug_trans_matrix = cas.TransformationMatrix.from_point_rotation_matrix(point=goal_point,
        #                                                                 rotation_matrix=root_R_goal)
        # debug_current_trans_matrix = cas.TransformationMatrix.from_point_rotation_matrix(point=r_T_c.to_position(),
        #                                                                         rotation_matrix=r_R_c)
        # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/goal_orientation', debug_trans_matrix)
        # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/current_orientation',
        #                                                       debug_current_trans_matrix)

        rotation_error = r_R_c.rotational_error(root_R_goal)
        self.observation_expression = cas.logic_and(
            cas.abs(rotation_error) < self.threshold,
            distance_to_goal < self.threshold,
        )


@validated_dataclass
class CartesianPositionVelocityLimit(Task):
    root_link: Body
    tip_link: Body
    max_linear_velocity: float = 0.2
    weight: float = WEIGHT_ABOVE_CA

    def __post_init__(self):
        """
        This goal will use put a strict limit on the Cartesian velocity. This will require a lot of constraints, thus
        slowing down the system noticeably.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        :param hard: Turn this into a hard constraint. This make create unsolvable optimization problems
        """
        r_P_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        self.add_translational_velocity_limit(
            frame_P_current=r_P_c,
            max_velocity=self.max_linear_velocity,
            weight=self.weight,
        )


@validated_dataclass
class CartesianRotationVelocityLimit(Task):
    root_link: Body
    tip_link: Body
    weight: float = WEIGHT_ABOVE_CA
    max_velocity: Optional[float] = None

    def __post_init__(self):
        """
        See CartesianVelocityLimit
        """
        r_R_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_rotation()

        r_R_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_rotation_matrix()

        self.add_rotational_velocity_limit(
            frame_R_current=r_R_c, max_velocity=self.max_velocity, weight=self.weight
        )


@validated_dataclass
class CartesianVelocityLimit(Task):
    root_link: Body
    tip_link: Body
    max_linear_velocity: float = 0.1
    max_angular_velocity: float = 0.5
    weight: float = WEIGHT_ABOVE_CA

    def __post_init__(self):
        """
        This goal will use put a strict limit on the Cartesian velocity. This will require a lot of constraints, thus
        slowing down the system noticeably.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        r_T_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        r_P_c = r_T_c.to_position()
        r_R_c = r_T_c.to_rotation_matrix()
        self.add_translational_velocity_limit(
            frame_P_current=r_P_c,
            max_velocity=self.max_linear_velocity,
            weight=self.weight,
        )
        self.add_rotational_velocity_limit(
            frame_R_current=r_R_c,
            max_velocity=self.max_angular_velocity,
            weight=self.weight,
        )


@validated_dataclass
class CartesianPositionVelocityTarget(Task):
    root_link: Body
    tip_link: Body
    x_vel: float
    y_vel: float
    z_vel: float
    weight: float = WEIGHT_ABOVE_CA

    def __post_init__(self):
        """
        This goal will use put a strict limit on the Cartesian velocity. This will require a lot of constraints, thus
        slowing down the system noticeably.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param root_group: if the root_link is not unique, use this to say to which group the link belongs
        :param tip_group: if the tip_link is not unique, use this to say to which group the link belongs
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        :param hard: Turn this into a hard constraint. This make create unsolvable optimization problems
        """
        r_P_c = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.name}/target",
            cas.Expression(self.y_vel),
            derivative=Derivatives.velocity,
            derivatives_to_plot=[
                # Derivatives.position,
                Derivatives.velocity
            ],
        )
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.name}/current",
            r_P_c.y,
            derivative=Derivatives.position,
            derivatives_to_plot=Derivatives.range(
                Derivatives.position, Derivatives.jerk
            ),
        )
        self.add_velocity_eq_constraint_vector(
            velocity_goals=cas.Expression([self.x_vel, self.y_vel, self.z_vel]),
            task_expression=r_P_c,
            reference_velocities=[
                CartesianPosition.default_reference_velocity,
                CartesianPosition.default_reference_velocity,
                CartesianPosition.default_reference_velocity,
            ],
            names=[
                f"{self.name}/x",
                f"{self.name}/y",
                f"{self.name}/z",
            ],
            weights=[self.weight] * 3,
        )


@validated_dataclass
class JustinTorsoLimitCart(Task):
    root_link: Body
    tip_link: Body
    forward_distance: float
    backward_distance: float
    weight: float = WEIGHT_ABOVE_CA

    def __post_init__(self):
        torso_root_T_torso_tip = god_map.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        torso_root_V_up = cas.Vector3(0, 0, 1)
        torso_root_V_up.reference_frame = self.root_link
        torso_root_V_up.vis_frame = self.root_link

        torso_root_V_left = cas.Vector3(0, 1, 0)
        torso_root_V_left.reference_frame = self.root_link
        torso_root_V_left.vis_frame = self.root_link

        torso_root_P_torso_tip = torso_root_T_torso_tip.to_position()

        nearest, distance = torso_root_P_torso_tip.project_to_plane(
            frame_V_plane_vector1=torso_root_V_left,
            frame_V_plane_vector2=torso_root_V_up,
        )
        # distance = cas.distance_point_to_line(torso_root_P_torso_tip, cas.Point3((0, 0, 0)), torso_root_V_up)

        # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/torso_root_V_up',
        #                                                       expression=torso_root_V_up)
        # god_map.debug_expression_manager.add_debug_expression(f'{self.name}/torso_root_P_torso_tip',
        #                                                       expression=torso_root_P_torso_tip)
        god_map.debug_expression_manager.add_debug_expression(
            f"{self.name}/distance", expression=distance
        )

        self.add_inequality_constraint(
            reference_velocity=CartesianPosition.default_reference_velocity,
            lower_error=-self.backward_distance - distance,
            upper_error=self.forward_distance - distance,
            weight=self.weight,
            task_expression=distance,
            name=f"{self.name}/distance",
        )
        self.observation_expression = distance <= self.forward_distance
