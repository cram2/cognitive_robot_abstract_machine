import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.spatial_types import Point3, Vector3, RotationMatrix
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class WiggleInsert(Task):
    root_link: Body
    tip_link: Body
    hole_point: Point3
    noise_translation: float
    noise_angle: float
    down_velocity: float
    hole_normal: Optional[Vector3] = None
    threshold: float = 0.01
    random_walk: bool = True
    vector_momentum_factor: float = 0.9
    angular_momentum_factor: float = 0.9
    center_pull_strength_angle: float = 0.1
    center_pull_strength_vector: float = 0.25
    weight: float = DefaultWeights.WEIGHT_ABOVE_CA

    def __post_init__(self):
        """
        Press down while wiggling the end effector.
        :param root_link:
        :param tip_link:
        :param hole_point: Center point of the hole
        :param hole_normal: Vector perpendicular to the hole. default = z-axis of map
        :param threshold: threshold for distance to hole_point to end task
        :param noise_translation: describes how strong the translation wiggle is.
        :param noise_angle: describes how strong the angular wiggle is.
        :param random_walk: determines if random walk or random sample strategy is used
        :param vector_momentum_factor: (only when random_walk=True) Higher value increases influence of momentum,
                                                                    creating smoother but less random translation movement
        :param angular_momentum_factor: (only when random_walk=True) Higher value increases influence of momentum,
                                                                     creating smoother but less random angular movement
        :param center_pull_strength_angle: (only when random_walk=True) Forces angular movement faster back towards
                                                                        starting angle
        :param center_pull_strength_vector: (only when random_walk=True) Forces translation movement faster back towards
                                                                         hole_point
        """
        if self.hole_normal is None:
            self.hole_normal = Vector3(0, 0, 0, reference_frame=context.world.root)

        # Random-Sample works better with control_dt and Random-Walk with throttling using self.dt in my testing
        if self.random_walk:
            self.dt = context.qp_controller.control_dt
        else:
            self.dt = context.qp_controller.control_dt
        self.hz = 1 / self.dt

        if self.random_walk:
            vector_function = ".get_rand_walk_vector()"
            angle_function = ".get_rand_walk_angle()"
        else:
            vector_function = ".get_rand_vector()"
            angle_function = ".get_rand_angle()"

        # Init-Values for angular random walk
        self.current_angle = 0
        self.center_angle = 0
        self.angular_momentum = 0.0
        self.last_angular_change = 0

        # Init-Values for vector random walk
        self.current_vector = np.zeros(3)
        self.center_vector = np.zeros(3)
        self.vector_momentum = np.zeros(3)
        self.last_vector_change = 0

        v1, v2 = self.calculate_vectors(self.hole_normal.to_np()[:3])
        self.v1 = Vector3(*v1, reference_frame=self.hole_normal.reference_frame)
        self.v2 = Vector3(*v2, reference_frame=self.hole_normal.reference_frame)

        r_P_c = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        ).to_position()
        r_P_g = context.world.transform(
            target_frame=self.root_link, spatial_object=self.hole_point
        )

        rand_v = symbol_manager.get_expr(
            self.ref_str + vector_function,
            input_type_hint=np.ndarray,
            output_type_hint=Vector3,
        )

        r_P_g_rand = r_P_g + rand_v

        self.add_point_goal_constraints(
            frame_P_current=r_P_c,
            frame_P_goal=r_P_g_rand,
            reference_velocity=self.down_velocity,
            weight=self.weight,
            name=f"{self.name}_point_goal",
        )

        angle = symbol_manager.get_expr(
            self.ref_str + angle_function,
            input_type_hint=float,
            output_type_hint=sm.FloatVariable,
        )

        tip_V_hole_normal = context.world.transform(
            target_frame=self.tip_link, spatial_object=self.hole_normal
        )
        tip_R_hole_normal = RotationMatrix.from_axis_angle(
            angle=angle, axis=tip_V_hole_normal
        )
        root_R_hole_normal = context.world.compute_fk(
            self.root_link, self.tip_link
        ).dot(tip_R_hole_normal)

        r_T_c = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        )
        r_R_c = r_T_c.to_rotation_matrix()

        self.add_rotation_goal_constraints(
            frame_R_current=r_R_c,
            frame_R_goal=root_R_hole_normal,
            reference_velocity=self.down_velocity,
            weight=self.weight + 1,
        )

        god_map.context.add_debug_expression(name="r_P_g", expression=r_P_g)
        god_map.context.add_debug_expression(name="r_P_g_rand", expression=r_P_g_rand)
        god_map.context.add_debug_expression(
            name="root_R_hole_normal", expression=root_R_hole_normal
        )
        god_map.context.add_debug_expression(
            name="tip_R_hole_normal", expression=tip_R_hole_normal
        )

        dist = r_P_c.euclidean_distance(r_P_g)
        end = dist <= self.threshold

        self.observation_expression = end

    def get_rand_angle(self) -> float:
        now = context.time
        if now - self.last_angular_change >= self.dt:
            self.last_angular_change = now

            self.current_angle = ((random.random() - 0.5) * self.noise_angle) / self.hz
        return self.current_angle

    def get_rand_walk_angle(self):
        now = context.time
        if now - self.last_angular_change >= self.dt:
            self.last_angular_change = now
            random_angular_change = (
                (random.random() - 0.5) * self.noise_angle
            ) / self.hz

            # Update angular momentum (weighted average of previous momentum and new random change)
            self.angular_momentum = (
                self.angular_momentum_factor * self.angular_momentum
                + (1 - self.angular_momentum_factor) * random_angular_change
            )

            self.current_angle += self.angular_momentum
            angle_diff = self.center_angle - self.current_angle
            angle_diff = (
                (angle_diff + np.pi) % (2 * np.pi)
            ) - np.pi  # Normalize angle difference to [-pi, pi] range
            self.current_angle += angle_diff * self.center_pull_strength_angle
            self.current_angle = self.current_angle % (
                2 * np.pi
            )  # Normalize current angle to [0, 2pi] range

        return self.current_angle

    def get_rand_vector(self) -> Vector3:
        now = context.time
        if now - self.last_vector_change >= self.dt:
            self.last_vector_change = now

            self.current_vector = (
                (random.random() - 0.5) * self.noise_translation * self.v1.to_np()
                + (random.random() - 0.5) * self.noise_translation * self.v2.to_np()
            ) / self.hz
        return self.current_vector

    def get_rand_walk_vector(self) -> Vector3:
        now = context.time
        if now - self.last_vector_change >= self.dt:
            self.last_vector_change = now

            random_vector_change = (
                (random.random() - 0.5) * self.noise_translation * self.v1.to_np()[:3]
                + (random.random() - 0.5) * self.noise_translation * self.v2.to_np()[:3]
            ) / self.hz
            self.vector_momentum = (
                self.vector_momentum_factor * self.vector_momentum
                + (1 - self.vector_momentum_factor) * random_vector_change
            )
            self.current_vector += self.vector_momentum
            vector_diff = self.center_vector - self.current_vector
            self.current_vector += vector_diff * self.center_pull_strength_vector

        return self.current_vector

    def calculate_vectors(self, n: np.ndarray):
        if abs(n[0]) >= abs(n[1]):
            u = np.array([-n[2], 0, n[0]])
        else:
            u = np.array([0, -n[2], n[1]])

        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)

        return u, v
