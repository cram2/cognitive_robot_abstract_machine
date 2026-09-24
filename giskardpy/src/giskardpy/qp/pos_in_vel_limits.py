"""
Velocity profiles that keep a degree of freedom within its position limits across the
prediction horizon.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.qp_data import QPDataExplicit
from giskardpy.qp.solvers.linear_program_solver_highs import LinearProgramSolverHighs
from giskardpy.utils.decorators import memoize
from krrood.symbolic_math.symbolic_math import (
    Scalar,
    Vector,
    substitution_cache,
)
from semantic_digital_twin.spatial_types.derivatives import Derivatives

# %% braking to rest


@dataclass
class BrakingProfile:
    """
    Velocity of a degree of freedom at each step of the prediction horizon while it
    brakes from a velocity to rest.
    """

    velocity: npt.NDArray
    """
    Velocity at each step of the prediction horizon.
    """

    time_step: float
    """
    Duration of a single step of the prediction horizon.
    """

    @classmethod
    def fastest(
        cls,
        initial_velocity: float,
        acceleration_limit: float,
        jerk_limit: float,
        time_step: float,
        prediction_horizon: int,
    ) -> BrakingProfile:
        """
        Creates the profile that brakes from ``initial_velocity`` to rest as fast as the
        limits allow, starting with zero acceleration.

        The profile solves a linear program to a vertex, so its velocity levels are
        exact rather than accurate only up to a solver tolerance.

        :param initial_velocity: Velocity the profile starts braking from.
        :param acceleration_limit: Acceleration limit applied at every horizon step.
        :param jerk_limit: Jerk limit applied at every horizon step.
        :param time_step: Duration of a single step of the prediction horizon.
        :param prediction_horizon: Number of steps in the prediction horizon.
        """
        return cls(
            velocity=cls._solve_fastest_braking(
                initial_velocity,
                acceleration_limit,
                jerk_limit,
                time_step,
                prediction_horizon,
            )[:prediction_horizon],
            time_step=time_step,
        )

    @staticmethod
    @memoize
    def _solve_fastest_braking(
        initial_velocity: float,
        acceleration_limit: float,
        jerk_limit: float,
        time_step: float,
        prediction_horizon: int,
    ) -> npt.NDArray:
        """
        Solves the linear program behind :meth:`fastest` and returns its solution: the
        velocity, acceleration, and jerk at each step, one derivative after the other.

        :param initial_velocity: Velocity the profile starts braking from.
        :param acceleration_limit: Acceleration limit applied at every horizon step.
        :param jerk_limit: Jerk limit applied at every horizon step.
        :param time_step: Duration of a single step of the prediction horizon.
        :param prediction_horizon: Number of steps in the prediction horizon.
        """
        limits = (initial_velocity, acceleration_limit, jerk_limit)
        number_of_derivatives = len(limits)
        upper_bounds = np.repeat(np.array(limits, dtype=float), prediction_horizon)
        lower_bounds = -upper_bounds
        for derivative in range(number_of_derivatives - 1):
            last_step = (derivative + 1) * prediction_horizon - 1
            lower_bounds[last_step] = upper_bounds[last_step] = 0
        link_model = BrakingProfile._derivative_link_model(
            time_step, prediction_horizon, number_of_derivatives
        )
        link_bounds = np.zeros(link_model.shape[0])
        link_bounds[0] = initial_velocity
        linear_weights = np.zeros(upper_bounds.shape[0])
        linear_weights[:prediction_horizon] = -1
        qp_data = QPDataExplicit(
            quadratic_weights=np.zeros(upper_bounds.shape[0]),
            linear_weights=linear_weights,
            box_lower_constraints=lower_bounds,
            box_upper_constraints=upper_bounds,
            equality_matrix=sp.csc_matrix(link_model),
            equality_bounds=link_bounds,
            inequality_matrix=sp.csc_matrix(np.zeros((0, link_model.shape[0]))),
            inequality_lower_bounds=np.array([]),
            inequality_upper_bounds=np.array([]),
            num_equality_slack_variables=0,
            num_inequality_slack_variables=0,
        )
        return LinearProgramSolverHighs().solver_call_explicit_interface(qp_data)

    @staticmethod
    def _derivative_link_model(
        time_step: float, prediction_horizon: int, number_of_derivatives: int
    ) -> npt.NDArray:
        """
        Returns the equality matrix that integrates each derivative into the one below
        it, ``x_k = x_(k-1) + time_step * xdot_k``, with ``x_(-1)`` the current state.

        :param time_step: Duration of a single step of the prediction horizon.
        :param prediction_horizon: Number of steps in the prediction horizon.
        :param number_of_derivatives: Number of derivatives, starting at velocity.
        """
        number_of_rows = prediction_horizon * (number_of_derivatives - 1)
        number_of_columns = prediction_horizon * number_of_derivatives
        link_model = np.zeros((number_of_rows, number_of_columns))
        link_model[:, :number_of_rows] += np.eye(number_of_rows)
        link_model[:, prediction_horizon:] += -np.eye(number_of_rows) * time_step
        previous_step_height = prediction_horizon - 1
        previous_step = -np.eye(previous_step_height)
        row_offset = 0
        column_offset = 0
        for _ in Derivatives.range(Derivatives.velocity, number_of_derivatives - 1):
            row_offset += 1
            link_model[
                row_offset : row_offset + previous_step_height,
                column_offset : column_offset + previous_step_height,
            ] += previous_step
            row_offset += previous_step_height
            column_offset += prediction_horizon
        return link_model

    def shifted_by(self, distance: Scalar) -> Vector:
        """
        Returns the part of this profile that remains when only ``distance`` is left to
        cover, padded with rest.

        Selects how far into the braking the motion already is by comparing the remaining
        ``distance`` against the distance covered by progressively truncated tails of the
        profile.

        :param distance: Remaining distance that determines how much of the profile is
            shifted out.
        """
        time_step = self.time_step
        velocity_profile = self.zero_negligible_velocities(self.velocity)
        velocity_if_cases = []
        for x in range(len(velocity_profile) - 1, -1, -1):
            condition = time_step * sum(velocity_profile[x:])
            velocity_result = np.concatenate(
                [velocity_profile[x + 1 :], np.zeros(x + 1)]
            )
            if condition > 0:
                velocity_if_cases.append((condition, sm.Vector(velocity_result)))
        # A distance shorter than the last braking step would leave only rest, so the degree
        # of freedom could never close it; the first step covers it instead.
        shortest_braking_distance, _ = velocity_if_cases[0]
        remaining_distance_profile = sm.Vector.zeros(velocity_profile.shape[0])
        remaining_distance_profile[0] = distance / time_step
        velocity_if_cases[0] = (shortest_braking_distance, remaining_distance_profile)
        velocity_if_cases.append(
            (
                2 * velocity_if_cases[-1][0] - velocity_if_cases[-2][0],
                sm.Vector(velocity_profile),
            )
        )
        default_velocity_profile = np.full(
            velocity_profile.shape[0], velocity_profile[0]
        )
        return sm.if_less_eq_cases(
            distance, velocity_if_cases, sm.Vector(default_velocity_profile)
        )

    @staticmethod
    def zero_negligible_velocities(
        velocity_profile: npt.NDArray, negligible_velocity=1e-4
    ) -> npt.NDArray:
        """
        Returns a copy of a braking profile in which every velocity below
        negligible_velocity is exactly zero.

        A profile that brakes to a standstill ends at rest, while one computed
        numerically ends at the solver's tolerance instead. Those leftovers become
        velocity bounds that are a hair apart rather than identical, which no interior
        point method can resolve.

        :param velocity_profile: Velocity values over the prediction horizon.
        :param negligible_velocity: Velocity below which a velocity is considered to be
            at rest. Sits above the absolute tolerance of every solver the controller
            can be configured with, and far below the smallest velocity a braking
            profile genuinely contains.
        """
        at_rest = copy(velocity_profile)
        at_rest[at_rest < negligible_velocity] = 0.0
        return at_rest


# %% slowing down as soon as possible


@dataclass
class SlowdownProfile:
    """
    Velocity and jerk of a degree of freedom at each step of the prediction horizon
    while it slows down towards a target velocity profile as soon as possible.
    """

    velocity: Vector
    """
    Velocity at each step of the prediction horizon.
    """

    jerk: Vector
    """
    Jerk at each step of the prediction horizon.
    """

    @classmethod
    def immediate(
        cls,
        current_velocity: Scalar,
        current_acceleration: Scalar,
        target_velocity_profile: Vector,
        jerk_limit: Scalar,
        time_step: Scalar,
        prediction_horizon: int,
        skip_first: Scalar,
    ) -> SlowdownProfile:
        """
        Creates the profile that slows down towards ``target_velocity_profile`` as soon
        as the jerk limit and the remaining horizon allow.

        :param current_velocity: Velocity at the start of the horizon.
        :param current_acceleration: Acceleration at the start of the horizon.
        :param target_velocity_profile: Per-step target velocities the motion is driven
            towards.
        :param jerk_limit: Maximum allowed change of acceleration per time step.
        :param time_step: Duration of a single time step.
        :param prediction_horizon: Number of time steps in the profile.
        :param skip_first: When truthy, the horizon cap is disabled for the first step.
        """
        velocity, jerk = cls._profiles(
            current_velocity,
            current_acceleration,
            target_velocity_profile,
            jerk_limit,
            time_step,
            prediction_horizon,
            skip_first,
        )
        return cls(velocity=velocity, jerk=jerk)

    @staticmethod
    @substitution_cache
    def _profiles(
        current_velocity: Scalar,
        current_acceleration: Scalar,
        target_velocity_profile: Vector,
        jerk_limit: Scalar,
        time_step: Scalar,
        prediction_horizon: int,
        skip_first: Scalar,
    ) -> tuple[Vector, Vector]:
        """
        Computes the velocity and jerk profiles of :meth:`immediate`.

        :param current_velocity: Velocity at the start of the horizon.
        :param current_acceleration: Acceleration at the start of the horizon.
        :param target_velocity_profile: Per-step target velocities the motion is driven
            towards.
        :param jerk_limit: Maximum allowed change of acceleration per time step.
        :param time_step: Duration of a single time step.
        :param prediction_horizon: Number of time steps in the profile.
        :param skip_first: When truthy, the horizon cap is disabled for the first step.
        """
        velocity_profile = []
        acceleration_profile = []
        next_velocity, next_acceleration = current_velocity, current_acceleration
        for i in range(prediction_horizon):
            next_velocity, next_acceleration = (
                SlowdownProfile._next_velocity_and_acceleration(
                    next_velocity,
                    next_acceleration,
                    target_velocity_profile[i],
                    jerk_limit,
                    time_step,
                    prediction_horizon - i - 1,
                    sm.logic_and(skip_first, sm.Scalar(i == 0)),
                )
            )
            velocity_profile.append(next_velocity)
            acceleration_profile.append(next_acceleration)
        acceleration_profile = copy(Vector(acceleration_profile))
        acceleration_profile2 = copy(Vector(acceleration_profile))
        acceleration_profile2[1:] = acceleration_profile[:-1]
        acceleration_profile2[0] = current_acceleration
        jerk_profile = (acceleration_profile - acceleration_profile2) / time_step

        return Vector(velocity_profile), jerk_profile

    @staticmethod
    @substitution_cache
    def _next_velocity_and_acceleration(
        current_velocity: Scalar,
        current_acceleration: Scalar,
        velocity_limit: Scalar,
        jerk_limit: Scalar,
        delta_time: Scalar,
        remaining_prediction_horizon: Scalar,
        no_cap: Scalar,
    ) -> tuple[Scalar, Scalar]:
        """
        Advance velocity and acceleration by one time step while respecting jerk and
        horizon limits.

        Picks the acceleration that drives the velocity towards ``velocity_limit`` as
        fast as allowed, bounded both by the jerk-reachable acceleration and by the
        acceleration still recoverable within the remaining horizon.

        :param current_velocity: Velocity at the current time step.
        :param current_acceleration: Acceleration at the current time step.
        :param velocity_limit: Target velocity the step moves towards.
        :param jerk_limit: Maximum allowed change of acceleration per time step.
        :param delta_time: Duration of a single time step.
        :param remaining_prediction_horizon: Number of time steps left in the horizon.
        :param no_cap: When truthy, skips the horizon-based acceleration capping.
        :return: The velocity and acceleration of the next time step.
        """
        acceleration_cap1 = SlowdownProfile._acceleration_cap(
            current_velocity, jerk_limit, delta_time
        )  # if we start at arbitrary horizon and jerk as strongly as possible, which acc do we have when we reach the vel limit
        acceleration_cap2 = (
            remaining_prediction_horizon * jerk_limit * delta_time
        )  # max acc reachable given horizon depending only on vel
        acceleration_prediction_horizon_max = sm.min(
            acceleration_cap1, acceleration_cap2
        )  # in reality we have a limited horizon, so we have to use the min of the two.
        acceleration_prediction_horizon_min = -acceleration_prediction_horizon_max

        next_acceleration_min = (
            current_acceleration - jerk_limit * delta_time
        )  # looking from the other side, these are the actual acc we can achieve with the jerk limits
        next_acceleration_max = current_acceleration + jerk_limit * delta_time

        acceleration_to_velocity = (
            velocity_limit - current_velocity
        ) / delta_time  # the total acc needed to reach vel target vel

        target_acceleratino = sm.max(next_acceleration_min, acceleration_to_velocity)
        target_acceleratino = sm.if_else(
            no_cap,
            target_acceleratino,
            sm.limit(
                target_acceleratino,
                acceleration_prediction_horizon_min,
                acceleration_prediction_horizon_max,
            ),
        )  # skip when vel_limit is negative
        next_acceleration = sm.limit(
            target_acceleratino, next_acceleration_min, next_acceleration_max
        )

        next_velocity = current_velocity + next_acceleration * delta_time
        return next_velocity, next_acceleration

    @staticmethod
    @substitution_cache
    def _acceleration_cap(
        current_velocity: Scalar, jerk_limit: Scalar, delta_time: Scalar
    ) -> Scalar:
        """
        Compute the largest acceleration that can be reached when braking to a stop
        under the jerk limit.

        Distributes the velocity that has to be removed across the jerk-limited
        acceleration steps and returns the peak acceleration of that braking ramp.

        :param current_velocity: Velocity that needs to be reduced to zero.
        :param jerk_limit: Maximum allowed change of acceleration per time step.
        :param delta_time: Duration of a single time step.
        :return: The peak acceleration of the jerk-limited braking ramp.
        """
        acceleration_integral = sm.abs(current_velocity) / delta_time
        jerk_step = jerk_limit * delta_time
        n = sm.floor(
            SlowdownProfile._reverse_gauss(sm.abs(acceleration_integral / jerk_step))
        )
        x = (-sm.gauss(n) * jerk_limit * delta_time + acceleration_integral) / (n + 1)
        return sm.abs(n * jerk_limit * delta_time + x)

    @staticmethod
    def _reverse_gauss(integral: Scalar) -> Scalar:
        """
        Invert the Gauss summation formula to recover the term count from a triangular
        sum.

        Solves ``n * (n + 1) / 2 == integral`` for ``n``, returning the continuous (non-
        floored) solution.

        :param integral: Value of the triangular sum.
        :return: The number of terms that produce the given sum.
        """
        return sm.sqrt(2 * integral + (1 / 4)) - 1 / 2
