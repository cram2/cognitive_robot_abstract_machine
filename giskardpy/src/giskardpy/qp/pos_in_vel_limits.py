from copy import copy

import numpy as np
from typing_extensions import Tuple, List

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import (
    Scalar,
    Vector,
    substitution_cache,
)


def shifted_velocity_profile(
    velocity_profile: Vector,
    acceleration_profile: Vector,
    distance: Scalar,
    delta_time: float,
) -> Tuple[Vector, Vector]:
    """
    Shift a velocity and acceleration profile forward in time based on a remaining
    distance.

    Selects how far into the braking profile the motion already is by comparing the remaining
    ``distance`` against the distance covered by progressively truncated tails of the profile.

    :param velocity_profile: Velocity values over the prediction horizon; negative values are clamped to zero.
    :param acceleration_profile: Acceleration values matching ``velocity_profile``.
    :param distance: Remaining distance that determines how much of the profile is shifted out.
    :param delta_time: Duration of a single time step.
    :return: The shifted velocity profile and the shifted acceleration profile.
    """
    velocity_profile = copy(velocity_profile)
    velocity_profile[velocity_profile < 0] = 0
    velocity_if_cases = []
    acceleration_if_cases = []
    for x in range(len(velocity_profile) - 1, -1, -1):
        condition = delta_time * sum(velocity_profile[x:])
        velocity_result = np.concatenate([velocity_profile[x + 1 :], np.zeros(x + 1)])
        acceleration_result = np.concatenate(
            [acceleration_profile[x + 1 :], np.zeros(x + 1)]
        )
        if condition > 0:
            velocity_if_cases.append((condition, sm.Vector(velocity_result)))
            acceleration_if_cases.append((condition, sm.Vector(acceleration_result)))
    velocity_if_cases.append(
        (
            2 * velocity_if_cases[-1][0] - velocity_if_cases[-2][0],
            sm.Vector(velocity_profile),
        )
    )
    default_velocity_profile = np.full(velocity_profile.shape[0], velocity_profile[0])

    shifted_velocity_profile = sm.if_less_eq_cases(
        distance, velocity_if_cases, sm.Vector(default_velocity_profile)
    )
    shifted_acceleration_profile = sm.if_less_eq_cases(
        distance, acceleration_if_cases, sm.Vector(acceleration_profile)
    )
    return shifted_velocity_profile, shifted_acceleration_profile


def reverse_gauss(integral: Scalar) -> Scalar:
    """
    Invert the Gauss summation formula to recover the term count from a triangular sum.

    Solves ``n * (n + 1) / 2 == integral`` for ``n``, returning the continuous (non-
    floored) solution.

    :param integral: Value of the triangular sum.
    :return: The number of terms that produce the given sum.
    """
    return sm.sqrt(2 * integral + (1 / 4)) - 1 / 2


@substitution_cache
def acceleration_cap(
    current_velocity: Scalar, jerk_limit: Scalar, delta_time: Scalar
) -> Scalar:
    """
    Compute the largest acceleration that can be reached when braking to a stop under
    the jerk limit.

    Distributes the velocity that has to be removed across the jerk-limited acceleration
    steps and returns the peak acceleration of that braking ramp.

    :param current_velocity: Velocity that needs to be reduced to zero.
    :param jerk_limit: Maximum allowed change of acceleration per time step.
    :param delta_time: Duration of a single time step.
    :return: The peak acceleration of the jerk-limited braking ramp.
    """
    acceleration_integral = sm.abs(current_velocity) / delta_time
    jerk_step = jerk_limit * delta_time
    n = sm.floor(reverse_gauss(sm.abs(acceleration_integral / jerk_step)))
    x = (-sm.gauss(n) * jerk_limit * delta_time + acceleration_integral) / (n + 1)
    return sm.abs(n * jerk_limit * delta_time + x)


@substitution_cache
def compute_next_velocity_and_acceleration(
    current_velocity: Scalar,
    current_acceleration: Scalar,
    velocity_limit: Scalar,
    jerk_limit: Scalar,
    delta_time: Scalar,
    remaining_prediction_horizon: Scalar,
    no_cap: Scalar,
) -> Tuple[Scalar, Scalar]:
    """
    Advance velocity and acceleration by one time step while respecting jerk and horizon
    limits.

    Picks the acceleration that drives the velocity towards ``velocity_limit`` as fast
    as allowed, bounded both by the jerk-reachable acceleration and by the acceleration
    still recoverable within the remaining horizon.

    :param current_velocity: Velocity at the current time step.
    :param current_acceleration: Acceleration at the current time step.
    :param velocity_limit: Target velocity the step moves towards.
    :param jerk_limit: Maximum allowed change of acceleration per time step.
    :param delta_time: Duration of a single time step.
    :param remaining_prediction_horizon: Number of time steps left in the horizon.
    :param no_cap: When truthy, skips the horizon-based acceleration capping.
    :return: The velocity and acceleration of the next time step.
    """
    acceleration_cap1 = acceleration_cap(
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


@substitution_cache
def compute_immediate_slowdown_profile(
    current_velocity: Scalar,
    current_acceleration: Scalar,
    target_velocity_profile: Vector,
    jerk_limit: Scalar,
    delta_time: Scalar,
    prediction_horizon: int,
    skip_first: Scalar,
) -> Tuple[Vector, Vector, Vector]:
    """
    Compute the velocity, acceleration and jerk profile for slowing down as soon as
    possible.

    Iterates :func:`compute_next_velocity_and_acceleration` over the whole prediction
    horizon and derives the jerk profile from the resulting accelerations.

    :param current_velocity: Velocity at the start of the horizon.
    :param current_acceleration: Acceleration at the start of the horizon.
    :param target_velocity_profile: Per-step target velocities the motion is driven
        towards.
    :param jerk_limit: Maximum allowed change of acceleration per time step.
    :param delta_time: Duration of a single time step.
    :param prediction_horizon: Number of time steps in the profile.
    :param skip_first: When truthy, the horizon cap is disabled for the first step.
    :return: The velocity profile, acceleration profile and jerk profile over the
        horizon.
    """
    velocity_profile = []
    acceleration_profile = []
    next_velocity, next_acceleration = current_velocity, current_acceleration
    for i in range(prediction_horizon):
        next_velocity, next_acceleration = compute_next_velocity_and_acceleration(
            next_velocity,
            next_acceleration,
            target_velocity_profile[i],
            jerk_limit,
            delta_time,
            prediction_horizon - i - 1,
            sm.logic_and(skip_first, sm.Scalar(i == 0)),
        )
        velocity_profile.append(next_velocity)
        acceleration_profile.append(next_acceleration)
    acceleration_profile = copy(Vector(acceleration_profile))
    acceleration_profile2 = copy(Vector(acceleration_profile))
    acceleration_profile2[1:] = acceleration_profile[:-1]
    acceleration_profile2[0] = current_acceleration
    jerk_profile = (acceleration_profile - acceleration_profile2) / delta_time

    return Vector(velocity_profile), acceleration_profile, jerk_profile


def implicit_velocity_profile(
    acceleration_limit: float,
    jerk_limit: float,
    delta_time: float,
    prediction_horizon: int,
) -> List[float]:
    """
    Build the velocity profile implied by ramping up acceleration under jerk and
    acceleration limits.

    Integrates jerk into acceleration and acceleration into velocity over the horizon
    and returns the profile reversed, so it represents the velocities to brake from
    while ending at rest.

    :param acceleration_limit: Maximum allowed acceleration.
    :param jerk_limit: Maximum allowed change of acceleration per time step.
    :param delta_time: Duration of a single time step.
    :param prediction_horizon: Number of time steps in the profile.
    :return: The implied velocity profile, ordered from the highest velocity down to
        rest.
    """
    velocity_profile = [0, 0]  # because last two vel are always 0
    velocity = 0
    acceleration = 0
    for i in range(prediction_horizon - 2):
        acceleration += jerk_limit * delta_time
        acceleration = min(acceleration, acceleration_limit)
        velocity += acceleration * delta_time
        velocity_profile.append(velocity)
    return list(reversed(velocity_profile))
