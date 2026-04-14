from copy import copy

import numpy as np
from typing_extensions import Tuple, List

import giskardpy.utils.math as gm
import krrood.symbolic_math.symbolic_math as sm

from giskardpy.utils.decorators import memoize
from krrood.symbolic_math.symbolic_math import (
    FloatVariable,
    Scalar,
    Vector,
    substitution_cache,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap


def shifted_velocity_profile(
    vel_profile: Vector, acc_profile: Vector, distance: Scalar, dt: float
) -> Tuple[Vector, Vector]:
    vel_profile = copy(vel_profile)
    vel_profile[vel_profile < 0] = 0
    vel_if_cases = []
    acc_if_cases = []
    for x in range(len(vel_profile) - 1, -1, -1):
        condition = dt * sum(vel_profile[x:])
        vel_result = np.concatenate([vel_profile[x + 1 :], np.zeros(x + 1)])
        acc_result = np.concatenate([acc_profile[x + 1 :], np.zeros(x + 1)])
        if condition > 0:
            vel_if_cases.append((condition, sm.Vector(vel_result)))
            acc_if_cases.append((condition, sm.Vector(acc_result)))
    vel_if_cases.append(
        (2 * vel_if_cases[-1][0] - vel_if_cases[-2][0], sm.Vector(vel_profile))
    )
    default_vel_profile = np.full(vel_profile.shape[0], vel_profile[0])

    shifted_vel_profile = sm.if_less_eq_cases(
        distance, vel_if_cases, sm.Vector(default_vel_profile)
    )
    shifted_acc_profile = sm.if_less_eq_cases(
        distance, acc_if_cases, sm.Vector(acc_profile)
    )
    return shifted_vel_profile, shifted_acc_profile


def r_gauss(integral: Scalar) -> Scalar:
    return sm.sqrt(2 * integral + (1 / 4)) - 1 / 2


@substitution_cache
def acc_cap(current_vel: Scalar, jerk_limit: Scalar, dt: Scalar) -> Scalar:
    acc_integral = sm.abs(current_vel) / dt
    jerk_step = jerk_limit * dt
    n = sm.floor(r_gauss(sm.abs(acc_integral / jerk_step)))
    x = (-sm.gauss(n) * jerk_limit * dt + acc_integral) / (n + 1)
    return sm.abs(n * jerk_limit * dt + x)


@substitution_cache
def compute_next_vel_and_acc(
    current_vel: Scalar,
    current_acc: Scalar,
    vel_limit: Scalar,
    jerk_limit: Scalar,
    dt: Scalar,
    remaining_ph: Scalar,
    no_cap: Scalar,
) -> Tuple[Scalar, Scalar]:
    acc_cap1 = acc_cap(
        current_vel, jerk_limit, dt
    )  # if we start at arbitrary horizon and jerk as strongly as possible, which acc do we have when we reach the vel limit
    acc_cap2 = (
        remaining_ph * jerk_limit * dt
    )  # max acc reachable given horizon depending only on vel
    acc_ph_max = sm.min(
        acc_cap1, acc_cap2
    )  # in reality we have a limited horizon, so we have to use the min of the two.
    acc_ph_min = -acc_ph_max

    next_acc_min = (
        current_acc - jerk_limit * dt
    )  # looking from the other side, these are the actual acc we can achieve with the jerk limits
    next_acc_max = current_acc + jerk_limit * dt

    acc_to_vel = (
        vel_limit - current_vel
    ) / dt  # the total acc needed to reach vel target vel

    target_acc = sm.max(next_acc_min, acc_to_vel)
    target_acc = sm.if_else(
        no_cap, target_acc, sm.limit(target_acc, acc_ph_min, acc_ph_max)
    )  # skip when vel_limit is negative
    next_acc = sm.limit(target_acc, next_acc_min, next_acc_max)

    next_vel = current_vel + next_acc * dt
    return next_vel, next_acc


@substitution_cache
def compute_slowdown_asap_vel_profile(
    current_vel: Scalar,
    current_acc: Scalar,
    target_vel_profile: Vector,
    jerk_limit: Scalar,
    dt: Scalar,
    ph: int,
    skip_first: Scalar,
) -> Tuple[Vector, Vector, Vector]:
    """
    Compute the vel, acc and jerk profile for slowing down asap.
    """
    vel_profile = []
    acc_profile = []
    next_vel, next_acc = current_vel, current_acc
    for i in range(ph):
        next_vel, next_acc = compute_next_vel_and_acc(
            next_vel,
            next_acc,
            target_vel_profile[i],
            jerk_limit,
            dt,
            ph - i - 1,
            sm.logic_and(skip_first, sm.Scalar(i == 0)),
        )
        vel_profile.append(next_vel)
        acc_profile.append(next_acc)
    acc_profile = copy(Vector(acc_profile))
    acc_profile2 = copy(Vector(acc_profile))
    acc_profile2[1:] = acc_profile[:-1]
    acc_profile2[0] = current_acc
    jerk_profile = (acc_profile - acc_profile2) / dt

    return Vector(vel_profile), acc_profile, jerk_profile


def implicit_vel_profile(
    acc_limit: float, jerk_limit: float, dt: float, ph: int
) -> List[float]:
    vel_profile = [0, 0]  # because last two vel are always 0
    vel = 0
    acc = 0
    for i in range(ph - 2):
        acc += jerk_limit * dt
        acc = min(acc, acc_limit)
        vel += acc * dt
        vel_profile.append(vel)
    return list(reversed(vel_profile))
