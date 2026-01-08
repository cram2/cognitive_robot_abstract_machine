from copy import copy
from typing import List

import numpy as np
from line_profiler import profile
from typing_extensions import Tuple

import giskardpy.utils.math as gm
import krrood.symbolic_math.symbolic_math as sm
from giskardpy.utils.decorators import memoize
from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar, Vector
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


def acc_cap(current_vel: Scalar, jerk_limit: Scalar, dt: Scalar) -> Scalar:
    acc_integral = sm.abs(current_vel) / dt
    jerk_step = jerk_limit * dt
    n = sm.floor(r_gauss(sm.abs(acc_integral / jerk_step)))
    x = (-sm.gauss(n) * jerk_limit * dt + acc_integral) / (n + 1)
    return sm.abs(n * jerk_limit * dt + x)


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
            sm.logic_and(skip_first, i == 0),
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


@memoize
@profile
def b_profile(
    dof_symbols: DerivativeMap[FloatVariable],
    lower_limits: DerivativeMap[float],
    upper_limits: DerivativeMap[float],
    solver_class,
    dt: float,
    ph: int,
    eps: float = 0.00001,
):
    vel_limit = upper_limits.velocity
    acc_limit = upper_limits.acceleration
    jerk_limit = upper_limits.jerk
    if lower_limits.position is not None:
        pos_range = upper_limits.position - lower_limits.position
        # reduce vel limit, if it can surpass position limits in one dt
        vel_limit = min(vel_limit * dt, pos_range / 2) / dt
        # %% compute max possible profile
        profile = gm.simple_mpc(
            vel_limit=vel_limit,
            acc_limit=acc_limit,
            jerk_limit=jerk_limit,
            current_vel=vel_limit,
            current_acc=0,
            dt=dt,
            ph=ph,
            q_weight=(0, 0, 0),
            lin_weight=(-1, 0, 0),
            solver_class=solver_class,
        )
        vel_profile_mpc = profile[:ph]
        acc_profile_mpc = profile[ph : ph * 2]
        pos_error_lb = lower_limits.position - dof_symbols.position
        pos_error_ub = upper_limits.position - dof_symbols.position
        # %% limits to profile, if vel integral bigger than remaining distance to pos limits
        pos_vel_profile_lb, _ = shifted_velocity_profile(
            vel_profile=vel_profile_mpc,
            acc_profile=acc_profile_mpc,
            distance=-pos_error_lb,
            dt=dt,
        )
        pos_vel_profile_lb *= -1
        pos_vel_profile_ub, _ = shifted_velocity_profile(
            vel_profile=vel_profile_mpc,
            acc_profile=acc_profile_mpc,
            distance=pos_error_ub,
            dt=dt,
        )
        # %% when limits are violated, compute the max velocity that can be reached in one step from zero and put it as
        # negative limits
        one_step_change_ = jerk_limit * dt**2
        one_step_change_lb = sm.min(
            sm.max(Scalar(0), pos_error_lb / dt), Scalar(one_step_change_)
        )
        one_step_change_lb = sm.limit(one_step_change_lb, -vel_limit, vel_limit)
        one_step_change_ub = sm.max(
            sm.min(Scalar(0), pos_error_ub / dt), -Scalar(one_step_change_)
        )
        one_step_change_ub = sm.limit(one_step_change_ub, -vel_limit, vel_limit)
        pos_vel_profile_lb[0] = sm.if_greater(
            pos_error_lb, 0, one_step_change_lb, copy(pos_vel_profile_lb[0])
        )
        pos_vel_profile_ub[0] = sm.if_less(
            pos_error_ub, 0, one_step_change_ub, copy(pos_vel_profile_ub[0])
        )

        # all 0, unless lower or upper position limits are violated
        goal_profile = sm.max(pos_vel_profile_lb, 0) + sm.min(pos_vel_profile_ub, 0)
        # skip first when lower or upper position limit are violated
        skip_first = sm.logic_or(pos_vel_profile_lb[0] >= 0, pos_vel_profile_ub[0] <= 0)
    else:
        goal_profile = sm.Vector.zeros(ph)
        pos_vel_profile_ub = sm.Vector.ones(ph) * vel_limit
        pos_vel_profile_lb = -pos_vel_profile_ub
        skip_first = 0

    acc_profile = sm.Vector.ones(pos_vel_profile_ub.shape[0]) * acc_limit
    jerk_profile = sm.Vector.ones(pos_vel_profile_ub.shape[0]) * jerk_limit

    # vel and acc profile for slowing down asap
    proj_vel_profile, proj_acc_profile, _ = compute_slowdown_asap_vel_profile(
        dof_symbols.velocity,
        dof_symbols.acceleration,
        goal_profile,
        Scalar(jerk_limit),
        Scalar(dt),
        ph,
        skip_first,
    )
    # jerk profile when slowing down without jerk limits
    _, _, proj_jerk_profile_violated = compute_slowdown_asap_vel_profile(
        dof_symbols.velocity,
        dof_symbols.acceleration,
        goal_profile,
        Scalar(np.inf),
        Scalar(dt),
        ph,
        skip_first,
    )
    # check if my projected vel profile violated position limits
    vel_lb_violated = sm.logic_or(
        sm.logic_any(proj_vel_profile < pos_vel_profile_lb - eps),
        sm.abs(proj_vel_profile[-1]) >= eps,
    )
    vel_ub_violated = sm.logic_or(
        sm.logic_any(proj_vel_profile > pos_vel_profile_ub + eps),
        sm.abs(proj_vel_profile[-1]) >= eps,
    )

    # if either lower or upper position limits would get violated, relax jerk constraints to max slow down.
    special_jerk_limits = sm.logic_or(vel_lb_violated, vel_ub_violated)
    # with 3 derivatives, slow down is possible in 3 steps
    jerk_profile[0] = sm.if_else(
        special_jerk_limits,
        sm.max(Scalar(jerk_limit), sm.abs(proj_jerk_profile_violated[0])),
        sm.Scalar(jerk_limit),
    )
    jerk_profile[1] = sm.if_else(
        special_jerk_limits,
        sm.max(Scalar(jerk_limit), sm.abs(proj_jerk_profile_violated[1])),
        sm.Scalar(jerk_limit),
    )
    jerk_profile[2] = sm.if_else(
        special_jerk_limits,
        sm.max(Scalar(jerk_limit), sm.abs(proj_jerk_profile_violated[2])),
        sm.Scalar(jerk_limit),
    )

    lb_profile = sm.concatenate(pos_vel_profile_lb, -acc_profile, -jerk_profile)
    ub_profile = sm.concatenate(pos_vel_profile_ub, acc_profile, jerk_profile)
    lb_profile = sm.min(lb_profile, ub_profile)
    ub_profile = sm.max(lb_profile, ub_profile)
    return lb_profile, ub_profile
