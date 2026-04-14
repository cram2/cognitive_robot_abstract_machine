from __future__ import annotations

import abc
import logging
from abc import ABC
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, TYPE_CHECKING, DefaultDict, Type

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.exceptions import (
    InfeasibleException,
    VelocityLimitUnreachableException,
)
from giskardpy.qp.pos_in_vel_limits import (
    shifted_velocity_profile,
    compute_slowdown_asap_vel_profile,
)
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.utils.decorators import memoize
from giskardpy.utils.math import mpc
from krrood.symbolic_math.symbolic_math import Vector, Matrix
from semantic_digital_twin.spatial_types.derivatives import Derivatives, DerivativeMap
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
import giskardpy.utils.math as gm
from krrood.utils import memoize

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller_config import QPControllerConfig


def max_velocity_from_horizon_and_jerk_qp(
    prediction_horizon: int,
    vel_limit: float,
    acc_limit: float,
    jerk_limit: float,
    dt: float,
    max_derivative: Derivatives,
    solver_class: Type[QPSolver],
):
    upper_limits = (
        (vel_limit,) * prediction_horizon,
        (acc_limit,) * prediction_horizon,
        (jerk_limit,) * prediction_horizon,
    )
    lower_limits = (
        (-vel_limit,) * prediction_horizon,
        (-acc_limit,) * prediction_horizon,
        (-jerk_limit,) * prediction_horizon,
    )
    return mpc(
        upper_limits=upper_limits,
        lower_limits=lower_limits,
        current_values=(0, 0),
        dt=dt,
        ph=prediction_horizon,
        q_weight=(0, 0, 0),
        lin_weight=(-1, 0, 0),
        solver_class=solver_class,
        link_to_current_vel=False,
    )


def _sorter(*args: dict) -> Tuple[List[sm.SymbolicScalar], np.ndarray]:
    """
    Sorts every arg dict individually and then appends all of them.
    :arg args: a bunch of dicts
    :return: list
    """
    result = []
    result_names = []
    for arg in args:
        result.extend(__helper(arg))
        result_names.extend(__helper_names(arg))
    return result, np.array(result_names)


def __helper(param: dict):
    return [x for _, x in sorted(param.items())]


def __helper_names(param: dict):
    return [x for x, _ in sorted(param.items())]


@dataclass(eq=False)
class ProblemDataPart(ABC):
    """
    min_x 0.5*x^T*diag(w)*x + g^T*x
    s.t.  lb <= x <= ub
               Ex = b
        lbA <= Ax <= ubA
    """

    degrees_of_freedom: List[DegreeOfFreedom]
    constraint_collection: ConstraintCollection
    config: QPControllerConfig

    def __post_init__(self):
        self.control_horizon = (
            self.config.prediction_horizon - self.config.max_derivative + 1
        )

    def __hash__(self):
        return id(self)

    @property
    def number_of_free_variables(self) -> int:
        return len(self.degrees_of_freedom)

    @property
    def number_ineq_slack_variables(self):
        return sum(
            self.control_horizon
            for c in self.constraint_collection.velocity_inequality_constraints
        )

    @abc.abstractmethod
    def construct_expression(
        self,
    ) -> Tuple[sm.Matrix, sm.Matrix]:
        pass

    def _remove_columns_columns_where_variables_are_zero(
        self, free_variable_model: sm.Matrix, max_derivative: Derivatives
    ) -> sm.Matrix:
        if np.prod(free_variable_model.shape) == 0:
            return free_variable_model
        column_ids = []
        end = 0
        for derivative in Derivatives.range(Derivatives.velocity, max_derivative - 1):
            last_non_zero_variable = self.config.prediction_horizon - (
                max_derivative - derivative
            )
            start = end + self.number_of_free_variables * last_non_zero_variable
            end += self.number_of_free_variables * self.config.prediction_horizon
            column_ids.extend(range(start, end))
        free_variable_model.remove([], column_ids)
        return free_variable_model

    def velocity_limit(
        self, v: DegreeOfFreedom, max_derivative: Derivatives
    ) -> Tuple[sm.Vector, sm.Vector]:
        lower_limits = DerivativeMap()
        upper_limits = DerivativeMap()

        # %% pos limits
        if not v.has_position_limits():
            lower_limits.position = upper_limits.position = None
        else:
            lower_limits.position = v.limits.lower.position
            upper_limits.position = v.limits.upper.position

        # %% vel limits
        lower_limits.velocity = v.limits.lower.velocity
        upper_limits.velocity = v.limits.upper.velocity
        if self.config.prediction_horizon == 1:
            return sm.Vector([lower_limits.velocity]), sm.Vector(
                [upper_limits.velocity]
            )

        # %% acc limits
        if v.limits.lower.acceleration is None:
            lower_limits.acceleration = -np.inf
        else:
            lower_limits.acceleration = v.limits.lower.acceleration
        if v.limits.upper.acceleration is None:
            upper_limits.acceleration = np.inf
        else:
            upper_limits.acceleration = v.limits.upper.acceleration

        # %% jerk limits
        if upper_limits.jerk is None:
            upper_limits.jerk = self.find_best_jerk_limit(
                self.config.prediction_horizon,
                self.config.mpc_dt,
                upper_limits.velocity,
                solver_class=self.config.qp_solver_class,
            )
            lower_limits.jerk = -upper_limits.jerk
        else:
            upper_limits.jerk = v.limits.upper.jerk
            lower_limits.jerk = v.limits.lower.jerk

        try:
            lb, ub = self.b_profile(
                dof_symbols=v.variables,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                solver_class=self.config.qp_solver_class,
                dt=self.config.mpc_dt,
                ph=self.config.prediction_horizon,
            )
        except InfeasibleException as e:
            max_reachable_vel = max_velocity_from_horizon_and_jerk_qp(
                prediction_horizon=self.config.prediction_horizon,
                vel_limit=100,
                acc_limit=upper_limits.acceleration,
                jerk_limit=upper_limits.jerk,
                dt=self.config.mpc_dt,
                max_derivative=max_derivative,
                solver_class=self.config.qp_solver_class,
            )[0]
            if max_reachable_vel < upper_limits.velocity:
                error_msg = (
                    f'Free variable "{v.name}" can\'t reach velocity limit of "{upper_limits.velocity}". '
                    f'Maximum reachable with prediction horizon = "{self.config.prediction_horizon}", '
                    f'jerk limit = "{upper_limits.jerk}" and dt = "{self.config.mpc_dt}" is "{max_reachable_vel}".'
                )
                logger.error(error_msg)
                raise VelocityLimitUnreachableException(error_msg)
            else:
                raise
        return lb, ub

    @memoize
    def find_best_jerk_limit(
        self,
        prediction_horizon: int,
        dt: float,
        target_vel_limit: float,
        solver_class: Type[QPSolver],
        eps: float = 0.0001,
    ) -> float:
        jerk_limit = (4 * target_vel_limit) / dt**2
        upper_bound = jerk_limit
        lower_bound = 0
        best_vel_limit = 0
        best_jerk_limit = 0
        i = -1
        for i in range(100):
            vel_limit = max_velocity_from_horizon_and_jerk_qp(
                prediction_horizon=prediction_horizon,
                vel_limit=1000,
                acc_limit=np.inf,
                jerk_limit=jerk_limit,
                dt=dt,
                max_derivative=Derivatives.jerk,
                solver_class=solver_class,
            )[0]
            if abs(vel_limit - target_vel_limit) < abs(
                best_vel_limit - target_vel_limit
            ):
                best_vel_limit = vel_limit
                best_jerk_limit = jerk_limit
            if abs(vel_limit - target_vel_limit) < eps:
                break
            if vel_limit > target_vel_limit:
                upper_bound = jerk_limit
                jerk_limit = round((jerk_limit + lower_bound) / 2, 4)
            else:
                lower_bound = jerk_limit
                jerk_limit = round((jerk_limit + upper_bound) / 2, 4)
        logger.debug(
            f"best velocity limit: {best_vel_limit} "
            f"(target = {target_vel_limit}) with jerk limit: {best_jerk_limit} after {i + 1} iterations"
        )
        return best_jerk_limit

    @memoize
    def b_profile(
        self,
        dof_symbols: DerivativeMap[sm.FloatVariable],
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
                sm.max(sm.Scalar(0), pos_error_lb / dt), sm.Scalar(one_step_change_)
            )
            one_step_change_lb = sm.limit(one_step_change_lb, -vel_limit, vel_limit)
            one_step_change_ub = sm.max(
                sm.min(sm.Scalar(0), pos_error_ub / dt), -sm.Scalar(one_step_change_)
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
            skip_first = sm.logic_or(
                pos_vel_profile_lb[0] >= 0, pos_vel_profile_ub[0] <= 0
            )
        else:
            goal_profile = sm.Vector.zeros(ph)
            pos_vel_profile_ub = sm.Vector.ones(ph) * vel_limit
            pos_vel_profile_lb = -pos_vel_profile_ub
            skip_first = sm.Scalar(0)

        acc_profile = sm.Vector.ones(pos_vel_profile_ub.shape[0]) * acc_limit
        jerk_profile = sm.Vector.ones(pos_vel_profile_ub.shape[0]) * jerk_limit

        # vel and acc profile for slowing down asap
        proj_vel_profile, proj_acc_profile, _ = compute_slowdown_asap_vel_profile(
            dof_symbols.velocity,
            dof_symbols.acceleration,
            goal_profile,
            sm.Scalar(jerk_limit),
            sm.Scalar(dt),
            ph,
            skip_first,
        )
        # jerk profile when slowing down without jerk limits
        _, _, proj_jerk_profile_violated = compute_slowdown_asap_vel_profile(
            dof_symbols.velocity,
            dof_symbols.acceleration,
            goal_profile,
            sm.Scalar(np.inf),
            sm.Scalar(dt),
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
            sm.max(sm.Scalar(jerk_limit), sm.abs(proj_jerk_profile_violated[0])),
            sm.Scalar(jerk_limit),
        )
        jerk_profile[1] = sm.if_else(
            special_jerk_limits,
            sm.max(sm.Scalar(jerk_limit), sm.abs(proj_jerk_profile_violated[1])),
            sm.Scalar(jerk_limit),
        )
        jerk_profile[2] = sm.if_else(
            special_jerk_limits,
            sm.max(sm.Scalar(jerk_limit), sm.abs(proj_jerk_profile_violated[2])),
            sm.Scalar(jerk_limit),
        )

        lb_profile = sm.concatenate(pos_vel_profile_lb, -acc_profile, -jerk_profile)
        ub_profile = sm.concatenate(pos_vel_profile_ub, acc_profile, jerk_profile)
        lb_profile = sm.min(lb_profile, ub_profile)
        ub_profile = sm.max(lb_profile, ub_profile)
        return lb_profile, ub_profile


@dataclass
class Weights(ProblemDataPart):
    """
    order:
        free_variable_velocity
        free_variable_acceleration
        free_variable_jerk
        eq integral constraints
        eq vel constraints
        neq integral constraints
        neq vel constraints
    """

    evaluated: bool = field(default=True)

    def construct_expression(self) -> Tuple[sm.Vector, sm.Vector]:
        components = []
        components.extend(self.free_variable_weights_expression())
        components.append(self.equality_weight_expressions())
        components.extend(self.eq_derivative_weight_expressions())
        components.append(self.inequality_weight_expressions())
        components.extend(self.derivative_weight_expressions())
        weights, _ = _sorter(*components)
        quadratic_weights = []
        linear_weights = []
        for quadratic_weight, linear_weight in weights:
            quadratic_weights.append(quadratic_weight)
            linear_weights.append(linear_weight)
        return sm.Vector(quadratic_weights), sm.Vector(linear_weights)

    def free_variable_weights_expression(self) -> List[tuple[defaultdict, defaultdict]]:
        max_derivative = self.config.max_derivative
        params = []
        weights = defaultdict(dict)  # maps order to joints
        for t in range(self.config.prediction_horizon):
            for v in self.degrees_of_freedom:
                for derivative in Derivatives.range(
                    Derivatives.velocity, max_derivative
                ):
                    if t >= self.config.prediction_horizon - (
                        max_derivative - derivative
                    ):
                        continue
                    if derivative == Derivatives.acceleration:
                        continue
                    normalized_weight = self.normalize_dof_weight(
                        limit=v.limits.upper[derivative],
                        base_weight=self.config.get_dof_weight(v.name, derivative),
                        t=t,
                        derivative=derivative,
                        horizon=self.config.prediction_horizon - 3,
                        alpha=self.config.horizon_weight_gain_scalar,
                    )
                    weights[derivative][
                        f"t{t:03}/{v.variables.position.dof.name}/{derivative}"
                    ] = (normalized_weight, 0)
        for _, weight in sorted(weights.items()):
            params.append(weight)
        return params

    def normalize_dof_weight(self, limit, base_weight, t, derivative, horizon, alpha):
        def linear(x_in: float, weight: float, h: int, alpha: float) -> float:
            start = weight * alpha
            a = (weight - start) / h
            return a * x_in + start

        if limit is None:
            return 0.0
        weight = linear(t, base_weight, horizon, alpha)

        return weight * (1 / limit) ** 2

    def derivative_weight_expressions(self) -> List[Dict[str, sm.Scalar]]:
        params = []
        for d in Derivatives.range(Derivatives.velocity, self.config.max_derivative):
            derivative_constr_weights = {}
            for t in range(self.config.prediction_horizon):
                d = Derivatives(d)
                for (
                    c
                ) in self.constraint_collection.get_inequality_constraints_by_derivative(
                    d
                ):
                    if t < self.control_horizon:
                        derivative_constr_weights[f"t{t:03}/{c.name}"] = (
                            c.normalized_weight(),
                            c.linear_weight,
                        )
            params.append(derivative_constr_weights)
        return params

    def eq_derivative_weight_expressions(self) -> List[Dict[str, sm.Scalar]]:
        params = []
        for d in Derivatives.range(Derivatives.velocity, self.config.max_derivative):
            derivative_constr_weights = {}
            for t in range(self.config.prediction_horizon):
                d = Derivatives(d)
                for (
                    c
                ) in self.constraint_collection.get_equality_constraints_by_derivative(
                    d
                ):
                    if t < self.control_horizon:
                        derivative_constr_weights[f"t{t:03}/{c.name}"] = (
                            c.normalized_weight(),
                            c.linear_weight,
                        )
            params.append(derivative_constr_weights)
        return params

    def equality_weight_expressions(self) -> dict:
        error_slack_weights = {
            f"{c.name}/error": (
                c.normalized_weight(self.control_horizon),
                c.linear_weight,
            )
            for c in self.constraint_collection.equality_constraints
        }
        return error_slack_weights

    def inequality_weight_expressions(self) -> dict:
        error_slack_weights = {
            f"{c.name}/error": (
                c.normalized_weight(self.control_horizon),
                c.linear_weight,
            )
            for c in self.constraint_collection.inequality_constraints
        }
        return error_slack_weights

    def get_free_variable_symbols(self, order: Derivatives) -> List[sm.FloatVariable]:
        return _sorter(
            {
                v.variables.position: v.variables.data[order]
                for v in self.degrees_of_freedom
            }
        )[0]


@dataclass(eq=False)
class FreeVariableBounds(ProblemDataPart):
    """
    order:
        free_variable_velocity
        free_variable_acceleration
        free_variable_jerk
        eq integral constraints
        eq vel constraints
        neq integral constraints
        neq vel constraints
    """

    names: np.ndarray = field(default=None)
    names_without_slack: np.ndarray = field(default=None)
    names_slack: np.ndarray = field(default=None)
    names_neq_slack: np.ndarray = field(default=None)
    names_derivative_slack: np.ndarray = field(default=None)
    names_eq_slack: np.ndarray = field(default=None)
    evaluated: bool = field(default=True)

    def free_variable_bounds(
        self,
    ) -> Tuple[List[Dict[str, sm.ScalarData]], List[Dict[str, sm.ScalarData]]]:
        max_derivative = self.config.max_derivative
        lb: DefaultDict[Derivatives, Dict[str, sm.ScalarData]] = defaultdict(dict)
        ub: DefaultDict[Derivatives, Dict[str, sm.ScalarData]] = defaultdict(dict)
        for v in self.degrees_of_freedom:
            lb_, ub_ = self.velocity_limit(v=v, max_derivative=max_derivative)
            for t in range(self.config.prediction_horizon):
                for derivative in Derivatives.range(
                    Derivatives.velocity, max_derivative
                ):
                    if t >= self.config.prediction_horizon - (
                        max_derivative - derivative
                    ):
                        continue
                    if derivative == Derivatives.acceleration:
                        continue
                    if derivative == Derivatives.jerk:
                        multiplier = self.config.mpc_dt**2
                    else:
                        multiplier = 1
                    index = t + self.config.prediction_horizon * (derivative - 1)
                    lb[derivative][f"t{t:03}/{v.name}/{derivative}"] = (
                        lb_[index] * multiplier
                    )
                    ub[derivative][f"t{t:03}/{v.name}/{derivative}"] = (
                        ub_[index] * multiplier
                    )
        lb_params = []
        ub_params = []
        for derivative, name_to_bound_map in sorted(lb.items()):
            lb_params.append(name_to_bound_map)
        for derivative, name_to_bound_map in sorted(ub.items()):
            ub_params.append(name_to_bound_map)
        return lb_params, ub_params

    def derivative_slack_limits(
        self, derivative: Derivatives
    ) -> Tuple[Dict[str, sm.Scalar], Dict[str, sm.Scalar]]:
        lower_slack = {}
        upper_slack = {}
        for t in range(self.config.prediction_horizon):
            for (
                c
            ) in self.constraint_collection.get_inequality_constraints_by_derivative(
                derivative
            ):
                if t < self.control_horizon:
                    lower_slack[f"t{t:03}/{c.name}"] = c.lower_slack_limit
                    upper_slack[f"t{t:03}/{c.name}"] = c.upper_slack_limit
        return lower_slack, upper_slack

    def eq_derivative_slack_limits(
        self, derivative: Derivatives
    ) -> Tuple[Dict[str, sm.Scalar], Dict[str, sm.Scalar]]:
        lower_slack = {}
        upper_slack = {}
        for t in range(self.config.prediction_horizon):
            for c in self.constraint_collection.get_equality_constraints_by_derivative(
                derivative
            ):
                if t < self.control_horizon:
                    lower_slack[f"t{t:03}/{c.name}"] = c.lower_slack_limit[t]
                    upper_slack[f"t{t:03}/{c.name}"] = c.upper_slack_limit[t]
        return lower_slack, upper_slack

    def equality_constraint_slack_lower_bound(self):
        return {
            f"{c.name}/error": c.lower_slack_limit
            for c in self.constraint_collection.equality_constraints
        }

    def equality_constraint_slack_upper_bound(self):
        return {
            f"{c.name}/error": c.upper_slack_limit
            for c in self.constraint_collection.equality_constraints
        }

    def inequality_constraint_slack_lower_bound(self):
        return {
            f"{c.name}/error": c.lower_slack_limit
            for c in self.constraint_collection.inequality_constraints
        }

    def inequality_constraint_slack_upper_bound(self):
        return {
            f"{c.name}/error": c.upper_slack_limit
            for c in self.constraint_collection.inequality_constraints
        }

    def construct_expression(
        self,
    ) -> Tuple[sm.Vector, sm.Vector]:
        # derivative model
        lb_params, ub_params = self.free_variable_bounds()
        num_free_variables = sum(len(x) for x in lb_params)

        # eq integral constraints
        equality_constraint_slack_lower_bounds = (
            self.equality_constraint_slack_lower_bound()
        )
        num_eq_slacks = len(equality_constraint_slack_lower_bounds)
        lb_params.append(equality_constraint_slack_lower_bounds)
        ub_params.append(self.equality_constraint_slack_upper_bound())

        # eq vel constraints
        num_eq_derivative_slack = 0
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative
        ):
            lower_slack, upper_slack = self.eq_derivative_slack_limits(derivative)
            num_eq_derivative_slack += len(lower_slack)
            lb_params.append(lower_slack)
            ub_params.append(upper_slack)

        # neq integral constraints
        lb_params.append(self.inequality_constraint_slack_lower_bound())
        ub_params.append(self.inequality_constraint_slack_upper_bound())

        # neq vel constraints
        num_derivative_slack = 0
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative
        ):
            lower_slack, upper_slack = self.derivative_slack_limits(derivative)
            num_derivative_slack += len(lower_slack)
            lb_params.append(lower_slack)
            ub_params.append(upper_slack)

        lb, self.names = _sorter(*lb_params)
        ub, _ = _sorter(*ub_params)
        self.names_without_slack = self.names[:num_free_variables]
        self.names_slack = self.names[num_free_variables:]

        derivative_slack_start = 0
        derivative_slack_stop = (
            derivative_slack_start + num_derivative_slack + num_eq_derivative_slack
        )
        self.names_derivative_slack = self.names_slack[
            derivative_slack_start:derivative_slack_stop
        ]

        eq_slack_start = derivative_slack_stop
        eq_slack_stop = eq_slack_start + num_eq_slacks
        self.names_eq_slack = self.names_slack[eq_slack_start:eq_slack_stop]

        neq_slack_start = eq_slack_stop
        self.names_neq_slack = self.names_slack[neq_slack_start:]
        return sm.Vector(lb), sm.Vector(ub)


@dataclass
class EqualityBounds(ProblemDataPart):
    """

    order:
        derivative model (optional)
        eq integral constraints
        eq vel constraints
    """

    names: np.ndarray = field(default=None)
    names_equality_constraints: np.ndarray = field(default=None)
    names_derivative_links: np.ndarray = field(default=None)
    evaluated: bool = field(default=True)

    def equality_constraint_bounds(self) -> Dict[str, sm.Scalar]:
        return {
            f"{c.name}": c.capped_bound(self.config.mpc_dt, self.control_horizon)
            for c in self.constraint_collection.equality_constraints
        }

    def last_derivative_values(
        self, derivative: Derivatives
    ) -> Dict[str, sm.ScalarData]:
        last_values = {}
        for v in self.degrees_of_freedom:
            last_values[f"{v.name}/last_{derivative}"] = v.variables.data[derivative]
        return last_values

    def derivative_links(self, derivative: Derivatives) -> Dict[str, sm.ScalarData]:
        derivative_link = {}
        for t in range(self.config.prediction_horizon - 1):
            if t >= self.config.prediction_horizon - (
                self.config.max_derivative - derivative
            ):
                continue  # this row is all zero in the model, because the system has to stop at 0 vel
            for v in self.degrees_of_freedom:
                derivative_link[f"t{t:03}/{derivative}/{v.name}/link"] = 0
        return derivative_link

    def eq_derivative_constraint_bounds(
        self, derivative: Derivatives
    ) -> Dict[str, sm.Vector]:
        bound = {}
        for t in range(self.config.prediction_horizon):
            for c in self.constraint_collection.get_equality_constraints_by_derivative(
                derivative
            ):
                if t < self.control_horizon:
                    bound[f"t{t:03}/{c.name}"] = c.bound[t] * self.config.mpc_dt
        return bound

    def construct_expression(
        self,
    ) -> sm.Vector:
        """
        explicit no acc
        -vc - ac*dt = -v0 + j0*dt**2
                 vc = -v1 + 2*v0 + j1*dt**2
                  0 = -vt + 2*vt-1 - vt-2 + jt*dt**2
                  0 =   0 + 2*vt-1 - vt-2 + jt*dt**2
                  0 =   0 +    0   - vt-2 + jt*dt**2
        :return:
        """
        bounds = []
        # derivative model
        derivative_link = {}
        for t in range(self.config.prediction_horizon):
            for v in self.degrees_of_freedom:
                name = f"t{t:03}/{Derivatives.jerk}/{v.name}/link"
                if t == 0:
                    derivative_link[name] = (
                        -v.variables.velocity
                        - v.variables.acceleration * self.config.mpc_dt
                    )
                elif t == 1:
                    derivative_link[name] = v.variables.velocity
                else:
                    derivative_link[name] = 0
        bounds.append(derivative_link)

        num_derivative_links = sum(len(x) for x in bounds)
        num_derivative_constraints = 0

        # eq integral constraints
        bounds.append(self.equality_constraint_bounds())

        # eq vel constraints
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative
        ):
            bound = self.eq_derivative_constraint_bounds(derivative)
            num_derivative_constraints += len(bound)
            bounds.append(bound)

        bounds, self.names = _sorter(*bounds)
        self.names_derivative_links = self.names[:num_derivative_links]
        # self.names_equality_constraints = self.names[num_derivative_links:]
        return sm.Vector(bounds)


@dataclass
class InequalityBounds(ProblemDataPart):
    """
    order:
        derivative model (optional)
        neq integral constraints
        neq vel constraints
    """

    default_limits: bool = field(default=False)
    names: np.ndarray = field(default=None)
    names_position_limits: np.ndarray = field(default=None)
    names_derivative_links: np.ndarray = field(default=None)
    names_neq_constraints: np.ndarray = field(default=None)
    names_non_position_limits: np.ndarray = field(default=None)
    evaluated: bool = field(default=True)

    def derivative_constraint_bounds(
        self, derivative: Derivatives
    ) -> Tuple[Dict[str, sm.Vector], Dict[str, sm.Vector]]:
        lower = {}
        upper = {}
        for t in range(self.config.prediction_horizon):
            for (
                c
            ) in self.constraint_collection.get_inequality_constraints_by_derivative(
                derivative
            ):
                if t < self.control_horizon:
                    lower[f"t{t:03}/{c.name}"] = c.lower_limit * self.config.mpc_dt
                    upper[f"t{t:03}/{c.name}"] = c.upper_limit * self.config.mpc_dt
        return lower, upper

    def lower_inequality_constraint_bound(self):
        bounds = {}
        for constraint in self.constraint_collection.inequality_constraints:
            if isinstance(constraint.lower_error, float) and np.isinf(
                constraint.lower_error
            ):
                bounds[f"{constraint.name}"] = constraint.lower_error
            else:
                bounds[f"{constraint.name}"] = constraint.capped_lower_error(
                    self.config.mpc_dt, self.control_horizon
                )
        return bounds

    def upper_inequality_constraint_bound(self):
        bounds = {}
        for constraint in self.constraint_collection.inequality_constraints:
            if isinstance(constraint.upper_error, float) and np.isinf(
                constraint.upper_error
            ):
                bounds[f"{constraint.name}"] = constraint.upper_error
            else:
                bounds[f"{constraint.name}"] = constraint.capped_upper_error(
                    self.config.mpc_dt, self.control_horizon
                )
        return bounds

    def implicit_pos_model_limits(
        self,
    ) -> Tuple[List[Dict[str, sm.Vector]], List[Dict[str, sm.Vector]]]:
        lb_acc, ub_acc = {}, {}
        lb_jerk, ub_jerk = {}, {}
        for v in self.degrees_of_freedom:
            lb_, ub_ = v.lower_limits.position, v.upper_limits.position
            for t in range(self.config.prediction_horizon - 2):
                ptc = v.variables.position
                lb_jerk[f"t{t:03}/{v.name}/{Derivatives.position}"] = lb_ - ptc
                ub_jerk[f"t{t:03}/{v.name}/{Derivatives.position}"] = ub_ - ptc
        return [lb_acc, lb_jerk], [ub_acc, ub_jerk]

    def implicit_model_limits(
        self,
    ) -> Tuple[List[Dict[str, sm.Vector]], List[Dict[str, sm.Vector]]]:
        lb_acc, ub_acc = {}, {}
        lb_jerk, ub_jerk = {}, {}
        for v in self.degrees_of_freedom:
            lb_, ub_ = self.velocity_limit(v=v, max_derivative=Derivatives.jerk)
            for t in range(self.config.prediction_horizon):
                # if self.config.max_derivative >= Derivatives.acceleration:
                #     a_min = v.get_lower_limit(Derivatives.acceleration)
                #     a_max = v.get_upper_limit(Derivatives.acceleration)
                #     if not ((np.isinf(a_min) or cas.is_inf(a_min)) and (np.isinf(a_max) or cas.is_inf(a_max))):
                #         vtc = v.symbols.velocity
                #         if t == 0:
                #             # vtc/dt + a_min <= vt0/dt <= vtc/dt + a_max
                #             lb_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = vtc / self.config.mpc_dt + a_min
                #             ub_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = vtc / self.config.mpc_dt + a_max
                #         else:
                #             lb_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = a_min
                #             ub_acc[f't{t:03}/{v.name}/{Derivatives.acceleration}'] = a_max
                if self.config.max_derivative >= Derivatives.jerk:
                    j_min = lb_[self.config.prediction_horizon * 2 + t]
                    j_max = ub_[self.config.prediction_horizon * 2 + t]
                    vtc = v.variables.velocity
                    atc = v.variables.acceleration
                    if t == 0:
                        # vtc/dt**2 + atc/dt + j_min <=    vt0/dt**2     <= vtc/dt**2 + atc/dt + j_max
                        lb_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = (
                            vtc / self.config.mpc_dt**2
                            + atc / self.config.mpc_dt
                            + j_min
                        )
                        ub_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = (
                            vtc / self.config.mpc_dt**2
                            + atc / self.config.mpc_dt
                            + j_max
                        )
                    elif t == 1:
                        # (- vtc)/dt**2 + j_min <= (vt1 - 2vt0)/dt**2 <= (- vtc)/dt**2 + j_max
                        lb_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = (
                            -vtc / self.config.mpc_dt**2 + j_min
                        )
                        ub_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = (
                            -vtc / self.config.mpc_dt**2 + j_max
                        )
                    else:
                        lb_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = j_min
                        ub_jerk[f"t{t:03}/{v.name}/{Derivatives.jerk}"] = j_max
        return [lb_acc, lb_jerk], [ub_acc, ub_jerk]

    def construct_expression(
        self,
    ) -> Tuple[sm.Vector, sm.Vector]:
        lb_params: List[Dict[str, sm.Vector]] = []
        ub_params: List[Dict[str, sm.Vector]] = []

        # neq integral constraints
        lower_inequality_constraint_bounds = self.lower_inequality_constraint_bound()
        lb_params.append(lower_inequality_constraint_bounds)
        ub_params.append(self.upper_inequality_constraint_bound())
        num_neq_constraints = len(lower_inequality_constraint_bounds)

        # neq vel constraints
        num_derivative_constraints = 0
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative
        ):
            lower, upper = self.derivative_constraint_bounds(derivative)
            num_derivative_constraints += len(lower)
            lb_params.append(lower)
            ub_params.append(upper)

        lbA, self.names = _sorter(*lb_params)
        ubA, _ = _sorter(*ub_params)

        self.names_derivative_links = self.names[:num_derivative_constraints]
        self.names_neq_constraints = self.names[
            num_derivative_constraints + num_neq_constraints :
        ]

        return sm.Vector(lbA), sm.Vector(ubA)


@dataclass
class EqualityModel(ProblemDataPart):
    """
    Format:
        last free variable velocity
        0
        last free variable acceleration
        0
        equality_constraint_bounds
    """

    def equality_constraint_expressions(self) -> List[sm.Matrix]:
        return _sorter(
            {
                c.name: c.expression
                for c in self.constraint_collection.equality_constraints
            }
        )[0]

    def get_free_variable_symbols(
        self, derivative: Derivatives
    ) -> List[sm.FloatVariable]:
        return _sorter(
            {
                v.variables.position.name: v.variables.data[derivative]
                for v in self.degrees_of_freedom
            }
        )[0]

    def get_eq_derivative_constraint_expressions(self, derivative: Derivatives):
        return _sorter(
            {
                c.name: c.expression
                for c in self.constraint_collection.derivative_equality_constraints
                if c.derivative == derivative
            }
        )[0]

    def velocity_constraint_model(self) -> Tuple[sm.Matrix, sm.Matrix]:
        """
        model
        |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   | prediction horizon
        |v1 v2 v3|v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|j1 j2 j3| free variables
        |--------------------------------------------------------------------------------|
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |--------------------------------------------------------------------------------|
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |--------------------------------------------------------------------------------|

        slack model
        |   t1 |   t2 | prediction horizon
        |s1 s2 |s1 s2 | slack
        |------|------|
        |sp    |      | vel constr 1
        |   sp |      | vel constr 2
        |-------------|
        |      |sp    | vel constr 1
        |      |   sp | vel constr 2
        |-------------|
        """
        number_of_vel_rows = len(
            self.constraint_collection.velocity_equality_constraints
        ) * (self.config.prediction_horizon - 2)
        if number_of_vel_rows > 0:
            expressions = sm.Matrix(
                self.get_eq_derivative_constraint_expressions(Derivatives.velocity)
            )
            parts = []
            for derivative in Derivatives.range(
                Derivatives.position, self.config.max_derivative - 1
            ):
                if derivative == Derivatives.velocity:
                    continue
                if derivative == Derivatives.acceleration:
                    continue
                J_vel = (
                    expressions.jacobian(
                        variables=self.get_free_variable_symbols(derivative)
                    )
                    * self.config.mpc_dt
                )
                missing_variables = self.config.max_derivative - derivative - 1
                eye = sm.Matrix.eye(self.config.prediction_horizon)[
                    :-2, : self.config.prediction_horizon - missing_variables
                ]
                J_vel_limit_block = eye.kron(J_vel)
                parts.append(J_vel_limit_block)

            # constraint slack
            model = sm.hstack(parts)
            num_slack_variables = sum(
                self.control_horizon
                for c in self.constraint_collection.velocity_equality_constraints
            )
            slack_model = sm.Matrix.eye(num_slack_variables) * self.config.mpc_dt
            return model, slack_model
        return sm.Matrix(), sm.Matrix()

    @property
    def number_of_non_slack_columns(self) -> int:
        vel_columns = self.number_of_free_variables * (
            self.config.prediction_horizon - 2
        )
        jerk_columns = self.number_of_free_variables * self.config.prediction_horizon
        result = vel_columns
        result += jerk_columns
        return result

    def derivative_link_model(self, max_derivative: Derivatives) -> sm.Matrix:
        """
        Layout for prediction horizon 5
        Slots are matrices of |controlled variables| x |controlled variables|
        | vt0 | vt1 | vt2 | at0 | at1 | at2 | at3 | jt0 | jt1 | jt2 | jt3 | jt4 |
        |-----------------------------------------------------------------------|
        |  1  |     |     | -dt |     |     |     |     |     |     |     |     | last_v =  vt0 - at0*cdt
        | -1  |  1  |     |     | -dt |     |     |     |     |     |     |     |      0 = -vt0 + vt1 - at1 * mdt
        |     | -1  |  1  |     |     | -dt |     |     |     |     |     |     |      0 = -vt1 + vt2 - at2 * mdt
        |     |     | -1  |     |     |     | -dt |     |     |     |     |     |      0 = -vt2 - at3 * mdt
        |=======================================================================|
        |     |     |     |  1  |     |     |     | -dt |     |     |     |     | last_a =  at0 - jt0*cdt
        |     |     |     | -1  |  1  |     |     |     | -dt |     |     |     |      0 = -at0 + at1 - jt1 * mdt
        |     |     |     |     | -1  |  1  |     |     |     | -dt |     |     |      0 = -at1 + at2 - jt2 * mdt
        |     |     |     |     |     | -1  |  1  |     |     |     | -dt |     |      0 = -at2 + at3 - jt3 * mdt
        |     |     |     |     |     |     | -1  |     |     |     |     | -dt |      0 = -at3 - jt4 * mdt
        |-----------------------------------------------------------------------|
        """
        num_rows = (
            self.number_of_free_variables
            * self.config.prediction_horizon
            * (max_derivative - 1)
        )
        num_columns = (
            self.number_of_free_variables
            * self.config.prediction_horizon
            * max_derivative
        )
        derivative_link_model = sm.Matrix.zeros(num_rows, num_columns)

        x_n = sm.Matrix.eye(num_rows)
        derivative_link_model[:, : x_n.shape[0]] += x_n

        xd_n = -sm.Matrix.eye(num_rows) * self.config.mpc_dt
        h_offset = self.number_of_free_variables * self.config.prediction_horizon
        derivative_link_model[:, h_offset:] += xd_n

        x_c_height = self.number_of_free_variables * (
            self.config.prediction_horizon - 1
        )
        x_c = -sm.Matrix.eye(x_c_height)
        offset_v = 0
        offset_h = 0
        for derivative in Derivatives.range(Derivatives.velocity, max_derivative - 1):
            offset_v += self.number_of_free_variables
            derivative_link_model[
                offset_v : offset_v + x_c_height, offset_h : offset_h + x_c_height
            ] += x_c
            offset_v += x_c_height
            offset_h += self.config.prediction_horizon * self.number_of_free_variables
        derivative_link_model = self._remove_rows_columns_where_variables_are_zero(
            derivative_link_model
        )
        return derivative_link_model

    def derivative_link_model_no_acc(self, max_derivative: Derivatives) -> sm.Matrix:
        """
        Layout for prediction horizon 5
        Slots are matrices of |controlled variables| x |controlled variables|
        | vt0 | vt1 | vt2 | jt0 | jt1 | jt2 | jt3 | jt4 |
        |-----------------------------------------------|
        | -1  |     |     |dt**2|     |     |     |     |       -vc - ac*dt = -v0 + j0*dt**2
        |  2  | -1  |     |     |dt**2|     |     |     |                vc = -v1 + 2*v0 + j1*dt**2
        | -1  |  2  | -1  |     |     |dt**2|     |     |                 0 = -vt + 2*vt-1 - vt-2 + jt*dt**2
        |     | -1  |  2  |     |     |     |dt**2|     |                 0 =   0 + 2*vt-1 - vt-2 + jt*dt**2
        |     |     | -1  |     |     |     |     |dt**2|                 0 =   0 +    0   - vt-2 + jt*dt**2
        |-----------------------------------------------|
        vt = vt-1 + at * dt     <=>  vt/dt - vt-1/dt = at
        at = at-1 + jt * dt     <=>  at/dt - at-1/dt = jt

        vt = vt-1 + (at-1 + jt * dt) * dt
        vt = vt-1 + at-1*dt + jt * dt**2
        vt = vt-1 + (vt-1/dt - vt-2/dt)*dt + jt * dt**2
        vt = vt-1 + vt-1 - vt-2 + jt*dt**2

        0 = -v1 + 2*v0 - vc + j1*dt**2
        vc = -v1 + 2*v0 + j1*dt**2

        a0/dt - ac/dt = j0
        (v0/dt - vc/dt)/dt - ac/dt = j0
        v0/dt**2 - vc/dt**2 - ac/dt = j0
        -vc/dt**2 - ac/dt = -v0/dt**2 + j0
        -vc - ac*dt = -v0 + j0*dt**2

        v0 = vc + ac*dt + j0*dt**2
        vc = -v1 + 2*v0 + j1*dt**2
        v1 = -vc + 2*v0 + j1*dt**2
        """
        n_vel = self.number_of_free_variables * (self.config.prediction_horizon - 2)
        n_jerk = self.number_of_free_variables * self.config.prediction_horizon
        model = sm.Matrix.zeros(rows=n_jerk, columns=n_jerk + n_vel)
        pre_previous = -sm.Matrix.eye(n_vel)
        same = pre_previous
        previous = -2 * pre_previous
        j_same = sm.Matrix.eye(n_jerk)  # * self.config.mpc_dt**2
        # j_same = sm.Matrix.eye(n_jerk) * self.config.mpc_dt**2
        model[: -self.number_of_free_variables * 2, :n_vel] += pre_previous
        model[
            self.number_of_free_variables : -self.number_of_free_variables, :n_vel
        ] += previous
        model[self.number_of_free_variables * 2 :, :n_vel] += same
        model[:, n_vel:] = j_same
        return model

    def _remove_rows_columns_where_variables_are_zero(
        self, derivative_link_model: sm.Matrix
    ) -> sm.Matrix:
        if np.prod(derivative_link_model.shape) == 0:
            return derivative_link_model
        row_ids = []
        end = 0
        for derivative in Derivatives.range(
            Derivatives.velocity, self.config.max_derivative - 1
        ):
            last_non_zero_variable = self.config.prediction_horizon - (
                self.config.max_derivative - derivative - 1
            )
            start = end + self.number_of_free_variables * last_non_zero_variable
            end += self.number_of_free_variables * self.config.prediction_horizon
            row_ids.extend(range(start, end))
        derivative_link_model.remove(row_ids, [])
        return derivative_link_model

    def equality_constraint_model(self) -> Tuple[sm.Matrix, sm.Matrix]:
        """
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------------------------|
        |  J1*sp |  J1*sp |  J2*sp |  J2*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------------------------|

        explicit no acc
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------|
        |  J1*sp |  J1*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------|
        """
        if len(self.constraint_collection.equality_constraints) > 0:
            model = sm.Matrix.zeros(
                len(self.constraint_collection.equality_constraints),
                self.number_of_non_slack_columns,
            )
            J_eq = (
                sm.Matrix(self.equality_constraint_expressions()).jacobian(
                    variables=self.get_free_variable_symbols(Derivatives.position)
                )
                * self.config.mpc_dt
            )
            J_hstack = sm.hstack(
                [J_eq for _ in range(self.config.prediction_horizon - 2)]
            )
            # set jacobian entry to 0 if control horizon shorter than prediction horizon
            horizontal_offset = J_hstack.shape[1]
            model[:, horizontal_offset * 0 : horizontal_offset * 1] = J_hstack

            # slack variable for total error
            slack_model = sm.Matrix.diag(
                [
                    self.config.mpc_dt
                    for _ in self.constraint_collection.equality_constraints
                ]
            )
            return model, slack_model
        return sm.Matrix(), sm.Matrix()

    def construct_expression(
        self,
    ) -> Tuple[sm.Matrix, sm.Matrix]:
        max_derivative = Derivatives.velocity
        derivative_link_model = sm.Matrix()
        max_derivative = Derivatives.velocity
        derivative_link_model = self.derivative_link_model_no_acc(
            self.config.max_derivative
        )
        equality_constraint_model, equality_constraint_slack_model = (
            self.equality_constraint_model()
        )
        equality_constraint_model = (
            self._remove_columns_columns_where_variables_are_zero(
                equality_constraint_model, max_derivative
            )
        )
        vel_constr_model, vel_constr_slack_model = self.velocity_constraint_model()

        model_parts = []
        slack_model_parts = []
        if len(derivative_link_model) > 0:
            model_parts.append(derivative_link_model)
        if len(equality_constraint_model) > 0:
            model_parts.append(equality_constraint_model)
            slack_model_parts.append(equality_constraint_slack_model)
        if len(vel_constr_model) > 0:
            model_parts.append(vel_constr_model)
            slack_model_parts.append(vel_constr_slack_model)

        if len(model_parts) == 0:
            # if there are no eq constraints, make an empty matrix with the right columns to prevent stacking issues
            model = sm.Matrix(np.empty((0, self.number_of_non_slack_columns)))
        else:
            model = sm.vstack(model_parts)
        slack_model = sm.diag_stack(slack_model_parts)

        slack_model = sm.vstack(
            [
                sm.Matrix.zeros(derivative_link_model.shape[0], slack_model.shape[1]),
                slack_model,
            ]
        )
        # model = self._remove_columns_columns_where_variables_are_zero(model, max_derivative)
        return model, slack_model


@dataclass
class InequalityModel(ProblemDataPart):
    """
    order:
        derivative model (optional)
        neq integral constraints
        neq vel constraints
    """

    @property
    def number_of_free_variables(self):
        return len(self.degrees_of_freedom)

    @property
    def number_of_non_slack_columns(self) -> int:
        return (
            self.number_of_free_variables * (self.config.prediction_horizon - 2)
            + self.number_of_free_variables * self.config.prediction_horizon
        )

    @memoize
    def num_position_limits(self):
        return self.number_of_free_variables - self.num_of_continuous_joints()

    @memoize
    def num_of_continuous_joints(self):
        return len([v for v in self.degrees_of_freedom if not v.has_position_limits()])

    def inequality_constraint_expressions(self) -> List[sm.Matrix]:
        return _sorter(
            {
                c.name: c.expression
                for c in self.constraint_collection.inequality_constraints
            }
        )[0]

    def get_derivative_constraint_expressions(self, derivative: Derivatives):
        return _sorter(
            {
                c.name: c.expression
                for c in self.constraint_collection.derivative_inequality_constraints
                if c.derivative == derivative
            }
        )[0]

    def get_free_variable_symbols(self, order: Derivatives):
        return _sorter(
            {
                v.variables.position.name: v.variables.data[order]
                for v in self.degrees_of_freedom
            }
        )[0]

    def velocity_constraint_model(self) -> Tuple[sm.Matrix, sm.Matrix]:
        """
        model
        |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   | prediction horizon
        |v1 v2 v3|v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|j1 j2 j3| free variables
        |--------------------------------------------------------------------------------|
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |        |
        |--------------------------------------------------------------------------------|
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |        |  Jv*sp |        |        |  Ja*sp |        |        |  Jj*sp |        |
        |--------------------------------------------------------------------------------|

        slack model
        |   t1 |   t2 | prediction horizon
        |s1 s2 |s1 s2 | slack
        |------|------|
        |sp    |      | vel constr 1
        |   sp |      | vel constr 2
        |-------------|
        |      |sp    | vel constr 1
        |      |   sp | vel constr 2
        |-------------|
        """
        number_of_vel_rows = len(
            self.constraint_collection.velocity_inequality_constraints
        ) * (self.config.prediction_horizon - 2)
        if number_of_vel_rows > 0:
            expressions = sm.Matrix(
                self.get_derivative_constraint_expressions(Derivatives.velocity)
            )
            parts = []
            for derivative in Derivatives.range(
                Derivatives.position, self.config.max_derivative - 1
            ):
                if derivative == Derivatives.velocity:
                    continue
                J_vel = (
                    expressions.jacobian(
                        variables=self.get_free_variable_symbols(derivative),
                    )
                    * self.config.mpc_dt
                )
                missing_variables = self.config.max_derivative - derivative - 1
                eye = sm.Matrix.eye(self.config.prediction_horizon)[
                    :-2, : self.config.prediction_horizon - missing_variables
                ]
                J_vel_limit_block = eye.kron(J_vel)
                parts.append(J_vel_limit_block)

            # constraint slack
            model = sm.hstack(parts)
            num_slack_variables = sum(
                self.control_horizon
                for c in self.constraint_collection.velocity_inequality_constraints
            )
            slack_model = sm.Matrix.eye(num_slack_variables) * self.config.mpc_dt
            return model, slack_model
        return sm.Matrix(), sm.Matrix()

    def inequality_constraint_model(
        self, max_derivative: Derivatives
    ) -> Tuple[sm.Matrix, sm.Matrix]:
        """
        |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   |   t1   |   t2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|s1 s2 s3|s1 s2 s3| free variables / slack
        |-----------------------------------------------------------------------|
        |  J1*sp |  J1*sp |  J2*sp |  J2*sp |  J3*sp | J3*sp  | sp*ch  | sp*ch  |
        |-----------------------------------------------------------------------|
        """
        if len(self.constraint_collection.inequality_constraints) > 0:
            model = sm.Matrix.zeros(
                len(self.constraint_collection.inequality_constraints),
                self.number_of_non_slack_columns,
            )
            J_neq = (
                sm.Matrix(self.inequality_constraint_expressions()).jacobian(
                    variables=self.get_free_variable_symbols(Derivatives.position),
                )
                * self.config.mpc_dt
            )
            J_hstack = sm.hstack(
                [J_neq for _ in range(self.config.prediction_horizon - 2)]
            )
            # set jacobian entry to 0 if control horizon shorter than prediction horizon
            horizontal_offset = J_hstack.shape[1]
            model[:, horizontal_offset * 0 : horizontal_offset * 1] = J_hstack

            # slack variable for total error
            slack_model = sm.Matrix.diag(
                [
                    self.config.mpc_dt
                    for _ in self.constraint_collection.inequality_constraints
                ]
            )
            return model, slack_model
        if len(self.constraint_collection.inequality_constraints) > 0:
            model = sm.Matrix.zeros(
                len(self.constraint_collection.inequality_constraints),
                self.number_of_non_slack_columns,
            )
            for derivative in Derivatives.range(
                Derivatives.position, max_derivative - 1
            ):
                J_neq = (
                    sm.Matrix(self.inequality_constraint_expressions()).jacobian(
                        variables=self.get_free_variable_symbols(derivative),
                    )
                    * self.config.mpc_dt
                )
                J_hstack = sm.hstack(
                    [J_neq for _ in range(self.config.prediction_horizon - 2)]
                )
                horizontal_offset = J_hstack.shape[1]
                model[:, horizontal_offset * 0 : horizontal_offset * 1] = J_hstack

            # slack variable for total error
            slack_model = sm.Matrix.diag(
                [
                    self.config.mpc_dt
                    for _ in self.constraint_collection.inequality_constraints
                ]
            )
            return model, slack_model
        return sm.Matrix(), sm.Matrix()

    def implicit_pos_limits(self) -> Tuple[sm.Matrix, sm.Matrix]:
        """
        pk = pk-1 + vk*dt
            p0 = pc + v0*dt
            pk/dt - pk-1/dt = vk
        vk = vk-1 + ak*dt
        ak = (vk - vk-1)/dt
        jk = (vk - 2vk-1 + vk-2)/dt**2

        Layout for prediction horizon 4
        Slots are matrices of |controlled variables| x |controlled variables|
        |  vt0   |  vt1   |  vt2   |  vt3   |
        |-----------------------------------|
        |  1*dt  |        |        |        |       pt0 - ptc = vt0*dt
        |  1*dt  |  1*dt  |        |        |       pt1 - ptc = vt0*dt + vt1*dt
        |  1*dt  |  1*dt  |  1*dt  |        |       pt2 - ptc = vt0*dt + vt1*dt + vt2*dt
        |  1*dt  |  1*dt  |  1*dt  |  1*dt  |       pt3 - ptc = vt0*dt + vt1*dt + vt2*dt + vt3*dt
        |-----------------------------------|

        :return:
        """
        n_vel = self.number_of_free_variables * (self.config.prediction_horizon - 2)
        model = sm.Matrix.tri(n_vel) * self.config.mpc_dt
        slack_model = sm.Matrix.zeros(model.shape[0], self.number_ineq_slack_variables)
        return model, slack_model

    def implicit_model(
        self, max_derivative: Derivatives
    ) -> Tuple[sm.Matrix, sm.Matrix]:
        """
        ak = (vk - vk-1)/dt
        jk = (vk - 2vk-1 + vk-2)/dt**2

        vt0 = vtc + at0 * cdt
        vt1 = vt0 + at1 * mdt
        vt = vt-1 + at * mdt

        at = vt-1 + jt * mdt

        Layout for prediction horizon 6
        Slots are matrices of |controlled variables| x |controlled variables|
        |  vt0   |  vt1   |  vt2   |  vt3   |
        |-----------------------------------|
        |  1/dt  |        |        |        |               vtc/cdt + at0 = vt0/cdt                   vtc/dt + a_min <= vt0/dt <= vtc/dt + a_max
        | -1/dt  |  1/dt  |        |        |                        at1 = (vt1 - vt0)/dt
        |        | -1/dt  |  1/dt  |        |                        at2 = (vt2 - vt1)/mdt
        |        |        | -1/dt  | 1/dt   |                        at3 = (vt3 - vt2)/mdt
        |===================================|
        | 1/dt**2|        |        |        |   vtc/dt**2 + atc/dt + jt0 = vt0/dt**2                vtc/dt**2 + atc/dt + j_min <=    vt0/dt**2     <= vtc/dt**2 + atc/dt + j_max
        |-2/dt**2| 1/dt**2|        |        |          - vtc/dt**2 + jt1 = (vt1 - 2vt0)/dt**2           (- vtc)/dt**2 + j_min <= (vt1 - 2vt0)/dt**2 <= (- vtc)/dt**2 + j_max
        | 1/dt**2|-2/dt**2| 1/dt**2|        |                        jt2 = (vt2 - 2vt1 + vt0)/dt**2
        |        | 1/dt**2|-2/dt**2| 1/dt**2|                        jt3 = (vt3 - 2vt2 + vt1)/dt**2
        |        |        | 1/dt**2|-2/dt**2|                        jt4 = (- 2vt3 + vt2)/dt**2
        |        |        |        | 1/dt**2|                        jt5 = (vt3)/dt**2
        |-----------------------------------|

        :param max_derivative:
        :return:
        """
        n_vel = self.number_of_free_variables * (self.config.prediction_horizon - 2)
        n_jerk = self.number_of_free_variables * (self.config.prediction_horizon)
        if max_derivative >= Derivatives.acceleration:
            model = sm.Matrix.zeros(rows=n_jerk, columns=n_vel)
            pre_previous = sm.Matrix.eye(n_vel) / self.config.mpc_dt**2
            previous = -2 * sm.Matrix.eye(n_vel) / self.config.mpc_dt**2
            same = pre_previous
            model[: -self.number_of_free_variables * 2, :] += pre_previous
            model[
                self.number_of_free_variables : -self.number_of_free_variables, :
            ] += previous
            model[self.number_of_free_variables * 2 :, :] += same
        else:
            model = sm.Matrix()
        slack_model = sm.Matrix.zeros(model.shape[0], self.number_ineq_slack_variables)
        return model, slack_model

    def construct_expression(
        self,
    ) -> Tuple[sm.Matrix, sm.Matrix]:
        model_parts = []
        slack_model_parts = []

        max_derivative = self.config.max_derivative

        inequality_model, inequality_slack_model = self.inequality_constraint_model(
            max_derivative
        )
        vel_constr_model, vel_constr_slack_model = self.velocity_constraint_model()

        # neq integral constraints
        if len(vel_constr_model) > 0:
            model_parts.append(vel_constr_model)
            slack_model_parts.append(vel_constr_slack_model)
        # neq vel constraints
        if len(inequality_model) > 0:
            model_parts.append(inequality_model)
            slack_model_parts.append(inequality_slack_model)

        combined_model = sm.vstack(model_parts)
        combined_slack_model = sm.diag_stack(slack_model_parts)
        return combined_model, combined_slack_model


@dataclass
class QPDataSymbolic:
    """
    Takes free variables and constraints and converts them to a QP problem in the following format, depending on the
    class attributes:
    min_x 0.5 x^T H x + g^T x
    s.t.  lb <= x <= ub     (box constraints)
          Edof x <= bE_dof          (equality constraints)
          Eslack x <= bE_slack        (equality constraints)
          lbA <= Adof x <= ubA_dof  (lower/upper inequality constraints)
          lbA <= Aslack x <= ubA_slack  (lower/upper inequality constraints)
    """

    quadratic_weights: Vector
    linear_weights: Vector

    box_lower_constraints: Vector
    box_upper_constraints: Vector

    eq_matrix_dofs: Matrix
    eq_matrix_slack: Matrix
    eq_bounds: Vector

    neq_matrix_dofs: Matrix
    neq_matrix_slack: Matrix
    neq_lower_bounds: Vector
    neq_upper_bounds: Vector

    _weights: Weights
    _free_variable_bounds: FreeVariableBounds
    _equality_model: EqualityModel
    _equality_bounds: EqualityBounds
    _inequality_model: InequalityModel
    _inequality_bounds: InequalityBounds

    @classmethod
    def from_giskard(
        cls,
        degrees_of_freedom: List[DegreeOfFreedom],
        constraint_collection: ConstraintCollection,
        config: QPControllerConfig,
    ):
        kwargs = {
            "degrees_of_freedom": degrees_of_freedom,
            "constraint_collection": constraint_collection,
            "config": config,
        }
        weights = Weights(**kwargs)
        free_variable_bounds = FreeVariableBounds(**kwargs)
        equality_model = EqualityModel(**kwargs)
        equality_bounds = EqualityBounds(**kwargs)
        inequality_model = InequalityModel(**kwargs)
        inequality_bounds = InequalityBounds(**kwargs)

        quadratic_weights, linear_weights = weights.construct_expression()
        box_lower_constraints, box_upper_constraints = (
            free_variable_bounds.construct_expression()
        )
        eq_matrix_dofs, eq_matrix_slack = equality_model.construct_expression()
        eq_bounds = equality_bounds.construct_expression()
        neq_matrix_dofs, neq_matrix_slack = inequality_model.construct_expression()
        neq_lower_bounds, neq_upper_bounds = inequality_bounds.construct_expression()
        return cls(
            quadratic_weights=quadratic_weights,
            linear_weights=linear_weights,
            box_lower_constraints=box_lower_constraints,
            box_upper_constraints=box_upper_constraints,
            eq_matrix_dofs=eq_matrix_dofs,
            eq_matrix_slack=eq_matrix_slack,
            eq_bounds=eq_bounds,
            neq_matrix_dofs=neq_matrix_dofs,
            neq_matrix_slack=neq_matrix_slack,
            neq_lower_bounds=neq_lower_bounds,
            neq_upper_bounds=neq_upper_bounds,
            _weights=weights,
            _free_variable_bounds=free_variable_bounds,
            _equality_model=equality_model,
            _equality_bounds=equality_bounds,
            _inequality_model=inequality_model,
            _inequality_bounds=inequality_bounds,
        )

    def __hash__(self):
        return hash(id(self))

    @property
    def num_eq_constraints(self) -> int:
        return len(self.constraint_collection.equality_constraints)

    @property
    def num_neq_constraints(self) -> int:
        return len(self.constraint_collection.inequality_constraints)

    @property
    def num_free_variable_constraints(self) -> int:
        return len(self.degrees_of_freedom)

    @property
    def num_eq_slack_variables(self) -> int:
        return self.eq_matrix_slack.shape[1]

    @property
    def num_neq_slack_variables(self) -> int:
        return self.neq_matrix_slack.shape[1]

    @property
    def num_slack_variables(self) -> int:
        return self.num_eq_slack_variables + self.num_neq_slack_variables

    @property
    def num_non_slack_variables(self) -> int:
        return self.num_free_variable_constraints - self.num_slack_variables
