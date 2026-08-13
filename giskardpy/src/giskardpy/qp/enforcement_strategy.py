"""
Enforcement strategies that turn high-level constraints into QP matrix rows, slack columns,
and bounds (:class:`EnforcementStrategy` and subclasses).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Callable

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint import (
    GiskardConstraint,
    GiskardEqualityConstraint,
    GiskardInequalityConstraint,
    LargeNumber,
)
from giskardpy.qp.dof_limits import DirectLimits
from giskardpy.qp.exceptions import ConstraintTypeMismatchError
from krrood.symbolic_math.symbolic_math import Vector, Matrix, Scalar
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller_config import QPControllerConfig


def normalize_slack_weight(
    weight: Scalar,
    normalization_factor: float,
    control_horizon: int = 1,
) -> Scalar:
    """
    Scales a slack weight so constraints with different units become comparable.
    The control horizon spreads the weight over the time steps a constraint is active.
    """
    return weight * (1 / (sm.Scalar(normalization_factor) ** 2 * control_horizon))


@dataclass
class EnforcementStrategy(ABC):
    """
    Turns a block of constraints into the QP building blocks that enforce them: the constraint
    matrix over the free variables, the slack columns and their limits, the bounds, and the row
    names. Subclasses define how a constraint is mapped onto the prediction horizon.
    """

    degrees_of_freedom: list[DegreeOfFreedom]
    """
    Free variables of the robot the constraints act on.
    """

    constraints: list[GiskardConstraint]
    """
    Constraints enforced by this strategy.
    """

    qp_controller_config: QPControllerConfig
    """
    Controller configuration providing horizon length and time step.
    """

    @abstractmethod
    def create_matrix(self) -> Matrix:
        """
        Builds the constraint matrix mapping the free variables onto the constraint rows.
        """

    @abstractmethod
    def create_slack_matrix(self) -> Matrix:
        """
        Builds the matrix coupling the slack variables to the constraint rows.
        """

    @abstractmethod
    def create_names(self) -> list[str]:
        """
        Creates a debug name for every constraint row.
        """

    @abstractmethod
    def create_slack_variables(self) -> DirectLimits:
        """
        Creates the limits and weights of the slack variables introduced by this strategy.
        """

    @property
    def number_of_free_variables(self) -> int:
        """
        Number of degrees of freedom controlled by this strategy.
        """
        return len(self.degrees_of_freedom)

    @property
    def number_of_velocity_columns(self) -> int:
        """
        Number of velocity decision variable columns across the horizon.
        """
        return self.number_of_free_variables * (
            self.qp_controller_config.prediction_horizon - 2
        )

    @property
    def number_of_jerk_columns(self) -> int:
        """
        Number of jerk decision variable columns across the horizon.
        """
        return (
            self.number_of_free_variables * self.qp_controller_config.prediction_horizon
        )

    @property
    def position_variables(self) -> Vector:
        """
        Symbolic position variables of the degrees of freedom.
        """
        return Vector([dof.variables.position for dof in self.degrees_of_freedom])

    @property
    def velocity_variables(self) -> Vector:
        """
        Symbolic velocity variables of the degrees of freedom.
        """
        return Vector([dof.variables.velocity for dof in self.degrees_of_freedom])

    @property
    def acceleration_variables(self) -> Vector:
        """
        Symbolic acceleration variables of the degrees of freedom.
        """
        return Vector([dof.variables.acceleration for dof in self.degrees_of_freedom])


@dataclass
class ExpressionEnforcementStrategy(EnforcementStrategy, ABC):
    """
    Base for strategies that map per-constraint expressions onto QP rows and read their bounds
    through a getter. Provides the inequality and equality bound builders shared by the
    expression-based strategies.
    """

    @abstractmethod
    def create_bounds(
        self, bounds_getter: Callable[[GiskardConstraint], Scalar]
    ) -> Vector:
        """
        Builds the constraint bounds, reading the relevant bound of each constraint via the getter.
        """

    def create_lower_bounds(self) -> Vector:
        """
        Builds the lower bounds of the inequality constraints.
        """
        self._require_constraint_type(GiskardInequalityConstraint)
        return self.create_bounds(lambda c: c.lower_bound)

    def create_upper_bounds(self) -> Vector:
        """
        Builds the upper bounds of the inequality constraints.
        """
        self._require_constraint_type(GiskardInequalityConstraint)
        return self.create_bounds(lambda c: c.upper_bound)

    def create_equality_bounds(self) -> Vector:
        """
        Builds the bounds of the equality constraints.
        """
        self._require_constraint_type(GiskardEqualityConstraint)
        return self.create_bounds(lambda c: c.bound)

    def _require_constraint_type(self, expected: type[GiskardConstraint]) -> None:
        """
        Ensures every constraint handled by this strategy is of the expected type.
        """
        for constraint in self.constraints:
            if not isinstance(constraint, expected):
                raise ConstraintTypeMismatchError(
                    strategy_name=type(self).__name__,
                    expected_type=expected,
                    actual_type=type(constraint),
                    constraint_name=constraint.name,
                )


@dataclass
class IntegralStrategy(ExpressionEnforcementStrategy):
    """
    Enforces a constraint on the integral of an expression's derivative over the prediction
    horizon, so a single row covers the whole horizon.

    Equality constraints have the form:
    .. math::
        f(q) = b

    where

    .. math::

        target - f = \\Delta t \\  \\sum_{k=0}^{N-3} J_{f} + \\Delta t \\epsilon

    ::

        |   k1   |   k2   |   k1   |   k2   |   k1   |   k2   | prediction horizon
        |v1 v2 v3|v1 v2 v3|j1 j2 j3|j1 j2 j3|   eps  |   eps  | free variables / slack
        |--------+--------+--------+--------+--------+--------|
        |  J1*dt |  J1*dt |  J3*dt | J3*dt  |   dt   |   dt   |
        |-----------------------------------------------------|
    """

    def create_matrix(self) -> Matrix:
        """
        Builds the constraint matrix by repeating the expression jacobian across the velocity
        horizon and padding the jerk columns with zeros.
        """
        if len(self.constraints) == 0:
            return sm.Matrix()
        jacobian = (
            sm.Vector([c.expression for c in self.constraints]).jacobian(
                variables=self.position_variables
            )
            * self.qp_controller_config.model_predictive_control_time_step
        )
        return sm.hstack(
            [jacobian for _ in range(self.qp_controller_config.control_horizon)]
            + [sm.Matrix.zeros(jacobian.shape[0], self.number_of_jerk_columns)]
        )

    def create_slack_matrix(self) -> Matrix:
        """
        Builds the diagonal slack matrix with one slack variable per constraint.
        """
        if len(self.constraints) == 0:
            return sm.Matrix()
        return sm.Matrix.diag(
            [
                self.qp_controller_config.model_predictive_control_time_step
                for _ in self.constraints
            ]
        )

    def create_slack_variables(self) -> DirectLimits:
        """
        Creates one slack variable per constraint with normalized weights.
        """
        number_of_slack_variables = len(self.constraints)
        return DirectLimits(
            lower_bounds=Vector([-LargeNumber] * number_of_slack_variables),
            upper_bounds=Vector([LargeNumber] * number_of_slack_variables),
            quadratic_weights=Vector(
                [
                    normalize_slack_weight(
                        c.quadratic_weight,
                        c.normalization_factor,
                        self.qp_controller_config.control_horizon,
                    )
                    for c in self.constraints
                ]
            ),
            linear_weights=Vector(
                [
                    normalize_slack_weight(
                        c.linear_weight,
                        c.normalization_factor,
                        self.qp_controller_config.control_horizon,
                    )
                    for c in self.constraints
                ]
            ),
            names=[c.name for c in self.constraints],
        )

    def _apply_cap(
        self,
        value: Scalar,
        dt: float,
        normalization_number: float,
        control_horizon: int,
    ) -> Scalar:
        """
        Clamps a bound to the largest change reachable within the control horizon.
        """
        return sm.limit(
            value,
            -normalization_number * dt * control_horizon,
            normalization_number * dt * control_horizon,
        )

    def capped_bound(
        self,
        equality_bound: Scalar,
        dt: float,
        normalization_number: float,
        control_horizon: int,
    ) -> Scalar:
        """
        Returns the bound capped to what is reachable within the control horizon.
        """
        return self._apply_cap(
            equality_bound, dt, normalization_number, control_horizon
        )

    def create_bounds(
        self, bounds_getter: Callable[[GiskardConstraint], Scalar]
    ) -> Vector:
        """
        Builds the capped bounds, one per constraint.
        """
        return Vector(
            [
                self.capped_bound(
                    bounds_getter(c),
                    self.qp_controller_config.model_predictive_control_time_step,
                    c.normalization_factor,
                    self.qp_controller_config.control_horizon,
                )
                for c in self.constraints
            ]
        )

    def create_names(self) -> list[str]:
        """
        Returns the constraint names, one row per constraint.
        """
        return [c.name for c in self.constraints]


@dataclass
class VelocityStrategy(ExpressionEnforcementStrategy):
    """
    The constraint will be applied to the derivative of the expression.
    Position constraints are implemented by constraining the integral of the expressions' derivative over a prediction horizon.
    All other constraints are applied directly to that derivative of the expression.
    As a result, position constraints are cheaper, as they only require a single constraint.

    Equality constraints have the form:
    .. math::
        f(q) = b

    where

    .. math::

        target - f = \\Delta t \\ \\sum_{k=0}^{N-1} J_{f}

    ::

        |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   |   t1   |   t2   |   t3   | prediction horizon
        |v1 v2 v3|v1 v2 v3|v1 v2 v3|a1 a2 a3|a1 a2 a3|a1 a2 a3|j1 j2 j3|j1 j2 j3|j1 j2 j3| free variables
        |--------------------------------------------------------------------------------|
        |  Jv*dt |        |        |  Ja*dt |        |        |  Jj*dt |        |        |
        |  Jv*dt |        |        |  Ja*dt |        |        |  Jj*dt |        |        |
        |--------------------------------------------------------------------------------|
        |        |  Jv*dt |        |        |  Ja*dt |        |        |  Jj*dt |        |
        |        |  Jv*dt |        |        |  Ja*dt |        |        |  Jj*dt |        |
        |--------------------------------------------------------------------------------|
    """

    def create_matrix(self) -> Matrix:
        """
        Builds the constraint matrix applying the expression jacobian at each velocity step of the
        horizon, padding the jerk columns with zeros.
        """
        number_of_vel_rows = len(self.constraints) * (
            self.qp_controller_config.prediction_horizon - 2
        )
        if number_of_vel_rows == 0:
            return sm.Matrix()
        jacobian = (
            sm.Vector([c.expression for c in self.constraints]).jacobian(
                variables=self.position_variables
            )
            * self.qp_controller_config.model_predictive_control_time_step
        )
        missing_variables = self.qp_controller_config.max_derivative - 1
        eye = sm.Matrix.eye(self.qp_controller_config.prediction_horizon)[
            :-2, : self.qp_controller_config.prediction_horizon - missing_variables
        ]
        J_vel_limit_block = eye.kron(jacobian)

        return sm.hstack(
            [
                J_vel_limit_block,
                sm.Matrix.zeros(
                    J_vel_limit_block.shape[0], self.number_of_jerk_columns
                ),
            ]
        )

    def create_slack_matrix(self) -> Matrix:
        """
        Builds the diagonal slack matrix with one slack variable per constraint and velocity step.
        """
        if len(self.constraints) == 0:
            return sm.Matrix()
        num_slack_variables = sum(
            self.qp_controller_config.prediction_horizon - 2 for c in self.constraints
        )
        return (
            sm.Matrix.eye(num_slack_variables)
            * self.qp_controller_config.model_predictive_control_time_step
        )

    def create_slack_variables(self) -> DirectLimits:
        """
        Creates one slack variable per constraint and velocity step with normalized weights.
        """
        lower_slack = []
        upper_slack = []
        quadratic_weights = []
        linear_weights = []
        names = []
        for t in range(self.qp_controller_config.control_horizon):
            for c in self.constraints:
                lower_slack.append(c.lower_slack_limit)
                upper_slack.append(c.upper_slack_limit)
                quadratic_weights.append(
                    normalize_slack_weight(c.quadratic_weight, c.normalization_factor)
                )
                linear_weights.append(c.linear_weight)
                names.append(f"t{t:03}/{c.name}")
        return DirectLimits(
            lower_bounds=sm.Vector(lower_slack),
            upper_bounds=sm.Vector(upper_slack),
            quadratic_weights=sm.Vector(quadratic_weights),
            linear_weights=sm.Vector(linear_weights),
            names=names,
        )

    def create_bounds(
        self, bounds_getter: Callable[[GiskardConstraint], Scalar]
    ) -> Vector:
        """
        Builds the bounds, one per constraint and control step.
        """
        bounds = []
        for _ in range(self.qp_controller_config.control_horizon):
            for c in self.constraints:
                bounds.append(
                    bounds_getter(c)
                    * self.qp_controller_config.model_predictive_control_time_step
                )
        return Vector(bounds)

    def create_names(self) -> list[str]:
        """
        Creates a name per constraint and control step, prefixed with the time step.
        """
        names = []
        for t in range(self.qp_controller_config.control_horizon):
            for c in self.constraints:
                names.append(f"t{t:03}/{c.name}")
        return names


@dataclass
class SystemDynamicsStrategy(EnforcementStrategy):
    """
    The constraints produced by this class describe the discrete-time relationships between variables
    in the prediction horizon :math:`N` using a semi-implicit euler integration method:

    .. math::

        v_k = v_{k-1} + a_{k} \\, \\Delta t

        a_k = a_{k-1} + j_{k} \\, \\Delta t

    Where v, a and j are velocity, acceleration and jerk, respectively, and k is the time step.
    Acceleration variables are removed using substitution.
    The first two rows links the MPC to the current state:

    .. math::

        -v_{c} - a_{c} \\, \\Delta t = -v_0 + j_0 \\, \\Delta t^2

        v_{c} = - v_1 + 2 v_0 + j_1 \\, \\Delta t^2

    Rows from 2 until k-2 have this form:

    .. math::

        0 = - v_k + 2 v_{k-1} - v_{k-2} + j_k \\, \\Delta t^2

    The final two rows have this form:

    .. math::

        0 = 2 v_{k-1} - v_{k-2} + j_k \\, \\Delta t^2

        0 = - v_{k-2} + j_k \\, \\Delta t^2

    For a prediction horizon of 5 with 1 degree of freedom, the matrix looks like this:

    ::

        |  equality_bounds |   |           equality constraint matrix          |   |    v_0    |
        |------------------|   |-----------------------------------------------|   |    v_1    |
        | - v_c - a_c * dt |   | -1  |     |     |  1  |     |     |     |     |   |    v_2    |
        |       v_c        |   |  2  | -1  |     |     |  1  |     |     |     |   | j_0*dt**2 |
        |        0         | = | -1  |  2  | -1  |     |     |  1  |     |     | @ | j_1*dt**2 |
        |        0         |   |     | -1  |  2  |     |     |     |  1  |     |   | j_2*dt**2 |
        |        0         |   |     |     | -1  |     |     |     |     |  1  |   | j_3*dt**2 |
        |------------------|   |-----------------------------------------------|   | j_4*dt**2 |

    This means that the QP does not optimize jerk, but jerk :math:`*\\Delta t^2`, this improves the conditioning of the
    constraint matrix, which helps some solvers.
    """

    def create_matrix(self) -> Matrix:
        """
        Builds the equality matrix encoding the velocity, acceleration, and jerk integration over
        the horizon.
        """
        matrix = np.zeros(
            (
                self.number_of_jerk_columns,
                self.number_of_velocity_columns + self.number_of_jerk_columns,
            )
        )
        for horizon_index in range(self.qp_controller_config.prediction_horizon):
            row_start = horizon_index * self.number_of_free_variables
            row_end = (horizon_index + 1) * self.number_of_free_variables

            # velocity at k
            if horizon_index < self.qp_controller_config.prediction_horizon - 2:
                col_start = horizon_index * self.number_of_free_variables
                col_end = (horizon_index + 1) * self.number_of_free_variables
                matrix[row_start:row_end, col_start:col_end] -= np.eye(
                    self.number_of_free_variables
                )

            # velocity at k-1
            if 0 < horizon_index < self.qp_controller_config.prediction_horizon - 1:
                col_start = (horizon_index - 1) * self.number_of_free_variables
                col_end = horizon_index * self.number_of_free_variables
                matrix[row_start:row_end, col_start:col_end] += 2 * np.eye(
                    self.number_of_free_variables
                )

            # velocity at k-2
            if horizon_index > 1:
                col_start = (horizon_index - 2) * self.number_of_free_variables
                col_end = (horizon_index - 1) * self.number_of_free_variables
                matrix[row_start:row_end, col_start:col_end] -= np.eye(
                    self.number_of_free_variables
                )

            # jerk at k
            col_start = self.number_of_velocity_columns + row_start
            col_end = self.number_of_velocity_columns + row_end
            matrix[row_start:row_end, col_start:col_end] += np.eye(
                self.number_of_free_variables
            )

        return sm.Matrix(matrix)

    def create_slack_matrix(self) -> Matrix:
        """
        Returns an empty slack matrix, as the system dynamics need no slack variables.
        """
        return sm.Matrix.zeros(self.number_of_jerk_columns, 0)

    def create_slack_variables(self) -> DirectLimits:
        """
        Returns empty limits, as the system dynamics introduce no slack variables.
        """
        return DirectLimits.empty()

    def create_names(self) -> list[str]:
        """
        Creates a name for every velocity/jerk integration row.
        """
        names = []
        for k in range(self.qp_controller_config.prediction_horizon):
            for dof in self.degrees_of_freedom:
                names.append(f"{dof.name} k_{k} vel/jerk link")
        return names

    def create_equality_bounds(self) -> Vector:
        """
        Builds the equality bounds linking the first horizon steps to the current state.
        """
        res = sm.Vector.zeros(self.number_of_jerk_columns)
        res[: self.number_of_free_variables] = (
            -self.velocity_variables
            - self.acceleration_variables
            * self.qp_controller_config.model_predictive_control_time_step
        )
        res[self.number_of_free_variables : self.number_of_free_variables * 2] = (
            self.velocity_variables
        )
        return res
