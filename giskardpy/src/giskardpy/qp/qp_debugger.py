from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from giskardpy.qp.qp_data_symbolic import QPDataSymbolic

logger = logging.getLogger(__name__)


class DebuggerColumn(StrEnum):
    """Column labels for the debugger's pandas frames."""

    LOWER_BOUNDS = "lower bounds"
    """Lower bounds of the decision variables."""

    UPPER_BOUNDS = "upper bounds"
    """Upper bounds of the decision variables."""

    SOLUTION = "solution"
    """The solution expanded to the full decision-variable layout."""

    QUADRATIC_WEIGHT = "quadratic weight"
    """Quadratic weight of each decision variable."""

    LINEAR_WEIGHT = "linear weight"
    """Linear weight of each decision variable."""

    CONSTRAINT_VALUE_WITHOUT_SLACK = "constraint value no slack"
    """Result of the constraint matrix times the decision variables, excluding slack."""

    SLACK = "slack"
    """How much a constraint is violated."""

    BOUNDS = "bounds"
    """Equality constraint bounds."""

    INEQUALITY_LOWER_BOUNDS = "lower_bounds"
    """Lower bounds of the inequality constraints."""

    INEQUALITY_UPPER_BOUNDS = "upper_bounds"
    """Upper bounds of the inequality constraints."""


@dataclass
class QuadraticProgramDebugger:
    """
    This class is designed to help you debug Giskard's quadratic programs (QP) by using names of constraints and
    degrees of freedom to create panda arrays with names rows and columns.
    """

    qp_data_symbolic: QPDataSymbolic
    """
    The symbolic casadi expressions for computing the QP components.
    """
    current_solution: np.ndarray | None = field(default=None)
    """
    The raw solver solution of the QP, None if there is none.
    """
    padded_solution: np.ndarray = field(init=False)
    """
    The solution expanded to the full decision-variable layout, with nan in filtered-out slots.
    """
    direct_limits: pd.DataFrame = field(init=False)
    """
    This panda array gives you insights into the decision variables of the QP.
    It contains columns for direct upper and lower bounds, last solution and weights.
    """
    equality_constraints: pd.DataFrame = field(init=False)
    """
    This panda array gives insight in the equality constraints.
    It contains columns for the equality bounds, the result of the equality matrix * decision variables (without slack),
    and the slack, which is essentially how much the constraints are violated. 
    """
    equality_matrix: pd.DataFrame = field(init=False)
    """
    Panda array representing the equality constraint matrix.
    """
    inequality_constraints: pd.DataFrame = field(init=False)
    """
    Panda array giving insights into the inequality constraints.
    It contains columns for the inequality bounds, the result of the inequality matrix * decision variables (without slack),
    and the slack, which is essentially how much the constraints are violated. 
    """
    inequality_matrix: pd.DataFrame = field(init=False)
    """
    Panda array representing the inequality constraint matrix.
    """

    def __post_init__(self):
        self.update(self.current_solution)

    def update(self, current_solution: np.ndarray | None) -> None:
        """
        Updates the debugger with a new solution.
        """
        self.current_solution = current_solution
        padded_solution = (
            np.ones(self.qp_data_symbolic.box_lower_constraints.shape[0]) * np.nan
        )
        if self.current_solution is not None:
            padded_solution[self.quadratic_weight_filter] = self.current_solution

        self.padded_solution = padded_solution
        self.create_direct_limits()
        self.create_equality_constraints()
        self.create_inequality_constraints()

    @property
    def quadratic_weight_filter(self) -> np.ndarray:
        """
        Returns a filter for the quadratic weights.
        """
        quadratic_weight_filter = np.ones(
            self.qp_data_symbolic.quadratic_weights.shape[0]
        )
        quadratic_weight_filter[self.qp_data_symbolic.number_non_slack_variables :] = (
            self.qp_data_symbolic.quadratic_weights.evaluate()[
                self.qp_data_symbolic.number_non_slack_variables :
            ]
            != 0
        )
        return quadratic_weight_filter.astype(bool)

    def create_direct_limits(self) -> None:
        """
        Creates a panda array for decision variable insights.
        """
        self.direct_limits = pd.DataFrame(
            {
                DebuggerColumn.LOWER_BOUNDS: self.qp_data_symbolic.box_lower_constraints.evaluate(),
                DebuggerColumn.SOLUTION: self.padded_solution,
                DebuggerColumn.UPPER_BOUNDS: self.qp_data_symbolic.box_upper_constraints.evaluate(),
                DebuggerColumn.QUADRATIC_WEIGHT: self.qp_data_symbolic.quadratic_weights.evaluate(),
                DebuggerColumn.LINEAR_WEIGHT: self.qp_data_symbolic.linear_weights.evaluate(),
            },
            self.free_variable_names,
            dtype=float,
        )

    def create_equality_constraints(self) -> None:
        """
        Creates panda arrays for equality constraint insights.
        """
        eq_matrix_dofs_np = (
            self.qp_data_symbolic.equality_matrix_degrees_of_freedom.evaluate()
        )
        constraint_value_without_slack = (
            eq_matrix_dofs_np @ self.padded_solution[: eq_matrix_dofs_np.shape[1]]
        )
        bounds = self.qp_data_symbolic.equality_bounds.evaluate()
        self.equality_constraints = pd.DataFrame(
            {
                DebuggerColumn.CONSTRAINT_VALUE_WITHOUT_SLACK: constraint_value_without_slack,
                DebuggerColumn.SLACK: bounds - constraint_value_without_slack,
                DebuggerColumn.BOUNDS: bounds,
            },
            self.equality_constraint_names,
            dtype=float,
        )
        self.equality_matrix = pd.DataFrame(
            eq_matrix_dofs_np,
            self.equality_constraint_names,
            self.degree_of_freedom_names,
            dtype=float,
        )

    def create_inequality_constraints(self) -> None:
        """
        Creates panda arrays for inequality constraint insights.
        """
        neq_matrix_dofs_np = (
            self.qp_data_symbolic.inequality_matrix_degrees_of_freedom.evaluate()
        )
        constraint_value_without_slack = (
            neq_matrix_dofs_np @ self.padded_solution[: neq_matrix_dofs_np.shape[1]]
        )
        lower_bounds = self.qp_data_symbolic.inequality_lower_bounds.evaluate()
        upper_bounds = self.qp_data_symbolic.inequality_upper_bounds.evaluate()
        if len(self.inequality_constraint_names) > 0:
            self.inequality_constraints = pd.DataFrame(
                {
                    DebuggerColumn.INEQUALITY_LOWER_BOUNDS: lower_bounds,
                    DebuggerColumn.CONSTRAINT_VALUE_WITHOUT_SLACK: constraint_value_without_slack,
                    DebuggerColumn.INEQUALITY_UPPER_BOUNDS: upper_bounds,
                },
                self.inequality_constraint_names,
                dtype=float,
            )
            self.inequality_matrix = pd.DataFrame(
                neq_matrix_dofs_np,
                self.inequality_constraint_names,
                self.degree_of_freedom_names,
                dtype=float,
            )
        else:
            self.inequality_constraints = pd.DataFrame()
            self.inequality_matrix = pd.DataFrame()

    @property
    def free_variable_names(self) -> list[str]:
        """
        Returns the names of all free variables.
        """
        return self.qp_data_symbolic.free_variable_names

    @property
    def degree_of_freedom_names(self) -> list[str]:
        """
        Returns the names of the degree-of-freedom decision variables, excluding slack variables, in
        the same column order that :class:`DofLimits` uses to build the QP matrices.
        """
        return self.free_variable_names[
            : self.qp_data_symbolic.number_non_slack_variables
        ]

    @property
    def equality_constraint_names(self) -> list[str]:
        """
        Returns the names of all equality constraints.
        """
        return self.qp_data_symbolic.equality_constraint_names

    @property
    def inequality_constraint_names(self) -> list[str]:
        """
        Returns the names of all inequality constraints.
        """
        return self.qp_data_symbolic.inequality_constraint_names
