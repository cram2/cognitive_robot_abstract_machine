"""
Tests for
:class:`giskardpy.qp.solvers.linear_program_solver_highs.LinearProgramSolverHighs`.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from giskardpy.qp.exceptions import (
    InfeasibleException,
    QuadraticObjectiveUnsupportedError,
)
from giskardpy.qp.qp_data import QPDataExplicit
from giskardpy.qp.solvers.linear_program_solver_highs import LinearProgramSolverHighs

VARIABLE_UPPER_BOUND = 0.3
VARIABLE_DIFFERENCE = 0.1


def _linear_program(
    quadratic_weights: np.ndarray,
    equality_bounds: np.ndarray,
    inequality_lower_bounds: np.ndarray,
    inequality_upper_bounds: np.ndarray,
) -> QPDataExplicit:
    """
    Maximizes the sum of two variables inside ``[0, VARIABLE_UPPER_BOUND]``, with one
    equality row on their difference and one two-sided inequality row on the first
    variable.
    """
    return QPDataExplicit(
        quadratic_weights=quadratic_weights,
        linear_weights=np.array([-1.0, -1.0]),
        box_lower_constraints=np.zeros(2),
        box_upper_constraints=np.full(2, VARIABLE_UPPER_BOUND),
        equality_matrix=sp.csc_matrix(np.array([[1.0, -1.0]])),
        equality_bounds=equality_bounds,
        inequality_matrix=sp.csc_matrix(np.array([[1.0, 0.0]])),
        inequality_lower_bounds=inequality_lower_bounds,
        inequality_upper_bounds=inequality_upper_bounds,
        num_equality_slack_variables=0,
        num_inequality_slack_variables=0,
    )


def test_solution_lies_on_the_binding_bounds():
    """
    The solution is the vertex itself, not a point within the solver's tolerance of it.
    """
    linear_program = _linear_program(
        quadratic_weights=np.zeros(2),
        equality_bounds=np.array([VARIABLE_DIFFERENCE]),
        inequality_lower_bounds=np.array([-np.inf]),
        inequality_upper_bounds=np.array([np.inf]),
    )

    solution = LinearProgramSolverHighs().solver_call(linear_program)

    assert solution[0] == VARIABLE_UPPER_BOUND
    assert solution[0] - solution[1] == VARIABLE_DIFFERENCE


def test_two_sided_inequality_rows_bound_the_solution():
    inequality_upper_bound = VARIABLE_UPPER_BOUND / 2
    linear_program = _linear_program(
        quadratic_weights=np.zeros(2),
        equality_bounds=np.array([0.0]),
        inequality_lower_bounds=np.array([-np.inf]),
        inequality_upper_bounds=np.array([inequality_upper_bound]),
    )

    solution = LinearProgramSolverHighs().solver_call(linear_program)

    assert solution[0] == inequality_upper_bound


def test_infeasible_program_raises():
    linear_program = _linear_program(
        quadratic_weights=np.zeros(2),
        equality_bounds=np.array([2 * VARIABLE_UPPER_BOUND]),
        inequality_lower_bounds=np.array([-np.inf]),
        inequality_upper_bounds=np.array([np.inf]),
    )

    with pytest.raises(InfeasibleException):
        LinearProgramSolverHighs().solver_call(linear_program)


def test_quadratic_objective_is_rejected():
    linear_program = _linear_program(
        quadratic_weights=np.ones(2),
        equality_bounds=np.array([VARIABLE_DIFFERENCE]),
        inequality_lower_bounds=np.array([-np.inf]),
        inequality_upper_bounds=np.array([np.inf]),
    )

    with pytest.raises(QuadraticObjectiveUnsupportedError):
        LinearProgramSolverHighs().solver_call(linear_program)
