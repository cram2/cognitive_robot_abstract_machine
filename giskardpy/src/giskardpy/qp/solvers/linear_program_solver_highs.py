from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np
import scipy.optimize
import scipy.sparse as sp

from giskardpy.qp.exceptions import (
    InfeasibleException,
    QuadraticObjectiveUnsupportedError,
    SolverReturnedFailureError,
)
from giskardpy.qp.qp_data import QPDataExplicit
from giskardpy.qp.solvers.qp_solver import QPSolver


class HighsMethod(StrEnum):
    """
    Algorithms of HiGHS selectable through :func:`scipy.optimize.linprog`.
    """

    DUAL_SIMPLEX = "highs-ds"
    """
    Dual simplex, which ends on a vertex of the feasible set.
    """


class LinearProgramStatus(IntEnum):
    """
    Exit statuses of :func:`scipy.optimize.linprog`.
    """

    OPTIMAL = 0
    """
    The optimization terminated successfully.
    """

    ITERATION_LIMIT = 1
    """
    The iteration limit was reached.
    """

    INFEASIBLE = 2
    """
    The problem has no feasible solution.
    """

    UNBOUNDED = 3
    """
    The objective is unbounded.
    """

    NUMERICAL_DIFFICULTIES = 4
    """
    The solver ran into numerical difficulties.
    """


@dataclass
class LinearProgramSolverHighs(QPSolver[QPDataExplicit]):
    """
    Solves problems without a quadratic objective exactly with the HiGHS simplex method.

    The solution is a vertex of the feasible set, so every binding bound holds up to
    floating point precision instead of up to a convergence tolerance.
    """

    def solver_call_explicit_interface(self, qp_data: QPDataExplicit) -> np.ndarray:
        if np.any(qp_data.quadratic_weights):
            raise QuadraticObjectiveUnsupportedError(solver_name=type(self).__name__)
        upper_rows, upper_bounds = self._one_sided_inequalities(qp_data)
        result = scipy.optimize.linprog(
            c=qp_data.linear_weights,
            A_ub=upper_rows,
            b_ub=upper_bounds,
            A_eq=qp_data.equality_matrix if qp_data.equality_bounds.size else None,
            b_eq=qp_data.equality_bounds if qp_data.equality_bounds.size else None,
            bounds=np.column_stack(
                (qp_data.box_lower_constraints, qp_data.box_upper_constraints)
            ),
            method=HighsMethod.DUAL_SIMPLEX,
        )
        status = LinearProgramStatus(result.status)
        if status == LinearProgramStatus.INFEASIBLE:
            raise InfeasibleException(solver_status=status.name)
        if status != LinearProgramStatus.OPTIMAL:
            raise SolverReturnedFailureError(solver_status=status.name)
        return result.x

    solver_call = solver_call_explicit_interface

    @staticmethod
    def _one_sided_inequalities(
        qp_data: QPDataExplicit,
    ) -> tuple[sp.csc_matrix | None, np.ndarray | None]:
        """
        Rewrites the two-sided inequality rows as rows bounded from above only, dropping
        every side that is unbounded.

        :param qp_data: Problem whose inequality rows are rewritten.
        :return: The rewritten rows and their upper bounds, or ``None`` for both when no
            bounded side remains.
        """
        lower_finite = np.isfinite(qp_data.inequality_lower_bounds)
        upper_finite = np.isfinite(qp_data.inequality_upper_bounds)
        if not (lower_finite.any() or upper_finite.any()):
            return None, None
        matrix = sp.csr_matrix(qp_data.inequality_matrix)
        rows = sp.vstack((matrix[upper_finite], -matrix[lower_finite]), format="csc")
        bounds = np.concatenate(
            (
                qp_data.inequality_upper_bounds[upper_finite],
                -qp_data.inequality_lower_bounds[lower_finite],
            )
        )
        return rows, bounds
