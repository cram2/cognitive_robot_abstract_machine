from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import qpSWIFT_sparse_bindings as qpSWIFT

from giskardpy.qp.exceptions import QPSolverException
from giskardpy.qp.qp_data import QPDataExplicit
from giskardpy.qp.solvers.qp_solver import QPSolver


class QPSWIFTExitFlags(IntEnum):
    Optimal = 0  # Solution Found
    KKT_Failure = 1  # Failure in factorising KKT matrix
    MAX_ITER_REACHED = 2  # Maximum Number of Iterations Reached
    ERROR = 3  # Unknown Problem in Solver


@dataclass
class QPSolverQPSwift(QPSolver[QPDataExplicit]):

    big_ball_mode: bool = False
    """
    qpSWIFT does not have infeasible detection and cannot differentiate suboptimal from infeasible.
    If you know you QP is actually feasible, you can just ignore the failures and use the suboptimal solution.
    .. warning:: This might lead to instability if the qp was actually infeasible.
    """

    opts = {
        "OUTPUT": 1,  # 0 = sol; 1 = sol + basicInfo; 2 = sol + basicInfo + advInfo
        "MAXITER": 100,  # 0 < MAXITER < 200; default 100. maximum number of iterations needed
        # 'ABSTOL': 9e-4,  # 0 < ABSTOL < 1; default 1e-6. absolute tolerance
        "RELTOL": 3.5e-5,  # 0 < RELTOL < 1; default 1e-6. relative tolerance
        # 'SIGMA': 0.01,  # default 100. maximum centering allowed
        "VERBOSE": 0,  # 0 = no print; 1 = print
    }

    def solver_call_explicit_interface(self, qp_data: QPDataExplicit) -> np.ndarray:
        result = qpSWIFT.solve_sparse_H_diag(
            H=qp_data.quadratic_weights,
            g=qp_data.linear_weights,
            lb=qp_data.box_lower_constraints,
            ub=qp_data.box_upper_constraints,
            E=qp_data.equality_matrix,
            b=qp_data.equality_bounds,
            A=qp_data.inequality_matrix,
            lbA=qp_data.inequality_lower_bounds,
            ubA=qp_data.inequality_upper_bounds,
            options=self.opts,
        )
        exit_flag = result.exit_flag
        if not self.big_ball_mode:
            if exit_flag != QPSWIFTExitFlags.Optimal:
                error_code = QPSWIFTExitFlags(exit_flag)
                raise QPSolverException(f"Failed to solve qp: {str(error_code)}")
        return result.x

    solver_call = solver_call_explicit_interface
