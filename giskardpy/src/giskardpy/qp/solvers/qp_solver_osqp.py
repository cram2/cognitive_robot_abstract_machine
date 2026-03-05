import numpy as np
import osqp
import scipy.sparse as sp

from giskardpy.qp.qp_data import QPDataExplicit, QPDataTwoSidedInequality
from giskardpy.qp.solvers.qp_solver import QPSolver


class QPSolverOSQP(QPSolver[QPDataTwoSidedInequality]):
    """
    min_x 0.5 x^T Q x + q^T x
    s.t.  lb <= Ax <= ub
    https://github.com/kul-optec/QPALM
    """

    def solver_call(self, qp_data: QPDataTwoSidedInequality) -> np.ndarray:
        prob = osqp.OSQP()
        prob.setup(
            P=sp.diags(qp_data.quadratic_weights, format="csc"),
            q=qp_data.linear_weights,
            A=qp_data.neq_matrix,
            l=qp_data.neq_lower_bounds,
            u=qp_data.neq_upper_bounds,
            verbose=False,
            polish=True,
        )
        solution = prob.solve()
        return solution.x

    def solver_call_explicit_interface(self, qp_data: QPDataExplicit) -> np.ndarray:
        return self.solver_call(qp_data.to_two_sided_inequality())
