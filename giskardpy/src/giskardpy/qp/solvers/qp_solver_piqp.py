from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import piqp

from giskardpy.qp.exceptions import InfeasibleException
from giskardpy.qp.qp_data import QPDataExplicit
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.utils.math import fast_sparse_diagonal


@dataclass
class QPSolverPIQP(QPSolver[QPDataExplicit]):
    solver: piqp.SparseSolver = field(default_factory=piqp.SparseSolver)
    """
    The solver object of piqp.
    """

    fraction_to_boundary: float = 0.95
    """
    Fraction of the distance to the nearest constraint that piqp is allowed to step.

    Our problems regularly put a constraint right where the solution lies: a joint braking
    at its position limit, or a box collapsed to almost nothing. At piqp's default of 0.99
    the iterates then press against that constraint and stop improving, and the solve ends
    at the iteration limit with a dual residual it can no longer reduce. Keeping the steps
    further inside the feasible region leaves the iterates room to move along such a
    constraint instead. Every value from 0.8 to 0.98 solves all of the problems recorded
    from the test suites; 0.95 sits in the middle of that range and costs about a sixth
    more iterations than 0.99 on the problems that converge either way.
    """

    infeasibility_threshold: float = 100.0
    """
    How far piqp's multipliers may drift from their last accepted values, scaled by the
    regularization, before piqp declares the problem infeasible.

    Our problems cannot be infeasible, but tasks that cannot be reached push heavily
    weighted slack variables far from zero, and their multipliers legitimately grow to
    about 1e6. At piqp's default of 0.9 that drift already counts as infeasibility after
    a few iterations. 100 solves every problem recorded from the test suites at any
    fraction to boundary from 0.8 to 0.99; 10 is not enough.
    """

    big_ball_mode: bool = False
    """
    If the QP is known to be feasible, ignore non-SOLVED solver statuses and return the
    (possibly suboptimal) solution instead of raising.

    .. warning:: This is unsafe because it might lead to instability if the QP was actually infeasible. Only enable it
        when you are certain the problem is feasible.
    """

    def __post_init__(self):
        self.solver.settings.eps_abs = 1e-6
        self.solver.settings.eps_rel = 1e-7
        self.solver.settings.eps_duality_gap_abs = 1e-5
        self.solver.settings.eps_duality_gap_rel = 1e-5
        self.solver.settings.reg_lower_limit = 1e-11
        self.solver.settings.tau = self.fraction_to_boundary
        self.solver.settings.infeasibility_threshold = self.infeasibility_threshold

    def solver_call_explicit_interface(self, qp_data: QPDataExplicit) -> np.ndarray:
        weight_matrix = fast_sparse_diagonal(qp_data.quadratic_weights)
        if len(qp_data.inequality_upper_bounds) == 0:
            self.solver.setup(
                P=weight_matrix,
                c=qp_data.linear_weights,
                A=qp_data.equality_matrix,
                b=qp_data.equality_bounds,
                x_l=qp_data.box_lower_constraints,
                x_u=qp_data.box_upper_constraints,
            )
        else:
            self.solver.setup(
                P=weight_matrix,
                c=qp_data.linear_weights,
                A=qp_data.equality_matrix,
                b=qp_data.equality_bounds,
                G=qp_data.inequality_matrix,
                h_l=qp_data.inequality_lower_bounds,
                h_u=qp_data.inequality_upper_bounds,
                x_l=qp_data.box_lower_constraints,
                x_u=qp_data.box_upper_constraints,
            )

        status = self.solver.solve()
        if status.value != piqp.PIQP_SOLVED and not self.big_ball_mode:
            raise InfeasibleException(solver_status=str(status.value))
        return self.solver.result.x

    solver_call = solver_call_explicit_interface
