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

    def __post_init__(self):
        self.solver.settings.eps_abs = 1e-6
        self.solver.settings.eps_rel = 1e-7
        self.solver.settings.eps_duality_gap_abs = 1e-5
        self.solver.settings.eps_duality_gap_rel = 1e-5
        self.solver.settings.reg_lower_limit = 1e-11
        # self.solver.settings.kkt_solver = piqp.KKTSolver.sparse_multistage

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
        if status.value != piqp.PIQP_SOLVED:
            raise InfeasibleException(f"Solver status: {status.value}")
        return self.solver.result.x

    solver_call = solver_call_explicit_interface
