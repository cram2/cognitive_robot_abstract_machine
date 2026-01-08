from __future__ import annotations

from typing import Tuple, List, TYPE_CHECKING

import numpy as np
from line_profiler import profile

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.adapters.qp_adapter import GiskardToQPAdapter
from giskardpy.qp.qp_data import QPData
from krrood.symbolic_math.symbolic_math import VariableParameters

if TYPE_CHECKING:
    pass


# @dataclass
class GiskardToExplicitQPAdapter(GiskardToQPAdapter):
    """
    Takes free variables and constraints and converts them to a QP problem in the following format, depending on the
    class attributes:

    min_x 0.5 x^T H x + g^T x
    s.t.  lb <= x <= ub     (box constraints)
          Ex <= bE          (equality constraints)
          lbA <= Ax <= ubA  (lower/upper inequality constraints)
    """

    bE_filter: np.ndarray
    bA_filter: np.ndarray
    aux_symbols: List[sm.FloatVariable]

    def general_qp_to_specific_qp(
        self,
        quadratic_weights: sm.Vector,
        linear_weights: sm.Vector,
        box_lower_constraints: sm.Vector,
        box_upper_constraints: sm.Vector,
        eq_matrix_dofs: sm.Matrix,
        eq_matrix_slack: sm.Matrix,
        eq_bounds: sm.Vector,
        neq_matrix_dofs: sm.Matrix,
        neq_matrix_slack: sm.Matrix,
        neq_lower_bounds: sm.Vector,
        neq_upper_bounds: sm.Vector,
    ):
        eq_matrix = sm.hstack(
            [
                eq_matrix_dofs,
                eq_matrix_slack,
                sm.Matrix.zeros(eq_matrix_slack.shape[0], self.num_neq_slack_variables),
            ]
        )
        neq_matrix = sm.hstack(
            [
                neq_matrix_dofs,
                sm.Matrix.zeros(neq_matrix_slack.shape[0], self.num_eq_slack_variables),
                neq_matrix_slack,
            ]
        )

        self.free_symbols = [
            self.world_state_symbols,
            self.life_cycle_symbols,
            self.external_collision_symbols,
            self.self_collision_symbols,
            self.auxiliary_variables,
        ]

        self.eq_matrix_compiled = eq_matrix.compile(
            parameters=VariableParameters.from_lists(*self.free_symbols),
            sparse=self.sparse,
        )
        self.neq_matrix_compiled = neq_matrix.compile(
            parameters=VariableParameters.from_lists(*self.free_symbols),
            sparse=self.sparse,
        )

        self.combined_vector_f = sm.CompiledFunctionWithViews(
            expressions=[
                quadratic_weights,
                linear_weights,
                box_lower_constraints,
                box_upper_constraints,
                eq_bounds,
                neq_lower_bounds,
                neq_upper_bounds,
            ],
            parameters=VariableParameters.from_lists(*self.free_symbols),
        )

        self.bE_filter = np.ones(eq_matrix.shape[0], dtype=bool)
        self.bA_filter = np.ones(neq_matrix.shape[0], dtype=bool)

    @profile
    def create_filters(
        self,
        quadratic_weights_np_raw: np.ndarray,
        num_slack_variables: int,
        num_eq_slack_variables: int,
        num_neq_slack_variables: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        zero_quadratic_weight_filter: np.ndarray = quadratic_weights_np_raw != 0
        # don't filter dofs with 0 weight
        zero_quadratic_weight_filter[:-num_slack_variables] = True
        slack_part = zero_quadratic_weight_filter[
            -(num_eq_slack_variables + num_neq_slack_variables) :
        ]
        bE_part = slack_part[:num_eq_slack_variables]
        bA_part = slack_part[num_eq_slack_variables:]

        self.bE_filter.fill(True)
        if len(bE_part) > 0:
            self.bE_filter[-len(bE_part) :] = bE_part

        self.bA_filter.fill(True)
        if len(bA_part) > 0:
            self.bA_filter[-len(bA_part) :] = bA_part
        return zero_quadratic_weight_filter, self.bE_filter, self.bA_filter

    @profile
    def evaluate(
        self,
        world_state: np.ndarray,
        life_cycle_state: np.ndarray,
        external_collision_data: np.ndarray,
        self_collision_data: np.ndarray,
        auxiliary_variables: np.ndarray,
    ) -> QPData:
        args = [
            world_state,
            life_cycle_state,
            external_collision_data,
            self_collision_data,
            auxiliary_variables,
        ]
        eq_matrix_np_raw = self.eq_matrix_compiled(*args)
        neq_matrix_np_raw = self.neq_matrix_compiled(*args)
        (
            quadratic_weights_np_raw,
            linear_weights_np_raw,
            box_lower_constraints_np_raw,
            box_upper_constraints_np_raw,
            eq_bounds_np_raw,
            neq_lower_bounds_np_raw,
            neq_upper_bounds_np_raw,
        ) = self.combined_vector_f(*args)

        self.qp_data_raw = QPData(
            quadratic_weights=quadratic_weights_np_raw,
            linear_weights=linear_weights_np_raw,
            box_lower_constraints=box_lower_constraints_np_raw,
            box_upper_constraints=box_upper_constraints_np_raw,
            eq_matrix=eq_matrix_np_raw,
            eq_bounds=eq_bounds_np_raw,
            neq_matrix=neq_matrix_np_raw,
            neq_lower_bounds=neq_lower_bounds_np_raw,
            neq_upper_bounds=neq_upper_bounds_np_raw,
            num_eq_constraints=self.num_eq_constraints,
            num_neq_constraints=self.num_neq_constraints,
        )

        zero_quadratic_weight_filter, bE_filter, bA_filter = self.create_filters(
            quadratic_weights_np_raw=quadratic_weights_np_raw,
            num_slack_variables=self.num_slack_variables,
            num_eq_slack_variables=self.num_eq_slack_variables,
            num_neq_slack_variables=self.num_neq_slack_variables,
        )

        self.qp_data_raw.apply_filters(
            zero_quadratic_weight_filter, bE_filter, bA_filter
        )
        return self.qp_data_raw
