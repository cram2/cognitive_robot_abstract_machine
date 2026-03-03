from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import scipy.sparse as sp


@dataclass
class QPData:
    quadratic_weights: np.ndarray = field(default=None)
    linear_weights: np.ndarray = field(default=None)

    box_lower_constraints: np.ndarray = field(default=None)
    box_upper_constraints: np.ndarray = field(default=None)

    eq_matrix: Union[sp.csc_matrix, np.ndarray] = field(default=None)
    eq_bounds: np.ndarray = field(default=None)

    neq_matrix: Union[sp.csc_matrix, np.ndarray] = field(default=None)
    neq_lower_bounds: np.ndarray = field(default=None)
    neq_upper_bounds: np.ndarray = field(default=None)

    num_eq_constraints: int = field(default=None)
    num_neq_constraints: int = field(default=None)

    def explicit_data(self):
        return (
            self.quadratic_weights,
            self.linear_weights,
            self.box_lower_constraints,
            self.box_upper_constraints,
            self.eq_matrix,
            self.eq_bounds,
            self.neq_matrix,
            self.neq_lower_bounds,
            self.neq_upper_bounds,
        )

    @property
    def sparse_hessian(self) -> sp.csc_matrix:
        import scipy.sparse as sp

        return sp.diags(self.quadratic_weights)

    @property
    def dense_hessian(self) -> np.ndarray:
        return np.diag(self.quadratic_weights)

    @property
    def dense_eq_matrix(self) -> np.ndarray:
        try:
            return self.eq_matrix.toarray()
        except Exception:
            return self.eq_matrix

    @property
    def dense_neq_matrix(self) -> np.ndarray:
        try:
            return self.neq_matrix.toarray()
        except Exception:
            return self.neq_matrix

    def apply_filters(
        self,
        zero_quadratic_weight_filter: np.ndarray,
        bE_filter: np.ndarray,
        bA_filter: np.ndarray,
    ):
        self.zero_quadratic_weight_filter = zero_quadratic_weight_filter
        self.bE_filter = bE_filter
        self.bA_filter = bA_filter
        qp_data_filtered = QPData()
        qp_data_filtered.quadratic_weights = self.quadratic_weights[
            zero_quadratic_weight_filter
        ]
        qp_data_filtered.linear_weights = self.linear_weights[
            zero_quadratic_weight_filter
        ]
        qp_data_filtered.box_lower_constraints = self.box_lower_constraints[
            zero_quadratic_weight_filter
        ]
        qp_data_filtered.box_upper_constraints = self.box_upper_constraints[
            zero_quadratic_weight_filter
        ]
        if (
            len(self.eq_matrix.shape) > 1
            and self.eq_matrix.shape[0] * self.eq_matrix.shape[1] > 0
        ):
            qp_data_filtered.eq_matrix = self.eq_matrix[bE_filter, :][
                :, zero_quadratic_weight_filter
            ]
        else:
            qp_data_filtered.eq_matrix = self.eq_matrix
        # when no eq constraints were filtered, we can just cut off at the end, because that section is always all 0
        # qp_data_filtered.eq_matrix = self.eq_matrix_np_raw[:, :self.zero_quadratic_weight_filter.sum()]
        qp_data_filtered.eq_bounds = self.eq_bounds[bE_filter]
        if (
            len(self.neq_matrix.shape) > 1
            and self.neq_matrix.shape[0] * self.neq_matrix.shape[1] > 0
        ):
            qp_data_filtered.neq_matrix = self.neq_matrix[
                :, zero_quadratic_weight_filter
            ][bA_filter, :]
        else:
            qp_data_filtered.neq_matrix = self.neq_matrix
        qp_data_filtered.neq_lower_bounds = self.neq_lower_bounds[bA_filter]
        qp_data_filtered.neq_upper_bounds = self.neq_upper_bounds[bA_filter]
        self.filtered = qp_data_filtered

    def relaxed(self) -> QPData:
        relaxed_qp_data = QPData(
            quadratic_weights=self.filtered.quadratic_weights,
            linear_weights=self.filtered.linear_weights,
            box_lower_constraints=self.filtered.box_lower_constraints.copy(),
            box_upper_constraints=self.filtered.box_upper_constraints.copy(),
            eq_matrix=self.filtered.eq_matrix,
            eq_bounds=self.filtered.eq_bounds,
            neq_matrix=self.filtered.neq_matrix,
            neq_lower_bounds=self.filtered.neq_lower_bounds,
            neq_upper_bounds=self.filtered.neq_upper_bounds,
        )

        relaxed_qp_data.box_lower_constraints[self.num_non_constraints :] -= 100
        relaxed_qp_data.box_upper_constraints[self.num_non_constraints :] += 100

        return relaxed_qp_data

    @property
    def num_non_constraints(self) -> int:
        return (
            len(self.quadratic_weights)
            - self.num_eq_constraints
            - self.num_neq_constraints
        )

    def partially_relaxed(self, relaxed_solution: np.ndarray) -> QPData:
        relaxed_qp_data = QPData(
            quadratic_weights=self.filtered.quadratic_weights.copy(),
            linear_weights=self.filtered.linear_weights,
            box_lower_constraints=self.filtered.box_lower_constraints.copy(),
            box_upper_constraints=self.filtered.box_upper_constraints.copy(),
            eq_matrix=self.filtered.eq_matrix,
            eq_bounds=self.filtered.eq_bounds,
            neq_matrix=self.filtered.neq_matrix,
            neq_lower_bounds=self.filtered.neq_lower_bounds,
            neq_upper_bounds=self.filtered.neq_upper_bounds,
        )
        lower_box_filter = relaxed_solution < self.filtered.box_lower_constraints
        upper_box_filter = relaxed_solution > self.filtered.box_upper_constraints
        relaxed_qp_data.box_lower_constraints[lower_box_filter] -= 100
        relaxed_qp_data.box_upper_constraints[upper_box_filter] += 100
        relaxed_qp_data.quadratic_weights[lower_box_filter | upper_box_filter] *= 1000

        return relaxed_qp_data

    def pretty_print_problem(self):
        print("QP data")
        large = int(1e10)
        if self.quadratic_weights is not None:
            print(
                f"H (quadratic_weights): \n{np.array2string(self.quadratic_weights, max_line_width=large)}"
            )
        if self.linear_weights is not None:
            print(
                f"g (linear_weights): \n{np.array2string(self.linear_weights, max_line_width=large)})"
            )

        if self.box_lower_constraints is not None:
            print(
                f"lb (box_lower_constraints): \n{np.array2string(self.box_lower_constraints, max_line_width=large)}"
            )
        if self.box_upper_constraints is not None:
            print(
                f"ub (box_upper_constraints): \n{np.array2string(self.box_upper_constraints, max_line_width=large)}"
            )

        if self.eq_matrix is not None:
            try:
                print(
                    f"E (eq_matrix): \n{np.array2string(self.eq_matrix.toarray(), max_line_width=large)}"
                )
            except:
                print(
                    f"E (eq_matrix): \n{np.array2string(self.eq_matrix, max_line_width=large)}"
                )
        if self.eq_bounds is not None:
            print(
                f"bE (eq_bounds): \n{np.array2string(self.eq_bounds, max_line_width=large)}"
            )

        if self.neq_matrix is not None:
            try:
                print(
                    f"A (neq_matrix): \n{np.array2string(self.neq_matrix.toarray(), max_line_width=large)}"
                )
            except:
                print(
                    f"A (neq_matrix): \n{np.array2string(self.neq_matrix, max_line_width=large)}"
                )
        if self.neq_lower_bounds is not None:
            print(
                f"lbA (neq_lower_bounds): \n{np.array2string(self.neq_lower_bounds, max_line_width=large)}"
            )
        if self.neq_upper_bounds is not None:
            print(
                f"ubA (neq_upper_bounds): \n{np.array2string(self.neq_upper_bounds, max_line_width=large)}"
            )
