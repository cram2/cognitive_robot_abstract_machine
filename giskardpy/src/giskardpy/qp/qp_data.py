from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
from typing_extensions import Self


@dataclass
class QPData(ABC):
    """
    Parent class for a container of input for a QP solver.
    Subclasses implement specific formats for the QP problem.
    """

    num_equality_slack_variables: int
    num_inequality_slack_variables: int

    @property
    def num_slack_variables(self) -> int:
        return self.num_inequality_slack_variables + self.num_equality_slack_variables

    @abstractmethod
    def apply_filters(self) -> Self:
        """
        Applies filters to the QP data to remove constraints that have slack-variables with 0 weight.
        """


@dataclass
class QPDataExplicit(QPData):
    """
    Represents a QP problem for solvers that require the following format:

    min_x 0.5 x^T diagonal(quadratic_weights) x + linear_weights^T x
    s.t.  box_lower_constraints   <=            x          <= box_upper_constraints
                                      equality_matrix @ x  == equality_bounds
          inequality_lower_bounds <= inequality_matrix @ x <= inequality_upper_bounds
    """

    quadratic_weights: np.ndarray
    """
    The diagonal of the QP's Hessian matrix.
    """
    linear_weights: np.ndarray
    """
    The linear part of the QP's objective function.
    """

    box_lower_constraints: np.ndarray
    """
    Lower bounds for x.
    """
    box_upper_constraints: np.ndarray
    """
    Upper bounds for x.
    """

    equality_matrix: sp.csc_matrix
    """
    Equality constraints matrix.
    """
    equality_bounds: np.ndarray
    """
    Constraints for the equality matrix multiplied with x.
    """

    inequality_matrix: sp.csc_matrix
    """
    Inequality constraints matrix.
    """
    inequality_lower_bounds: np.ndarray
    """
    Lower bounds for the inequality matrix multiplied with x.
    """
    inequality_upper_bounds: np.ndarray
    """
    Upper bounds for the inequality matrix multiplied with x.
    """

    @property
    def dense_eq_matrix(self) -> np.ndarray:
        return self.equality_matrix.toarray()

    @property
    def dense_neq_matrix(self) -> np.ndarray:
        return self.inequality_matrix.toarray()

    def to_two_sided_inequality(self) -> QPDataTwoSidedInequality:
        A2 = sp.eye(len(self.box_upper_constraints), format="csc")
        if self.equality_matrix.shape[0] * self.equality_matrix.shape[1] != 0:
            A2 = sp.vstack((A2, self.equality_matrix))
        if self.inequality_matrix.shape[0] * self.inequality_matrix.shape[1] != 0:
            A2 = sp.vstack((A2, self.inequality_matrix))
        return QPDataTwoSidedInequality(
            quadratic_weights=self.quadratic_weights,
            linear_weights=self.linear_weights,
            inequality_matrix=A2,
            inequality_lower_bounds=np.concatenate(
                (
                    self.box_lower_constraints,
                    self.equality_bounds,
                    self.inequality_lower_bounds,
                )
            ),
            inequality_upper_bounds=np.concatenate(
                (
                    self.box_upper_constraints,
                    self.equality_bounds,
                    self.inequality_upper_bounds,
                )
            ),
            num_equality_slack_variables=self.num_equality_slack_variables,
            num_inequality_slack_variables=self.num_inequality_slack_variables,
        )

    def apply_filters(self) -> Self:
        zero_quadratic_weight_filter: np.ndarray = self.quadratic_weights != 0
        # don't filter dofs with 0 weight
        zero_quadratic_weight_filter[: -self.num_slack_variables] = True
        slack_part = zero_quadratic_weight_filter[
            -(self.num_equality_slack_variables + self.num_inequality_slack_variables) :
        ]
        bE_part = slack_part[: self.num_equality_slack_variables]
        bA_part = slack_part[self.num_equality_slack_variables :]

        bE_filter = np.ones(self.equality_matrix.shape[0], dtype=bool)
        bE_filter.fill(True)
        if len(bE_part) > 0:
            bE_filter[-len(bE_part) :] = bE_part

        bA_filter = np.ones(self.inequality_matrix.shape[0], dtype=bool)
        bA_filter.fill(True)
        if len(bA_part) > 0:
            bA_filter[-len(bA_part) :] = bA_part

        return QPDataExplicit(
            quadratic_weights=self.quadratic_weights[zero_quadratic_weight_filter],
            linear_weights=self.linear_weights[zero_quadratic_weight_filter],
            box_lower_constraints=self.box_lower_constraints[
                zero_quadratic_weight_filter
            ],
            box_upper_constraints=self.box_upper_constraints[
                zero_quadratic_weight_filter
            ],
            equality_matrix=self._filter_eq_matrix(
                self.equality_matrix, bE_filter, zero_quadratic_weight_filter
            ),
            equality_bounds=self.equality_bounds[bE_filter],
            inequality_matrix=self._filter_neq_matrix(
                self.inequality_matrix, bA_filter, zero_quadratic_weight_filter
            ),
            inequality_lower_bounds=self.inequality_lower_bounds[bA_filter],
            inequality_upper_bounds=self.inequality_upper_bounds[bA_filter],
            num_equality_slack_variables=self.num_equality_slack_variables,
            num_inequality_slack_variables=self.num_inequality_slack_variables,
        )

    def _filter_eq_matrix(
        self,
        eq_matrix: sp.csc_matrix,
        bE_filter: np.ndarray,
        zero_quadratic_weight_filter: np.ndarray,
    ) -> sp.csc_matrix:
        if len(eq_matrix.shape) > 1 and eq_matrix.shape[0] * eq_matrix.shape[1] > 0:
            return eq_matrix[bE_filter, :][:, zero_quadratic_weight_filter]
        return eq_matrix

    def _filter_neq_matrix(
        self,
        neq_matrix: sp.csc_matrix,
        bA_filter: np.ndarray,
        zero_quadratic_weight_filter: np.ndarray,
    ) -> sp.csc_matrix:
        if len(neq_matrix.shape) > 1 and neq_matrix.shape[0] * neq_matrix.shape[1] > 0:
            return neq_matrix[:, zero_quadratic_weight_filter][bA_filter, :]
        return neq_matrix

    def pretty_print_problem(self):
        return (
            f"QPDataExplicit(\n"
            f"    quadratic_weights={self._np_array_to_str(self.quadratic_weights)},\n"
            f"    linear_weights={self._np_array_to_str(self.linear_weights)},\n"
            f"    box_lower_constraints={self._np_array_to_str(self.box_lower_constraints)},\n"
            f"    box_upper_constraints={self._np_array_to_str(self.box_upper_constraints)},\n"
            f"    eq_matrix={self._sparse_matrix_to_str(self.equality_matrix)},\n"
            f"    eq_bounds={self._np_array_to_str(self.equality_bounds)},\n"
            f"    neq_matrix={self._sparse_matrix_to_str(self.inequality_matrix)},\n"
            f"    neq_lower_bounds={self._np_array_to_str(self.inequality_lower_bounds)},\n"
            f"    neq_upper_bounds={self._np_array_to_str(self.inequality_upper_bounds)},\n"
            f"    num_equality_slack_variables={self.num_equality_slack_variables},\n"
            f"    num_inequality_slack_variables={self.num_inequality_slack_variables},\n"
            ")"
        )

    def _np_array_to_str(self, array: np.ndarray, dtype: str = "float") -> str:
        return f"np.array({array.tolist()}, dtype={dtype})".replace("inf", "np.inf")

    def _sparse_matrix_to_str(self, matrix: sp.csc_matrix, spaces: int = 4) -> str:
        return (
            f"sp.csc_matrix(\n"
            f"{' '*spaces}(\n"
            f"{' '*spaces}    {self._np_array_to_str(matrix.data)},\n"
            f"{' '*spaces}    {self._np_array_to_str(matrix.indices, dtype='int')},\n"
            f"{' '*spaces}    {self._np_array_to_str(matrix.indptr, dtype='int')},\n"
            f"{' '*spaces}),\n"
            f"{' '*spaces}shape={matrix.shape},\n"
            f"{' '*spaces})"
        )

    def analyze_well_posedness(self):
        """
        Analyzes the QP problem data for numerical issues and poor posing.
        Prints statistics and warnings for potentially ill-posed problems.
        """
        print("--- QP Well-Posedness Analysis ---")
        self._analyze_hessian()
        self._analyze_constraints()
        print("----------------------------------")

    def _analyze_hessian(self):
        """
        Checks the condition number of the Hessian.
        """
        if self.quadratic_weights is not None:
            max_weight = np.max(np.abs(self.quadratic_weights))
            min_weight = np.min(
                np.abs(self.quadratic_weights)[np.abs(self.quadratic_weights) > 0]
            )
            condition_number = max_weight / min_weight
            print(f"  Weight Matrix max singular value: {max_weight}")
            print(f"  Weight Matrix min singular value: {min_weight}")
            print(f"  Weight Matrix Condition Number: {condition_number}")
            if condition_number > 1_000:
                print("  Warning: Weight Matrix is poorly conditioned.")

    def _analyze_constraints(self):
        """
        Checks for scale imbalances and potential rank issues in constraints.
        """
        self._check_matrix_condition(
            self.equality_matrix, "Equality Constraint Matrix (E)"
        )
        self._check_matrix_condition(
            self.inequality_matrix, "Inequality Constraint Matrix (A)"
        )

        # Simple infeasibility check for box constraints
        if (
            self.box_lower_constraints is not None
            and self.box_upper_constraints is not None
        ):
            violations = self.box_lower_constraints > self.box_upper_constraints
            if np.any(violations):
                print(
                    f"  WARNING: Box constraints are infeasible for indices {np.where(violations)[0]}."
                )

    def _check_matrix_condition(self, matrix: sp.csc_matrix | np.ndarray, name: str):
        if issparse(matrix):
            matrix = matrix.toarray()
        if matrix.shape[0] * matrix.shape[1] == 0:
            print(f"  {name} is empty.")
            return
        singular_value_decomposition = np.linalg.svd(matrix, compute_uv=False)
        condition_number = (
            singular_value_decomposition[0] / singular_value_decomposition[-1]
        )
        print(f"  {name} max singular value: {singular_value_decomposition[0]}")
        print(f"  {name} min singular value: {singular_value_decomposition[-1]}")
        print(f"  {name} Condition Number: {condition_number}")
        if condition_number > 1_000:
            print(f"        WARNING: this is very large.")


@dataclass
class QPDataTwoSidedInequality(QPData):
    """
    Represents a QP problem for solvers that require the following format:

    min_x 0.5 x^T diagonal(quadratic_weights) x + linear_weights^T x
    s.t.  inequality_lower_bounds <= inequality_matrix @ x <= inequality_upper_bounds

    Box constraints and equality constraints must be integrated into the inequality constraints.
    """

    quadratic_weights: np.ndarray
    linear_weights: np.ndarray

    inequality_matrix: sp.csc_matrix
    inequality_lower_bounds: np.ndarray
    inequality_upper_bounds: np.ndarray

    @property
    def num_box_constraints(self) -> int:
        return self.quadratic_weights.shape[0]

    @property
    def box_lower_constraints(self) -> np.ndarray:
        return self.inequality_lower_bounds[: self.num_box_constraints]

    @property
    def box_upper_constraints(self) -> np.ndarray:
        return self.inequality_upper_bounds[: self.num_box_constraints]

    @property
    def eq_matrix(self) -> sp.csc_matrix:
        return self.inequality_matrix[self.bE_start : self.bA_start, :]

    @property
    def bE_start(self) -> int:
        return self.num_box_constraints

    @property
    def bA_start(self) -> int:
        return (
            self.inequality_lower_bounds.shape[0] - self.num_inequality_slack_variables
        )

    def apply_filters(self) -> Self:
        b_bE_bA_filter = np.ones(
            self.inequality_lower_bounds.shape[0],
            dtype=bool,
        )
        b_zero_inf_filter_view = b_bE_bA_filter[: self.num_box_constraints]
        bE_filter_view = b_bE_bA_filter[self.bE_start : self.bA_start]
        bA_filter_view = b_bE_bA_filter[self.bA_start :]
        bE_bA_filter = b_bE_bA_filter[self.bE_start :]

        zero_quadratic_weight_filter = self.quadratic_weights != 0
        zero_quadratic_weight_filter[: -self.num_slack_variables] = True

        slack_part = zero_quadratic_weight_filter[-self.num_slack_variables :]
        bE_part = slack_part[: self.num_equality_slack_variables]
        if len(bE_part) > 0:
            bE_filter_view[-len(bE_part) :] = bE_part

        bA_part = slack_part[self.num_equality_slack_variables :]
        if len(bA_part) > 0:
            bA_filter_view[-len(bA_part) :] = bA_part

        b_finite_filter = np.isfinite(self.box_lower_constraints) | np.isfinite(
            self.box_upper_constraints
        )
        b_zero_inf_filter_view[::] = zero_quadratic_weight_filter & b_finite_filter
        Ai_inf_filter = b_finite_filter  # [zero_quadratic_weight_filter]

        neq_matrix = self.inequality_matrix[:, zero_quadratic_weight_filter][
            bE_bA_filter, :
        ]

        self._nAi_Ai_cache = {}
        box_matrix = self._direct_limit_model(
            self.quadratic_weights.shape[0], Ai_inf_filter, two_sided=True
        )[:, zero_quadratic_weight_filter][zero_quadratic_weight_filter, :]

        return QPDataTwoSidedInequality(
            quadratic_weights=self.quadratic_weights[zero_quadratic_weight_filter],
            linear_weights=self.linear_weights[zero_quadratic_weight_filter],
            inequality_matrix=sp.vstack((box_matrix, neq_matrix)),
            inequality_lower_bounds=self.inequality_lower_bounds[b_bE_bA_filter],
            inequality_upper_bounds=self.inequality_upper_bounds[b_bE_bA_filter],
            num_equality_slack_variables=self.num_equality_slack_variables,
            num_inequality_slack_variables=self.num_inequality_slack_variables,
        )

    def _direct_limit_model(
        self,
        dimensions_after_zero_filter: int,
        Ai_inf_filter: np.ndarray | None = None,
        two_sided: bool = True,
    ) -> sp.csc_matrix:
        """
        These models are often identical, yet the computation is expensive. Caching to the rescue
        """
        if Ai_inf_filter is None:
            key = hash(dimensions_after_zero_filter)
        else:
            key = hash((dimensions_after_zero_filter, Ai_inf_filter.tobytes()))
        if key not in self._nAi_Ai_cache:
            nI_I = self._cached_eyes(dimensions_after_zero_filter, two_sided)
            if Ai_inf_filter is None:
                self._nAi_Ai_cache[key] = nI_I
            else:
                self._nAi_Ai_cache[key] = nI_I[Ai_inf_filter]
        return self._nAi_Ai_cache[key]

    def _cached_eyes(self, dimensions: int, two_sided: bool = False) -> sp.csc_matrix:
        if two_sided:
            data = np.ones(dimensions, dtype=float)
            row_indices = np.arange(dimensions)
            col_indices = np.arange(dimensions + 1)
            return sp.csc_matrix((data, row_indices, col_indices))
        else:
            d2 = dimensions * 2
            data = np.ones(d2, dtype=float)
            data[::2] *= -1
            r1 = np.arange(dimensions)
            r2 = np.arange(dimensions, d2)
            row_indices = np.empty((d2,), dtype=int)
            row_indices[0::2] = r1
            row_indices[1::2] = r2
            col_indices = np.arange(0, d2 + 1, 2)
            return sp.csc_matrix((data, row_indices, col_indices))
