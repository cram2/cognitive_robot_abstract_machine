from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.sparse import issparse
from typing_extensions import Self

logger = logging.getLogger(__name__)


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
        """
        Returns the total number of slack variables.
        """
        return self.num_inequality_slack_variables + self.num_equality_slack_variables

    @abstractmethod
    def apply_filters(self) -> Self:
        """
        Applies filters to the QP data to remove constraints that have slack-variables
        with 0 weight.
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
        """
        Converts the sparse equality matrix into a dense array.
        """
        return self.equality_matrix.toarray()

    @property
    def dense_neq_matrix(self) -> np.ndarray:
        """
        Converts the sparse inequality matrix into a dense array.
        """
        return self.inequality_matrix.toarray()

    @property
    def hessian(self) -> sp.csc_matrix:
        """
        The QP's Hessian matrix, a sparse diagonal matrix of the quadratic weights.

        .. note:: Built directly in CSC form, which is faster than
            :func:`scipy.sparse.diags`.
        """
        number_of_variables = len(self.quadratic_weights)
        return sp.csc_matrix(
            (
                self.quadratic_weights,
                np.arange(number_of_variables),
                np.arange(number_of_variables + 1),
            ),
            shape=(number_of_variables, number_of_variables),
        )

    def to_two_sided_inequality(self) -> QPDataTwoSidedInequality:
        """
        Converts the explicit QP format to a format with only two-sided inequalities.
        """
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
        number_non_slack_variables = (
            zero_quadratic_weight_filter.shape[0] - self.num_slack_variables
        )
        # don't filter dofs with 0 weight
        zero_quadratic_weight_filter[:number_non_slack_variables] = True
        slack_part = zero_quadratic_weight_filter[number_non_slack_variables:]
        bE_part = slack_part[: self.num_equality_slack_variables]
        bA_part = slack_part[self.num_equality_slack_variables :]

        bE_filter = np.ones(self.equality_matrix.shape[0], dtype=bool)
        if len(bE_part) > 0:
            bE_filter[-len(bE_part) :] = bE_part

        bA_filter = np.ones(self.inequality_matrix.shape[0], dtype=bool)
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
            equality_matrix=self._submatrix(
                self.equality_matrix, bE_filter, zero_quadratic_weight_filter
            ),
            equality_bounds=self.equality_bounds[bE_filter],
            inequality_matrix=self._submatrix(
                self.inequality_matrix, bA_filter, zero_quadratic_weight_filter
            ),
            inequality_lower_bounds=self.inequality_lower_bounds[bA_filter],
            inequality_upper_bounds=self.inequality_upper_bounds[bA_filter],
            num_equality_slack_variables=self.num_equality_slack_variables,
            num_inequality_slack_variables=self.num_inequality_slack_variables,
        )

    def without_variables(self, removed_variables: NDArray[np.bool_]) -> QPDataExplicit:
        """
        :param removed_variables: Which variables to leave out, one flag per variable.
        :return: This problem without the flagged variables and without the
            constraints that have no coefficient left once they are gone.
        """
        kept_variables = ~removed_variables
        kept_equalities = self._rows_keeping_coefficients(
            self.equality_matrix, kept_variables
        )
        kept_inequalities = self._rows_keeping_coefficients(
            self.inequality_matrix, kept_variables
        )
        return QPDataExplicit(
            quadratic_weights=self.quadratic_weights[kept_variables],
            linear_weights=self.linear_weights[kept_variables],
            box_lower_constraints=self.box_lower_constraints[kept_variables],
            box_upper_constraints=self.box_upper_constraints[kept_variables],
            equality_matrix=self._submatrix(
                self.equality_matrix, kept_equalities, kept_variables
            ),
            equality_bounds=self.equality_bounds[kept_equalities],
            inequality_matrix=self._submatrix(
                self.inequality_matrix, kept_inequalities, kept_variables
            ),
            inequality_lower_bounds=self.inequality_lower_bounds[kept_inequalities],
            inequality_upper_bounds=self.inequality_upper_bounds[kept_inequalities],
            num_equality_slack_variables=self.num_equality_slack_variables,
            num_inequality_slack_variables=self.num_inequality_slack_variables,
        )

    def variables_without_coefficients(
        self, number_of_variables: int, number_of_ignored_equalities: int
    ) -> NDArray[np.bool_]:
        """
        :param number_of_variables: How many variables are checked, from the first on.
        :param number_of_ignored_equalities: How many equalities, from the first on, are
            not checked.
        :return: Which of the checked variables have a coefficient of zero in every
            checked equality and in every inequality, one flag per variable.
        """
        with_coefficients = self._columns_with_coefficients(
            self.equality_matrix, number_of_ignored_equalities, number_of_variables
        ) | self._columns_with_coefficients(
            self.inequality_matrix, 0, number_of_variables
        )
        return ~with_coefficients

    @classmethod
    def _columns_with_coefficients(
        cls, matrix: sp.csc_matrix, first_row: int, number_of_columns: int
    ) -> NDArray[np.bool_]:
        """
        :param matrix: A constraint matrix.
        :param first_row: The first row of `matrix` that is checked.
        :param number_of_columns: How many columns of `matrix` are checked, from the
            first on.
        :return: Which of the checked columns have a nonzero value in a checked row, one
            flag per column.
        """
        with_coefficients = np.zeros(number_of_columns, dtype=bool)
        # A matrix without rows does not necessarily have a column per variable.
        if matrix.shape[0] == 0:
            return with_coefficients
        entries_in_checked_columns = matrix.indptr[number_of_columns]
        coefficients = (matrix.data[:entries_in_checked_columns] != 0) & (
            matrix.indices[:entries_in_checked_columns] >= first_row
        )
        column_of_entry = cls._column_of_each_entry(matrix)
        with_coefficients[
            column_of_entry[:entries_in_checked_columns][coefficients]
        ] = True
        return with_coefficients

    @classmethod
    def _rows_keeping_coefficients(
        cls, matrix: sp.csc_matrix, kept_variables: np.ndarray
    ) -> np.ndarray:
        """
        :param matrix: A constraint matrix.
        :param kept_variables: Which columns of `matrix` are kept, one flag per column.
        :return: Which rows do not lose all of their entries with the columns that are
            not kept, one flag per row.
        """
        if matrix.shape[0] == 0:
            return np.ones(0, dtype=bool)
        kept_entries = kept_variables[cls._column_of_each_entry(matrix)]
        had_entries = np.bincount(matrix.indices, minlength=matrix.shape[0]) > 0
        has_entries = (
            np.bincount(matrix.indices[kept_entries], minlength=matrix.shape[0]) > 0
        )
        return has_entries | ~had_entries

    @classmethod
    def _submatrix(
        cls, matrix: sp.csc_matrix, kept_rows: np.ndarray, kept_columns: np.ndarray
    ) -> sp.csc_matrix:
        """
        :param matrix: A constraint matrix.
        :param kept_rows: Which rows of `matrix` are kept, one flag per row.
        :param kept_columns: Which columns of `matrix` are kept, one flag per column.
        :return: `matrix` with only the kept rows and columns, or `matrix` itself if it
            has no rows or columns.
        """
        if matrix.shape[0] * matrix.shape[1] == 0:
            return matrix
        column_of_entry = cls._column_of_each_entry(matrix)
        kept_entries = kept_columns[column_of_entry] & kept_rows[matrix.indices]
        new_row_of_row = np.cumsum(kept_rows) - 1
        new_column_of_column = np.cumsum(kept_columns) - 1
        number_of_kept_columns = int(new_column_of_column[-1]) + 1
        entries_per_kept_column = np.bincount(
            new_column_of_column[column_of_entry[kept_entries]],
            minlength=number_of_kept_columns,
        )
        return sp.csc_matrix(
            (
                matrix.data[kept_entries],
                new_row_of_row[matrix.indices[kept_entries]],
                np.concatenate(([0], np.cumsum(entries_per_kept_column))),
            ),
            shape=(int(new_row_of_row[-1]) + 1, number_of_kept_columns),
        )

    @staticmethod
    def _column_of_each_entry(matrix: sp.csc_matrix) -> np.ndarray:
        """
        :param matrix: A matrix in compressed sparse column format.
        :return: The column of every stored entry of `matrix`, in storage order.
        """
        return np.repeat(np.arange(matrix.shape[1]), np.diff(matrix.indptr))

    def pretty_print_problem(self):
        """
        Returns a human-readable string representation of the QP problem.

        Use this to create test cases for QPs that cannot be solved.
        """
        return (
            f"QPDataExplicit(\n"
            f"    quadratic_weights={self._np_array_to_str(self.quadratic_weights)},\n"
            f"    linear_weights={self._np_array_to_str(self.linear_weights)},\n"
            f"    box_lower_constraints={self._np_array_to_str(self.box_lower_constraints)},\n"
            f"    box_upper_constraints={self._np_array_to_str(self.box_upper_constraints)},\n"
            f"    equality_matrix={self._sparse_matrix_to_str(self.equality_matrix)},\n"
            f"    equality_bounds={self._np_array_to_str(self.equality_bounds)},\n"
            f"    inequality_matrix={self._sparse_matrix_to_str(self.inequality_matrix)},\n"
            f"    inequality_lower_bounds={self._np_array_to_str(self.inequality_lower_bounds)},\n"
            f"    inequality_upper_bounds={self._np_array_to_str(self.inequality_upper_bounds)},\n"
            f"    num_equality_slack_variables={self.num_equality_slack_variables},\n"
            f"    num_inequality_slack_variables={self.num_inequality_slack_variables},\n"
            ")"
        )

    def _np_array_to_str(self, array: np.ndarray, dtype: str = "float") -> str:
        """
        Converts a numpy array to a string representation for debugging.
        """
        return f"np.array({array.tolist()}, dtype={dtype})".replace("inf", "np.inf")

    def _sparse_matrix_to_str(self, matrix: sp.csc_matrix, spaces: int = 4) -> str:
        """
        Converts a sparse matrix to a string representation for debugging.
        """
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

        Logs statistics and warnings for potentially ill-posed problems.
        """
        logger.info("--- QP Well-Posedness Analysis ---")
        self._analyze_hessian()
        self._analyze_constraints()
        logger.info("----------------------------------")

    def _analyze_hessian(self):
        """
        Checks the condition number of the Hessian.
        """
        if self.quadratic_weights is not None:
            weights = np.abs(self.quadratic_weights)
            if not np.any(weights > 0):
                logger.warning("  Weight Matrix is all zero.")
                return
            max_weight = np.max(weights)
            min_weight = np.min(weights[weights > 0])
            condition_number = max_weight / min_weight
            logger.info(f"  Weight Matrix max singular value: {max_weight}")
            logger.info(f"  Weight Matrix min singular value: {min_weight}")
            logger.info(f"  Weight Matrix Condition Number: {condition_number}")
            if condition_number > 1_000:
                logger.warning("  Weight Matrix is poorly conditioned.")

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
                logger.warning(
                    f"  Box constraints are infeasible for indices {np.where(violations)[0]}."
                )

    def _check_matrix_condition(self, matrix: sp.csc_matrix | np.ndarray, name: str):
        if issparse(matrix):
            matrix = matrix.toarray()
        if matrix.shape[0] * matrix.shape[1] == 0:
            logger.info(f"  {name} is empty.")
            return
        singular_value_decomposition = np.linalg.svd(matrix, compute_uv=False)
        condition_number = (
            singular_value_decomposition[0] / singular_value_decomposition[-1]
        )
        logger.info(f"  {name} max singular value: {singular_value_decomposition[0]}")
        logger.info(f"  {name} min singular value: {singular_value_decomposition[-1]}")
        logger.info(f"  {name} Condition Number: {condition_number}")
        if condition_number > 1_000:
            logger.warning(f"  {name} Condition Number is very large.")


@dataclass(eq=False)
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

    def __hash__(self):
        return id(self)

    @property
    def num_box_constraints(self) -> int:
        """
        Returns the number of box constraints.
        """
        return self.quadratic_weights.shape[0]

    @property
    def box_lower_constraints(self) -> np.ndarray:
        """
        Returns the lower bounds of the box constraints.
        """
        return self.inequality_lower_bounds[: self.num_box_constraints]

    @property
    def box_upper_constraints(self) -> np.ndarray:
        """
        Returns the upper bounds of the box constraints.
        """
        return self.inequality_upper_bounds[: self.num_box_constraints]

    @property
    def eq_matrix(self) -> sp.csc_matrix:
        """
        Extracts the equality constraint matrix from the combined inequality matrix.
        """
        return self.inequality_matrix[
            self.start_index_of_equality_constraints : self.start_index_of_inequality_constraints,
            :,
        ]

    @property
    def start_index_of_equality_constraints(self) -> int:
        """
        Returns the starting index of the equality constraints in the combined bounds.
        """
        return self.num_box_constraints

    @property
    def start_index_of_inequality_constraints(self) -> int:
        """
        Returns the starting index of the inequality constraints in the combined bounds.
        """
        return (
            self.inequality_lower_bounds.shape[0] - self.num_inequality_slack_variables
        )

    def apply_filters(self) -> Self:
        combined_constraint_filter = np.ones(
            self.inequality_lower_bounds.shape[0],
            dtype=bool,
        )
        box_zero_inf_filter_view = combined_constraint_filter[
            : self.num_box_constraints
        ]
        equality_filter_view = combined_constraint_filter[
            self.start_index_of_equality_constraints : self.start_index_of_inequality_constraints
        ]
        inequality_filter_view = combined_constraint_filter[
            self.start_index_of_inequality_constraints :
        ]
        constraint_filter = combined_constraint_filter[
            self.start_index_of_equality_constraints :
        ]

        zero_quadratic_weight_filter = self.quadratic_weights != 0
        number_non_slack_variables = (
            zero_quadratic_weight_filter.shape[0] - self.num_slack_variables
        )
        zero_quadratic_weight_filter[:number_non_slack_variables] = True

        slack_part = zero_quadratic_weight_filter[number_non_slack_variables:]
        equality_part = slack_part[: self.num_equality_slack_variables]
        if len(equality_part) > 0:
            equality_filter_view[-len(equality_part) :] = equality_part

        inequality_part = slack_part[self.num_equality_slack_variables :]
        if len(inequality_part) > 0:
            inequality_filter_view[-len(inequality_part) :] = inequality_part

        box_finite_filter = np.isfinite(self.box_lower_constraints) | np.isfinite(
            self.box_upper_constraints
        )
        box_zero_inf_filter_view[::] = zero_quadratic_weight_filter & box_finite_filter
        box_identity_matrix_inf_filter = (
            box_finite_filter  # [zero_quadratic_weight_filter]
        )

        inequality_matrix = self.inequality_matrix[
            self.num_box_constraints :, zero_quadratic_weight_filter
        ][constraint_filter, :]

        box_matrix = self._direct_limit_model(
            self.quadratic_weights.shape[0],
            box_identity_matrix_inf_filter,
        )[:, zero_quadratic_weight_filter][zero_quadratic_weight_filter, :]

        return QPDataTwoSidedInequality(
            quadratic_weights=self.quadratic_weights[zero_quadratic_weight_filter],
            linear_weights=self.linear_weights[zero_quadratic_weight_filter],
            inequality_matrix=sp.vstack((box_matrix, inequality_matrix)),
            inequality_lower_bounds=self.inequality_lower_bounds[
                combined_constraint_filter
            ],
            inequality_upper_bounds=self.inequality_upper_bounds[
                combined_constraint_filter
            ],
            num_equality_slack_variables=self.num_equality_slack_variables,
            num_inequality_slack_variables=self.num_inequality_slack_variables,
        )

    def _direct_limit_model(
        self,
        dimensions_after_zero_filter: int,
        Ai_inf_filter: np.ndarray | None = None,
    ) -> sp.csc_matrix:
        """
        These models are often identical, yet the computation is expensive.

        Caching to the rescue
        """
        nI_I = self._cached_eyes(dimensions_after_zero_filter)
        if Ai_inf_filter is None:
            return nI_I
        return nI_I[Ai_inf_filter]

    @lru_cache
    def _cached_eyes(self, dimensions: int) -> sp.csc_matrix:
        """
        Creates and caches identity or selection matrices for constraint models.
        """
        data = np.ones(dimensions, dtype=float)
        row_indices = np.arange(dimensions)
        col_indices = np.arange(dimensions + 1)
        return sp.csc_matrix((data, row_indices, col_indices))
