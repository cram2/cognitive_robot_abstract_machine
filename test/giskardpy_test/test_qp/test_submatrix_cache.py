"""
Tests of :class:`~giskardpy.qp.submatrix_cache.SubmatrixCache` and of the QP filters
that take their constraint matrices through it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from giskardpy.qp.qp_data import QPDataExplicit
from giskardpy.qp.submatrix_cache import SubmatrixCache


# %% helpers


def constraint_matrix() -> sp.csc_matrix:
    """
    :return: A matrix of three rows and four columns with one stored zero.
    """
    matrix = sp.csc_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0, 3.0],
                [0.0, 4.0, 0.0, 5.0],
                [6.0, 7.0, 8.0, 0.0],
            ]
        )
    )
    matrix.data[matrix.data == 7.0] = 0.0
    return matrix


def dense_submatrix(
    matrix: sp.csc_matrix, kept_rows: np.ndarray, kept_columns: np.ndarray
) -> list[list[float]]:
    """
    :return: The kept rows and columns of `matrix` as nested lists, taken by scipy.
    """
    return matrix.toarray()[kept_rows][:, kept_columns].tolist()


# %% taking a submatrix


def test_submatrix_holds_the_kept_rows_and_columns() -> None:
    """
    The submatrix holds the entries of the kept rows and columns, in their order.
    """
    matrix = constraint_matrix()
    kept_rows = np.array([True, False, True])
    kept_columns = np.array([True, True, False, True])

    submatrix = SubmatrixCache().take_submatrix(
        matrix=matrix, kept_rows=kept_rows, kept_columns=kept_columns
    )

    assert submatrix.toarray().tolist() == dense_submatrix(
        matrix, kept_rows, kept_columns
    )


def test_stored_zeros_stay_stored_in_the_submatrix() -> None:
    """
    Entries are kept by their position, so a stored zero keeps its place for the values
    of later control cycles.
    """
    matrix = constraint_matrix()

    submatrix = SubmatrixCache().take_submatrix(
        matrix=matrix,
        kept_rows=np.ones(3, dtype=bool),
        kept_columns=np.ones(4, dtype=bool),
    )

    assert submatrix.nnz == matrix.nnz


def test_taking_the_same_submatrix_again_reads_the_new_values() -> None:
    """
    The same rows and columns of a matrix with new values give the same submatrix,
    holding the new values.
    """
    cache = SubmatrixCache()
    matrix = constraint_matrix()
    kept_rows = np.array([False, True, True])
    kept_columns = np.array([True, True, True, False])
    first = cache.take_submatrix(
        matrix=matrix, kept_rows=kept_rows, kept_columns=kept_columns
    )

    matrix.data[:] = np.arange(1.0, matrix.nnz + 1.0)
    second = cache.take_submatrix(
        matrix=matrix, kept_rows=kept_rows, kept_columns=kept_columns
    )

    assert second is first
    assert second.toarray().tolist() == dense_submatrix(matrix, kept_rows, kept_columns)


def test_other_rows_and_columns_give_their_own_submatrix() -> None:
    """
    A second selection of the same matrix leaves the first one correct.
    """
    cache = SubmatrixCache()
    matrix = constraint_matrix()
    kept_rows = np.array([True, True, False])
    first_columns = np.array([True, False, True, True])
    second_columns = np.array([False, True, True, False])

    first = cache.take_submatrix(
        matrix=matrix, kept_rows=kept_rows, kept_columns=first_columns
    )
    second = cache.take_submatrix(
        matrix=matrix, kept_rows=kept_rows, kept_columns=second_columns
    )

    assert first.toarray().tolist() == dense_submatrix(matrix, kept_rows, first_columns)
    assert second.toarray().tolist() == dense_submatrix(
        matrix, kept_rows, second_columns
    )


def test_keeping_no_row_gives_a_submatrix_without_rows() -> None:
    """
    Keeping no row keeps every column, without rows.
    """
    matrix = constraint_matrix()

    submatrix = SubmatrixCache().take_submatrix(
        matrix=matrix,
        kept_rows=np.zeros(3, dtype=bool),
        kept_columns=np.ones(4, dtype=bool),
    )

    assert submatrix.shape == (0, 4)


def test_a_matrix_without_rows_is_its_own_submatrix() -> None:
    """
    A matrix without rows is returned as it is.
    """
    matrix = sp.csc_matrix((0, 4))

    submatrix = SubmatrixCache().take_submatrix(
        matrix=matrix,
        kept_rows=np.zeros(0, dtype=bool),
        kept_columns=np.ones(4, dtype=bool),
    )

    assert submatrix is matrix


# %% forgetting selections


def test_the_least_recently_used_selection_is_forgotten() -> None:
    """
    Past the maximum, the selection used least recently is forgotten, and a selection
    used again in between is kept.
    """
    cache = SubmatrixCache(maximum_number_of_selections=2)
    matrix = constraint_matrix()
    kept_rows = np.ones(3, dtype=bool)
    columns = [
        np.array([True, False, False, False]),
        np.array([False, True, False, False]),
        np.array([False, False, True, False]),
    ]
    first = cache.take_submatrix(
        matrix=matrix, kept_rows=kept_rows, kept_columns=columns[0]
    )
    cache.take_submatrix(matrix=matrix, kept_rows=kept_rows, kept_columns=columns[1])
    cache.take_submatrix(matrix=matrix, kept_rows=kept_rows, kept_columns=columns[0])
    cache.take_submatrix(matrix=matrix, kept_rows=kept_rows, kept_columns=columns[2])

    assert len(cache.selections) == cache.maximum_number_of_selections
    assert (
        cache.take_submatrix(
            matrix=matrix, kept_rows=kept_rows, kept_columns=columns[0]
        )
        is first
    )


# %% filtering a problem in consecutive control cycles


def qp_with_one_slack_per_constraint(
    equality_matrix: sp.csc_matrix, inequality_matrix: sp.csc_matrix
) -> QPDataExplicit:
    """
    :return: A problem over two degree-of-freedom variables, one equality slack and one
        inequality slack, whose slack weights are both nonzero.
    """
    return QPDataExplicit(
        quadratic_weights=np.array([1.0, 1.0, 10.0, 10.0]),
        linear_weights=np.zeros(4),
        box_lower_constraints=-np.ones(4),
        box_upper_constraints=np.ones(4),
        equality_matrix=equality_matrix,
        equality_bounds=np.array([0.1, 0.2]),
        inequality_matrix=inequality_matrix,
        inequality_lower_bounds=np.array([-0.3, -0.4]),
        inequality_upper_bounds=np.array([0.3, 0.4]),
        num_equality_slack_variables=1,
        num_inequality_slack_variables=1,
    )


def test_filtering_through_a_cache_gives_the_problem_of_every_control_cycle() -> None:
    """
    The constraint matrices keep their sparsity pattern from one control cycle to the
    next while their values and the slack weights change.
    """
    cache = SubmatrixCache()
    equality_matrix = sp.csc_matrix(
        np.array([[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 1.0, 0.0]])
    )
    inequality_matrix = sp.csc_matrix(
        np.array([[4.0, 5.0, 0.0, 0.0], [0.0, 6.0, 0.0, 1.0]])
    )
    qp_data = qp_with_one_slack_per_constraint(
        equality_matrix=equality_matrix, inequality_matrix=inequality_matrix
    )
    for cycle in range(3):
        equality_matrix.data[:] = equality_matrix.data + cycle
        inequality_matrix.data[:] = inequality_matrix.data * (cycle + 2)
        qp_data.quadratic_weights[2:] = [10.0, 0.0] if cycle == 1 else [10.0, 10.0]

        cached = qp_data.apply_filters(submatrix_cache=cache)
        expected = qp_data.apply_filters()

        assert cached.dense_eq_matrix.tolist() == expected.dense_eq_matrix.tolist()
        assert cached.dense_neq_matrix.tolist() == expected.dense_neq_matrix.tolist()


def test_removing_variables_through_a_cache_reads_the_new_values() -> None:
    """
    Removing the same variables again after the values changed gives the problem
    removing them from scratch gives.
    """
    cache = SubmatrixCache()
    equality_matrix = sp.csc_matrix(
        np.array([[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 1.0, 0.0]])
    )
    inequality_matrix = sp.csc_matrix(
        np.array([[4.0, 5.0, 0.0, 0.0], [0.0, 6.0, 0.0, 1.0]])
    )
    qp_data = qp_with_one_slack_per_constraint(
        equality_matrix=equality_matrix, inequality_matrix=inequality_matrix
    )
    removed_variables = np.array([True, False, False, False])
    qp_data.without_variables(
        removed_variables=removed_variables, submatrix_cache=cache
    )

    equality_matrix.data[:] = equality_matrix.data * 3.0
    inequality_matrix.data[:] = inequality_matrix.data - 1.0
    cached = qp_data.without_variables(
        removed_variables=removed_variables, submatrix_cache=cache
    )
    expected = qp_data.without_variables(removed_variables=removed_variables)

    assert cached.dense_eq_matrix.tolist() == expected.dense_eq_matrix.tolist()
    assert cached.dense_neq_matrix.tolist() == expected.dense_neq_matrix.tolist()
