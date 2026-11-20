"""
Tests for the edge matrix whose rows with the same child columns form full blocks.
"""

from __future__ import annotations

# %% imports
import numpy as np
import pytest

from probabilistic_model.probabilistic_circuit.tensorized.matrix_shape import (
    MatrixShape2D,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    BlockEdgeMatrix,
    RowReduction,
    SparseEdgeMatrix,
    SparseEntries,
)

# %% blocks


@pytest.fixture
def rows_of_two_blocks_and_a_row_without_edges() -> SparseEntries:
    """
    Rows 0 and 3 mix columns 1 and 2, rows 1 and 4 mix columns 0, 3 and 4, row 2 has no
    edges; the entries are listed out of order.
    """
    rows = np.array([3, 0, 4, 1, 1, 4, 0, 3, 1, 4])
    columns = np.array([2, 1, 0, 3, 0, 4, 2, 1, 4, 3])
    return SparseEntries(np.arange(10.0), rows, columns)


def test_rows_with_the_same_columns_form_one_block(
    rows_of_two_blocks_and_a_row_without_edges: SparseEntries,
):
    """
    The rows that mix the same columns are stacked into one block per set of columns.
    """
    entries = rows_of_two_blocks_and_a_row_without_edges
    matrix = BlockEdgeMatrix.from_edge_positions(
        rows=entries.rows,
        columns=entries.columns,
        shape=MatrixShape2D(number_of_rows=5, number_of_columns=5),
    )
    shapes = sorted(
        stack.block_rows.shape + stack.block_columns.shape[1:]
        for stack in matrix.stacks
    )
    assert shapes == [(1, 2, 2), (1, 2, 3)]


@pytest.mark.parametrize("reduction", list(RowReduction), ids=str)
def test_block_reduction_is_the_sparse_reduction(
    rows_of_two_blocks_and_a_row_without_edges: SparseEntries, reduction: RowReduction
):
    """
    Reducing values of the columns by row gives what the sparse matrix of the same edges
    gives, the identity for the row without edges.
    """
    entries = rows_of_two_blocks_and_a_row_without_edges
    values = np.random.default_rng(0).normal(size=(3, 5))
    results = []
    for matrix_type in (SparseEdgeMatrix, BlockEdgeMatrix):
        matrix = matrix_type.from_edge_positions(
            rows=entries.rows,
            columns=entries.columns,
            shape=MatrixShape2D(number_of_rows=5, number_of_columns=5),
        )
        stored = matrix.stored_values(entries=entries, padding=0.0)
        results.append(
            matrix.reduce_by_row(
                values=matrix.values_of_columns(values=values) + stored,
                reduction=reduction,
            )
        )
    np.testing.assert_allclose(results[1], results[0])


def test_stored_values_follow_their_edges(
    rows_of_two_blocks_and_a_row_without_edges: SparseEntries,
):
    """
    The value of every edge read back is the value given for it, whatever order the
    entries were given in.
    """
    entries = rows_of_two_blocks_and_a_row_without_edges
    matrix = BlockEdgeMatrix.from_edge_positions(
        rows=entries.rows,
        columns=entries.columns,
        shape=MatrixShape2D(number_of_rows=5, number_of_columns=5),
    )
    edge_values = matrix.edge_values(
        stored=matrix.stored_values(entries=entries, padding=0.0)
    )
    expected = {
        (int(row), int(column)): value
        for row, column, value in zip(entries.rows, entries.columns, entries.data)
    }
    assert {
        (int(row), int(column)): value
        for row, column, value in zip(matrix.rows, matrix.columns, edge_values)
    } == expected


def test_sums_of_one_shape_are_one_stack():
    """
    Sums like those of a region graph, every one mixing all products of its region, are
    stacked into a single stack.
    """
    regions, sums_per_region, products_per_region = 4, 3, 9
    rows = np.repeat(np.arange(regions * sums_per_region), products_per_region)
    columns = np.repeat(
        np.arange(regions), sums_per_region * products_per_region
    ) * products_per_region + np.tile(
        np.arange(products_per_region), regions * sums_per_region
    )
    matrix = BlockEdgeMatrix.from_edge_positions(
        rows=rows,
        columns=columns,
        shape=MatrixShape2D(
            number_of_rows=regions * sums_per_region,
            number_of_columns=regions * products_per_region,
        ),
    )
    [stack] = matrix.stacks
    assert stack.block_rows.shape == (regions, sums_per_region)
    assert stack.block_columns.shape == (regions, products_per_region)
