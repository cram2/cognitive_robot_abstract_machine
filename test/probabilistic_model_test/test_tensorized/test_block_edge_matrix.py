"""
Tests for the edge matrix whose rows with the same child columns form full blocks.
"""

from __future__ import annotations

# %% imports
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probabilistic_model.probabilistic_circuit.tensorized.matrix_shape import (
    MatrixShape2D,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    BlockEdgeMatrix,
    EdgeBlockStackShape,
    EdgeWeighting,
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


@pytest.fixture
def sums_of_a_region_graph() -> BlockEdgeMatrix:
    """
    Four regions of three sums, every sum mixing the nine products of its region, the
    products of the regions laid out one region after another.
    """
    regions, sums_per_region, products_per_region = 4, 3, 9
    rows = np.repeat(np.arange(regions * sums_per_region), products_per_region)
    columns = np.repeat(
        np.arange(regions), sums_per_region * products_per_region
    ) * products_per_region + np.tile(
        np.arange(products_per_region), regions * sums_per_region
    )
    return BlockEdgeMatrix.from_edge_positions(
        rows=rows,
        columns=columns,
        shape=MatrixShape2D(
            number_of_rows=regions * sums_per_region,
            number_of_columns=regions * products_per_region,
        ),
    )


def test_sums_of_one_shape_are_one_stack(sums_of_a_region_graph: BlockEdgeMatrix):
    """
    Sums like those of a region graph, every one mixing all products of its region, are
    stacked into a single stack of one block per region.
    """
    [stack] = sums_of_a_region_graph.stacks
    assert stack.block_rows.shape == (4, 3)
    assert stack.block_columns.shape == (4, 9)


def test_blocks_of_consecutive_regions_have_contiguous_columns(
    sums_of_a_region_graph: BlockEdgeMatrix,
    rows_of_two_blocks_and_a_row_without_edges: SparseEntries,
):
    """
    The stack of a region graph's sums reads the columns of its blocks as one range, as
    does a block of the columns 1 and 2, while the block of the columns 0, 3 and 4 skips
    some.
    """
    entries = rows_of_two_blocks_and_a_row_without_edges
    scattered = BlockEdgeMatrix.from_edge_positions(
        rows=entries.rows,
        columns=entries.columns,
        shape=MatrixShape2D(number_of_rows=5, number_of_columns=5),
    )
    [stack] = sums_of_a_region_graph.stacks
    assert stack.has_contiguous_columns
    assert {
        tuple(stack.block_columns[0].tolist()): stack.has_contiguous_columns
        for stack in scattered.stacks
    } == {(1, 2): True, (0, 3, 4): False}


@pytest.mark.parametrize("reduction", list(RowReduction), ids=str)
@pytest.mark.parametrize(
    "combination", [None, RowReduction.SUM, RowReduction.PRODUCT], ids=str
)
def test_block_reduction_of_child_values_is_the_sparse_reduction(
    sums_of_a_region_graph: BlockEdgeMatrix,
    rows_of_two_blocks_and_a_row_without_edges: SparseEntries,
    reduction: RowReduction,
    combination: RowReduction | None,
):
    """
    Reducing the values of the children of every node, weighted by the values of their
    edges or not, gives what the sparse matrix of the same edges gives, whether the
    columns of the blocks are contiguous or not.
    """
    entries = rows_of_two_blocks_and_a_row_without_edges
    scattered = BlockEdgeMatrix.from_edge_positions(
        rows=entries.rows,
        columns=entries.columns,
        shape=MatrixShape2D(number_of_rows=5, number_of_columns=5),
    )
    generator = np.random.default_rng(0)
    for block in (scattered, sums_of_a_region_graph):
        sparse = SparseEdgeMatrix.from_edge_positions(
            rows=block.rows, columns=block.columns, shape=block.shape
        )
        edge_entries = SparseEntries(
            generator.normal(size=block.number_of_edges), block.rows, block.columns
        )
        values = generator.normal(size=(3, block.shape.number_of_columns))
        results = [
            matrix.reduce_child_values_by_row(
                values=values,
                reduction=reduction,
                weighting=(
                    None
                    if combination is None
                    else EdgeWeighting(
                        stored=matrix.stored_values(entries=edge_entries, padding=0.0),
                        combination=combination,
                    )
                ),
            )
            for matrix in (sparse, block)
        ]
        np.testing.assert_allclose(results[1], results[0])


def test_gradients_of_block_reduction_are_those_of_the_sparse_reduction(
    sums_of_a_region_graph: BlockEdgeMatrix,
):
    """
    The gradients of a log-weighted sum with respect to the values of the children and
    the weights of the edges are those the sparse matrix of the same edges gives.
    """
    block = sums_of_a_region_graph
    sparse = SparseEdgeMatrix.from_edge_positions(
        rows=block.rows, columns=block.columns, shape=block.shape
    )
    generator = np.random.default_rng(1)
    log_weights = generator.normal(size=block.number_of_edges)
    values = generator.normal(size=(5, block.shape.number_of_columns))

    def gradients(matrix: BlockEdgeMatrix | SparseEdgeMatrix) -> tuple:
        """
        :return: The gradients of the summed reduction by the values and the weights,
            in double precision.
        """

        def total(values: jnp.ndarray, log_weights: jnp.ndarray) -> jnp.ndarray:
            """
            :return: The sum of the log-weighted sums of all nodes and rows.
            """
            return jnp.sum(
                matrix.reduce_child_values_by_row(
                    values=values,
                    reduction=RowReduction.LOG_SUM_EXP,
                    weighting=EdgeWeighting(
                        stored=log_weights, combination=RowReduction.SUM
                    ),
                )
            )

        return jax.grad(total, argnums=(0, 1))(
            jnp.asarray(values), jnp.asarray(log_weights)
        )

    with jax.enable_x64(True):
        for expected, result in zip(gradients(sparse), gradients(block)):
            np.testing.assert_allclose(np.asarray(result), np.asarray(expected))


def test_stack_shape_counts_the_blocks_and_their_rows_and_columns(
    rows_of_two_blocks_and_a_row_without_edges: SparseEntries,
):
    """
    Every stack reports how many blocks it holds and how many rows and columns each
    block has, and its number of edges is their product.
    """
    entries = rows_of_two_blocks_and_a_row_without_edges
    matrix = BlockEdgeMatrix.from_edge_positions(
        rows=entries.rows,
        columns=entries.columns,
        shape=MatrixShape2D(number_of_rows=5, number_of_columns=5),
    )
    shapes = sorted(
        (stack.stack_shape for stack in matrix.stacks),
        key=lambda shape: shape.columns_per_block,
    )
    assert shapes == [
        EdgeBlockStackShape(number_of_blocks=1, rows_per_block=2, columns_per_block=2),
        EdgeBlockStackShape(number_of_blocks=1, rows_per_block=2, columns_per_block=3),
    ]
    assert sum(stack.number_of_edges for stack in matrix.stacks) == len(entries.rows)
