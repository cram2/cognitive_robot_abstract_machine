"""
The edges of an inner layer of a layered circuit, as the entries of a matrix whose rows
are the nodes of the layer and whose columns are the nodes of its child layers, and the
reductions that combine values along the edges of every node.
"""

from __future__ import annotations

import functools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType

import numpy as np
import numpy.typing as npt
import array_api_compat
from array_api_compat import array_namespace
from typing_extensions import Self

from probabilistic_model.probabilistic_circuit.tensorized.matrix_shape import (
    MatrixShape2D,
)
from probabilistic_model.probabilistic_circuit.tensorized.gradient_stop import (
    GradientStop,
)
from probabilistic_model.probabilistic_circuit.tensorized.structure_array_copies import (
    StructureArrayCopies,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
    EdgeValues,
    NodeColumnMask,
    NodeValues,
)

# %% reductions


class RowReduction(Enum):
    """
    An associative and commutative operation that combines the values along the edges of
    a node.
    """

    SUM = "sum"
    """
    Add the values.
    """

    PRODUCT = "product"
    """
    Multiply the values.
    """

    LOG_SUM_EXP = "log sum exp"
    """
    The logarithm of the sum of the exponentials of the values.
    """

    @property
    def identity(self) -> float:
        """
        :return: The result of combining no values.
        """
        match self:
            case RowReduction.SUM:
                return 0.0
            case RowReduction.PRODUCT:
                return 1.0
            case RowReduction.LOG_SUM_EXP:
                return -np.inf

    def combine(
        self, namespace: ModuleType, first: BackendArray, second: BackendArray
    ) -> BackendArray:
        """
        :param namespace: The array namespace of both arrays.
        :param first: Values to combine element by element.
        :param second: Values of the same shape as ``first``.
        :return: Both values combined element by element.
        """
        match self:
            case RowReduction.SUM:
                return namespace.add(first, second)
            case RowReduction.PRODUCT:
                return namespace.multiply(first, second)
            case RowReduction.LOG_SUM_EXP:
                return namespace.logaddexp(first, second)

    def reduce_last_axis(
        self, namespace: ModuleType, values: BackendArray
    ) -> BackendArray:
        """
        :param namespace: The array namespace of the values.
        :param values: The values to combine, along the last axis.
        :return: The values combined along the last axis, which is removed.
        """
        match self:
            case RowReduction.SUM:
                return namespace.sum(values, axis=-1)
            case RowReduction.PRODUCT:
                return namespace.prod(values, axis=-1)
            case RowReduction.LOG_SUM_EXP:
                shift = self.finite_largest_value(values=values)
                total = namespace.sum(namespace.exp(values - shift), axis=-1)
                return self.shifted_logarithm(
                    total=total, shift=namespace.squeeze(shift, axis=-1)
                )

    @staticmethod
    def finite_largest_value(*, values: BackendArray) -> BackendArray:
        """
        :param values: Logarithms of values to add up along the last axis.
        :return: The largest value along the last axis, kept as an axis of length one,
            or zero where every value is ``-inf``, a constant of automatic
            differentiation, as subtracting it and adding it back cancels out.
        """
        namespace = array_namespace(values)
        largest = namespace.max(values, axis=-1, keepdims=True)
        return GradientStop.from_arrays(values).stop_gradient(
            values=namespace.where(namespace.isfinite(largest), largest, 0.0)
        )

    @staticmethod
    def shifted_logarithm(*, total: BackendArray, shift: BackendArray) -> BackendArray:
        """
        :param total: Non-negative sums of the exponentials of values minus ``shift``.
        :param shift: What was subtracted from the logarithms before they were summed.
        :return: The logarithm of the sums plus ``shift``, ``-inf`` where a sum is zero.
        """
        # a zero total's logarithm is taken of one instead to keep it free of warnings
        namespace = array_namespace(total, shift)
        has_mass = total > 0.0
        return namespace.where(
            has_mass,
            namespace.log(namespace.where(has_mass, total, 1.0)) + shift,
            -np.inf,
        )


# %% sparse entries


@dataclass
class SparseEntries:
    """
    Entries of a two dimensional matrix, as one array per coordinate.

    The ``k``-th entry of every array belongs to the ``k``-th entry.
    """

    data: BackendArray
    """
    The value of every entry.
    """

    rows: npt.NDArray[np.int64]
    """
    The row of every entry.
    """

    columns: npt.NDArray[np.int64]
    """
    The column of every entry.
    """

    @classmethod
    def concatenate(cls, parts: list[SparseEntries]) -> Self:
        """
        :param parts: The entries to join, in order.
        :return: All entries of the parts.
        """
        return cls(
            np.concatenate([part.data for part in parts]),
            np.concatenate([part.rows for part in parts]),
            np.concatenate([part.columns for part in parts]),
        )


# %% edge weighting


@dataclass(kw_only=True)
class EdgeWeighting:
    """
    Values along the edges that the value of the child of every edge is joined with
    before the values along the edges of a node are reduced.
    """

    stored: BackendArray
    """
    The values along the edges, as the edge matrix stores them.
    """

    combination: RowReduction
    """
    The reduction whose :meth:`RowReduction.combine` joins the value of a child with the
    value of its edge.
    """


# %% edge positions


@dataclass(kw_only=True)
class EdgePositions:
    """
    The row and the column of every edge of an edge matrix.

    The ``k``-th entry of both arrays belongs to the ``k``-th edge.
    """

    rows: npt.NDArray[np.int64]
    """
    The node of every edge.
    """

    columns: npt.NDArray[np.int64]
    """
    The column of the child node of every edge.
    """


# %% edge matrices


@dataclass(eq=False)
class EdgeMatrix(ABC):
    """
    Which entries of the matrix of shape (#nodes, total number of nodes of the child
    layers) are edges, and how values along the edges of every node are combined.

    The edges are structure, fixed once constructed. Values along the edges are laid out
    as the subclass stores them, which only the methods of the subclass read.
    """

    @property
    @abstractmethod
    def shape(self) -> MatrixShape2D:
        """
        :return: The number of nodes and the total number of nodes of the child layers.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def rows(self) -> npt.NDArray[np.int64]:
        """
        :return: The node of every edge, the edges in the order of this matrix.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def columns(self) -> npt.NDArray[np.int64]:
        """
        :return: The column of the child node of every edge, the edges in the order of
            this matrix.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def stored_shape(self) -> tuple[int, ...]:
        """
        :return: The shape of the values along the edges as this matrix stores them.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_edge_positions(
        cls,
        *,
        rows: npt.NDArray[np.int64],
        columns: npt.NDArray[np.int64],
        shape: MatrixShape2D,
    ) -> Self:
        """
        :param rows: The node of every edge.
        :param columns: The column of the child node of every edge.
        :param shape: The number of nodes and of child nodes.
        :return: The matrix with these edges.
        """
        raise NotImplementedError

    @abstractmethod
    def stored_values(self, *, entries: SparseEntries, padding: float) -> BackendArray:
        """
        :param entries: Values at edges of this matrix, in any order if this matrix
            stores entries that are not edges, in its own order otherwise.
        :param padding: The value of the stored entries that are not edges.
        :return: The values as this matrix stores them, an array of their backend.
        """
        raise NotImplementedError

    @abstractmethod
    def edge_values(self, *, stored: BackendArray) -> EdgeValues:
        """
        :param stored: Values as this matrix stores them.
        :return: The value of every edge, in the order of this matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def values_of_columns(self, *, values: BackendArray) -> BackendArray:
        """
        :param values: One value per column, the columns in the last axis.
        :return: The value of the column of every edge, as this matrix stores values.
        """
        raise NotImplementedError

    @abstractmethod
    def values_of_rows(self, *, values: NodeValues) -> BackendArray:
        """
        :param values: One value per node of a layer.
        :return: The value of the node of every edge, as this matrix stores values.
        """
        raise NotImplementedError

    @abstractmethod
    def reduce_by_row(
        self, *, values: BackendArray, reduction: RowReduction
    ) -> BackendArray:
        """
        Combine the values along the edges of every node, giving the identity of the
        reduction to a node without edges.

        :param values: Values along the edges, as this matrix stores them.
        :param reduction: How the values along the edges of a node are combined.
        :return: The combined values with the nodes in the last axis, an array of the
            backend of ``values``.
        """
        raise NotImplementedError

    def reduce_child_values_by_row(
        self,
        *,
        values: BackendArray,
        reduction: RowReduction,
        weighting: EdgeWeighting | None = None,
    ) -> BackendArray:
        """
        Combine the values of the children of every node, each first joined with the
        value of its edge if a weighting is given, giving the identity of the reduction
        to a node without edges.

        :param values: One value per column, the columns in the last axis.
        :param reduction: How the values along the edges of a node are combined.
        :param weighting: The values along the edges and how they join the values of the
            children.
        :return: The combined values with the nodes in the last axis, an array of the
            backend of ``values``.
        """
        edge_values = self.values_of_columns(values=values)
        if weighting is not None:
            edge_values = weighting.combination.combine(
                array_namespace(values), edge_values, weighting.stored
            )
        return self.reduce_by_row(values=edge_values, reduction=reduction)

    @property
    def number_of_edges(self) -> int:
        """
        :return: The number of edges.
        """
        return len(self.rows)

    @functools.cached_property
    def structure_array_copies(self) -> StructureArrayCopies:
        """
        :return: The copies of the arrays of this matrix on the devices of the values
            they are used with.
        """
        return StructureArrayCopies()

    @functools.cached_property
    def gather(self) -> npt.NDArray[np.int64]:
        """
        The positions of the edges of every node in the order of this matrix, as a
        rectangular index matrix of shape (#nodes, largest number of edges of a node).

        Nodes with fewer edges are padded with the position one past the last edge,
        which :meth:`pad` fills with a neutral value.
        """
        counts = np.bincount(self.rows, minlength=self.shape.number_of_rows)
        width = max(int(counts.max()) if len(counts) else 0, 1)

        result = np.full(
            (self.shape.number_of_rows, width), self.number_of_edges, dtype=np.int64
        )
        # the position of every edge among the edges of its node
        order = np.argsort(self.rows, kind="stable")
        offsets = np.arange(self.number_of_edges) - np.repeat(
            np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
        )
        result[self.rows[order], offsets] = order
        return result

    @staticmethod
    def pad(values: EdgeValues, padding: float) -> EdgeValues:
        """
        Append the slot that :attr:`gather` pads with.

        :param values: One value per edge.
        :param padding: The value of the padding slot.
        :return: The values with one extra entry.
        """
        return np.concatenate([values, [padding]])


@dataclass(eq=False)
class SparseEdgeMatrix(EdgeMatrix):
    """
    An edge matrix that stores one value per edge.

    Reducing values by node gathers the values of the ``k``-th edge of every node with
    more than ``k`` edges at once, so it never pads a node to the one with most edges.
    """

    edge_rows: npt.NDArray[np.int64]
    """
    The node of every edge.
    """

    edge_columns: npt.NDArray[np.int64]
    """
    The column of the child node of every edge.
    """

    matrix_shape: MatrixShape2D
    """
    The number of nodes and of child nodes.
    """

    rows_by_entry_count: npt.NDArray[np.int64] = field(init=False)
    """
    The nodes ordered by their number of edges, the one with most edges first.
    """

    entry_count_ranks: npt.NDArray[np.int64] = field(init=False)
    """
    The position of every node in :attr:`rows_by_entry_count`.
    """

    entry_positions_by_rank: list[npt.NDArray[np.int64]] = field(init=False)
    """
    Element ``k`` lists the position of the ``k``-th edge of every node with more than
    ``k`` edges, the nodes ordered as :attr:`rows_by_entry_count`. For edges of the
    nodes ``[2, 0, 2, 1, 2, 0]``:

    .. code-block:: python

        rows_by_entry_count     = [2, 0, 1, 3]
        entry_positions_by_rank = [[0, 1, 3],  # the first edge of nodes 2, 0 and 1
                                  [2, 5],     # the second edge of nodes 2 and 0
                                  [4]]        # the third edge of node 2
    """

    stores_every_column_once_in_order: bool = field(init=False)
    """
    Whether the ``k``-th edge is in column ``k`` and every column holds one edge, so
    that taking the columns of the edges from a row of values is the identity.
    """

    def __post_init__(self) -> None:
        """
        Derive the order of the nodes by their number of edges and the positions of
        their edges by rank.
        """
        self.edge_rows = np.asarray(self.edge_rows, dtype=np.int64)
        self.edge_columns = np.asarray(self.edge_columns, dtype=np.int64)

        counts = np.bincount(self.edge_rows, minlength=self.matrix_shape.number_of_rows)
        self.rows_by_entry_count = np.argsort(-counts, kind="stable")
        self.entry_count_ranks = np.argsort(self.rows_by_entry_count)

        gather_by_entry_count = self.gather[self.rows_by_entry_count]
        rows_per_rank = (gather_by_entry_count < self.number_of_edges).sum(axis=0)
        self.entry_positions_by_rank = [
            gather_by_entry_count[:number_of_rows, rank]
            for rank, number_of_rows in enumerate(rows_per_rank)
            if number_of_rows > 0
        ]
        self.stores_every_column_once_in_order = len(
            self.edge_columns
        ) == self.matrix_shape.number_of_columns and np.array_equal(
            self.edge_columns, np.arange(len(self.edge_columns))
        )

    @property
    def shape(self) -> MatrixShape2D:
        return self.matrix_shape

    @property
    def rows(self) -> npt.NDArray[np.int64]:
        return self.edge_rows

    @property
    def columns(self) -> npt.NDArray[np.int64]:
        return self.edge_columns

    @property
    def stored_shape(self) -> tuple[int, ...]:
        return (self.number_of_edges,)

    @classmethod
    def from_edge_positions(
        cls,
        *,
        rows: npt.NDArray[np.int64],
        columns: npt.NDArray[np.int64],
        shape: MatrixShape2D,
    ) -> Self:
        return cls(rows, columns, shape)

    def stored_values(self, *, entries: SparseEntries, padding: float) -> BackendArray:
        return entries.data

    def edge_values(self, *, stored: BackendArray) -> EdgeValues:
        return stored

    def values_of_columns(self, *, values: BackendArray) -> BackendArray:
        # usually every column is an edge once and in order, in which case the gather
        # would copy an array with one entry per event per column for nothing
        if self.stores_every_column_once_in_order:
            return values
        return self.structure_array_copies.values_at_positions(
            values=values, positions=self.edge_columns
        )

    def values_of_rows(self, *, values: NodeValues) -> BackendArray:
        return self.structure_array_copies.values_at_positions(
            values=values, positions=self.edge_rows
        )

    def reduce_by_row(
        self, *, values: BackendArray, reduction: RowReduction
    ) -> BackendArray:
        namespace = array_namespace(values)
        batch_shape = tuple(values.shape[:-1])
        result = namespace.full(
            batch_shape + (0,),
            reduction.identity,
            dtype=values.dtype,
            device=array_api_compat.device(values),
        )
        for positions in reversed(self.entry_positions_by_rank):
            entries = self.structure_array_copies.values_at_positions(
                values=values, positions=positions
            )
            number_of_reduced_rows = result.shape[-1]
            result = namespace.concat(
                [
                    reduction.combine(
                        namespace, entries[..., :number_of_reduced_rows], result
                    ),
                    entries[..., number_of_reduced_rows:],
                ],
                axis=-1,
            )
        rows_without_edges = namespace.full(
            batch_shape + (self.matrix_shape.number_of_rows - result.shape[-1],),
            reduction.identity,
            dtype=values.dtype,
            device=array_api_compat.device(values),
        )
        result = namespace.concat([result, rows_without_edges], axis=-1)
        return self.structure_array_copies.values_at_positions(
            values=result, positions=self.entry_count_ranks
        )


@dataclass(eq=False)
class DenseEdgeMatrix(EdgeMatrix):
    """
    An edge matrix that stores a value for every entry of the matrix, edge or not.

    Reducing values by node combines whole rows at once, the entries that are not edges
    replaced by the identity of the reduction.
    """

    is_edge: NodeColumnMask
    """
    Whether each entry of the matrix is an edge.
    """

    @property
    def shape(self) -> MatrixShape2D:
        return MatrixShape2D(
            number_of_rows=self.is_edge.shape[0],
            number_of_columns=self.is_edge.shape[1],
        )

    @functools.cached_property
    def edge_positions(self) -> EdgePositions:
        """
        :return: The row and the column of every edge, row by row.
        """
        rows, columns = np.nonzero(self.is_edge)
        return EdgePositions(
            rows=rows.astype(np.int64), columns=columns.astype(np.int64)
        )

    @property
    def rows(self) -> npt.NDArray[np.int64]:
        return self.edge_positions.rows

    @property
    def columns(self) -> npt.NDArray[np.int64]:
        return self.edge_positions.columns

    @property
    def stored_shape(self) -> tuple[int, ...]:
        return self.shape.to_tuple()

    @classmethod
    def from_edge_positions(
        cls,
        *,
        rows: npt.NDArray[np.int64],
        columns: npt.NDArray[np.int64],
        shape: MatrixShape2D,
    ) -> Self:
        is_edge = np.zeros(shape.to_tuple(), dtype=bool)
        is_edge[
            np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)
        ] = True
        return cls(is_edge)

    def stored_values(self, *, entries: SparseEntries, padding: float) -> BackendArray:
        result = np.full(self.shape.to_tuple(), padding)
        result[
            np.asarray(entries.rows, dtype=np.int64),
            np.asarray(entries.columns, dtype=np.int64),
        ] = np.asarray(entries.data)
        return array_namespace(entries.data).asarray(result)

    def edge_values(self, *, stored: BackendArray) -> EdgeValues:
        return stored[self.rows, self.columns]

    def values_of_columns(self, *, values: BackendArray) -> BackendArray:
        return array_namespace(values).expand_dims(values, axis=-2)

    def values_of_rows(self, *, values: NodeValues) -> BackendArray:
        return array_namespace(values).expand_dims(values, axis=-1)

    def reduce_by_row(
        self, *, values: BackendArray, reduction: RowReduction
    ) -> BackendArray:
        namespace = array_namespace(values)
        masked = namespace.where(
            self.structure_array_copies.placed(array=self.is_edge, values=values),
            values,
            reduction.identity,
        )
        return reduction.reduce_last_axis(namespace, masked)


@dataclass(kw_only=True)
class EdgeBlockStackShape:
    """
    The number of blocks of an :class:`EdgeBlockStack` and the shape of each block.
    """

    number_of_blocks: int
    """
    The number of blocks in the stack.
    """

    rows_per_block: int
    """
    The number of rows of every block.
    """

    columns_per_block: int
    """
    The number of columns of every block.
    """

    @property
    def number_of_entries(self) -> int:
        """
        :return: The number of entries of all blocks together.
        """
        return self.number_of_blocks * self.rows_per_block * self.columns_per_block

    def to_tuple(self) -> tuple[int, int, int]:
        """
        :return: The shape as array libraries take it.
        """
        return self.number_of_blocks, self.rows_per_block, self.columns_per_block


@dataclass(kw_only=True)
class CrossProductRanges:
    """
    Groups of equally long ranges of columns, one group after another without gaps,
    whose blocks each have one row that combines one column of every range of a group,
    every combination once.
    """

    first_column: int
    """
    The first column of the first range of the first group.
    """

    number_of_groups: int
    """
    The number of groups of ranges.
    """

    ranges_per_group: int
    """
    The number of ranges in every group, the columns every block combines.
    """

    columns_per_range: int
    """
    The number of columns of every range.
    """

    @property
    def number_of_columns(self) -> int:
        """
        :return: The number of columns of all ranges together.
        """
        return self.number_of_groups * self.ranges_per_group * self.columns_per_range

    @property
    def combinations_per_group(self) -> int:
        """
        :return: The number of blocks of every group.
        """
        return self.columns_per_range**self.ranges_per_group

    @property
    def block_columns(self) -> npt.NDArray[np.int64]:
        """
        :return: The columns of every block, shape (#blocks, :attr:`ranges_per_group`),
            the blocks in the order of their group and then of their columns.
        """
        grids = np.meshgrid(
            *[np.arange(self.columns_per_range)] * self.ranges_per_group, indexing="ij"
        )
        offsets = np.stack(
            [
                range_index * self.columns_per_range + grid.reshape(-1)
                for range_index, grid in enumerate(grids)
            ],
            axis=-1,
        )
        group_starts = (
            self.first_column
            + np.arange(self.number_of_groups)
            * self.ranges_per_group
            * self.columns_per_range
        )
        return (group_starts[:, None, None] + offsets[None]).reshape(
            -1, self.ranges_per_group
        )

    @classmethod
    def from_block_columns(
        cls, *, block_columns: npt.NDArray[np.int64]
    ) -> CrossProductRanges | None:
        """
        :param block_columns: The columns of every block of one row, shape (#blocks,
            #columns of a block).
        :return: The ranges of at least two columns whose cross products the blocks
            are, in the order of :attr:`block_columns`, or ``None`` if they are none.
        """
        number_of_blocks, ranges_per_group = block_columns.shape
        if ranges_per_group < 2:
            return None
        first_column = int(block_columns.min())
        width = int(block_columns.max()) - first_column + 1
        columns_per_range = round(
            (number_of_blocks * ranges_per_group / width)
            ** (1 / (ranges_per_group - 1))
        )
        if columns_per_range < 2 or width % (ranges_per_group * columns_per_range):
            return None
        candidate = cls(
            first_column=first_column,
            number_of_groups=width // (ranges_per_group * columns_per_range),
            ranges_per_group=ranges_per_group,
            columns_per_range=columns_per_range,
        )
        if (
            candidate.number_of_groups * candidate.combinations_per_group
            != number_of_blocks
            or not np.array_equal(candidate.block_columns, block_columns)
        ):
            return None
        return candidate

    def combine_ranges(
        self, *, values: BackendArray, reduction: RowReduction
    ) -> BackendArray:
        """
        :param values: One value per column, the columns in the last axis.
        :param reduction: How the values of the columns of a block are combined.
        :return: The combined values of every block in the order of
            :attr:`block_columns`, in the last axis.
        """
        namespace = array_namespace(values)
        batch_shape = tuple(values.shape[:-1])
        ranges = namespace.reshape(
            values[..., self.first_column : self.first_column + self.number_of_columns],
            batch_shape
            + (self.number_of_groups, self.ranges_per_group, self.columns_per_range),
        )
        combined = ranges[..., 0, :]
        for range_index in range(1, self.ranges_per_group):
            combined = namespace.reshape(
                reduction.combine(
                    namespace,
                    namespace.expand_dims(combined, axis=-1),
                    namespace.expand_dims(ranges[..., range_index, :], axis=-2),
                ),
                batch_shape
                + (
                    self.number_of_groups,
                    self.columns_per_range ** (range_index + 1),
                ),
            )
        return namespace.reshape(
            combined,
            batch_shape + (self.number_of_groups * self.combinations_per_group,),
        )


@dataclass(kw_only=True)
class EdgeBlockStack:
    """
    Full blocks of one shape of an edge matrix: every row of a block has an edge to
    every column of the block.
    """

    block_rows: npt.NDArray[np.int64]
    """
    The rows of every block, shape (#blocks, #rows of a block).
    """

    block_columns: npt.NDArray[np.int64]
    """
    The columns of every block, shape (#blocks, #columns of a block).
    """

    first_position: int
    """
    The position of the first edge of this stack among the edges of the matrix.
    """

    @property
    def stack_shape(self) -> EdgeBlockStackShape:
        """
        :return: The number of blocks and the shape of each block.
        """
        return EdgeBlockStackShape(
            number_of_blocks=int(self.block_rows.shape[0]),
            rows_per_block=int(self.block_rows.shape[1]),
            columns_per_block=int(self.block_columns.shape[1]),
        )

    @property
    def number_of_edges(self) -> int:
        """
        :return: The number of edges of all blocks of this stack.
        """
        return self.stack_shape.number_of_entries

    @property
    def edge_rows(self) -> npt.NDArray[np.int64]:
        """
        :return: The row of every edge, block by block and row by row.
        """
        return np.repeat(
            self.block_rows, self.stack_shape.columns_per_block, axis=1
        ).reshape(-1)

    @property
    def edge_columns(self) -> npt.NDArray[np.int64]:
        """
        :return: The column of every edge, block by block and row by row.
        """
        return np.tile(
            self.block_columns, (1, self.stack_shape.rows_per_block)
        ).reshape(-1)

    @functools.cached_property
    def has_contiguous_columns(self) -> bool:
        """
        :return: Whether the columns of all blocks form one range without gaps, those of
            every block following those of the block before it.
        """
        first_column = self.block_columns[0, 0]
        return bool(
            np.array_equal(
                self.block_columns,
                first_column
                + np.arange(self.block_columns.size).reshape(self.block_columns.shape),
            )
        )

    def block_column_values(self, *, values: BackendArray) -> BackendArray:
        """
        :param values: One value per column, the columns in the last axis.
        :return: The values of the columns of every block, an array of the backend of
            ``values`` with shape (..., #blocks, #columns of a block).
        """
        namespace = array_namespace(values)
        block_shape = tuple(values.shape[:-1]) + self.block_columns.shape
        if not self.has_contiguous_columns:
            return namespace.reshape(
                self.structure_array_copies.values_at_positions(
                    values=values, positions=self.block_columns
                ),
                block_shape,
            )
        first_column = int(self.block_columns[0, 0])
        return namespace.reshape(
            values[..., first_column : first_column + self.block_columns.size],
            block_shape,
        )

    @functools.cached_property
    def structure_array_copies(self) -> StructureArrayCopies:
        """
        :return: The copies of the columns of the blocks on the devices of the values
            they index.
        """
        return StructureArrayCopies()

    @functools.cached_property
    def cross_product_ranges(self) -> CrossProductRanges | None:
        """
        :return: The ranges whose cross products the blocks of this stack are, ``None``
            if a block has several rows or the blocks are no such cross products.
        """
        if self.stack_shape.rows_per_block != 1:
            return None
        return CrossProductRanges.from_block_columns(block_columns=self.block_columns)

    def reduce_child_values_by_row(
        self,
        *,
        values: BackendArray,
        reduction: RowReduction,
        weighting: EdgeWeighting | None = None,
    ) -> BackendArray:
        """
        :meth:`EdgeMatrix.reduce_child_values_by_row` for the rows of this stack only.

        :return: The combined values of the rows of every block, block by block, in the
            last axis.
        """
        namespace = array_namespace(values)
        batch_shape = tuple(values.shape[:-1])
        shape = self.stack_shape
        if weighting is None and self.cross_product_ranges is not None:
            return self.cross_product_ranges.combine_ranges(
                values=values, reduction=reduction
            )
        children = self.block_column_values(values=values)
        if weighting is None:
            combined = namespace.broadcast_to(
                namespace.expand_dims(
                    reduction.reduce_last_axis(namespace, children), axis=-1
                ),
                batch_shape + (shape.number_of_blocks, shape.rows_per_block),
            )
        else:
            combined = self.weighted_reduction(
                children=children,
                edge_values=namespace.reshape(
                    weighting.stored[
                        self.first_position : self.first_position + self.number_of_edges
                    ],
                    shape.to_tuple(),
                ),
                reduction=reduction,
                combination=weighting.combination,
            )
        return namespace.reshape(
            combined,
            batch_shape + (shape.number_of_blocks * shape.rows_per_block,),
        )

    def weighted_reduction(
        self,
        *,
        children: BackendArray,
        edge_values: BackendArray,
        reduction: RowReduction,
        combination: RowReduction,
    ) -> BackendArray:
        """
        :param children: The values of the columns of every block, shape (...,
            #blocks, #columns of a block).
        :param edge_values: The value of every edge, the shape of this stack.
        :param reduction: How the joined values along the edges of a row are combined.
        :param combination: How the value of a child joins the value of its edge.
        :return: The combined values of every row, shape (..., #blocks, #rows of a
            block).
        """
        namespace = array_namespace(children)
        match reduction, combination:
            case RowReduction.SUM, RowReduction.PRODUCT:
                return self.weighted_sums(children=children, weights=edge_values)
            case RowReduction.LOG_SUM_EXP, RowReduction.SUM:
                return self.log_weighted_sums(
                    children=children, log_weights=edge_values
                )
        return reduction.reduce_last_axis(
            namespace,
            combination.combine(
                namespace, namespace.expand_dims(children, axis=-2), edge_values
            ),
        )

    def weighted_sums(
        self, *, children: BackendArray, weights: BackendArray
    ) -> BackendArray:
        """
        :param children: The values of the columns of every block, shape (...,
            #blocks, #columns of a block).
        :param weights: The weight of every edge, the shape of this stack.
        :return: The sum of the weighted values of the children of every row, shape
            (..., #blocks, #rows of a block).
        """
        namespace = array_namespace(children)
        batch_shape = tuple(children.shape[:-2])
        number_of_blocks, rows_per_block, columns_per_block = (
            self.stack_shape.to_tuple()
        )
        blocks_first = namespace.permute_dims(
            namespace.reshape(
                children,
                (math.prod(batch_shape), number_of_blocks, columns_per_block),
            ),
            (1, 0, 2),
        )
        sums = namespace.matmul(
            blocks_first, namespace.permute_dims(weights, (0, 2, 1))
        )
        return namespace.reshape(
            namespace.permute_dims(sums, (1, 0, 2)),
            batch_shape + (number_of_blocks, rows_per_block),
        )

    def log_weighted_sums(
        self, *, children: BackendArray, log_weights: BackendArray
    ) -> BackendArray:
        """
        :param children: The logarithms of the values of the columns of every block,
            shape (..., #blocks, #columns of a block).
        :param log_weights: The logarithm of the weight of every edge, the shape of
            this stack.
        :return: The logarithm of the sum of the weighted values of the children of
            every row, shape (..., #blocks, #rows of a block).

        .. warning::
            A row whose children and weights both spread over more than the exponent
            range of their floating point type can come out as ``-inf``.
        """
        namespace = array_namespace(children)
        child_shift = RowReduction.finite_largest_value(values=children)
        weight_shift = RowReduction.finite_largest_value(values=log_weights)
        total = self.weighted_sums(
            children=namespace.exp(children - child_shift),
            weights=namespace.exp(log_weights - weight_shift),
        )
        return RowReduction.shifted_logarithm(
            total=total, shift=child_shift + namespace.squeeze(weight_shift, axis=-1)
        )


@dataclass(eq=False, kw_only=True)
class BlockEdgeMatrix(EdgeMatrix):
    """
    An edge matrix whose rows with the same child columns form full blocks, the blocks
    of one shape stacked.

    Reducing by node takes one batched operation per shape of block, which suits layers
    whose nodes each mix all nodes of a region, such as the folded sums of a RAT-SPN.
    """

    stacks: list[EdgeBlockStack]
    """
    The blocks of every shape, their edges in this order.
    """

    matrix_shape: MatrixShape2D
    """
    The number of nodes and of child nodes.
    """

    @property
    def shape(self) -> MatrixShape2D:
        return self.matrix_shape

    @functools.cached_property
    def edge_positions(self) -> EdgePositions:
        """
        :return: The row and the column of every edge, stack by stack.
        """
        if not self.stacks:
            return EdgePositions(
                rows=np.zeros(0, dtype=np.int64), columns=np.zeros(0, dtype=np.int64)
            )
        return EdgePositions(
            rows=np.concatenate([stack.edge_rows for stack in self.stacks]),
            columns=np.concatenate([stack.edge_columns for stack in self.stacks]),
        )

    @property
    def rows(self) -> npt.NDArray[np.int64]:
        return self.edge_positions.rows

    @property
    def columns(self) -> npt.NDArray[np.int64]:
        return self.edge_positions.columns

    @property
    def stored_shape(self) -> tuple[int, ...]:
        return (self.number_of_edges,)

    @functools.cached_property
    def node_positions(self) -> npt.NDArray[np.int64]:
        """
        :return: The position of every node in the result of reducing the stacks one
            after another, followed by the nodes without edges.
        """
        reduced_rows = [stack.block_rows.reshape(-1) for stack in self.stacks]
        nodes_with_edges = (
            np.concatenate(reduced_rows) if reduced_rows else np.zeros(0, np.int64)
        )
        nodes_without_edges = np.setdiff1d(
            np.arange(self.matrix_shape.number_of_rows), nodes_with_edges
        )
        order = np.concatenate([nodes_with_edges, nodes_without_edges])
        return np.argsort(order).astype(np.int64)

    @classmethod
    def from_edge_positions(
        cls,
        *,
        rows: npt.NDArray[np.int64],
        columns: npt.NDArray[np.int64],
        shape: MatrixShape2D,
    ) -> Self:
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        order = np.lexsort((columns, rows))
        counts = np.bincount(rows, minlength=shape.number_of_rows)
        columns_of_rows = np.split(columns[order], np.cumsum(counts)[:-1])

        rows_by_columns: dict[tuple[int, ...], list[int]] = {}
        for row, row_columns in enumerate(columns_of_rows):
            if len(row_columns):
                rows_by_columns.setdefault(tuple(row_columns.tolist()), []).append(row)

        blocks_by_shape: dict[
            MatrixShape2D, list[tuple[list[int], tuple[int, ...]]]
        ] = {}
        for block_columns, block_rows in rows_by_columns.items():
            block_shape = MatrixShape2D(
                number_of_rows=len(block_rows), number_of_columns=len(block_columns)
            )
            blocks_by_shape.setdefault(block_shape, []).append(
                (block_rows, block_columns)
            )

        stacks = []
        first_position = 0
        for blocks in blocks_by_shape.values():
            stack = EdgeBlockStack(
                block_rows=np.array([rows for rows, _ in blocks], dtype=np.int64),
                block_columns=np.array([cols for _, cols in blocks], dtype=np.int64),
                first_position=first_position,
            )
            stacks.append(stack)
            first_position += stack.number_of_edges
        return cls(stacks=stacks, matrix_shape=shape)

    def stored_values(self, *, entries: SparseEntries, padding: float) -> BackendArray:
        # the value of an edge is found by its key row * #columns + column
        width = self.matrix_shape.number_of_columns
        entry_keys = np.asarray(entries.rows, dtype=np.int64) * width + np.asarray(
            entries.columns, dtype=np.int64
        )
        order = np.argsort(entry_keys)
        positions = order[
            np.searchsorted(entry_keys[order], self.rows * width + self.columns)
        ]
        namespace = array_namespace(entries.data)
        return namespace.take(
            entries.data,
            namespace.asarray(positions, device=array_api_compat.device(entries.data)),
        )

    def edge_values(self, *, stored: BackendArray) -> EdgeValues:
        return stored

    def values_of_columns(self, *, values: BackendArray) -> BackendArray:
        return self.structure_array_copies.values_at_positions(
            values=values, positions=self.columns
        )

    def values_of_rows(self, *, values: NodeValues) -> BackendArray:
        return self.structure_array_copies.values_at_positions(
            values=values, positions=self.rows
        )

    def reduce_by_row(
        self, *, values: BackendArray, reduction: RowReduction
    ) -> BackendArray:
        namespace = array_namespace(values)
        batch_shape = tuple(values.shape[:-1])
        reduced = []
        for stack in self.stacks:
            stack_shape = stack.stack_shape
            blocks = namespace.reshape(
                values[
                    ...,
                    stack.first_position : stack.first_position + stack.number_of_edges,
                ],
                batch_shape + stack_shape.to_tuple(),
            )
            reduced.append(
                namespace.reshape(
                    reduction.reduce_last_axis(namespace, blocks),
                    batch_shape
                    + (stack_shape.number_of_blocks * stack_shape.rows_per_block,),
                )
            )
        return self.nodes_in_order(reduced=reduced, reduction=reduction, values=values)

    def reduce_child_values_by_row(
        self,
        *,
        values: BackendArray,
        reduction: RowReduction,
        weighting: EdgeWeighting | None = None,
    ) -> BackendArray:
        """
        Read the values of the children of every block once and combine them one stack
        at a time, see :meth:`EdgeMatrix.reduce_child_values_by_row`.
        """
        return self.nodes_in_order(
            reduced=[
                stack.reduce_child_values_by_row(
                    values=values, reduction=reduction, weighting=weighting
                )
                for stack in self.stacks
            ],
            reduction=reduction,
            values=values,
        )

    @functools.cached_property
    def reduces_nodes_in_order(self) -> bool:
        """
        :return: Whether reducing the stacks one after another puts every node at its
            own position already.
        """
        return bool(
            np.array_equal(
                self.node_positions, np.arange(self.matrix_shape.number_of_rows)
            )
        )

    def nodes_in_order(
        self,
        *,
        reduced: list[BackendArray],
        reduction: RowReduction,
        values: BackendArray,
    ) -> BackendArray:
        """
        :param reduced: The reduced values of the nodes of every stack, the stacks in
            order.
        :param reduction: The reduction, whose identity the nodes without edges get.
        :param values: The values that were reduced.
        :return: The reduced values of all nodes in the order of the nodes, an array of
            the backend, type and device of ``values``.
        """
        namespace = array_namespace(values)
        number_without_edges = self.matrix_shape.number_of_rows - sum(
            part.shape[-1] for part in reduced
        )
        if number_without_edges or not reduced:
            reduced = reduced + [
                namespace.full(
                    tuple(values.shape[:-1]) + (number_without_edges,),
                    reduction.identity,
                    dtype=values.dtype,
                    device=array_api_compat.device(values),
                )
            ]
        result = reduced[0] if len(reduced) == 1 else namespace.concat(reduced, axis=-1)
        if self.reduces_nodes_in_order:
            return result
        return self.structure_array_copies.values_at_positions(
            values=result, positions=self.node_positions
        )
