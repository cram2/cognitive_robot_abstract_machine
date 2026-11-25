"""
The edges of an inner layer of a layered circuit, as the entries of a matrix whose rows
are the nodes of the layer and whose columns are the nodes of its child layers, and the
reductions that combine values along the edges of every node.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType

import numpy as np
import numpy.typing as npt
import array_api_compat
from array_api_compat import array_namespace
from typing_extensions import Self

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
                largest = namespace.max(values, axis=-1, keepdims=True)
                # a row of only -inf has no finite value to shift by and sums to zero,
                # whose logarithm is taken of one instead to keep it free of warnings
                shift = namespace.where(namespace.isfinite(largest), largest, 0.0)
                total = namespace.sum(namespace.exp(values - shift), axis=-1)
                has_mass = total > 0.0
                return namespace.where(
                    has_mass,
                    namespace.log(namespace.where(has_mass, total, 1.0))
                    + namespace.squeeze(shift, axis=-1),
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


# %% edge matrices


@dataclass(eq=False)
class EdgeMatrix(ABC):
    """
    Which entries of the matrix of shape (#nodes, total number of nodes of the child
    layers) are edges, and how values along the edges of every node are combined.

    The edges are structure, fixed once constructed. Values along the edges are laid out
    as the subclass stores them, which only the methods of the subclass read.

    .. note::
        Values are gathered by indexing with numpy integer arrays, which numpy, jax and
        torch support although the Array API standard does not require it.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
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
        rows: npt.NDArray[np.int64],
        columns: npt.NDArray[np.int64],
        shape: tuple[int, int],
    ) -> Self:
        """
        :param rows: The node of every edge.
        :param columns: The column of the child node of every edge.
        :param shape: The number of nodes and of child nodes.
        :return: The matrix with these edges.
        """
        raise NotImplementedError

    @abstractmethod
    def stored_values(self, entries: SparseEntries, padding: float) -> BackendArray:
        """
        :param entries: Values at edges of this matrix, in any order if this matrix
            stores entries that are not edges, in its own order otherwise.
        :param padding: The value of the stored entries that are not edges.
        :return: The values as this matrix stores them, an array of their backend.
        """
        raise NotImplementedError

    @abstractmethod
    def edge_values(self, stored: BackendArray) -> EdgeValues:
        """
        :param stored: Values as this matrix stores them.
        :return: The value of every edge, in the order of this matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def values_of_columns(self, values: BackendArray) -> BackendArray:
        """
        :param values: One value per column, the columns in the last axis.
        :return: The value of the column of every edge, as this matrix stores values.
        """
        raise NotImplementedError

    @abstractmethod
    def values_of_rows(self, values: NodeValues) -> BackendArray:
        """
        :param values: One value per node of a layer.
        :return: The value of the node of every edge, as this matrix stores values.
        """
        raise NotImplementedError

    @abstractmethod
    def reduce_by_row(
        self, values: BackendArray, reduction: RowReduction
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

    @property
    def number_of_edges(self) -> int:
        """
        :return: The number of edges.
        """
        return len(self.rows)

    @functools.cached_property
    def gather(self) -> npt.NDArray[np.int64]:
        """
        The positions of the edges of every node in the order of this matrix, as a
        rectangular index matrix of shape (#nodes, largest number of edges of a node).

        Nodes with fewer edges are padded with the position one past the last edge,
        which :meth:`pad` fills with a neutral value.
        """
        counts = np.bincount(self.rows, minlength=self.shape[0])
        width = max(int(counts.max()) if len(counts) else 0, 1)

        result = np.full((self.shape[0], width), self.number_of_edges, dtype=np.int64)
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

    matrix_shape: tuple[int, int]
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

    def __post_init__(self):
        self.edge_rows = np.asarray(self.edge_rows, dtype=np.int64)
        self.edge_columns = np.asarray(self.edge_columns, dtype=np.int64)
        self.matrix_shape = (int(self.matrix_shape[0]), int(self.matrix_shape[1]))

        counts = np.bincount(self.edge_rows, minlength=self.matrix_shape[0])
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
        ) == self.matrix_shape[1] and np.array_equal(
            self.edge_columns, np.arange(len(self.edge_columns))
        )

    @property
    def shape(self) -> tuple[int, int]:
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
        rows: npt.NDArray[np.int64],
        columns: npt.NDArray[np.int64],
        shape: tuple[int, int],
    ) -> Self:
        return cls(rows, columns, shape)

    def stored_values(self, entries: SparseEntries, padding: float) -> BackendArray:
        return entries.data

    def edge_values(self, stored: BackendArray) -> EdgeValues:
        return stored

    def values_of_columns(self, values: BackendArray) -> BackendArray:
        # usually every column is an edge once and in order, in which case the gather
        # would copy an array with one entry per event per column for nothing
        if self.stores_every_column_once_in_order:
            return values
        return values[..., self.edge_columns]

    def values_of_rows(self, values: NodeValues) -> BackendArray:
        return values[..., self.edge_rows]

    def reduce_by_row(
        self, values: BackendArray, reduction: RowReduction
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
            entries = values[..., positions]
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
            batch_shape + (self.matrix_shape[0] - result.shape[-1],),
            reduction.identity,
            dtype=values.dtype,
            device=array_api_compat.device(values),
        )
        result = namespace.concat([result, rows_without_edges], axis=-1)
        return result[..., self.entry_count_ranks]


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
    def shape(self) -> tuple[int, int]:
        return (int(self.is_edge.shape[0]), int(self.is_edge.shape[1]))

    @functools.cached_property
    def edge_positions(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """
        :return: The row and the column of every edge, row by row.
        """
        rows, columns = np.nonzero(self.is_edge)
        return rows.astype(np.int64), columns.astype(np.int64)

    @property
    def rows(self) -> npt.NDArray[np.int64]:
        return self.edge_positions[0]

    @property
    def columns(self) -> npt.NDArray[np.int64]:
        return self.edge_positions[1]

    @property
    def stored_shape(self) -> tuple[int, ...]:
        return self.shape

    @classmethod
    def from_edge_positions(
        cls,
        rows: npt.NDArray[np.int64],
        columns: npt.NDArray[np.int64],
        shape: tuple[int, int],
    ) -> Self:
        is_edge = np.zeros((int(shape[0]), int(shape[1])), dtype=bool)
        is_edge[
            np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)
        ] = True
        return cls(is_edge)

    def stored_values(self, entries: SparseEntries, padding: float) -> BackendArray:
        result = np.full(self.shape, padding)
        result[
            np.asarray(entries.rows, dtype=np.int64),
            np.asarray(entries.columns, dtype=np.int64),
        ] = np.asarray(entries.data)
        return array_namespace(entries.data).asarray(result)

    def edge_values(self, stored: BackendArray) -> EdgeValues:
        return stored[self.rows, self.columns]

    def values_of_columns(self, values: BackendArray) -> BackendArray:
        return array_namespace(values).expand_dims(values, axis=-2)

    def values_of_rows(self, values: NodeValues) -> BackendArray:
        return array_namespace(values).expand_dims(values, axis=-1)

    def reduce_by_row(
        self, values: BackendArray, reduction: RowReduction
    ) -> BackendArray:
        namespace = array_namespace(values)
        masked = namespace.where(
            namespace.asarray(self.is_edge, device=array_api_compat.device(values)),
            values,
            reduction.identity,
        )
        return reduction.reduce_last_axis(namespace, masked)
