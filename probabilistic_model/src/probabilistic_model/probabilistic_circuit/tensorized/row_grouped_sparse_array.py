from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array
from typing_extensions import Any, Self


@dataclass
class SparseEntries:
    """
    The stored entries of a two dimensional sparse array, as one array per coordinate.

    The ``k``-th entry of every array belongs to the ``k``-th stored entry.
    """

    data: npt.NDArray
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

    def to_coo_array(self, shape: tuple[int, int]) -> coo_array:
        """
        :param shape: The shape of the dense array.
        :return: The sparse array that stores these entries.
        """
        return coo_array(
            (
                np.asarray(self.data),
                (
                    np.asarray(self.rows, dtype=np.int64),
                    np.asarray(self.columns, dtype=np.int64),
                ),
            ),
            shape=shape,
        )


@dataclass(eq=False)
class RowGroupedSparsity:
    """
    Which entries of a two dimensional sparse array are stored, with the index arrays
    that reduce per-entry values row by row.

    The sparsity is structure: it is fixed once constructed and every index array is
    computed at construction, so reducing values never computes an index.
    """

    rows: npt.NDArray[np.int64]
    """
    The row of every stored entry.
    """

    columns: npt.NDArray[np.int64]
    """
    The column of every stored entry.
    """

    shape: tuple[int, int]
    """
    The shape of the dense array.
    """

    gather: npt.NDArray[np.int64] = field(init=False)
    """
    The positions of the stored entries of every row, as a rectangular index matrix of
    shape (#rows, largest number of entries of a row).

    Rows with fewer entries are padded with the position one past the last entry, which
    :meth:`pad` fills with a neutral value.
    """

    rows_by_entry_count: npt.NDArray[np.int64] = field(init=False)
    """
    The rows ordered by their number of stored entries, the fullest first.
    """

    entry_count_ranks: npt.NDArray[np.int64] = field(init=False)
    """
    The position of every row in :attr:`rows_by_entry_count`.
    """

    entry_positions_by_rank: list[npt.NDArray[np.int64]] = field(init=False)
    """
    Element ``k`` lists the position of the ``k``-th entry of every row with more than
    ``k`` entries, the rows ordered as :attr:`rows_by_entry_count`. For entries stored
    in rows ``[2, 0, 2, 1, 2, 0]``:

    .. code-block:: python

        rows_by_entry_count     = [2, 0, 1, 3]
        entry_positions_by_rank = [[0, 1, 3],  # the first entry of rows 2, 0 and 1
                                   [2, 5],     # the second entry of rows 2 and 0
                                   [4]]        # the third entry of row 2
    """

    stores_every_column_once_in_order: bool = field(init=False)
    """
    Whether the ``k``-th stored entry is in column ``k`` and every column holds one
    entry, so that taking the stored columns of a row of values is the identity.
    """

    def __post_init__(self):
        self.rows = np.asarray(self.rows, dtype=np.int64)
        self.columns = np.asarray(self.columns, dtype=np.int64)
        self.shape = (int(self.shape[0]), int(self.shape[1]))

        counts = np.bincount(self.rows, minlength=self.shape[0])
        self.gather = self.gather_of_rows(counts)
        self.rows_by_entry_count = np.argsort(-counts, kind="stable")
        self.entry_count_ranks = np.argsort(self.rows_by_entry_count)

        gather_by_entry_count = self.gather[self.rows_by_entry_count]
        rows_per_rank = (gather_by_entry_count < self.number_of_stored_entries).sum(
            axis=0
        )
        self.entry_positions_by_rank = [
            gather_by_entry_count[:number_of_rows, rank]
            for rank, number_of_rows in enumerate(rows_per_rank)
            if number_of_rows > 0
        ]
        self.stores_every_column_once_in_order = len(self.columns) == self.shape[
            1
        ] and np.array_equal(self.columns, np.arange(len(self.columns)))

    def gather_of_rows(self, counts: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """
        :param counts: The number of stored entries of every row.
        :return: The value of :attr:`gather`.
        """
        number_of_entries = len(self.rows)
        width = max(int(counts.max()) if len(counts) else 0, 1)

        gather = np.full((self.shape[0], width), number_of_entries, dtype=np.int64)
        # the position of every entry inside its row
        order = np.argsort(self.rows, kind="stable")
        offsets = np.arange(number_of_entries) - np.repeat(
            np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
        )
        gather[self.rows[order], offsets] = order
        return gather

    @property
    def number_of_stored_entries(self) -> int:
        """
        :return: The number of entries the array stores.
        """
        return len(self.rows)

    def values_of_entries(self, values: npt.NDArray) -> npt.NDArray:
        """
        :param values: One value per column, the columns in the last axis.
        :return: The value of the column of every stored entry, the entries in the last
            axis.
        """
        # usually every column is stored once and in order, in which case the gather
        # would copy an array with one entry per event per column for nothing
        if self.stores_every_column_once_in_order:
            return values
        return values[..., self.columns]

    def pad(self, values: npt.NDArray, padding: float) -> npt.NDArray:
        """
        Append the slot that :attr:`gather` pads with.

        :param values: Per-entry values with the entries in the last axis.
        :param padding: The value of the padding slot. ``-inf`` is neutral for a
            logarithmic reduction, ``0`` for a linear one.
        :return: The values with one extra entry in the last axis.
        """
        return np.concatenate(
            [values, np.full(values.shape[:-1] + (1,), padding)], axis=-1
        )

    def reduce_by_row(self, values: npt.NDArray, operation: np.ufunc) -> npt.NDArray:
        """
        Combine the entries of every row with ``operation``, giving its identity for a
        row without entries, without padding any row to the fullest one.

        :param values: Per-entry values with the entries in the last axis.
        :param operation: An associative and commutative numpy universal function of two
            arrays, applied element by element, that has an identity.
        :return: The values with shape ``(..., #rows)``.
        """
        result = np.full(values.shape[:-1] + (0,), operation.identity, dtype=float)
        for positions in reversed(self.entry_positions_by_rank):
            entries = values[..., positions]
            number_of_reduced_rows = result.shape[-1]
            result = np.concatenate(
                [
                    operation(entries[..., :number_of_reduced_rows], result),
                    entries[..., number_of_reduced_rows:],
                ],
                axis=-1,
            )
        rows_without_entries = np.full(
            values.shape[:-1] + (self.shape[0] - result.shape[-1],),
            operation.identity,
            dtype=float,
        )
        result = np.concatenate([result, rows_without_entries], axis=-1)
        return result[..., self.entry_count_ranks]


@dataclass(eq=False)
class RowGroupedSparseArray:
    """
    A two dimensional sparse array whose stored entries can be reduced row by row.

    The values of the stored entries are the only part that may change; which entries
    are stored is fixed by :attr:`sparsity`.
    """

    data: npt.NDArray
    """
    The value of every stored entry.
    """

    sparsity: RowGroupedSparsity
    """
    Which entries are stored.
    """

    @classmethod
    def from_entries(cls, entries: SparseEntries, shape: tuple[int, int]) -> Self:
        """
        :param entries: The stored entries.
        :param shape: The shape of the dense array.
        :return: The sparse array.
        """
        return cls(
            np.asarray(entries.data),
            RowGroupedSparsity(entries.rows, entries.columns, shape),
        )

    def with_data(self, data: npt.NDArray) -> Self:
        """
        :param data: The new value of every stored entry.
        :return: A sparse array with the entries of this one and the given values.
        """
        return self.__class__(data, self.sparsity)

    def copy(self) -> Self:
        """
        :return: A copy of this array whose values share no memory with it.
        """
        return self.with_data(self.data.copy())

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self.copy()
