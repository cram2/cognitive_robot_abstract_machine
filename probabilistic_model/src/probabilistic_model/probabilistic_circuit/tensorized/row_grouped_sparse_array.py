from __future__ import annotations

import functools
from dataclasses import dataclass

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
class RowGroupedSparseArray:
    """
    A two dimensional sparse array whose stored entries can be reduced row by row.
    """

    array: coo_array
    """
    The stored entries in coordinate format.

    The rows and columns of the entries are fixed; only their values may change.
    """

    @classmethod
    def from_entries(cls, entries: SparseEntries, shape: tuple[int, int]) -> Self:
        """
        :param entries: The stored entries.
        :param shape: The shape of the dense array.
        :return: The sparse array.
        """
        return cls(entries.to_coo_array(shape))

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape

    @property
    def number_of_stored_entries(self) -> int:
        return self.array.nnz

    @property
    def rows(self) -> npt.NDArray[np.int64]:
        """
        :return: The row of every stored entry.
        """
        return self.array.row

    @property
    def columns(self) -> npt.NDArray[np.int64]:
        """
        :return: The column of every stored entry.
        """
        return self.array.col

    @property
    def data(self) -> npt.NDArray:
        """
        :return: The value of every stored entry.
        """
        return self.array.data

    @data.setter
    def data(self, value: npt.NDArray):
        self.array.data = value

    @functools.cached_property
    def gather(self) -> npt.NDArray[np.int64]:
        """
        The positions of the stored entries of every row, as a rectangular index matrix
        of shape (#rows, largest number of entries of a row).

        Rows with fewer entries are padded with the position one past the last entry,
        which :meth:`pad` fills with a neutral value.
        """
        rows = self.rows
        number_of_entries = len(rows)
        counts = np.bincount(rows, minlength=self.shape[0])
        width = max(int(counts.max()) if len(counts) else 0, 1)

        gather = np.full((self.shape[0], width), number_of_entries, dtype=np.int64)
        # the position of every entry inside its row
        order = np.argsort(rows, kind="stable")
        offsets = np.arange(number_of_entries) - np.repeat(
            np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
        )
        gather[rows[order], offsets] = order
        return gather

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

    @functools.cached_property
    def rows_by_entry_count(self) -> npt.NDArray[np.int64]:
        """
        The rows ordered by their number of stored entries, the fullest first.
        """
        counts = np.bincount(self.rows, minlength=self.shape[0])
        return np.argsort(-counts, kind="stable")

    @functools.cached_property
    def entry_count_ranks(self) -> npt.NDArray[np.int64]:
        """
        The position of every row in :attr:`rows_by_entry_count`.
        """
        return np.argsort(self.rows_by_entry_count)

    @functools.cached_property
    def entry_positions_by_rank(self) -> list[npt.NDArray[np.int64]]:
        """
        Element ``k`` holds, for every row with more than ``k`` stored entries, the
        position of that row's ``k``-th entry among the stored entries. The rows follow
        :attr:`rows_by_entry_count`, so the rows of each element are a prefix of the rows
        of the element before it. For stored entries in rows ``[2, 0, 2, 1, 2, 0]``::

            rows_by_entry_count     = [2, 0, 1, 3]
            entry_positions_by_rank = [[0, 1, 3], [2, 5], [4]]
        """
        gather = self.gather[self.rows_by_entry_count]
        rows_per_rank = (gather < self.number_of_stored_entries).sum(axis=0)
        return [
            gather[:number_of_rows, rank]
            for rank, number_of_rows in enumerate(rows_per_rank)
            if number_of_rows > 0
        ]

    def reduce_by_row(self, values: npt.NDArray, operation: np.ufunc) -> npt.NDArray:
        """
        Combine the entries of every row with ``operation``, giving its identity for a
        row without entries, without padding any row to the fullest one.

        :param values: Per-entry values with the entries in the last axis.
        :param operation: An associative and commutative binary ufunc with an identity.
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

    def with_data(self, data: npt.NDArray) -> Self:
        """
        :param data: The new value of every stored entry.
        :return: A sparse array with the entries of this one and the given values.
        """
        return self.from_entries(
            SparseEntries(data, self.rows, self.columns), self.shape
        )

    def copy(self) -> Self:
        """
        :return: A copy of this array that shares no memory with it.
        """
        return self.__class__(self.array.copy())

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        return self.copy()
