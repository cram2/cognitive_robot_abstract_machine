from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array
from typing_extensions import List, Self, Tuple


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
    def concatenate(cls, parts: List[SparseEntries]) -> Self:
        """
        :param parts: The entries to join, in order.
        :return: All entries of the parts.
        """
        return cls(
            np.concatenate([part.data for part in parts]),
            np.concatenate([part.rows for part in parts]),
            np.concatenate([part.columns for part in parts]),
        )

    def to_coo_array(self, shape: Tuple[int, int]) -> coo_array:
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
    A two dimensional sparse array whose stored entries can be grouped by row.

    Grouping lays the entries of every row out in one row of a rectangular array, padded
    where a row has fewer entries than the fullest one, so that a reduction over the
    entries of every row becomes one reduction over the last axis of that array.
    """

    array: coo_array
    """
    The stored entries in coordinate format.

    The rows and columns of the entries are fixed; only their values may change.
    """

    @classmethod
    def from_entries(cls, entries: SparseEntries, shape: Tuple[int, int]) -> Self:
        """
        :param entries: The stored entries.
        :param shape: The shape of the dense array.
        :return: The sparse array.
        """
        return cls(entries.to_coo_array(shape))

    @property
    def shape(self) -> Tuple[int, int]:
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

    @functools.cached_property
    def is_contiguous(self) -> bool:
        """
        :return: Whether the entries are stored row by row with the same number of
            entries per row, in which case grouping them is a reshape rather than a
            gather.
        """
        gather = self.gather
        return bool(
            self.number_of_stored_entries == gather.size
            and np.array_equal(gather, np.arange(gather.size).reshape(gather.shape))
        )

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

    def group_by_row(
        self, values: npt.NDArray, padding: float = -np.inf
    ) -> npt.NDArray:
        """
        Rearrange per-entry values into one row per row of this array.

        :param values: Per-entry values with the entries in the last axis.
        :param padding: The value for rows with fewer entries than the fullest one.
        :return: The values with shape ``(..., #rows, entries per row)``.
        """
        if self.is_contiguous:
            return values.reshape(values.shape[:-1] + self.gather.shape)
        return self.pad(values, padding)[..., self.gather]

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

    def __deepcopy__(self, memo=None) -> Self:
        return self.copy()
