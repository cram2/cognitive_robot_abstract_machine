from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from typing_extensions import Any, Self, Tuple


@dataclass(eq=False)
class SparseArray:
    """
    A minimal two dimensional sparse array in coordinate (COO) format.

    An explicitly stored zero is kept as a stored entry: the product layers hold *child
    node indices* as values, where an index of ``0`` is a meaningful edge.

    ``eq=False`` because the default field-tuple comparison would call ``==`` on the
    ``data``/``indices`` arrays, which raises once they hold more than one element.

    It is a plain dataclass of numpy arrays, so
    :class:`krrood.adapters.json_serializer.DataclassJSONSerializer` serializes it
    without this class, or the layers that hold one, writing anything of their own.
    """

    data: npt.NDArray
    """
    The values of the stored entries with shape (#entries,).
    """

    indices: npt.NDArray
    """
    The coordinates of the stored entries with shape (#entries, 2).

    The first column contains row indices, the second column contains column indices.
    """

    shape: Tuple[int, int]
    """
    The shape of the dense array this sparse array describes.
    """

    def __post_init__(self):
        self.data = np.asarray(self.data)
        self.indices = np.asarray(self.indices, dtype=np.int64).reshape(-1, 2)
        self.shape = (int(self.shape[0]), int(self.shape[1]))

    @property
    def number_of_stored_entries(self) -> int:
        """
        :return: The number of explicitly stored entries.
        """
        return len(self.data)

    @property
    def rows(self) -> npt.NDArray:
        """
        :return: The row coordinates of the stored entries.
        """
        return self.indices[:, 0]

    @property
    def columns(self) -> npt.NDArray:
        """
        :return: The column coordinates of the stored entries.
        """
        return self.indices[:, 1]

    @classmethod
    def from_dense(cls, array: npt.NDArray, fill_value: Any = 0) -> Self:
        """
        Create a sparse array from a dense one by storing every entry that differs from
        ``fill_value``.

        :param array: The dense array.
        :param fill_value: The value that is treated as "not stored".
        :return: The sparse array.
        """
        array = np.asarray(array)
        if fill_value is None or (
            isinstance(fill_value, float) and np.isnan(fill_value)
        ):
            mask = ~np.isnan(array)
        else:
            mask = array != fill_value
        rows, columns = np.nonzero(mask)
        return cls(array[rows, columns], np.stack([rows, columns], axis=1), array.shape)

    @classmethod
    def from_coordinates(
        cls,
        rows: npt.NDArray,
        columns: npt.NDArray,
        data: npt.NDArray,
        shape: Tuple[int, int],
    ) -> Self:
        """
        Create a sparse array from separate row, column and value sequences.

        :param rows: The row coordinates.
        :param columns: The column coordinates.
        :param data: The values.
        :param shape: The dense shape.
        :return: The sparse array.
        """
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        columns = np.asarray(columns, dtype=np.int64).reshape(-1)
        return cls(np.asarray(data), np.stack([rows, columns], axis=1), shape)

    def to_dense(self, fill_value: Any = 0) -> npt.NDArray:
        """
        Materialize this array as a dense array.

        :param fill_value: The value of the entries that are not stored.
        :return: The dense array.
        """
        result = np.full(self.shape, fill_value, dtype=self.data.dtype)
        if self.number_of_stored_entries:
            result[self.rows, self.columns] = self.data
        return result

    def sort_indices(self) -> Self:
        """
        :return: A copy of this array whose entries are sorted by (row, column).
        """
        order = np.lexsort((self.columns, self.rows))
        return self.__class__(self.data[order], self.indices[order], self.shape)

    def copy(self) -> Self:
        """
        :return: A copy of this array that shares no memory with it.
        """
        return self.__class__(self.data.copy(), self.indices.copy(), self.shape)

    def __deepcopy__(self, memo=None) -> Self:
        return self.copy()


def embedded_logsumexp(values: npt.NDArray, axis: int) -> npt.NDArray:
    """
    Numerically stable ``log(sum(exp(values)))`` that maps an all ``-inf`` reduction to
    ``-inf`` instead of ``nan``.

    :param values: The values in log space.
    :param axis: The axis to reduce.
    :return: The reduced array.
    """
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)

    # the subtraction and the exponentiation are done into the same buffer: these arrays
    # hold one entry per edge per event, so every avoided temporary matters
    shifted = values - maximum
    with np.errstate(over="ignore"):
        np.exp(shifted, out=shifted)
    summed = np.sum(shifted, axis=axis, keepdims=True)

    with np.errstate(divide="ignore"):
        result = np.where(summed > 0, np.log(summed) + maximum, -np.inf)
    return np.squeeze(result, axis=axis)


def remap_indices(
    keep_mask: npt.NDArray,
) -> Tuple[npt.NDArray, int]:
    """
    Create an index remapping for a prune operation.

    :param keep_mask: A boolean mask of the entries that survive.
    :return: An array that maps old indices to new indices (``-1`` for removed entries)
        and the number of surviving entries.
    """
    remap = np.full(len(keep_mask), -1, dtype=np.int64)
    number_of_kept = int(keep_mask.sum())
    remap[keep_mask] = np.arange(number_of_kept, dtype=np.int64)
    return remap, number_of_kept
