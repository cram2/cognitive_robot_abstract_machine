"""
Submatrices of sparse matrices whose sparsity pattern stays the same from one control
cycle to the next, such as the constraint matrices a compiled QP writes every cycle.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


# %% submatrix of one sparsity pattern


@dataclass(kw_only=True, frozen=True)
class SubmatrixKey:
    """
    Identifies a submatrix by the sparsity pattern it is taken from and by the rows and
    columns it keeps.
    """

    pattern_identity: int
    """
    The identity of the row index array of the matrix the submatrix is taken from.
    """

    kept_rows: bytes
    """
    The flags of the kept rows, one byte per row.
    """

    kept_columns: bytes
    """
    The flags of the kept columns, one byte per column.
    """


@dataclass(kw_only=True)
class SubmatrixSelection:
    """
    The entries of a sparse matrix that lie in the kept rows and columns, and the
    submatrix they form.
    """

    source_row_indices: NDArray[np.int32]
    """
    The row index array of the matrix the submatrix is taken from.

    Held so that its identity in a :class:`SubmatrixKey` names no other array.
    """

    kept_entries: NDArray[np.intp]
    """
    The positions of the kept entries among the stored entries of the source matrix.
    """

    submatrix: sp.csc_matrix
    """
    The submatrix, holding the values of the source matrix it was last taken from.
    """

    @classmethod
    def from_matrix(
        cls,
        *,
        matrix: sp.csc_matrix,
        kept_rows: NDArray[np.bool_],
        kept_columns: NDArray[np.bool_],
    ) -> SubmatrixSelection:
        """
        :param matrix: The matrix the submatrix is taken from.
        :param kept_rows: Which rows of `matrix` are kept, one flag per row.
        :param kept_columns: Which columns of `matrix` are kept, one flag per column.
        :return: The selection of the kept rows and columns of `matrix`.
        """
        column_of_entry = cls.column_of_each_entry(matrix=matrix)
        kept_entries = np.flatnonzero(
            kept_columns[column_of_entry] & kept_rows[matrix.indices]
        )
        new_row_of_row = np.cumsum(kept_rows) - 1
        new_column_of_column = np.cumsum(kept_columns) - 1
        number_of_kept_columns = int(new_column_of_column[-1]) + 1
        entries_per_kept_column = np.bincount(
            new_column_of_column[column_of_entry[kept_entries]],
            minlength=number_of_kept_columns,
        )
        submatrix = sp.csc_matrix(
            (
                matrix.data[kept_entries],
                new_row_of_row[matrix.indices[kept_entries]],
                np.concatenate(([0], np.cumsum(entries_per_kept_column))),
            ),
            shape=(int(new_row_of_row[-1]) + 1, number_of_kept_columns),
        )
        return cls(
            source_row_indices=matrix.indices,
            kept_entries=kept_entries,
            submatrix=submatrix,
        )

    def take_values(self, *, matrix: sp.csc_matrix) -> sp.csc_matrix:
        """
        :param matrix: A matrix with the sparsity pattern this selection was made for.
        :return: :attr:`submatrix`, now holding the values of `matrix`.
        """
        np.take(matrix.data, self.kept_entries, out=self.submatrix.data)
        return self.submatrix

    @staticmethod
    def column_of_each_entry(*, matrix: sp.csc_matrix) -> NDArray[np.intp]:
        """
        :param matrix: A matrix in compressed sparse column format.
        :return: The column of every stored entry of `matrix`, in storage order.
        """
        return np.repeat(np.arange(matrix.shape[1]), np.diff(matrix.indptr))


# %% remembered submatrices


@dataclass(kw_only=True)
class SubmatrixCache:
    """
    Remembers which entries form a submatrix, so that taking the same rows and columns
    of the same sparsity pattern again only copies values.

    .. warning:: A submatrix it returns is overwritten when the same rows and columns of
        the same sparsity pattern are taken again.
    """

    maximum_number_of_selections: int = 32
    """
    How many selections are remembered; the least recently used one is forgotten first.
    """

    selections: OrderedDict[SubmatrixKey, SubmatrixSelection] = field(
        default_factory=OrderedDict
    )
    """
    The remembered selections, the most recently used one last.
    """

    def submatrix(
        self,
        *,
        matrix: sp.csc_matrix,
        kept_rows: NDArray[np.bool_],
        kept_columns: NDArray[np.bool_],
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
        key = SubmatrixKey(
            pattern_identity=id(matrix.indices),
            kept_rows=kept_rows.tobytes(),
            kept_columns=kept_columns.tobytes(),
        )
        selection = self.selections.get(key)
        if selection is not None:
            self.selections.move_to_end(key)
            return selection.take_values(matrix=matrix)
        selection = SubmatrixSelection.from_matrix(
            matrix=matrix, kept_rows=kept_rows, kept_columns=kept_columns
        )
        self.selections[key] = selection
        if len(self.selections) > self.maximum_number_of_selections:
            self.selections.popitem(last=False)
        return selection.submatrix
