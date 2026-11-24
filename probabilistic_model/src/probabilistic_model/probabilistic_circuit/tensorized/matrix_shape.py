"""
The shape of a matrix of the layers of a layered circuit, such as an edge matrix or a
probability table.
"""

from __future__ import annotations

# %% matrix shape
from dataclasses import dataclass


@dataclass(kw_only=True, unsafe_hash=True)
class MatrixShape2D:
    """
    The number of rows and of columns of a matrix.
    """

    number_of_rows: int
    """
    The number of rows.
    """

    number_of_columns: int
    """
    The number of columns.
    """

    def __post_init__(self) -> None:
        """
        Store the numbers as Python integers, whatever integer type they were given as.
        """
        self.number_of_rows = int(self.number_of_rows)
        self.number_of_columns = int(self.number_of_columns)

    def to_tuple(self) -> tuple[int, int]:
        """
        :return: The shape as array libraries take it.
        """
        return self.number_of_rows, self.number_of_columns
