"""
Turning the columns of a table into a few levels each, so that rows can be counted per
combination of values.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype

# %% discretizing a table


@dataclass
class Discretization:
    """
    How the columns of a table are turned into a few integer levels each: a numeric
    column with many values into quantile bins, every other column into one level per
    value.
    """

    bin_count: int = 4
    """
    Into how many quantile bins a numeric column with more than
    :attr:`maximum_levels` values is cut.
    """

    maximum_levels: int = 6
    """
    How many values a numeric column may have and still keep one level per value.
    """

    def apply(self, rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        :param rows: A table.
        :param columns: The names of the columns to discretize.
        :return: Those columns, each as integer levels, in the rows' order.
        """
        return pd.DataFrame(
            {column: self.levels_of(rows[column]) for column in columns},
            index=rows.index,
        )

    def levels_of(self, values: pd.Series) -> pd.Series:
        """
        :param values: One column.
        :return: Its values as integer levels.
        """
        if is_numeric_dtype(values) and values.nunique() > self.maximum_levels:
            return pd.qcut(values, self.bin_count, labels=False, duplicates="drop")
        return pd.Series(pd.factorize(values)[0], index=values.index)
