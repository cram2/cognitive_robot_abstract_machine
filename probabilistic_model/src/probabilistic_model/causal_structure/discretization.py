"""
Turning the columns of a table into a few levels each, so that rows can be counted per
combination of values.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder

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
        binned = [
            column
            for column in columns
            if is_numeric_dtype(rows[column])
            and rows[column].nunique() > self.maximum_levels
        ]
        kept = [column for column in columns if column not in binned]
        levels = pd.DataFrame(index=rows.index)
        if binned:
            levels[binned] = KBinsDiscretizer(
                n_bins=self.bin_count,
                encode="ordinal",
                strategy="quantile",
                quantile_method="averaged_inverted_cdf",
            ).fit_transform(rows[binned])
        if kept:
            levels[kept] = OrdinalEncoder().fit_transform(rows[kept])
        return levels[columns].astype(int)
