"""
Turning the columns of a table into a few levels each, so that rows can be counted per
combination of values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self

# %% discretizing a table


@dataclass
class Discretizer(TransformerMixin, BaseEstimator):
    """
    A scikit-learn transformer that turns the columns of a table into a few integer
    levels each: a numeric column with many values into quantile bins, every other
    column into one level per value, whatever type the values have.
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

    column_encoder: ColumnTransformer | None = field(
        init=False, default=None, repr=False
    )
    """
    The bins and values learned by :meth:`fit`, or ``None`` before it.
    """

    def __sklearn_is_fitted__(self) -> bool:
        return self.column_encoder is not None

    def fit(self, rows: pd.DataFrame, labels: None = None) -> Self:
        """
        Learn which columns to bin and where, and the values of every other column.

        :param rows: A table.
        :param labels: Unused; scikit-learn hands it to every step of a pipeline.
        :return: The discretizer.
        """
        binned = [
            column
            for column in rows.columns
            if is_numeric_dtype(rows[column])
            and rows[column].nunique() > self.maximum_levels
        ]
        self.column_encoder = (
            ColumnTransformer(
                [
                    (
                        "binned",
                        KBinsDiscretizer(
                            n_bins=self.bin_count,
                            encode="ordinal",
                            strategy="quantile",
                            quantile_method="averaged_inverted_cdf",
                        ),
                        binned,
                    )
                ],
                remainder=make_pipeline(
                    FunctionTransformer(
                        pd.DataFrame.astype,
                        kw_args=dict(dtype=str),
                        feature_names_out="one-to-one",
                    ),
                    OrdinalEncoder(),
                ),
                verbose_feature_names_out=False,
            )
            .set_output(transform="pandas")
            .fit(rows)
        )
        return self

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
        :param rows: A table with the columns the discretizer was fitted on.
        :return: Each column as integer levels, in the order of the rows and columns.
        :raises sklearn.exceptions.NotFittedError: If the discretizer was not fitted.
        """
        check_is_fitted(self)
        return self.column_encoder.transform(rows)[list(rows.columns)].astype(int)
