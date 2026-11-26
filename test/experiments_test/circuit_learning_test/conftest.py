"""
Fixtures shared by the tests of the circuit learning benchmark.
"""

from __future__ import annotations

# %% imports
import numpy as np
import pandas as pd
import pytest

from experiments.circuit_learning.datasets import BenchmarkDataset, DatasetName

# %% datasets


@pytest.fixture
def dataset() -> BenchmarkDataset:
    """
    :return: Two continuous columns and one symbolic column whose rows form two clusters.
    """
    generator = np.random.default_rng(0)

    def frame(number_of_rows: int) -> pd.DataFrame:
        """
        :param number_of_rows: The number of rows.
        :return: Rows drawn from the two clusters.
        """
        cluster = generator.integers(0, 2, number_of_rows)
        return pd.DataFrame(
            {
                "a": generator.normal(cluster * 4.0, 1.0),
                "b": generator.normal(-cluster * 3.0, 1.0),
                "c": np.where(cluster == 1, "high", "low"),
            }
        )

    return BenchmarkDataset(
        name=DatasetName.NLTCS, train=frame(200), validation=frame(50), test=frame(50)
    )
