"""
Tests of fitting a Gaussian kernel on every row.
"""

from __future__ import annotations

# %% imports
import numpy as np
import pandas as pd
import pytest

from probabilistic_model.learning.exceptions import (
    KernelOfNonContinuousVariableError,
    NoRowsToFitError,
)
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.learning.kernel_density import GaussianKernelDensity

# %% fixtures


@pytest.fixture
def two_rows() -> pd.DataFrame:
    """
    Two rows far apart along both of their columns.
    """
    return pd.DataFrame({"angle": [0.0, 10.0], "speed": [1.0, -3.0]})


# %% kernels on the rows


def test_samples_lie_near_the_rows_fitted_on(two_rows):
    """
    With narrow kernels, every sample lies within five kernel widths of one row.
    """
    method = GaussianKernelDensity(bandwidth=0.01)
    variables = infer_variables_from_dataframe(two_rows)
    circuit = method.fit(two_rows, variables)
    np.random.seed(0)
    samples = circuit.sample(50)
    columns = [variable.name for variable in circuit.variables]
    widths = np.array(
        [
            method.bandwidth * annotated.standard_deviation
            for annotated in sorted(variables, key=lambda each: each.variable.name)
        ]
    )
    rows = two_rows[columns].to_numpy()

    for sample in samples:
        assert min(np.max(np.abs(sample - row) / widths) for row in rows) < 5


def test_every_row_gets_a_kernel_of_equal_weight(two_rows):
    """
    The mixture has one kernel per row, each as likely as the others.
    """
    circuit = GaussianKernelDensity().fit(two_rows)
    weights = [weight for weight, _ in circuit.root.log_weighted_subcircuits]

    assert len(weights) == len(two_rows)
    assert np.allclose(np.exp(weights), 1 / len(two_rows))


def test_kernels_are_as_wide_as_the_annotating_rows_vary(two_rows):
    """
    Fitted on one row but annotated from both, each kernel is the bandwidth times the
    standard deviation of both rows wide.
    """
    method = GaussianKernelDensity(bandwidth=0.5)
    variables = infer_variables_from_dataframe(two_rows)

    circuit = method.fit(two_rows.iloc[:1], variables)

    assert sorted(leaf.distribution.scale for leaf in circuit.leaves) == sorted(
        method.bandwidth * annotated.standard_deviation for annotated in variables
    )


# %% what cannot be fitted


def test_a_dataframe_without_rows_is_refused(two_rows):
    """
    There is nothing to put a kernel on without any row.
    """
    with pytest.raises(NoRowsToFitError):
        GaussianKernelDensity().fit(two_rows.iloc[:0])


def test_a_column_that_is_not_continuous_is_refused():
    """
    A Gaussian kernel cannot be put on a symbolic value.
    """
    rows = pd.DataFrame({"angle": [0.0, 1.0], "thrower": ["Alice", "Bob"]})
    with pytest.raises(KernelOfNonContinuousVariableError):
        GaussianKernelDensity().fit(rows)
