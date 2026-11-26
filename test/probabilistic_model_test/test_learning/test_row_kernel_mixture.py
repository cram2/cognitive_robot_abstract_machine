"""
Tests of fitting a mixture of one kernel per row.
"""

from __future__ import annotations

# %% imports
import numpy as np
import pandas as pd
import pytest

from probabilistic_model.distributions.binomial import BinomialDistribution
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.learning.structure.exceptions import NoRowsToFitError
from probabilistic_model.learning.structure.jpt.variables import (
    infer_variables_from_dataframe,
)
from probabilistic_model.learning.structure.row_kernel_mixture import RowKernelMixture

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
    method = RowKernelMixture(bandwidth=0.01)
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
    circuit = RowKernelMixture().fit(two_rows)
    weights = [weight for weight, _ in circuit.root.log_weighted_subcircuits]

    assert len(weights) == len(two_rows)
    assert np.allclose(np.exp(weights), 1 / len(two_rows))


def test_kernels_are_as_wide_as_the_annotating_rows_vary(two_rows):
    """
    Fitted on one row but annotated from both, each kernel is the bandwidth times the
    standard deviation of both rows wide.
    """
    method = RowKernelMixture(bandwidth=0.5)
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
        RowKernelMixture().fit(two_rows.iloc[:0])


# %% kernels on integers and symbols


def test_an_integer_kernel_is_a_binomial_around_the_row():
    """
    The kernel of an integer column is a binomial whose mean is the row's value.
    """
    rows = pd.DataFrame({"count": [2, 9, 5]})
    [annotated] = infer_variables_from_dataframe(rows)
    kernel = RowKernelMixture.kernel(
        annotated_variable=annotated, value=5, bandwidth=1.0
    )

    assert isinstance(kernel, BinomialDistribution)
    assert sum(value * p for value, p in kernel.probabilities.items()) == (
        pytest.approx(5)
    )


def test_a_symbol_kernel_keeps_most_probability_on_the_rows_symbol():
    """
    The kernel of a symbol puts all but the spread on the row's symbol and shares the
    spread among the others.
    """
    rows = pd.DataFrame({"thrower": ["Alice", "Bob", "Carol"]})
    [annotated] = infer_variables_from_dataframe(rows)
    method = RowKernelMixture(bandwidth=0.3)
    kernel = RowKernelMixture.kernel(
        annotated_variable=annotated, value="Bob", bandwidth=method.bandwidth
    )
    symbols = list(annotated.variable.domain.simple_sets)
    spread = method.bandwidth * (len(symbols) - 1) / len(symbols)

    assert isinstance(kernel, SymbolicDistribution)
    assert {
        symbol: kernel.probabilities[hash(symbol)] for symbol in symbols
    } == pytest.approx(
        {
            symbol: (
                1 - spread if symbol.element == "Bob" else spread / (len(symbols) - 1)
            )
            for symbol in symbols
        }
    )


def test_rows_of_every_variable_type_are_fitted(two_rows):
    """
    Continuous, integer and symbolic columns are fitted together, one kernel per row.
    """
    rows = two_rows.assign(count=[1, 4], thrower=["Alice", "Bob"])
    circuit = RowKernelMixture().fit(rows)
    assert len(circuit.root.subcircuits) == len(rows)
