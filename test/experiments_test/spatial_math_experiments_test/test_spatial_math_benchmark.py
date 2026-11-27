"""
Tests of the spatial math benchmark: every operation it times runs, and it reports one
row per operation in microseconds.
"""

from __future__ import annotations

import pytest

from experiments.experiment_definitions import Unit
from experiments.spatial_math_experiments.benchmark import (
    SpatialMathBenchmark,
    SpatialOperands,
    SpatialOperation,
)

# %% the operations


@pytest.fixture(scope="module")
def operands() -> SpatialOperands:
    return SpatialOperands.from_driven_base()


@pytest.mark.parametrize("operation", list(SpatialOperation))
def test_every_benchmarked_operation_builds_a_value(operands, operation):
    """
    Each operation runs on the operands and builds something.
    """
    assert operands.run(operation=operation) is not None


# %% the table


def test_the_benchmark_reports_one_row_per_operation_in_microseconds():
    table = SpatialMathBenchmark(repetitions=1, calls_per_repetition=1).measure()

    assert [row.operation for row in table.experiments] == list(SpatialOperation)
    assert {row.duration.unit for row in table.experiments} == {Unit.MICROSECONDS}
