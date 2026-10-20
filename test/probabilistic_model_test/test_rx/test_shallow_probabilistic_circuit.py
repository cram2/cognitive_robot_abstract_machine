from __future__ import annotations

import numpy as np
import pytest
from random_events.interval import SimpleInterval
from random_events.variable import Continuous

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProductUnit,
    ShallowProbabilisticCircuit,
    SumUnit,
    leaf,
)

# %% fixtures


@pytest.fixture
def variable() -> Continuous:
    return Continuous("x")


def shallow_uniform_mixture(
    variable: Continuous, intervals: list[SimpleInterval]
) -> ShallowProbabilisticCircuit:
    """
    Create a shallow circuit that mixes uniform distributions over the given intervals
    with equal weights.
    """
    circuit = ShallowProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for interval in intervals:
        product = ProductUnit(probabilistic_circuit=circuit)
        product.add_subcircuit(
            leaf(UniformDistribution(variable=variable, interval=interval), circuit)
        )
        root.add_subcircuit(product, np.log(1 / len(intervals)))
    return circuit


# %% total variation distance


def test_total_variation_distance_of_overlapping_models(variable: Continuous):
    model = shallow_uniform_mixture(variable, [SimpleInterval.from_data(0, 2)])
    other = shallow_uniform_mixture(variable, [SimpleInterval.from_data(1, 3)])

    # Both densities are 1/2 and overlap on [1, 2), so half of the mass differs.
    assert model.total_variation_distance(other) == pytest.approx(0.5)


def test_total_variation_distance_of_mixtures(variable: Continuous):
    model = shallow_uniform_mixture(
        variable, [SimpleInterval.from_data(0, 1), SimpleInterval.from_data(1, 2)]
    )
    other = shallow_uniform_mixture(variable, [SimpleInterval.from_data(0, 4)])

    # The model has density 1/2 on [0, 2), the other 1/4 on [0, 4).
    assert model.total_variation_distance(other) == pytest.approx(0.5)


def test_total_variation_distance_of_disjoint_models_is_one(variable: Continuous):
    model = shallow_uniform_mixture(variable, [SimpleInterval.from_data(0, 1)])
    other = shallow_uniform_mixture(variable, [SimpleInterval.from_data(2, 3)])

    assert model.total_variation_distance(other) == pytest.approx(1)


def test_total_variation_distance_of_identical_models_is_zero(variable: Continuous):
    model = shallow_uniform_mixture(variable, [SimpleInterval.from_data(0, 1)])

    assert model.total_variation_distance(model) == pytest.approx(0)


def test_total_variation_distance_where_a_component_boundary_splits_an_overlap(
    variable: Continuous,
):
    model = shallow_uniform_mixture(
        variable, [SimpleInterval.from_data(0, 2), SimpleInterval.from_data(1, 3)]
    )
    other = shallow_uniform_mixture(variable, [SimpleInterval.from_data(0, 3)])

    # The model has density 1/4, 1/2, 1/4 on [0, 1), [1, 2), [2, 3); the other has 1/3.
    assert model.total_variation_distance(other) == pytest.approx(1 / 6)
