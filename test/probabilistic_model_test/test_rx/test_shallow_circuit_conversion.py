from __future__ import annotations

import numpy as np
import pytest
from random_events.interval import SimpleInterval
from random_events.variable import Continuous

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    ShallowProbabilisticCircuit,
    SumUnit,
    leaf,
)

# %% fixtures


@pytest.fixture
def x() -> Continuous:
    return Continuous("x")


@pytest.fixture
def y() -> Continuous:
    return Continuous("y")


@pytest.fixture
def mixture_weights() -> list[float]:
    return [0.3, 0.7]


@pytest.fixture
def product_of_mixture_and_leaf(
    x: Continuous, y: Continuous, mixture_weights: list[float]
) -> ProbabilisticCircuit:
    """
    A product of a mixture over :math:`x` and a single leaf over :math:`y`.
    """
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)
    mixture = SumUnit(probabilistic_circuit=circuit)
    for weight, interval in zip(
        mixture_weights,
        [SimpleInterval.from_data(0, 1), SimpleInterval.from_data(1, 3)],
    ):
        mixture.add_subcircuit(
            leaf(UniformDistribution(variable=x, interval=interval), circuit),
            np.log(weight),
        )
    root.add_subcircuit(mixture)
    root.add_subcircuit(
        leaf(
            UniformDistribution(variable=y, interval=SimpleInterval.from_data(0, 1)),
            circuit,
        )
    )
    return circuit


@pytest.fixture
def mixture_of_products_of_mixtures(
    x: Continuous, y: Continuous
) -> ProbabilisticCircuit:
    """
    A mixture of two products, each of a mixture over :math:`x` and one over :math:`y`.
    """
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for log_weight, offset in [(np.log(0.4), 0.0), (np.log(0.6), 5.0)]:
        product = ProductUnit(probabilistic_circuit=circuit)
        for variable in [x, y]:
            mixture = SumUnit(probabilistic_circuit=circuit)
            for weight, lower, upper in [(0.2, 0, 1), (0.8, 1, 4)]:
                mixture.add_subcircuit(
                    leaf(
                        UniformDistribution(
                            variable=variable,
                            interval=SimpleInterval.from_data(
                                lower + offset, upper + offset
                            ),
                        ),
                        circuit,
                    ),
                    np.log(weight),
                )
            product.add_subcircuit(mixture)
        root.add_subcircuit(product, log_weight)
    return circuit


def points_on_grid() -> np.ndarray:
    """
    Points covering the supports of the fixture circuits, with :math:`x` first.
    """
    coordinates = np.linspace(-0.5, 9.5, 41)
    return np.array(np.meshgrid(coordinates, coordinates)).reshape(2, -1).T


def assert_is_shallow(circuit: ShallowProbabilisticCircuit):
    assert isinstance(circuit.root, SumUnit)
    for product in circuit.root.subcircuits:
        assert isinstance(product, ProductUnit)
        assert all(subcircuit.is_leaf for subcircuit in product.subcircuits)


# %% conversion of a product of a mixture


def test_conversion_has_one_product_per_mixture_component(
    product_of_mixture_and_leaf: ProbabilisticCircuit, mixture_weights: list[float]
):
    shallow = ShallowProbabilisticCircuit.from_probabilistic_circuit(
        product_of_mixture_and_leaf
    )

    assert_is_shallow(shallow)
    assert sorted(
        np.exp(log_weight) for log_weight, _ in shallow.root.log_weighted_subcircuits
    ) == pytest.approx(sorted(mixture_weights))


def test_conversion_keeps_the_likelihood_of_a_product_of_a_mixture(
    product_of_mixture_and_leaf: ProbabilisticCircuit,
):
    shallow = ShallowProbabilisticCircuit.from_probabilistic_circuit(
        product_of_mixture_and_leaf
    )

    assert shallow.likelihood(points_on_grid()) == pytest.approx(
        product_of_mixture_and_leaf.likelihood(points_on_grid())
    )


# %% conversion of nested mixtures


def test_conversion_of_nested_mixtures_is_shallow(
    mixture_of_products_of_mixtures: ProbabilisticCircuit,
):
    shallow = ShallowProbabilisticCircuit.from_probabilistic_circuit(
        mixture_of_products_of_mixtures
    )

    assert_is_shallow(shallow)
    # Each of the two products expands into two times two leaf combinations.
    assert len(shallow.root.subcircuits) == 8


def test_conversion_keeps_the_likelihood_of_nested_mixtures(
    mixture_of_products_of_mixtures: ProbabilisticCircuit,
):
    shallow = ShallowProbabilisticCircuit.from_probabilistic_circuit(
        mixture_of_products_of_mixtures
    )

    assert shallow.likelihood(points_on_grid()) == pytest.approx(
        mixture_of_products_of_mixtures.likelihood(points_on_grid())
    )


def test_conversion_leaves_the_original_circuit_unchanged(
    mixture_of_products_of_mixtures: ProbabilisticCircuit,
):
    likelihood_before = mixture_of_products_of_mixtures.likelihood(points_on_grid())

    ShallowProbabilisticCircuit.from_probabilistic_circuit(
        mixture_of_products_of_mixtures
    )

    assert mixture_of_products_of_mixtures.likelihood(
        points_on_grid()
    ) == pytest.approx(likelihood_before)
