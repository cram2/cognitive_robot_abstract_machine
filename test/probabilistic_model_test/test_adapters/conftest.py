"""
Fixtures shared by the tests of the conversions between circuit representations.
"""

from __future__ import annotations

import pytest
from random_events.set import Set
from random_events.variable import Continuous, Symbolic
from sortedcontainers import SortedSet

from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    ClassificationCircuit,
    DifferentiableLayeredCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

from .differentiable_circuits import (
    Color,
    Size,
    layered_circuit_over,
    region_graph_circuit,
)

# %% variables


@pytest.fixture
def w() -> Continuous:
    """
    A fourth continuous variable, so that every region of a region graph can be split.
    """
    return Continuous("w")


@pytest.fixture
def color() -> Symbolic:
    """
    A symbolic variable over :class:`Color`.
    """
    return Symbolic(name="color", domain=Set.from_iterable(Color))


@pytest.fixture
def size() -> Symbolic:
    """
    A symbolic variable over :class:`Size`.
    """
    return Symbolic(name="size", domain=Set.from_iterable(Size))


@pytest.fixture
def single_precision_tolerance() -> float:
    """
    The tolerance of a comparison with a differentiable circuit, which evaluates in
    single precision.
    """
    return 1e-4


# %% circuits


@pytest.fixture
def continuous_region_graph_circuit(w, x, y, z) -> DifferentiableLayeredCircuit:
    """
    A randomly initialized differentiable circuit over four continuous variables.
    """
    return region_graph_circuit(SortedSet([w, x, y, z]))


@pytest.fixture
def classification_circuit(w, x, y, z) -> ClassificationCircuit:
    """
    A randomly initialized circuit that classifies into two classes.
    """
    return (
        RegionGraph(
            SortedSet([w, x, y, z]), partitions=2, depth=1, repetitions=2, classes=2
        )
        .create_random_region_graph()
        .as_probabilistic_circuit()
    )


@pytest.fixture
def layered_circuit_over_color_and_x(color, x) -> LayeredProbabilisticCircuit:
    """
    A layered circuit over *color* and *x* that uses every input layer a differentiable
    circuit supports.
    """
    return layered_circuit_over(color, x)
