"""
Tests for the regions a causal circuit splits a cause into: a range of the cause is one
region however the circuit's support happens to write it.
"""

from __future__ import annotations

import math

import pytest
from random_events.interval import closed
from random_events.variable import Continuous

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
    SupportRegion,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    LeafUnit,
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from typing_extensions import Tuple


@pytest.fixture
def cause() -> Continuous:
    """
    The variable the regions are read for.
    """
    return Continuous("x")


@pytest.fixture
def effect() -> Continuous:
    """
    The variable the cause is asked about.
    """
    return Continuous("y")


@pytest.fixture
def circuit_writing_one_range_two_ways(
    cause: Continuous, effect: Continuous
) -> ProbabilisticCircuit:
    """
    A circuit whose two branches write the same range of the cause differently: one
    holds it inside a union of two ranges, the other holds it on its own.

    The first branch mixes ``x`` in ``[0, 1]`` and in ``[2, 3]`` equally beside ``y`` in
    ``[0, 1]``; the second holds ``x`` in ``[0, 1]`` alone beside ``y`` in ``[5, 6]``.
    Both branches carry half the weight, so ``[0, 1]`` holds three quarters of the
    circuit and ``[2, 3]`` the remaining quarter.
    """
    circuit = ProbabilisticCircuit()

    def uniform(variable: Continuous, lower: float, upper: float) -> LeafUnit:
        return leaf(
            UniformDistribution(
                variable=variable, interval=closed(lower, upper).simple_sets[0]
            ),
            circuit,
        )

    mixed_ranges = SumUnit(probabilistic_circuit=circuit)
    mixed_ranges.add_subcircuit(uniform(cause, 0, 1), math.log(0.5))
    mixed_ranges.add_subcircuit(uniform(cause, 2, 3), math.log(0.5))
    union_branch = ProductUnit(probabilistic_circuit=circuit)
    union_branch.add_subcircuit(mixed_ranges)
    union_branch.add_subcircuit(uniform(effect, 0, 1))
    single_branch = ProductUnit(probabilistic_circuit=circuit)
    single_branch.add_subcircuit(uniform(cause, 0, 1))
    single_branch.add_subcircuit(uniform(effect, 5, 6))
    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(union_branch, math.log(0.5))
    root.add_subcircuit(single_branch, math.log(0.5))
    return circuit


@pytest.fixture
def regions(
    circuit_writing_one_range_two_ways: ProbabilisticCircuit,
    cause: Continuous,
    effect: Continuous,
) -> Tuple[SupportRegion, ...]:
    """
    The regions the causal circuit splits that cause into.
    """
    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        circuit_writing_one_range_two_ways,
        MarginalDeterminismTreeNode.from_causal_graph([cause], [effect]),
        [cause],
        [effect],
    )
    return tuple(causal_circuit.disjoint_support_regions_of(cause))


# %% one region per range, however the support writes it


def test_a_range_written_two_ways_is_one_region(
    regions: Tuple[SupportRegion, ...],
) -> None:
    """
    The circuit holds two ranges of the cause, so the regions are two: the range both
    branches hold is not counted once per branch.
    """
    assert len(regions) == 2


def test_the_regions_hold_the_whole_circuit_between_them(
    regions: Tuple[SupportRegion, ...],
) -> None:
    assert sum(region.probability for region in regions) == pytest.approx(1.0)


def test_each_region_holds_the_share_of_the_circuit_its_branches_carry(
    regions: Tuple[SupportRegion, ...],
) -> None:
    """
    The range both branches hold carries three quarters of the circuit and the range
    only one branch holds carries the remaining quarter.
    """
    assert sorted(region.probability for region in regions) == pytest.approx(
        [0.25, 0.75]
    )
