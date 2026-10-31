"""
Tests for the regions a causal circuit splits a cause into: a range of the cause is one
region however the circuit's support happens to write it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
from random_events.product_algebra import Event, SimpleEvent
from typing_extensions import Any


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
) -> tuple[SupportRegion, ...]:
    """
    The regions the causal circuit splits that cause into.
    """
    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        circuit_writing_one_range_two_ways,
        MarginalDeterminismTreeNode.from_causal_graph([cause], [effect]),
        [cause],
        [effect],
    )
    return tuple(causal_circuit._extract_disjoint_regions_for_variable(cause))


# %% one region per range, however the support writes it


def test_a_range_written_two_ways_is_one_region(
    regions: tuple[SupportRegion, ...],
) -> None:
    """
    The circuit holds two ranges of the cause, so the regions are two: the range both
    branches hold is not counted once per branch.
    """
    assert len(regions) == 2


def test_the_regions_hold_the_whole_circuit_between_them(
    regions: tuple[SupportRegion, ...],
) -> None:
    assert sum(region.probability for region in regions) == pytest.approx(1.0)


def test_each_region_holds_the_share_of_the_circuit_its_branches_carry(
    regions: tuple[SupportRegion, ...],
) -> None:
    """
    The range both branches hold carries three quarters of the circuit and the range
    only one branch holds carries the remaining quarter.
    """
    assert sorted(region.probability for region in regions) == pytest.approx(
        [0.25, 0.75]
    )


# %% the order the regions come in


@dataclass
class SupportListedInReverse:
    """
    A support whose simple sets come in the reverse of their sorted order, standing in
    for the order a support happens to list the branches of a circuit in.
    """

    support: Event
    """
    The support to list in reverse.
    """

    @property
    def simple_sets(self) -> list[SimpleEvent]:
        """
        :return: The simple sets of the support, last one first.
        """
        return list(reversed(self.support.simple_sets))


@pytest.fixture
def two_equally_weighted_regions(
    cause: Continuous, effect: Continuous
) -> CausalCircuit:
    """
    A causal circuit whose two cause regions carry exactly the same weight, so that
    nothing but the order of the regions can decide which of them a search picks.
    """
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for lower in (2.0, 0.0):
        branch = ProductUnit(probabilistic_circuit=circuit)
        for variable in (cause, effect):
            branch.add_subcircuit(
                leaf(
                    UniformDistribution(
                        variable=variable,
                        interval=closed(lower, lower + 1.0).simple_sets[0],
                    ),
                    circuit,
                )
            )
        root.add_subcircuit(branch, math.log(0.5))
    return CausalCircuit.from_probabilistic_circuit(
        circuit,
        MarginalDeterminismTreeNode.from_causal_graph([cause], [effect]),
        [cause],
        [effect],
    )


def values_of(regions: list[SupportRegion], cause: Continuous) -> list[Any]:
    """
    :param regions: The regions a cause was split into.
    :param cause: The cause the regions are of.
    :return: The value of the cause in every region.
    """
    return [region.event.simple_sets[0][cause] for region in regions]


def test_the_regions_come_in_the_order_of_their_values(
    two_equally_weighted_regions: CausalCircuit,
    cause: Continuous,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The order a support happens to list its branches in must not reach the regions: a
    search over regions of equal probability would otherwise pick between them by
    whatever that order was, which is not fixed.
    """
    circuit = two_equally_weighted_regions.probabilistic_circuit
    reversed_support = SupportListedInReverse(circuit.support)
    monkeypatch.setattr(type(circuit), "support", property(lambda _: reversed_support))
    regions = two_equally_weighted_regions._extract_disjoint_regions_for_variable(cause)
    values = values_of(regions, cause)
    assert values == sorted(values)


def test_the_best_of_two_equally_good_regions_is_the_first_one(
    two_equally_weighted_regions: CausalCircuit, cause: Continuous, effect: Continuous
) -> None:
    interventional = two_equally_weighted_regions.backdoor_adjustment(cause, effect)
    best = two_equally_weighted_regions._best_disjoint_region(cause, interventional)
    [first, _] = two_equally_weighted_regions._extract_disjoint_regions_for_variable(
        cause
    )
    assert best == first.event
