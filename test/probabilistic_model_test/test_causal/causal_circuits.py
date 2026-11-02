"""
Circuits with known causal structure that the tests of causal inference share.

Every builder returns the circuit followed by its variables.
"""

from __future__ import annotations

import math
from enum import Enum, auto

from random_events.set import Set
from random_events.variable import Continuous, Symbolic

from probabilistic_model.distributions.distributions import (
    DiracDeltaDistribution,
    SymbolicDistribution,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    LeafUnit,
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.utils import MissingDict

from ..circuit_building import mixture_of_intervals, uniform_leaf


class Season(Enum):
    WARM = auto()
    COLD = auto()


class Treatment(Enum):
    LOW = auto()
    HIGH = auto()


class Outcome(Enum):
    GOOD = auto()
    BAD = auto()


def point_mass(
    variable: Symbolic, value: Enum, circuit: ProbabilisticCircuit
) -> LeafUnit:
    """
    :param variable: A symbolic variable.
    :param value: The value the variable always takes.
    :param circuit: The circuit to add the leaf to.
    :return: A leaf whose variable always takes the value.
    """
    return leaf(
        SymbolicDistribution(
            variable=variable, probabilities=MissingDict(float, {hash(value): 1.0})
        ),
        circuit,
    )


# %% continuous circuits


def independent_circuit() -> tuple[ProbabilisticCircuit, Continuous, Continuous]:
    """
    ProductUnit(SumUnit_x, SumUnit_y) — x and y are independent.

        SumUnit_x  [x∈[0,1] w=0.6,  x∈[1,2] w=0.4]
        SumUnit_y  [y∈[0,1] w=0.5,  y∈[1,2] w=0.5]

    Ground truth:
        P(y∈[0,1]) = 0.5 regardless of x.
        P(y | do(x)) = P(y) for all x in the support.
    """
    x, y = Continuous("x"), Continuous("y")
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(mixture_of_intervals(circuit, x, [(0.6, 0, 1), (0.4, 1, 2)]))
    root.add_subcircuit(mixture_of_intervals(circuit, y, [(0.5, 0, 1), (0.5, 1, 2)]))
    return circuit, x, y


def three_variable_circuit() -> (
    tuple[ProbabilisticCircuit, Continuous, Continuous, Continuous]
):
    """
    ProductUnit(SumUnit_x, SumUnit_y, SumUnit_z) — all three independent.

        SumUnit_x [x∈[0,1] w=0.7, x∈[1,2] w=0.3]
        SumUnit_y [y∈[0,1] w=0.4, y∈[1,2] w=0.6]
        SumUnit_z [z∈[0,1] w=0.8, z∈[1,2] w=0.2]

    Ground truth:
        P(z∈[0,1]) = 0.8 regardless of x or y.
        P(z | do(x)) = P(z) with or without adjustment on y.
    """
    x, y, z = Continuous("x"), Continuous("y"), Continuous("z")
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)
    for variable, first_weight in [(x, 0.7), (y, 0.4), (z, 0.8)]:
        root.add_subcircuit(
            mixture_of_intervals(
                circuit, variable, [(first_weight, 0, 1), (1 - first_weight, 1, 2)]
            )
        )
    return circuit, x, y, z


def correlated_circuit() -> (
    tuple[ProbabilisticCircuit, Continuous, Continuous, Continuous]
):
    """
    SumUnit-rooted mixture where x fully determines which y region is active.

    Two equal-weight components:
        Low:  x∈[0,1], w∈[0,1], y∈[0,0.4]
        High: x∈[1,2], w∈[0,1], y∈[9.6,10]

    Ground truth:
        y∈[0,0.4] and y∈[9.6,10] each occur with marginal probability 0.5.
        w has the same marginal in both components — it carries no information about y.
    """
    x, w, y = Continuous("x"), Continuous("w"), Continuous("y")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for x_range, y_range in [((0, 1), (0, 0.4)), ((1, 2), (9.6, 10))]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(uniform_leaf(x, *x_range, circuit))
        component.add_subcircuit(uniform_leaf(w, 0, 1, circuit))
        component.add_subcircuit(uniform_leaf(y, *y_range, circuit))
        root.add_subcircuit(component, math.log(0.5))
    return circuit, x, w, y


def nested_overlap_circuit() -> tuple[ProbabilisticCircuit, Continuous, Continuous]:
    """
    SumUnit-rooted mixture whose components all overlap on x without being identical:
    the kind of circuit Monte-Carlo grounding builds when it mixes one copy of a part
    template per sampled aggregation value.

    Three equal-weight components:
        Wide:   x∈[0,3], y∈[0,1]
        Narrow: x∈[0,1], y∈[1,2]
        Middle: x∈[1,3], y∈[0,2]

    Wide overlaps both others, so no pair of components is disjoint, yet the
    components do not share one support either: x∈[0,1] lies in Wide and Narrow but
    not Middle, so the mixture is not support-deterministic over x.
    """
    x, y = Continuous("x"), Continuous("y")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for x_range, y_range in [((0, 3), (0, 1)), ((0, 1), (1, 2)), ((1, 3), (0, 2))]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(uniform_leaf(x, *x_range, circuit))
        component.add_subcircuit(uniform_leaf(y, *y_range, circuit))
        root.add_subcircuit(component, math.log(1 / 3))
    return circuit, x, y


def confounded_circuit() -> (
    tuple[ProbabilisticCircuit, Continuous, Continuous, Continuous]
):
    """
    Circuit with confounder z that drives both x and y.

    Two equal-weight strata:
        Low stratum:  x∈[0,1], y∈[0,1], z∈[0,1]
        High stratum: x∈[1,2], y∈[1,2], z∈[1,2]

    Ground truth:
        P(y∈[0,1] | x∈[0,1]) = 1.0  — spurious, driven by z.
        P(y∈[0,1] | do(x))    = 0.5  — causal truth after adjusting for z.
    """
    x, y, z = Continuous("x"), Continuous("y"), Continuous("z")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for lower, upper in [(0, 1), (1, 2)]:
        component = ProductUnit(probabilistic_circuit=circuit)
        for variable in (x, y, z):
            component.add_subcircuit(uniform_leaf(variable, lower, upper, circuit))
        root.add_subcircuit(component, math.log(0.5))
    return circuit, x, y, z


def unnormalized_circuit() -> tuple[ProbabilisticCircuit, Continuous, Continuous]:
    """
    ProductUnit circuit whose SumUnit over x has weights summing to 1.6.
    """
    x, y = Continuous("x"), Continuous("y")
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(mixture_of_intervals(circuit, x, [(0.8, 0, 1), (0.8, 1, 2)]))
    root.add_subcircuit(mixture_of_intervals(circuit, y, [(0.5, 0, 1), (0.5, 1, 2)]))
    return circuit, x, y


def shared_cause_subcircuit_circuit() -> (
    tuple[ProbabilisticCircuit, Continuous, Continuous]
):
    """
    A DAG, not a tree: the SumUnit over the cause x is mounted as a shared child of two
    different ProductUnits, matching the mounting pattern relational grounding uses when
    one exchangeable instance is attached under several class-level product nodes.

        SumUnit(root)
          ProductUnit_a(SumUnit_x [shared], leaf y∈[0,1])
          ProductUnit_b(SumUnit_x [shared], leaf y∈[2,3])

        SumUnit_x [x∈[0,1] w=0.6, x∈[1,2] w=0.4]  -- same node object under both parents

    Ground truth: x's two branches are disjoint regardless of which parent reaches
    them, so the circuit is support-deterministic.
    """
    x, y = Continuous("x"), Continuous("y")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    shared_sum_x = mixture_of_intervals(circuit, x, [(0.6, 0, 1), (0.4, 1, 2)])
    for y_range in [(0, 1), (2, 3)]:
        product = ProductUnit(probabilistic_circuit=circuit)
        product.add_subcircuit(shared_sum_x)
        product.add_subcircuit(uniform_leaf(y, *y_range, circuit))
        root.add_subcircuit(product, math.log(0.5))
    return circuit, x, y


def circuit_with_a_rounded_cause_region() -> (
    tuple[ProbabilisticCircuit, Continuous, Continuous, Continuous]
):
    """
    Two equal-weight strata whose cause ``x`` is a Dirac point each, one of them at a
    value single precision cannot represent exactly.

    A support region is read off the circuit in single precision, so the region of the
    Dirac at ``0.2`` does not hold that location exactly. Adjusting for ``z`` has to
    intervene on it all the same, within the tolerance of the Dirac.
    """
    x, y, z = Continuous("x"), Continuous("y"), Continuous("z")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for x_location, y_range in [(0.2, (0, 1)), (1.0, (1, 2))]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(
            leaf(DiracDeltaDistribution(variable=x, location=x_location), circuit)
        )
        component.add_subcircuit(uniform_leaf(y, *y_range, circuit))
        component.add_subcircuit(uniform_leaf(z, 0, 1, circuit))
        root.add_subcircuit(component, math.log(0.5))
    return circuit, x, y, z


# %% symbolic circuits


def symbolic_variables() -> tuple[Symbolic, Symbolic, Symbolic]:
    """
    :return: The treatment, season and outcome variables.
    """
    return (
        Symbolic("treatment", domain=Set.from_iterable(Treatment)),
        Symbolic("season", domain=Set.from_iterable(Season)),
        Symbolic("outcome", domain=Set.from_iterable(Outcome)),
    )


def discrete_confounded_circuit() -> (
    tuple[ProbabilisticCircuit, Symbolic, Symbolic, Symbolic]
):
    """
    Circuit with a discrete confounder (season) driving both a discrete cause
    (treatment) and the effect (outcome), where treatment has no causal effect of its
    own. Unlike :func:`confounded_circuit`'s continuous, deterministically branch-tied
    cause, treatment's own distribution *overlaps* across confounder states (both values
    have positive probability in both seasons).

    Two strata:
        Warm stratum (p=0.6): P(treatment=HIGH)=0.8, outcome=GOOD (deterministic)
        Cold stratum (p=0.4): P(treatment=HIGH)=0.3, outcome=BAD (deterministic)

    Ground truth:
        P(outcome=GOOD | treatment=HIGH) = 0.8 -- spurious, driven by season.
        P(outcome=GOOD | do(treatment=HIGH)) = 0.6 -- causal truth after adjusting
            for season; equal to P(outcome=GOOD | do(treatment=LOW)), since treatment
            has no real effect on outcome.
    """
    treatment, season, outcome = symbolic_variables()
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for season_value, treatment_high_probability, stratum_weight, outcome_value in [
        (Season.WARM, 0.8, 0.6, Outcome.GOOD),
        (Season.COLD, 0.3, 0.4, Outcome.BAD),
    ]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(point_mass(season, season_value, circuit))
        treatment_distribution = SumUnit(probabilistic_circuit=circuit)
        treatment_distribution.add_subcircuit(
            point_mass(treatment, Treatment.HIGH, circuit),
            math.log(treatment_high_probability),
        )
        treatment_distribution.add_subcircuit(
            point_mass(treatment, Treatment.LOW, circuit),
            math.log(1 - treatment_high_probability),
        )
        component.add_subcircuit(treatment_distribution)
        component.add_subcircuit(point_mass(outcome, outcome_value, circuit))
        root.add_subcircuit(component, math.log(stratum_weight))
    return circuit, treatment, season, outcome


def cause_specific_effect_circuit() -> (
    tuple[ProbabilisticCircuit, Symbolic, Symbolic, Symbolic]
):
    """
    Circuit where treatment has a real, differentiated causal effect on outcome within
    each season stratum. Unlike :func:`discrete_confounded_circuit`, where
    P(outcome | do(treatment)) is identical for every treatment value by construction,
    it distinguishes a correctly cause-restricted adjustment from one that silently
    drops the cause restriction.

    Two strata, each further split by treatment:
        Warm stratum (p=0.6): P(treatment=HIGH)=0.8; treatment=HIGH pairs with
            outcome=GOOD (deterministic), treatment=LOW pairs with outcome=BAD.
        Cold stratum (p=0.4): P(treatment=HIGH)=0.3; same treatment/outcome pairing.

    Ground truth:
        P(outcome=GOOD | do(treatment=HIGH)) = 1.0
        P(outcome=GOOD | do(treatment=LOW)) = 0.0
    """
    treatment, season, outcome = symbolic_variables()
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for season_value, treatment_high_probability, stratum_weight in [
        (Season.WARM, 0.8, 0.6),
        (Season.COLD, 0.3, 0.4),
    ]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(point_mass(season, season_value, circuit))
        treatment_outcome = SumUnit(probabilistic_circuit=circuit)
        for treatment_value, outcome_value, branch_weight in [
            (Treatment.HIGH, Outcome.GOOD, treatment_high_probability),
            (Treatment.LOW, Outcome.BAD, 1 - treatment_high_probability),
        ]:
            branch = ProductUnit(probabilistic_circuit=circuit)
            branch.add_subcircuit(point_mass(treatment, treatment_value, circuit))
            branch.add_subcircuit(point_mass(outcome, outcome_value, circuit))
            treatment_outcome.add_subcircuit(branch, math.log(branch_weight))
        component.add_subcircuit(treatment_outcome)
        root.add_subcircuit(component, math.log(stratum_weight))
    return circuit, treatment, season, outcome
