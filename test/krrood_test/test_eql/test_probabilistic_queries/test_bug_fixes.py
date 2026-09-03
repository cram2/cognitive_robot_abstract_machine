from dataclasses import dataclass

import pytest
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    leaf,
)
from random_events.interval import closed
from random_events.variable import Continuous

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import average, probability_of, variable
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.parametrization.exceptions import (
    JointQueryAcrossClassesNotSupported,
    RelationalCircuitRegistryRequiresMatch,
)
from krrood.parametrization.model_registries import DictRegistry, RelationalCircuitRegistry


@dataclass
class Arm:
    battery: float


@dataclass
class Robot:
    arm: Arm


def _build_robot_circuit():
    """
    A circuit for the multi-hop attribute `Robot.arm.battery`, named to match exactly
    what that chain resolves to (see Attribute._name_) -- not `Robot.battery`, which a
    single-hop-only class resolution would incorrectly look for.
    """
    x = variable(Robot)
    var_battery = Continuous(x.arm.battery._name_)
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(
        leaf(UniformDistribution(variable=var_battery, interval=closed(0, 100).simple_sets[0]), circuit)
    )
    return circuit, var_battery


def test_multi_hop_attribute_resolves_to_root_class():
    """
    probability_of/average must key the model registry lookup by the attribute
    chain's root class (Robot), not its immediate parent (Arm) -- a Robot is what's
    registered in the ModelRegistry, an Arm never is.
    """
    circuit, _ = _build_robot_circuit()
    backend = ProbabilisticBackend(model_registry=DictRegistry({Robot: circuit}))

    x = variable(Robot)
    assert probability_of(x.arm.battery < 50).first(backend=backend) == pytest.approx(0.5)

    x2 = variable(Robot)
    assert average(x2.arm.battery).first(backend=backend) == pytest.approx(50.0)


def test_average_distinct_falls_through_to_native_evaluation():
    """
    average(..., distinct=True) must not silently take the probabilistic closed-form
    shortcut: native evaluation deduplicates values before averaging, which the
    closed-form expectation has no notion of.
    """
    circuit, _ = _build_robot_circuit()
    backend = ProbabilisticBackend(model_registry=DictRegistry({Robot: circuit}))

    x = variable(Robot)
    with pytest.raises(Exception):
        average(x.arm.battery, distinct=True).first(backend=backend)


def test_probability_of_bool_condition_verbalizes_but_does_not_evaluate():
    """
    probability_of(True)'s condition is normalized to a Literal on construction (the
    same normalization and_/or_/not_ already apply to a bare-bool operand), so it
    verbalizes cleanly like any other condition -- but it still can't evaluate: a
    content-free condition names no class, so there is no model to resolve it against.
    """
    text = verbalize_expression(probability_of(True))
    assert text == "the probability that True"

    circuit, _ = _build_robot_circuit()
    backend = ProbabilisticBackend(model_registry=DictRegistry({Robot: circuit}))
    with pytest.raises(JointQueryAcrossClassesNotSupported):
        probability_of(True).first(backend=backend)


def test_relational_circuit_registry_rejects_non_match_parameters():
    circuit, _ = _build_robot_circuit()
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=None)
    backend = ProbabilisticBackend(model_registry=registry)

    x = variable(Robot)
    with pytest.raises(RelationalCircuitRegistryRequiresMatch):
        probability_of(x.arm.battery < 50).first(backend=backend)
