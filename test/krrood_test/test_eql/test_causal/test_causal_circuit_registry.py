from dataclasses import dataclass

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    leaf,
)
from random_events.interval import closed
from random_events.variable import Continuous

from krrood.entity_query_language.factories import a, cause
from krrood.parametrization.model_registries import CausalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters


@dataclass
class Pick:
    arm: float
    success: float


@dataclass
class Place:
    height: float


def _build_pick_causal_circuit() -> CausalCircuit:
    arm = Continuous("Pick.arm")
    success = Continuous("Pick.success")
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(
        leaf(
            UniformDistribution(variable=arm, interval=closed(0, 1).simple_sets[0]),
            circuit,
        )
    )
    root.add_subcircuit(
        leaf(
            UniformDistribution(variable=success, interval=closed(0, 1).simple_sets[0]),
            circuit,
        )
    )
    return CausalCircuit.from_probabilistic_circuit(
        circuit,
        MarginalDeterminismTreeNode.from_causal_graph([arm], [success]),
        [arm],
        [success],
    )


def test_registry_resolves_the_circuit_registered_for_the_queried_class():
    pick_circuit = _build_pick_causal_circuit()
    registry = CausalCircuitRegistry({Pick: pick_circuit})

    match = a(Pick)(arm=cause, success=...)
    match.causes_effect(match.variable.success == 0.5)
    parameters = UnderspecifiedParameters(match)

    assert registry.get_model(parameters) is pick_circuit


def test_registry_distinguishes_between_multiple_registered_classes():
    pick_circuit = _build_pick_causal_circuit()
    place_circuit = _build_pick_causal_circuit()
    registry = CausalCircuitRegistry({Pick: pick_circuit, Place: place_circuit})

    pick_match = a(Pick)(arm=cause, success=...)
    pick_match.causes_effect(pick_match.variable.success == 0.5)
    assert registry.get_model(UnderspecifiedParameters(pick_match)) is pick_circuit
