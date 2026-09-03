from dataclasses import dataclass

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    leaf,
)
from random_events.interval import closed
from random_events.variable import Continuous


@dataclass
class Coin:
    a: float
    b: float
    c: float


@dataclass
class OtherClass:
    d: float


def build_three_independent_variables_circuit() -> tuple:
    """
    A circuit over three mutually independent uniformly distributed variables:

        Coin.a ~ Uniform([0, 1])
        Coin.b ~ Uniform([0, 2])
        Coin.c ~ Uniform([0, 3])

    Since the variables are independent, any operation on `a`/`b` alone must not depend
    on `c`, and vice versa -- used throughout test_probabilistic_queries/ to check EQL
    results against the same operation called directly on the circuit.
    """
    var_a = Continuous("Coin.a")
    var_b = Continuous("Coin.b")
    var_c = Continuous("Coin.c")

    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(leaf(UniformDistribution(variable=var_a, interval=closed(0, 1).simple_sets[0]), circuit))
    root.add_subcircuit(leaf(UniformDistribution(variable=var_b, interval=closed(0, 2).simple_sets[0]), circuit))
    root.add_subcircuit(leaf(UniformDistribution(variable=var_c, interval=closed(0, 3).simple_sets[0]), circuit))
    return circuit, var_a, var_b, var_c
