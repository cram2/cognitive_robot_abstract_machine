"""
Starting point for exploring better ways to visualize probabilistic circuits.

Builds a small toy circuit and renders it with the current baseline
visualization (ProbabilisticCircuit.plot_structure, which wraps
rustworkx.visualization.mpl_draw) so alternative renderers can be compared
against it.
"""

import matplotlib.pyplot as plt
from random_events.variable import Continuous

from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    UnivariateContinuousLeaf,
)


def toy_circuit() -> ProbabilisticCircuit:
    x = Continuous("x")
    y = Continuous("y")

    pc = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=pc)

    p1 = ProductUnit(probabilistic_circuit=pc)
    p1.add_subcircuit(UnivariateContinuousLeaf(GaussianDistribution(0, 1, variable=x), probabilistic_circuit=pc))
    p1.add_subcircuit(UnivariateContinuousLeaf(GaussianDistribution(0, 1, variable=y), probabilistic_circuit=pc))

    p2 = ProductUnit(probabilistic_circuit=pc)
    p2.add_subcircuit(UnivariateContinuousLeaf(GaussianDistribution(3, 1, variable=x), probabilistic_circuit=pc))
    p2.add_subcircuit(UnivariateContinuousLeaf(GaussianDistribution(3, 1, variable=y), probabilistic_circuit=pc))

    root.add_subcircuit(p1, 0.5)
    root.add_subcircuit(p2, 0.5)

    return pc


if __name__ == "__main__":
    circuit = toy_circuit()
    circuit.plot_structure()
    plt.savefig("baseline_plot_structure.png")
    print("Wrote baseline_plot_structure.png")
