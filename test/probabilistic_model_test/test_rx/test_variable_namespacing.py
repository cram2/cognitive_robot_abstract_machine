from __future__ import annotations

from random_events.interval import SimpleInterval
from random_events.variable import Continuous

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    leaf,
)


# %% helpers
def _product_circuit(*variables: Continuous) -> ProbabilisticCircuit:
    """
    :return: A product of one uniform leaf per variable in *variables*.
    """
    circuit = ProbabilisticCircuit()
    product = ProductUnit(probabilistic_circuit=circuit)
    for variable in variables:
        product.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=variable, interval=SimpleInterval.from_data(0, 1)
                ),
                probabilistic_circuit=circuit,
            )
        )
    return circuit


# %% namespacing
def test_renaming_with_prefix_namespaces_every_variable() -> None:
    circuit = _product_circuit(Continuous("x"), Continuous("y"))

    circuit.rename_variables_with_prefix("part")

    assert sorted(variable.name for variable in circuit.variables) == [
        "part.x",
        "part.y",
    ]


def test_renaming_with_prefix_leaves_variables_already_in_the_namespace() -> None:
    """
    A nested part's variables are namespaced before the part holding them is, so
    namespacing the outer part must not prefix them a second time.
    """
    circuit = _product_circuit(Continuous("x"), Continuous("part.inner[0].y"))

    circuit.rename_variables_with_prefix("part")

    assert sorted(variable.name for variable in circuit.variables) == [
        "part.inner[0].y",
        "part.x",
    ]


def test_renaming_with_prefix_leaves_excluded_variables() -> None:
    latent = Continuous("count")
    circuit = _product_circuit(Continuous("x"), latent)

    circuit.rename_variables_with_prefix("part", excluded_variables=[latent])

    assert sorted(variable.name for variable in circuit.variables) == [
        "count",
        "part.x",
    ]
