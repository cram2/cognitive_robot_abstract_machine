"""
Fixtures shared by the tests of the layered circuits.
"""

from __future__ import annotations

import pytest

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)

from ..circuit_building import mixture_of_products_of_mixtures, uniform_leaf

# %% circuits


@pytest.fixture
def shared_children_circuit(x, y) -> ProbabilisticCircuit:
    """
    A circuit whose leaves are shared by several sum units, so the layered circuit is a
    directed acyclic graph rather than a tree.
    """
    circuit = ProbabilisticCircuit()
    return mixture_of_products_of_mixtures(
        circuit,
        (uniform_leaf(x, 0, 1, circuit), uniform_leaf(x, 1, 2, circuit)),
        (uniform_leaf(y, 0, 1, circuit), uniform_leaf(y, 1, 2, circuit)),
        [((0.8, 0.2), (0.6, 0.4)), ((0.3, 0.7), (0.1, 0.9))],
    )
