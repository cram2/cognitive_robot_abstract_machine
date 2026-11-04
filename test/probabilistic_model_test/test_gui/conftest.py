"""
Fixtures shared by the tests of the graphical user interface.
"""

from __future__ import annotations

import sys

import pytest
from PySide6.QtWidgets import QApplication
from random_events.interval import closed
from random_events.set import Set
from random_events.variable import Continuous, Symbolic

from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.gui.controller import ModelController
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    leaf,
)
from probabilistic_model.utils import MissingDict

from ..circuit_building import uniform_leaf

# %% application


@pytest.fixture(scope="session", autouse=True)
def application() -> QApplication:
    """
    The application every widget needs to exist before it is created.
    """
    return QApplication.instance() or QApplication(sys.argv)


# %% model


@pytest.fixture
def continuous_variable() -> Continuous:
    """
    A continuous variable over [0, 1].
    """
    return Continuous(name="v1", domain=closed(0, 1))


@pytest.fixture
def symbolic_variable() -> Symbolic:
    """
    A symbolic variable over *a* and *b*.
    """
    return Symbolic(name="v2", domain=Set.from_iterable(["a", "b"]))


@pytest.fixture
def model(continuous_variable, symbolic_variable) -> ProbabilisticCircuit:
    """
    The product of a uniform distribution over [0, 1] and an equal distribution over *a*
    and *b*.
    """
    circuit = ProbabilisticCircuit()
    product = ProductUnit(probabilistic_circuit=circuit)
    product.add_subcircuit(uniform_leaf(continuous_variable, 0, 1, circuit))
    product.add_subcircuit(
        leaf(
            SymbolicDistribution(
                variable=symbolic_variable,
                probabilities=MissingDict(float, {hash("a"): 0.5, hash("b"): 0.5}),
            ),
            circuit,
        )
    )
    return circuit


@pytest.fixture
def controller(model) -> ModelController:
    """
    A controller of :func:`model`.
    """
    controller = ModelController()
    controller.set_model(model)
    return controller
