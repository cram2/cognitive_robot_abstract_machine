import enum
from dataclasses import dataclass

import numpy as np
import pytest

from probabilistic_model.distributions.distributions import (
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    leaf,
)
from probabilistic_model.utils import MissingDict

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a
from krrood.entity_query_language.query.match import (
    AbstractMatchExpression,
    AttributeMatch,
)
from krrood.parametrization.exceptions import (
    AmbiguousVariableName,
    DomainElementsIndistinguishableInSamples,
)
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters


class Color(enum.Enum):
    """
    Colors with string values, whose hashes are too large for a float to hold exactly.
    """

    RED = "red"
    BLUE = "blue"


class LargeNumber(enum.IntEnum):
    """
    Members whose hashes become the same float.
    """

    FIRST = 2**53
    SECOND = 2**53 + 1


@dataclass
class Counter:
    number: LargeNumber


@dataclass
class Die:
    face: int
    color: Color


def die_query():
    return a(Die)(face=..., color=...)


def backend_of_dice() -> ProbabilisticBackend:
    """
    :return: A backend whose model gives the faces 1 and 6 and both colors, built over
        the variables the backend creates for :func:`die_query`.
    """
    variables = UnderspecifiedParameters(die_query()).variables
    face, color = variables["Die.face"], variables["Die.color"]
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(
        leaf(
            IntegerDistribution(
                variable=face, probabilities=MissingDict(float, {1: 0.5, 6: 0.5})
            ),
            circuit,
        )
    )
    root.add_subcircuit(
        leaf(
            SymbolicDistribution(
                variable=color,
                probabilities=MissingDict(
                    float, {hash(Color.RED): 0.5, hash(Color.BLUE): 0.5}
                ),
            ),
            circuit,
        )
    )
    return ProbabilisticBackend(model_registry=DictRegistry({Die: circuit}))


def test_integer_attributes_are_answered_as_integers():
    dice = list(die_query().evaluate(backend=backend_of_dice()))
    assert {die.face for die in dice} <= {1, 6}
    assert all(type(die.face) is int for die in dice)


def test_symbolic_attributes_are_answered_as_domain_elements():
    dice = list(die_query().evaluate(backend=backend_of_dice()))
    assert all(isinstance(die.color, Color) for die in dice)


def number_of_match_walks_for(amount: int, monkeypatch) -> int:
    """
    :param amount: How many dice to ask for.
    :return: How often the attribute matches of the query were walked to answer it.
    """
    walks = []
    original = AbstractMatchExpression._matches_with_variables_

    def counted(self):
        walks.append(self)
        return original.fget(self)

    backend = backend_of_dice()
    query = die_query()
    query.limit(amount)
    monkeypatch.setattr(
        AbstractMatchExpression, "_matches_with_variables_", property(counted)
    )
    dice = list(query.evaluate(backend=backend))
    monkeypatch.undo()
    assert len(dice) == amount
    return len(walks)


def test_the_attributes_are_looked_up_once_per_query(monkeypatch):
    assert number_of_match_walks_for(5, monkeypatch) == number_of_match_walks_for(
        200, monkeypatch
    )


def test_a_variable_named_like_several_attributes_is_refused(monkeypatch):
    """
    The attributes of a query have distinct names, so two of them are given the same
    name here.
    """
    parameters = UnderspecifiedParameters(die_query())
    variable = parameters.variables["Die.face"]
    monkeypatch.setattr(
        AttributeMatch, "name_from_variable_access_path", property(lambda _: "Die.face")
    )
    with pytest.raises(AmbiguousVariableName) as error:
        list(
            parameters.construct_instances_from_model_samples(
                [variable], np.array([[1.0]])
            )
        )
    assert error.value.variable is variable
    assert len(error.value.attribute_matches) == 2


def test_domain_elements_that_samples_cannot_tell_apart_are_refused():
    query = a(Counter)(number=...)
    number = UnderspecifiedParameters(query).variables["Counter.number"]
    circuit = ProbabilisticCircuit()
    leaf(
        SymbolicDistribution(
            variable=number,
            probabilities=MissingDict(float, {hash(LargeNumber.FIRST): 1.0}),
        ),
        circuit,
    )
    backend = ProbabilisticBackend(model_registry=DictRegistry({Counter: circuit}))
    with pytest.raises(DomainElementsIndistinguishableInSamples) as error:
        list(query.evaluate(backend=backend))
    assert error.value.variable == number
    assert set(error.value.elements) == set(LargeNumber)
