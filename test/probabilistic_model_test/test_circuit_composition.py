"""
Tests for the two compositions every circuit type offers, a mixture and a product, run
against both circuit types.
"""

from __future__ import annotations

import math

import pytest
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous, Variable
from typing_extensions import List, Type

from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.exceptions import (
    EmptyCompositionError,
    MixtureOverDifferentVariablesError,
    NumberOfWeightsMismatchError,
    ProductOverSharedVariablesError,
)
from probabilistic_model.probabilistic_circuit.probabilistic_circuit_representation import (
    ProbabilisticCircuitRepresentation,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    leaf,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

# %% circuits of either type


@pytest.fixture(
    params=[ProbabilisticCircuit, LayeredProbabilisticCircuit],
    ids=lambda circuit_type: circuit_type.__name__,
)
def circuit_type(request) -> Type[ProbabilisticCircuitRepresentation]:
    """
    Every circuit type the compositions are offered by.
    """
    return request.param


def uniform_circuit(
    circuit_type: Type[ProbabilisticCircuitRepresentation],
    variable: Continuous,
    lower: float,
    upper: float,
) -> ProbabilisticCircuitRepresentation:
    """
    A circuit of one uniform distribution.

    :param circuit_type: The type to build the circuit in.
    :param variable: The variable of the distribution.
    :param lower: The lower bound of its interval.
    :param upper: The upper bound of its interval.
    :return: The circuit.
    """
    circuit = ProbabilisticCircuit()
    leaf(
        UniformDistribution(
            variable=variable, interval=closed(lower, upper).simple_sets[0]
        ),
        circuit,
    )
    if circuit_type is ProbabilisticCircuit:
        return circuit
    return RustworkxCircuitToLayeredCircuitConverter.convert(circuit)


def event_of(**ranges: tuple) -> Event:
    """
    An event that restricts every named variable to a range.

    :param ranges: The lower and upper bound per variable name.
    :return: The event.
    """
    return SimpleEvent.from_data(
        {
            Continuous(name): closed(lower, upper)
            for name, (lower, upper) in ranges.items()
        }
    ).as_composite_set()


def variables_of(circuit: ProbabilisticCircuitRepresentation) -> List[Variable]:
    """
    :param circuit: A circuit.
    :return: Its variables, as a list that can be compared and read in a report.
    """
    return list(circuit.variables)


# %% mixtures


def test_a_mixture_weights_every_component(circuit_type, x):
    lower_half = uniform_circuit(circuit_type, x, 0, 1)
    upper_half = uniform_circuit(circuit_type, x, 1, 2)
    mixture = circuit_type.mixture_of(
        [lower_half, upper_half], [math.log(0.6), math.log(0.4)]
    )
    assert mixture.probability(event_of(x=(0, 1))) == pytest.approx(0.6)
    assert mixture.probability(event_of(x=(1, 2))) == pytest.approx(0.4)


def test_a_mixture_normalizes_weights_that_do_not_sum_to_one(circuit_type, x):
    mixture = circuit_type.mixture_of(
        [
            uniform_circuit(circuit_type, x, 0, 1),
            uniform_circuit(circuit_type, x, 1, 2),
        ],
        [math.log(3.0), math.log(1.0)],
    )
    assert mixture.probability(event_of(x=(0, 1))) == pytest.approx(0.75)


def test_a_mixture_is_over_the_variables_of_its_components(circuit_type, x):
    component = uniform_circuit(circuit_type, x, 0, 1)
    mixture = circuit_type.mixture_of([component, component], [0.0, 0.0])
    assert variables_of(mixture) == variables_of(component)


def test_a_mixture_of_one_component_is_that_component(circuit_type, x):
    component = uniform_circuit(circuit_type, x, 0, 2)
    mixture = circuit_type.mixture_of([component], [math.log(0.5)])
    event = event_of(x=(0, 1))
    assert mixture.probability(event) == pytest.approx(component.probability(event))


def test_a_mixture_leaves_its_components_as_they_are(circuit_type, x):
    component = uniform_circuit(circuit_type, x, 0, 2)
    event = event_of(x=(0, 1))
    probability_before = component.probability(event)
    mixture = circuit_type.mixture_of(
        [component, uniform_circuit(circuit_type, x, 0, 2)], [0.0, 0.0]
    )
    mixture.truncated(event_of(x=(0, 0.5)))
    assert variables_of(component) == [x]
    assert component.probability(event) == pytest.approx(probability_before)


def test_a_mixture_of_no_component_raises(circuit_type):
    with pytest.raises(EmptyCompositionError):
        circuit_type.mixture_of([], [])


def test_a_mixture_without_one_weight_per_component_raises(circuit_type, x):
    with pytest.raises(NumberOfWeightsMismatchError):
        circuit_type.mixture_of([uniform_circuit(circuit_type, x, 0, 1)], [0.0, 0.0])


def test_a_mixture_of_components_over_different_variables_raises(circuit_type, x, y):
    with pytest.raises(MixtureOverDifferentVariablesError):
        circuit_type.mixture_of(
            [
                uniform_circuit(circuit_type, x, 0, 1),
                uniform_circuit(circuit_type, y, 0, 1),
            ],
            [0.0, 0.0],
        )


# %% products


def test_a_product_multiplies_the_probabilities_of_its_factors(circuit_type, x, y):
    along_x = uniform_circuit(circuit_type, x, 0, 2)
    along_y = uniform_circuit(circuit_type, y, 0, 4)
    product = circuit_type.product_of([along_x, along_y])
    assert product.probability(event_of(x=(0, 1), y=(0, 1))) == pytest.approx(
        along_x.probability(event_of(x=(0, 1)))
        * along_y.probability(event_of(y=(0, 1)))
    )


def test_a_product_is_over_the_variables_of_all_of_its_factors(circuit_type, x, y):
    product = circuit_type.product_of(
        [uniform_circuit(circuit_type, y, 0, 1), uniform_circuit(circuit_type, x, 0, 1)]
    )
    assert variables_of(product) == [x, y]


def test_a_product_of_one_factor_is_that_factor(circuit_type, x):
    factor = uniform_circuit(circuit_type, x, 0, 2)
    product = circuit_type.product_of([factor])
    event = event_of(x=(0, 1))
    assert product.probability(event) == pytest.approx(factor.probability(event))


def test_a_product_leaves_its_factors_as_they_are(circuit_type, x, y):
    factor = uniform_circuit(circuit_type, x, 0, 2)
    event = event_of(x=(0, 1))
    probability_before = factor.probability(event)
    circuit_type.product_of([factor, uniform_circuit(circuit_type, y, 0, 2)])
    assert variables_of(factor) == [x]
    assert factor.probability(event) == pytest.approx(probability_before)


def test_a_product_of_no_factor_raises(circuit_type):
    with pytest.raises(EmptyCompositionError):
        circuit_type.product_of([])


def test_a_product_of_factors_over_one_variable_raises(circuit_type, x):
    with pytest.raises(ProductOverSharedVariablesError) as raised:
        circuit_type.product_of(
            [
                uniform_circuit(circuit_type, x, 0, 1),
                uniform_circuit(circuit_type, x, 1, 2),
            ]
        )
    assert raised.value.shared_variables == [x]


# %% the two circuit types compose the same distribution


def mixture_of_products(
    circuit_type: Type[ProbabilisticCircuitRepresentation], x: Continuous, y: Continuous
) -> ProbabilisticCircuitRepresentation:
    """
    A mixture of two products, which is a composition of both kinds at once.

    :param circuit_type: The type to build the circuit in.
    :param x: The variable the first factor of every product is over.
    :param y: The variable the second factor of every product is over.
    :return: The circuit.
    """
    return circuit_type.mixture_of(
        [
            circuit_type.product_of(
                [
                    uniform_circuit(circuit_type, x, 0, 1),
                    uniform_circuit(circuit_type, y, 0, 3),
                ]
            ),
            circuit_type.product_of(
                [
                    uniform_circuit(circuit_type, x, 1, 2),
                    uniform_circuit(circuit_type, y, 2, 4),
                ]
            ),
        ],
        [math.log(0.7), math.log(0.3)],
    )


@pytest.mark.parametrize(
    "event",
    [
        event_of(x=(0, 1), y=(0, 1)),
        event_of(x=(0, 2), y=(2, 3)),
        event_of(x=(1, 2), y=(0, 4)),
        event_of(x=(0.5, 1.5), y=(1, 3)),
    ],
)
def test_both_circuit_types_give_a_mixture_of_products_the_same_probability(
    event, x, y
):
    assert mixture_of_products(LayeredProbabilisticCircuit, x, y).probability(
        event
    ) == pytest.approx(
        mixture_of_products(ProbabilisticCircuit, x, y).probability(event)
    )
