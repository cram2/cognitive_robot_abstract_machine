from __future__ import annotations

import numpy as np
import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent

from probabilistic_model.exceptions import UnboundedEventError
from probabilistic_model.probabilistic_circuit.rx.helper import (
    fully_factorized,
    uniform_measure_of_event,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)

# %% fully factorized circuits


@pytest.fixture
def fully_factorized_circuit(x, y) -> ProbabilisticCircuit:
    """
    A product of a Gaussian over *x* and a Gaussian over *y*.
    """
    return fully_factorized([x, y], {x: 1, y: 0}, {x: 2, y: 1})


def test_fully_factorized_circuit_is_a_product_of_one_leaf_per_variable(
    fully_factorized_circuit,
):
    assert len(list(fully_factorized_circuit.nodes())) == 3
    assert len(list(fully_factorized_circuit.edges())) == 2


@pytest.mark.xfail(
    strict=True,
    reason="The mode of the truncated circuit lies on the boundary of the event, "
    "outside it by numeric imprecision",
)
def test_mode_of_a_truncated_fully_factorized_circuit_lies_in_the_event(
    fully_factorized_circuit, x, y
):
    event = (
        SimpleEvent.from_data({x: closed(0, 2), y: closed(0, 2)})
        .as_composite_set()
        .complement()
    )
    truncated = fully_factorized_circuit.truncated(event).distribution
    mode = truncated.mode().event
    assert (mode - event).is_empty()


# %% uniform measures


def test_uniform_measure_of_a_bounded_event(x, y):
    event = SimpleEvent.from_data(
        {x: closed(0, 1) | closed(3, 4), y: closed(0, 2)}
    ).as_composite_set()
    measure = uniform_measure_of_event(event)
    half = SimpleEvent.from_data({x: closed(0, 1), y: closed(0, 2)}).as_composite_set()
    assert measure.probability(half) == pytest.approx(0.5)


def test_an_unbounded_event_has_no_uniform_measure(x, y):
    event = SimpleEvent.from_data(
        {x: closed(0, np.inf), y: closed(0, 2)}
    ).as_composite_set()
    with pytest.raises(UnboundedEventError) as raised:
        uniform_measure_of_event(event)
    assert raised.value.variable == x
