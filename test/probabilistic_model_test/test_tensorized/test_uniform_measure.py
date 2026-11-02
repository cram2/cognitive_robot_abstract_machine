from __future__ import annotations

from enum import Enum

import numpy as np
import pytest
from random_events.interval import closed, closed_open, singleton
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic

from probabilistic_model.exceptions import (
    EventMixesPointsAndIntervalsError,
    UnboundedEventError,
)
from probabilistic_model.probabilistic_circuit.tensorized.factories import (
    uniform_measure_of_event,
)


# %% events over intervals
def test_the_uniform_measure_of_a_box_has_its_volume_as_inverse_density() -> None:
    x, y = Continuous("x"), Continuous("y")
    event = SimpleEvent.from_data(
        {x: closed(0.0, 2.0), y: closed(1.0, 5.0)}
    ).as_composite_set()

    model = uniform_measure_of_event(event)

    assert np.exp(model.log_likelihood(np.array([[1.0, 2.0]]))) == pytest.approx(
        [1 / 8]
    )


def test_the_uniform_measure_of_separate_regions_samples_inside_them() -> None:
    x, y = Continuous("x"), Continuous("y")
    event = (
        SimpleEvent.from_data(
            {x: closed(0.0, 1.0), y: closed(0.0, 1.0)}
        ).as_composite_set()
        | SimpleEvent.from_data(
            {x: closed(2.0, 3.0), y: closed(2.0, 4.0)}
        ).as_composite_set()
    )

    samples = uniform_measure_of_event(event).sample(200)

    assert all(event.contains(sample) for sample in samples)


def test_pieces_are_drawn_in_proportion_to_their_width() -> None:
    x = Continuous("x")
    event = SimpleEvent.from_data(
        {x: closed(0.0, 1.0) | closed(2.0, 5.0)}
    ).as_composite_set()

    np.random.seed(0)
    samples = uniform_measure_of_event(event).sample(4000)

    assert np.mean(samples[:, 0] >= 2.0) == pytest.approx(0.75, abs=0.03)


def test_a_variable_without_an_upper_bound_is_rejected() -> None:
    x = Continuous("x")
    event = SimpleEvent.from_data({x: closed(0.0, float("inf"))}).as_composite_set()

    with pytest.raises(UnboundedEventError):
        uniform_measure_of_event(event)


# %% events pinning variables to points
def test_the_uniform_measure_of_an_event_with_points_samples_inside_it() -> None:
    x, y = Continuous("x"), Continuous("y")
    event = (
        SimpleEvent.from_data(
            {x: singleton(0.5), y: closed(0.0, 1.0)}
        ).as_composite_set()
        | SimpleEvent.from_data(
            {x: singleton(0.5), y: closed(2.0, 3.0)}
        ).as_composite_set()
    )

    samples = uniform_measure_of_event(event).sample(50)

    assert np.all(samples[:, 0] == 0.5)
    assert all(event.contains(sample) for sample in samples)


def test_the_uniform_measure_of_pieces_sharing_a_point_samples_inside_them() -> None:
    x, y, yaw = Continuous("x"), Continuous("y"), Continuous("yaw")
    event = (
        SimpleEvent.from_data(
            {x: closed_open(-0.3, -0.03), y: closed(-0.26, -0.22), yaw: singleton(0.0)}
        ).as_composite_set()
        | SimpleEvent.from_data(
            {x: closed(-0.03, 0.3), y: closed(-0.26, 0.3), yaw: singleton(0.0)}
        ).as_composite_set()
    )

    samples = uniform_measure_of_event(event).sample(50)

    assert np.all(samples[:, 2] == 0.0)
    assert all(event.contains(sample) for sample in samples)


def test_a_set_of_only_points_draws_each_point_equally_often() -> None:
    x = Continuous("x")
    points = [0.25, 0.5, 0.75]
    event = SimpleEvent.from_data(
        {x: singleton(points[0]) | singleton(points[1]) | singleton(points[2])}
    ).as_composite_set()

    np.random.seed(0)
    samples = uniform_measure_of_event(event).sample(3000)

    for point in points:
        assert np.mean(samples[:, 0] == point) == pytest.approx(
            1 / len(points), abs=0.03
        )


def test_a_variable_that_is_a_point_beside_an_interval_is_rejected() -> None:
    x = Continuous("x")
    event = SimpleEvent.from_data(
        {x: closed(0.0, 1.0) | singleton(5.0)}
    ).as_composite_set()

    with pytest.raises(EventMixesPointsAndIntervalsError):
        uniform_measure_of_event(event)


# %% discrete variables
class Colour(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


def test_the_uniform_measure_draws_every_allowed_integer_equally_often() -> None:
    number = Integer("number")
    allowed = [1, 3, 4]
    event = SimpleEvent.from_data(
        {number: singleton(1) | singleton(3) | singleton(4)}
    ).as_composite_set()

    model = uniform_measure_of_event(event)

    for value in allowed:
        assert np.exp(model.log_likelihood(np.array([[value]]))) == pytest.approx(
            [1 / len(allowed)]
        )
    assert np.exp(model.log_likelihood(np.array([[2]]))) == pytest.approx([0.0])


def test_the_uniform_measure_draws_every_allowed_symbol_equally_often() -> None:
    colour = Symbolic(name="colour", domain=Set.from_iterable(Colour))
    event = SimpleEvent.from_data(
        {colour: (Colour.RED, Colour.BLUE)}
    ).as_composite_set()

    np.random.seed(0)
    samples = uniform_measure_of_event(event).sample(2000)

    assert set(samples[:, 0]) == {float(hash(Colour.RED)), float(hash(Colour.BLUE))}
    assert np.mean(samples[:, 0] == float(hash(Colour.RED))) == pytest.approx(
        0.5, abs=0.04
    )
