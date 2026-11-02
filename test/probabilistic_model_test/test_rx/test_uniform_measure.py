from __future__ import annotations

import numpy as np
import pytest
from random_events.interval import closed, closed_open, singleton
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous

from probabilistic_model.exceptions import EventMixesPointsAndIntervalsError
from probabilistic_model.probabilistic_circuit.rx.helper import (
    uniform_measure_of_event,
)


# %% events pinning variables to points
def test_the_uniform_measure_of_an_event_with_points_samples_inside_it() -> None:
    """
    A mode of a circuit pins most variables to single points and leaves a few as
    intervals, in several disjoint pieces.
    """
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
    assert np.all(
        ((samples[:, 1] >= 0.0) & (samples[:, 1] <= 1.0))
        | ((samples[:, 1] >= 2.0) & (samples[:, 1] <= 3.0))
    )


def test_the_uniform_measure_of_a_single_point_is_that_point() -> None:
    x = Continuous("x")
    event = SimpleEvent.from_data({x: singleton(0.25)}).as_composite_set()

    samples = uniform_measure_of_event(event).sample(5)

    assert samples[:, 0].tolist() == [0.25] * 5


def test_the_uniform_measure_of_pieces_sharing_a_point_samples_inside_them() -> None:
    """
    The mode of a placement query pins the yaw of the placed object to a point in every
    piece of its free space.
    """
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


# %% weights of pieces
def test_pieces_are_drawn_in_proportion_to_their_width() -> None:
    x = Continuous("x")
    event = SimpleEvent.from_data(
        {x: closed(0.0, 1.0) | closed(2.0, 5.0)}
    ).as_composite_set()

    np.random.seed(0)
    samples = uniform_measure_of_event(event).sample(4000)

    assert np.mean(samples[:, 0] >= 2.0) == pytest.approx(0.75, abs=0.03)


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


# %% events mixing points and intervals in one variable
def test_a_piece_pinning_a_variable_another_piece_spans_gets_no_samples() -> None:
    """
    A point has no width next to an interval, as in the edge piece that closed boxes
    sharing a side produce.
    """
    x, y = Continuous("x"), Continuous("y")
    event = (
        SimpleEvent.from_data(
            {x: singleton(0.5), y: closed(2.0, 3.0)}
        ).as_composite_set()
        | SimpleEvent.from_data(
            {x: closed(0.0, 1.0), y: closed(0.0, 1.0)}
        ).as_composite_set()
    )

    samples = uniform_measure_of_event(event).sample(200)

    assert np.all(samples[:, 1] <= 1.0)


def test_a_variable_that_is_a_point_beside_an_interval_is_rejected() -> None:
    x = Continuous("x")
    event = SimpleEvent.from_data(
        {x: closed(0.0, 1.0) | singleton(5.0)}
    ).as_composite_set()

    with pytest.raises(EventMixesPointsAndIntervalsError):
        uniform_measure_of_event(event)
