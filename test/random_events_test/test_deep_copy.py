from __future__ import annotations

import copy

import pytest

from random_events.interval import closed, SimpleInterval, Bound
from random_events.product_algebra import SimpleEvent, Event
from random_events.set import Set, SetElement
from random_events.variable import Continuous, Symbolic, Integer, Variable

# %% fixtures


@pytest.fixture
def horizontal() -> Continuous:
    return Continuous("horizontal")


@pytest.fixture
def color() -> Symbolic:
    return Symbolic("color", domain=Set.from_iterable(["red", "green"]))


@pytest.fixture
def event(horizontal: Continuous, color: Symbolic) -> Event:
    return SimpleEvent.from_data(
        {horizontal: closed(0, 1), color: "red"}
    ).as_composite_set()


# %% variables


@pytest.mark.parametrize(
    "variable",
    [
        Continuous("horizontal"),
        Integer("count"),
        Symbolic("color", domain=Set.from_iterable(["red", "green"])),
    ],
)
def test_a_deep_copy_of_a_variable_is_the_variable(variable: Variable):
    assert copy.deepcopy(variable) is variable


# %% sets


def test_a_deep_copy_of_a_simple_interval_is_an_equal_simple_interval():
    simple_interval = SimpleInterval.from_data(0, 1, Bound.CLOSED, Bound.OPEN)
    copied = copy.deepcopy(simple_interval)
    assert copied == simple_interval
    assert copied is not simple_interval


def test_a_deep_copy_of_an_interval_is_an_equal_interval():
    interval = closed(0, 1) | closed(2, 3)
    copied = copy.deepcopy(interval)
    assert copied == interval
    assert copied is not interval


def test_a_deep_copy_of_a_set_element_is_an_equal_set_element(color: Symbolic):
    element = SetElement.from_data("red", color.domain.all_elements)
    copied = copy.deepcopy(element)
    assert copied == element
    assert copied is not element


def test_a_deep_copy_of_an_event_is_an_equal_event(event: Event):
    copied = copy.deepcopy(event)
    assert copied == event
    assert copied is not event


def test_a_deep_copy_of_an_event_shares_its_variables(event: Event):
    [original] = event.simple_sets
    [copied] = copy.deepcopy(event).simple_sets
    assert all(
        copied_variable is original_variable
        for copied_variable, original_variable in zip(copied.keys(), original.keys())
    )


def test_a_deep_copy_of_an_event_shares_none_of_its_assignments(
    event: Event, horizontal: Continuous
):
    [original] = event.simple_sets
    [copied] = copy.deepcopy(event).simple_sets
    assert copied[horizontal] is not original[horizontal]


def test_a_container_of_sets_deep_copies(event: Event):
    container = {"event": event, "interval": closed(0, 1)}
    copied = copy.deepcopy(container)
    assert copied == container
    assert copied["interval"] is not container["interval"]
