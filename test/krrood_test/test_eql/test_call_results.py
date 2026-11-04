"""
Tests that one evaluation calls each predicate and symbolic function once per distinct
tuple of arguments and reuses the result, without changing the answers.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.factories import (
    an,
    contains,
    entity,
    inference,
    literal,
    variable,
)
from krrood.entity_query_language.predicate import symbolic_function
from krrood.planning.state import State

from ..dataset.counted_calls import (
    IsWithinReach,
    Place,
    TravelTime,
    Visit,
    distance_to,
    distances_on_the_way_to,
)

# %% fixtures


@pytest.fixture
def reach() -> float:
    """
    The greatest distance that is within reach.
    """
    return 10.0


@pytest.fixture
def places(reach: float) -> list[Place]:
    """
    A place inside reach, one at its edge and one beyond it.
    """
    return [Place(reach - 1), Place(reach), Place(reach + 1)]


@pytest.fixture
def visits(places: list[Place]) -> list[Visit]:
    """
    Two visits to each place.
    """
    return [Visit(place) for place in places for _ in range(2)]


def times_asked(places: list[Place]) -> list[int]:
    """
    :param places: The places to read.
    :return: How often a counted procedure was called with each place.
    """
    return [place.times_asked for place in places]


# %% predicates


def test_predicate_is_called_once_per_distinct_argument(visits, places, reach):
    """
    Two visits to one place ask the predicate about that place once.
    """
    visit = variable(Visit, domain=visits)

    answers = an(entity(visit).where(IsWithinReach(visit.place, reach))).tolist()

    assert answers == [v for v in visits if v.place.distance <= reach]
    assert times_asked(places) == [1, 1, 1]


def test_predicate_is_called_again_in_a_new_evaluation(visits, places, reach):
    """
    Each evaluation calls the predicate afresh.
    """
    visit = variable(Visit, domain=visits)
    reachable = an(entity(visit).where(IsWithinReach(visit.place, reach)))

    reachable.tolist()
    reachable.tolist()

    assert times_asked(places) == [2, 2, 2]


def test_state_assigned_predicate_is_read_from_the_state(visits, places, reach):
    """
    A predicate the state assigns is read from the state and never called.
    """
    farthest = places[-1]
    state = State().with_facts(IsWithinReach(literal(farthest), reach))
    visit = variable(Visit, domain=visits)

    answers = an(entity(visit).where(IsWithinReach(visit.place, reach))).tolist(
        state=state
    )

    assert answers == visits
    assert farthest.times_asked == 0


def test_query_started_during_an_evaluation_reuses_its_calls(visits, places, reach):
    """
    A query a symbolic function runs for each visit asks about each place once over the
    whole evaluation.
    """

    @symbolic_function
    def is_reachable(visited: Visit) -> bool:
        """
        :param visited: The visit asked about.
        :return: Whether a query finds the place of *visited* within reach.
        """
        place = variable(Place, domain=[visited.place])
        return an(entity(place).where(IsWithinReach(place, reach))).tolist() != []

    visit = variable(Visit, domain=visits)

    answers = an(entity(visit).where(is_reachable(visit))).tolist()

    assert answers == [v for v in visits if v.place.distance <= reach]
    assert times_asked(places) == [1, 1, 1]


# %% symbolic functions


def test_symbolic_function_class_is_called_once_per_distinct_argument(
    visits, places, reach
):
    """
    A symbolic function class is called once per place.
    """
    visit = variable(Visit, domain=visits)

    answers = an(entity(visit).where(TravelTime(visit.place) < reach)).tolist()

    assert answers == [v for v in visits if v.place.distance < reach]
    assert times_asked(places) == [1, 1, 1]


def test_decorated_function_is_called_once_per_distinct_argument(visits, places, reach):
    """
    A function decorated with
    :func:`~krrood.entity_query_language.predicate.symbolic_function` is called once per
    place.
    """
    visit = variable(Visit, domain=visits)

    answers = an(entity(visit).where(distance_to(visit.place) < reach)).tolist()

    assert answers == [v for v in visits if v.place.distance < reach]
    assert times_asked(places) == [1, 1, 1]


def test_iterator_result_is_not_reused(visits, places, reach):
    """
    A result that is an iterator is consumed by its reader, so every visit calls again.
    """
    visit = variable(Visit, domain=visits)

    answers = an(
        entity(visit).where(contains(distances_on_the_way_to(visit.place), reach))
    ).tolist()

    assert answers == [v for v in visits if v.place.distance >= reach]
    assert times_asked(places) == [2, 2, 2]


# %% constructors


def test_constructor_is_called_for_every_binding(visits):
    """
    Constructing an object gives a new object for every binding, even for equal
    arguments.
    """
    visit = variable(Visit, domain=visits)

    revisits = an(entity(inference(Visit)(place=visit.place))).tolist()

    assert len({id(revisit) for revisit in revisits}) == len(visits)
