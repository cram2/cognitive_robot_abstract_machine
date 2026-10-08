"""
Tests for the do() questions put to the clutter circuit.

Everything here fits on the closed-form stand-in for the simulator rather than on
recorded attempts, so the whole module keeps running in CI without a simulator or
network access.
"""

from __future__ import annotations

from dataclasses import fields

import experiments.orm.ormatic_interface  # type: ignore  # noqa: F401
import numpy as np
import pytest
from krrood.entity_query_language.factories import variable
from typing_extensions import List

from experiments.causal_reasoning.tracy_clutter_picking.do_query import (
    AttemptAttribute,
    AttemptCount,
    ClosingAxisSideCausesDisturbance,
    ClutterDoQuery,
    CrowdingCausesLift,
    DoQueryAnswer,
    FrictionCausesLift,
    NeighbourAttribute,
    question_catalogue,
)
from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutteredObject,
    ClutterPickScene,
    ClutterPickSceneAggregations,
    PartField,
)
from experiments.causal_reasoning.tracy_clutter_picking.synthetic import (
    synthetic_clutter_pick_scenes,
)


@pytest.fixture(scope="module")
def probability_tolerance() -> float:
    """
    How far outside ``[0, 1]`` a probability read off a circuit may land through
    floating-point arithmetic alone.
    """
    return 1e-9


@pytest.fixture(scope="module")
def attempt_count() -> int:
    """
    Enough attempts for a cause to take several values, few enough to fit quickly.
    """
    return 60


@pytest.fixture(scope="module")
def object_count() -> int:
    """
    A small clutter, so grounding stays quick.
    """
    return 4


@pytest.fixture(scope="module")
def attempts(attempt_count: int, object_count: int) -> List[ClutterPickScene]:
    """
    Attempts drawn from the closed-form stand-in, the same ones for every test.
    """
    return synthetic_clutter_pick_scenes(
        np.random.default_rng(0),
        scene_count=attempt_count,
        object_count=object_count,
    )


@pytest.fixture(scope="module")
def neighbour_count(attempts: List[ClutterPickScene]) -> int:
    """
    How many neighbours every drawn attempt holds.
    """
    [count] = {len(attempt.neighbours) for attempt in attempts}
    return count


# %% what an attempt offers as a cause


def test_an_attempt_attribute_is_named_after_its_own_field() -> None:
    assert AttemptAttribute.FRICTION_COEFFICIENT.circuit_variable_name == (
        variable(ClutterPickScene).friction_coefficient._name_
    )


def test_a_count_is_named_after_its_own_aggregation() -> None:
    assert AttemptCount.CROWDING.circuit_variable_name == (
        variable(ClutterPickSceneAggregations).crowding_count()._name_
    )


def test_every_neighbour_attribute_is_a_field_a_neighbour_carries() -> None:
    """
    A part template names its columns by the bare attribute, so each member's value has
    to be a field of a neighbour for a fit to be able to group by it.
    """
    carried = {field.name for field in fields(ClutteredObject)}

    assert {attribute.value for attribute in NeighbourAttribute} <= carried


def test_every_attempt_attribute_is_a_field_an_attempt_carries() -> None:
    carried = {field.name for field in fields(ClutterPickScene)}

    assert {attribute.value for attribute in AttemptAttribute} <= carried


def test_every_attribute_and_count_reads_as_words() -> None:
    assert all(attribute.noun for attribute in AttemptAttribute)
    assert all(attribute.noun for attribute in NeighbourAttribute)
    assert all(count.noun for count in AttemptCount)


# %% where a question groups the fit


def test_an_attempt_level_cause_groups_the_class_circuit(
    neighbour_count: int,
) -> None:
    stratification = FrictionCausesLift(neighbour_count=neighbour_count).stratification

    assert stratification.part_field is None
    assert stratification.column == (
        AttemptAttribute.FRICTION_COEFFICIENT.circuit_variable_name
    )


def test_a_count_cause_groups_the_class_circuit_by_the_statistic(
    neighbour_count: int,
) -> None:
    stratification = CrowdingCausesLift(neighbour_count=neighbour_count).stratification

    assert stratification.part_field is None
    assert stratification.column == AttemptCount.CROWDING.circuit_variable_name


def test_a_neighbour_level_cause_groups_that_part_field_instead(
    neighbour_count: int,
) -> None:
    """
    The cause lives on one neighbour, so the template holding the neighbours is what has
    to be grouped; the class circuit has no column for it.
    """
    stratification = ClosingAxisSideCausesDisturbance(
        neighbour_count=neighbour_count
    ).stratification

    assert stratification.part_field is PartField.NEIGHBOURS
    assert stratification.column == NeighbourAttribute.CLOSING_AXIS_SIDE.value


def test_the_catalogue_asks_each_question_once(neighbour_count: int) -> None:
    names = [question.name for question in question_catalogue(neighbour_count)]

    assert len(names) == len(set(names))


def test_a_question_says_which_clutter_it_is_about(neighbour_count: int) -> None:
    question = FrictionCausesLift(neighbour_count=neighbour_count)

    assert str(neighbour_count) in question.name
    assert str(neighbour_count) in question.asked


# %% asking the friction question


@pytest.fixture(scope="module")
def friction_answer(
    attempts: List[ClutterPickScene], neighbour_count: int
) -> DoQueryAnswer:
    """
    The answer to the friction question, read once.
    """
    return ClutterDoQuery(
        question=FrictionCausesLift(neighbour_count=neighbour_count)
    ).run(attempts)


def test_the_answer_reports_the_question_and_what_it_was_fitted_on(
    friction_answer: DoQueryAnswer, attempt_count: int, neighbour_count: int
) -> None:
    question = FrictionCausesLift(neighbour_count=neighbour_count)

    assert friction_answer.asked == question.asked
    assert friction_answer.training_example_count == attempt_count


def test_one_region_per_friction_the_attempts_used(
    friction_answer: DoQueryAnswer, attempts: List[ClutterPickScene]
) -> None:
    used = {attempt.friction_coefficient for attempt in attempts}

    assert len(friction_answer.regions) == len(used)


def test_the_regions_hold_every_attempt_between_them(
    friction_answer: DoQueryAnswer,
) -> None:
    total = sum(region.probability for region in friction_answer.regions)

    assert total == pytest.approx(1.0)


def test_every_answer_on_a_region_is_a_probability(
    friction_answer: DoQueryAnswer, probability_tolerance: float
) -> None:
    for region in friction_answer.regions:
        assert (
            -probability_tolerance
            <= region.conditioned_probability
            <= 1.0 + probability_tolerance
        )
        assert (
            -probability_tolerance
            <= region.adjusted_probability
            <= 1.0 + probability_tolerance
        )


# %% adjusting for nothing changes nothing


def test_adjusting_for_nothing_leaves_the_conditioned_answer_alone(
    attempts: List[ClutterPickScene], neighbour_count: int
) -> None:
    """
    With no confounder to adjust for, the adjusted circuit is the conditioned one, so
    every region's two answers agree exactly.
    """
    answer = ClutterDoQuery(
        question=CrowdingCausesLift(neighbour_count=neighbour_count, adjusted_for=())
    ).run(attempts)

    assert answer.largest_shift_from_adjusting == pytest.approx(0.0)


def test_adjusting_for_the_environment_moves_the_crowding_answer(
    attempts: List[ClutterPickScene], neighbour_count: int
) -> None:
    """
    The environment decides both how slippery and how crowded an attempt is, so
    adjusting for it has to change what the crowding appears to cause.
    """
    answer = ClutterDoQuery(
        question=CrowdingCausesLift(neighbour_count=neighbour_count)
    ).run(attempts)

    assert answer.largest_shift_from_adjusting > 0.0


# %% a cause and an effect on one neighbour


def test_a_neighbour_level_question_is_answered_per_side_of_the_closing_axis(
    attempts: List[ClutterPickScene], neighbour_count: int
) -> None:
    """
    The cause is one neighbour's own position, which the circuit splits into the sides
    the recorded neighbours stood on.
    """
    answer = ClutterDoQuery(
        question=ClosingAxisSideCausesDisturbance(neighbour_count=neighbour_count)
    ).run(attempts)

    assert len(answer.regions) > 1
    assert sum(region.probability for region in answer.regions) == pytest.approx(1.0)


# %% every question in the catalogue can actually be asked


def test_every_question_of_the_catalogue_is_answered(
    attempts: List[ClutterPickScene],
    neighbour_count: int,
    probability_tolerance: float,
) -> None:
    """
    Each question has to ground, adjust and come back with regions that hold every
    attempt between them: a cause or a confounder the fitted circuit names differently
    would fail here rather than in a run.
    """
    for question in question_catalogue(neighbour_count):
        answer = ClutterDoQuery(question=question).run(attempts)

        assert answer.regions, question.name
        assert sum(region.probability for region in answer.regions) == pytest.approx(
            1.0
        ), question.name
        for region in answer.regions:
            assert (
                -probability_tolerance
                <= region.conditioned_probability
                <= 1.0 + probability_tolerance
            ), question.name
            assert (
                -probability_tolerance
                <= region.adjusted_probability
                <= 1.0 + probability_tolerance
            ), question.name
