"""
Tests for :mod:`experiments.causal_reasoning.tracy_clutter_picking.domain`.
"""

from __future__ import annotations

import math

import pytest

from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClosingAxisSide,
    ClutterEnvironment,
    ClutterPickOutcome,
    ClutterPickSceneAggregations,
    ClutterSceneLayout,
    DistanceBand,
    NeighbourThresholds,
    ObjectCategory,
    PlacedObject,
)


def _milk(x: float, y: float) -> PlacedObject:
    return PlacedObject(category=ObjectCategory.MILK, x=x, y=y, yaw=0.0)


@pytest.fixture
def thresholds() -> NeighbourThresholds:
    return NeighbourThresholds()


@pytest.fixture
def layout(thresholds) -> ClutterSceneLayout:
    """
    A target at the origin with one adjacent, one near and one far neighbour.
    """
    return ClutterSceneLayout(
        environment=ClutterEnvironment.TABLE,
        objects=[
            _milk(0.0, 0.0),
            _milk(thresholds.adjacent_distance / 2, 0.0),
            _milk(0.0, (thresholds.adjacent_distance + thresholds.near_distance) / 2),
            _milk(thresholds.near_distance * 2, 0.0),
        ],
        target_index=0,
        friction_coefficient=0.6,
        grasp_yaw=0.0,
    )


# %% distance bands


def test_distance_band_boundaries(thresholds):
    assert (
        thresholds.distance_band(thresholds.adjacent_distance / 2)
        == DistanceBand.ADJACENT
    )
    assert thresholds.distance_band(thresholds.adjacent_distance) == DistanceBand.NEAR
    assert thresholds.distance_band(thresholds.near_distance) == DistanceBand.FAR


# %% layouts


def test_layout_neighbours_exclude_the_target(layout):
    assert layout.target is layout.objects[0]
    assert layout.neighbours == layout.objects[1:]


def test_placed_object_distance_is_planar(layout, thresholds):
    target, adjacent = layout.objects[0], layout.objects[1]
    assert target.distance_to(adjacent) == pytest.approx(
        thresholds.adjacent_distance / 2
    )


# %% recording an outcome as a scene


def test_outcome_records_neighbours_relative_to_the_target(thresholds):
    target = _milk(0.8, 0.2)
    neighbour = _milk(0.85, 0.3)
    shifted_layout = ClutterSceneLayout(
        environment=ClutterEnvironment.BIN,
        objects=[neighbour, target],
        target_index=1,
        friction_coefficient=0.4,
        grasp_yaw=math.pi / 2,
    )
    outcome = ClutterPickOutcome(
        lift_height=0.2, lifted=True, neighbour_displacements=[0.0]
    )

    scene = outcome.to_scene(shifted_layout)

    [recorded] = scene.neighbours
    assert scene.environment == ClutterEnvironment.BIN
    assert scene.target_x == pytest.approx(0.8)
    assert scene.target_y == pytest.approx(0.2)
    assert scene.friction_coefficient == 0.4
    assert scene.grasp_yaw == math.pi / 2
    assert scene.lifted is True
    assert scene.lift_height == 0.2
    assert recorded.x == pytest.approx(0.05)
    assert recorded.y == pytest.approx(0.1)
    assert recorded.distance_to_target == pytest.approx(math.hypot(0.05, 0.1))
    assert recorded.distance_band == thresholds.distance_band(math.hypot(0.05, 0.1))


def test_aggregations_count_adjacent_neighbours(layout):
    outcome = ClutterPickOutcome(
        lift_height=0.0, lifted=False, neighbour_displacements=[0.0, 0.0, 0.0]
    )
    scene = outcome.to_scene(layout)

    assert ClutterPickSceneAggregations(instance=scene).crowding_count() == 1


def test_outcome_marks_a_neighbour_disturbed_past_the_threshold(layout, thresholds):
    outcome = ClutterPickOutcome(
        lift_height=0.0,
        lifted=False,
        neighbour_displacements=[thresholds.disturbance_threshold * 2, 0.0, 0.0],
    )
    scene = outcome.to_scene(layout)

    assert [neighbour.disturbed for neighbour in scene.neighbours] == [
        True,
        False,
        False,
    ]


def test_outcome_records_which_side_of_the_closing_axis_a_neighbour_stands(layout):
    """
    With the fingers closing along the table's x-axis, the neighbour standing along x is
    in their way and the one standing along y is not; turning the grasp a quarter turn
    swaps the two.
    """
    outcome = ClutterPickOutcome(
        lift_height=0.0, lifted=False, neighbour_displacements=[0.0, 0.0, 0.0]
    )
    along_x = outcome.to_scene(layout)
    layout.grasp_yaw = math.pi / 2
    along_y = outcome.to_scene(layout)

    assert along_x.neighbours[0].closing_axis_side == ClosingAxisSide.ALONG
    assert along_x.neighbours[1].closing_axis_side == ClosingAxisSide.ACROSS
    assert along_y.neighbours[0].closing_axis_side == ClosingAxisSide.ACROSS
    assert along_y.neighbours[1].closing_axis_side == ClosingAxisSide.ALONG
