"""
A closed-form stand-in for the MuJoCo attempt, with the same shape and the same causal
structure the ten-milk mock is built to produce, so the pipelines can be exercised
without a simulator.

Friction lets the fingers hold the target; every adjacent neighbour is in the way of the
descending fingers and takes a share of that hold away; and the environment drives both
(see
:meth:`~experiments.causal_reasoning.tracy_clutter_picking.layout_sampler.EnvironmentDistribution.of_mock_environments`),
which is what makes it a confounder of friction and success.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List

from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutterPickOutcome,
    ClutterPickScene,
    ClutterSceneLayout,
    DistanceBand,
    FrictionLadder,
    NeighbourThresholds,
)
from experiments.causal_reasoning.tracy_clutter_picking.layout_sampler import (
    ClutterLayoutSampler,
)


@dataclass
class SyntheticPickOutcomes:
    """
    Draws the outcome of an attempt on a layout from the closed-form model above.
    """

    random_state: np.random.Generator
    """
    Source of randomness for the hold's own coin flip.
    """

    ladder: FrictionLadder = field(default_factory=FrictionLadder)
    """
    The friction levels attempts draw from; the highest alone holds the target for
    certain.
    """

    thresholds: NeighbourThresholds = field(default_factory=NeighbourThresholds)
    """
    What makes a neighbour adjacent, and so in the fingers' way.
    """

    hold_lost_per_adjacent_neighbour: float = 0.25
    """
    How much of the hold each adjacent neighbour takes away.
    """

    lifted_height: float = 0.25
    """
    How far, in metres, a held target rises: the hover clearance the gripper returns to,
    less the closing swing it descended below it.
    """

    adjacent_neighbour_displacement: float = 0.03
    """
    How far, in metres, an adjacent neighbour is shoved when the fingers land on it.
    """

    def simulate(self, layout: ClutterSceneLayout) -> ClutterPickOutcome:
        """
        :param layout: The layout to attempt.
        :return: The attempt's outcome.
        """
        target = layout.target
        adjacent = [
            self.thresholds.distance_band(neighbour.distance_to(target))
            == DistanceBand.ADJACENT
            for neighbour in layout.neighbours
        ]
        hold_probability = float(
            np.clip(
                layout.friction_coefficient / self.ladder.highest
                - sum(adjacent) * self.hold_lost_per_adjacent_neighbour,
                0.0,
                1.0,
            )
        )
        lifted = bool(self.random_state.uniform() < hold_probability)
        return ClutterPickOutcome(
            lift_height=self.lifted_height if lifted else 0.0,
            lifted=lifted,
            neighbour_displacements=[
                self.adjacent_neighbour_displacement if is_adjacent else 0.0
                for is_adjacent in adjacent
            ],
        )


def synthetic_clutter_pick_scenes(
    random_state: np.random.Generator, scene_count: int, object_count: int = 10
) -> List[ClutterPickScene]:
    """
    Draw random layouts and attempt each with :class:`SyntheticPickOutcomes`.

    :param random_state: Source of randomness for layouts and outcomes alike.
    :param scene_count: How many attempts to record.
    :param object_count: How many objects each layout holds, the target included.
    :return: The recorded attempts.
    """
    sampler = ClutterLayoutSampler(random_state, object_count=object_count)
    outcomes = SyntheticPickOutcomes(random_state)
    scenes = []
    for _ in range(scene_count):
        layout = sampler.sample()
        scenes.append(outcomes.simulate(layout).to_scene(layout))
    return scenes
