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
from typing_extensions import Dict, List, Optional, Sequence, Tuple

from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutterPickOutcome,
    ClutterPickScene,
    ClutterSceneLayout,
    DistanceBand,
    FrictionLadder,
    NeighbourThresholds,
)
from experiments.causal_reasoning.comparison.evaluation import (
    InterventionalEffect,
    KnownTruth,
)
from experiments.causal_reasoning.comparison.queries import CausalQueryCase
from experiments.causal_reasoning.tracy_clutter_picking.domain import ClutterEnvironment
from experiments.causal_reasoning.tracy_clutter_picking.exceptions import (
    UnreadableQuestionError,
)
from experiments.causal_reasoning.tracy_clutter_picking.layout_sampler import (
    ClutterLayoutSampler,
    EnvironmentDistribution,
)
from experiments.causal_reasoning.tracy_clutter_picking.queries import (
    CrowdingCausesLift,
    FrictionCausesLift,
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
    random_state: np.random.Generator,
    scene_count: int,
    object_count: int = 10,
    outcomes: Optional[SyntheticPickOutcomes] = None,
    distributions: Optional[Dict[ClutterEnvironment, EnvironmentDistribution]] = None,
) -> List[ClutterPickScene]:
    """
    Draw random layouts and attempt each with :class:`SyntheticPickOutcomes`.

    :param random_state: Source of randomness for layouts and outcomes alike.
    :param scene_count: How many attempts to record.
    :param object_count: How many objects each layout holds, the target included.
    :param outcomes: The mechanism the attempts are drawn from; the default one if not
        given.
    :param distributions: How each environment lays its clutter out and what friction it
        gives; the mock's own if not given.
    :return: The recorded attempts.
    """
    sampler = ClutterLayoutSampler(
        random_state,
        object_count=object_count,
        **({} if distributions is None else {"distributions": distributions}),
    )
    outcomes = SyntheticPickOutcomes(random_state) if outcomes is None else outcomes
    scenes = []
    for _ in range(scene_count):
        layout = sampler.sample()
        scenes.append(outcomes.simulate(layout).to_scene(layout))
    return scenes


# %% the mechanism as known truth


@dataclass(frozen=True)
class MechanismSetting:
    """
    One setting of the closed-form mechanism.
    """

    hold_lost_per_adjacent_neighbour: float
    """
    How much of the hold each adjacent neighbour takes away.
    """

    confounded: bool = True
    """
    Whether the environment drives the friction as well as the crowding, which is what
    opens the backdoor path between friction and a lift.
    """

    neighbour_count: int = 9
    """
    How many neighbours every attempt holds.
    """

    def outcomes(self, random_state: np.random.Generator) -> SyntheticPickOutcomes:
        """
        :param random_state: Source of randomness for the hold's coin flip.
        :return: The mechanism under this setting.
        """
        return SyntheticPickOutcomes(
            random_state,
            hold_lost_per_adjacent_neighbour=self.hold_lost_per_adjacent_neighbour,
        )

    def distributions(self) -> Dict[ClutterEnvironment, EnvironmentDistribution]:
        """
        :return: How each environment lays its clutter out and what friction it gives.
            Without the confounding both environments draw from the whole ladder, so
            the environment still decides how crowded an attempt is and no longer
            decides how slippery it is.
        """
        ladder = FrictionLadder()
        mock = EnvironmentDistribution.of_mock_environments(ladder)
        if self.confounded:
            return mock
        return {
            environment: EnvironmentDistribution(
                distribution.minimum_spacing,
                distribution.maximum_spacing,
                tuple(ladder.levels),
            )
            for environment, distribution in mock.items()
        }


@dataclass
class ClutterTruth(KnownTruth):
    """
    The interventional probabilities the closed-form mechanism implies.

    The mechanism gives an attempt's hold probability in closed form, so forcing the
    friction or the crowding leaves an expectation over layouts alone and the truth is
    the mean of that expression over sampled layouts, exact in the outcome and
    approximate only in the layouts drawn.
    """

    settings: Sequence[MechanismSetting] = (
        MechanismSetting(0.10),
        MechanismSetting(0.25),
        MechanismSetting(0.40),
        MechanismSetting(0.25, confounded=False),
    )
    """
    The settings to run.
    """

    random_seed: int = 0
    """
    Seed of the layouts the expectation is taken over.
    """

    layout_count: int = 20_000
    """
    How many layouts each probability is averaged over.
    """

    known: Dict[Tuple[MechanismSetting, str, float], float] = field(
        default_factory=dict
    )
    """
    The probabilities computed so far, by setting, cause and forced value.
    """

    @property
    def configurations(self) -> Sequence[MechanismSetting]:
        return self.settings

    def describe(self, configuration: MechanismSetting) -> Dict[str, str]:
        return {
            "hold lost per neighbour": f"{configuration.hold_lost_per_adjacent_neighbour:.2f}",
            "confounded": "yes" if configuration.confounded else "no",
        }

    def examples(
        self,
        configuration: MechanismSetting,
        count: int,
        random_state: np.random.Generator,
    ) -> List[ClutterPickScene]:
        return synthetic_clutter_pick_scenes(
            random_state,
            count,
            object_count=configuration.neighbour_count + 1,
            outcomes=configuration.outcomes(random_state),
            distributions=configuration.distributions(),
        )

    def probability(
        self,
        configuration: MechanismSetting,
        case: CausalQueryCase,
        effect: InterventionalEffect,
    ) -> float:
        cause = self._cause_of(case)
        forced = float(effect.cause_region)
        key = (configuration, cause, forced)
        if key not in self.known:
            self.known[key] = self._interventional_probability(
                configuration, cause, forced
            )
        return self.known[key]

    @staticmethod
    def _cause_of(case: CausalQueryCase) -> str:
        """
        :param case: A question of the catalogue.
        :return: Which variable it forces, in the mechanism's terms.
        :raises UnreadableQuestionError: If the mechanism has no reading of it.
        """
        if isinstance(case, FrictionCausesLift):
            return "friction"
        if isinstance(case, CrowdingCausesLift):
            return "crowding"
        raise UnreadableQuestionError(type(case).__name__)

    def _interventional_probability(
        self, configuration: MechanismSetting, cause: str, forced: float
    ) -> float:
        """
        :param configuration: One setting.
        :param cause: Which variable is forced.
        :param forced: The value it is forced to.
        :return: The probability of a lift under that intervention.
        """
        random_state = np.random.default_rng(self.random_seed)
        outcomes = configuration.outcomes(random_state)
        sampler = ClutterLayoutSampler(
            random_state,
            object_count=configuration.neighbour_count + 1,
            distributions=configuration.distributions(),
        )
        frictions = np.empty(self.layout_count)
        adjacent = np.empty(self.layout_count)
        for index in range(self.layout_count):
            layout = sampler.sample()
            frictions[index] = layout.friction_coefficient
            adjacent[index] = sum(
                outcomes.thresholds.distance_band(neighbour.distance_to(layout.target))
                is DistanceBand.ADJACENT
                for neighbour in layout.neighbours
            )
        if cause == "friction":
            frictions = np.full(self.layout_count, forced)
        else:
            adjacent = np.full(self.layout_count, forced)
        held = np.clip(
            frictions / outcomes.ladder.highest
            - adjacent * outcomes.hold_lost_per_adjacent_neighbour,
            0.0,
            1.0,
        )
        return float(held.mean())
