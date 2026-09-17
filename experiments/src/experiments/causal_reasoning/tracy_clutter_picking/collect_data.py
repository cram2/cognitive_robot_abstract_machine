"""
Record many random pick attempts in MuJoCo into a
:class:`~experiments.causal_reasoning.tracy_clutter_picking.dataset.ClutterPickDataset`.

Run with (the ``iai_tracy_description`` ROS package must be built and sourced)::

    python -m experiments.causal_reasoning.tracy_clutter_picking.collect_data OUTPUT.json
        [--attempts N] [--seed N] [--objects N]

The file is rewritten after every attempt, so a run stopped early still leaves what it
recorded.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from experiments.causal_reasoning.tracy_clutter_picking.dataset import ClutterPickDataset
from experiments.causal_reasoning.tracy_clutter_picking.episode import PickEpisode
from experiments.causal_reasoning.tracy_clutter_picking.exceptions import (
    EpisodePlanningFailedError,
)
from experiments.causal_reasoning.tracy_clutter_picking.layout_sampler import (
    ClutterLayoutSampler,
)

logger = logging.getLogger(__name__)


@dataclass
class DataCollection:
    """
    Runs random attempts one after another and keeps what they recorded.
    """

    output: Path
    """
    The file the dataset is written to after every attempt.
    """

    sampler: ClutterLayoutSampler
    """
    Where the layouts come from.
    """

    episode: PickEpisode = field(default_factory=PickEpisode)
    """
    Runs each layout.
    """

    dataset: ClutterPickDataset = field(default_factory=ClutterPickDataset)
    """
    Everything recorded so far.
    """

    planning_failure_count: int = 0
    """
    How many layouts were dropped because a motion could not be planned.
    """

    def run(self, attempt_count: int) -> ClutterPickDataset:
        """
        Record the given number of attempts.

        :param attempt_count: How many attempts to record; dropped layouts do not count.
        :return: The dataset.
        """
        while len(self.dataset.scenes) < attempt_count:
            layout = self.sampler.sample()
            try:
                outcome = self.episode.run(layout)
            except EpisodePlanningFailedError as error:
                self.planning_failure_count += 1
                logger.warning("Dropped a layout: %s", error)
                continue
            self.dataset.scenes.append(outcome.to_scene(layout))
            self.dataset.save(self.output)
            logger.info(
                "Recorded %d/%d attempts, %.0f%% lifted so far.",
                len(self.dataset.scenes),
                attempt_count,
                100 * self.dataset.success_rate,
            )
        return self.dataset


def main(output: Path, attempt_count: int, seed: int, object_count: int) -> None:
    """
    Record random attempts headless and as fast as the machine allows.

    :param output: The file the dataset is written to.
    :param attempt_count: How many attempts to record.
    :param seed: Seed of the layouts' randomness.
    :param object_count: How many cartons each layout holds, the target included.
    """
    collection = DataCollection(
        output=output,
        sampler=ClutterLayoutSampler(
            np.random.default_rng(seed), object_count=object_count
        ),
    )
    collection.run(attempt_count)
    logger.info(
        "Done: %d attempts recorded, %d layouts dropped.",
        len(collection.dataset.scenes),
        collection.planning_failure_count,
    )


if __name__ == "__main__":
    # force=True: an imported package has configured logging already, which would
    # otherwise silence this module's own INFO lines.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--attempts", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objects", type=int, default=10)
    arguments = parser.parse_args()
    main(arguments.output, arguments.attempts, arguments.seed, arguments.objects)
