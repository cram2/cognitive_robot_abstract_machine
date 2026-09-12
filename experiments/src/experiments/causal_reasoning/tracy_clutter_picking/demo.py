"""
Watch Tracy pick one milk carton out of a ten-carton clutter in MuJoCo, held by contact
friction alone.

Run with (the ``iai_tracy_description`` ROS package must be built and sourced)::

    python -m experiments.causal_reasoning.tracy_clutter_picking.demo [--seed N] [--headless]
        [--screenshots DIRECTORY]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from typing_extensions import Optional

from experiments.causal_reasoning.tracy_clutter_picking.episode import PickEpisode
from experiments.causal_reasoning.tracy_clutter_picking.layout_sampler import (
    ClutterLayoutSampler,
)

logger = logging.getLogger(__name__)


def main(
    seed: int = 0, headless: bool = False, screenshots: Optional[Path] = None
) -> None:
    """
    Draw one random ten-carton layout and pick its target, at wall-clock speed.

    :param seed: Seed of the layout's randomness.
    :param headless: Whether to run without MuJoCo's viewer window.
    :param screenshots: Where to save a screenshot before and after the pick.
    """
    layout = ClutterLayoutSampler(np.random.default_rng(seed)).sample()
    logger.info(
        "Layout: %s, friction %.2f, grasp yaw %.2f rad, target %d",
        layout.environment,
        layout.friction_coefficient,
        layout.grasp_yaw,
        layout.target_index,
    )
    episode = PickEpisode(
        headless=headless,
        real_time_factor=None if headless else 1.0,
        screenshot_directory=screenshots,
        keep_viewer_open=True,
    )
    outcome = episode.run(layout)
    logger.info("Outcome: %s", outcome)


if __name__ == "__main__":
    # force=True: an imported package has configured logging already, which would
    # otherwise silence this module's own INFO lines.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--screenshots", type=Path, default=None)
    arguments = parser.parse_args()
    main(arguments.seed, arguments.headless, arguments.screenshots)
