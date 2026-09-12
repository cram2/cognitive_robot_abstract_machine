"""
Tests running one pick attempt in MuJoCo; skipped where Tracy's description is not
installed, and the screenshots where MuJoCo cannot render offscreen.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.causal_reasoning.tracy_clutter_picking.domain import FrictionLadder
from experiments.causal_reasoning.tracy_clutter_picking.episode import PickEpisode
from experiments.causal_reasoning.tracy_clutter_picking.layout_sampler import (
    ClutterLayoutSampler,
)
from experiments.causal_reasoning.tracy_clutter_picking.scene import MilkClutterWorld
from physics_simulators.mujoco_simulator import MujocoSimulator
from semantic_digital_twin.utils import tracy_installed

pytestmark = pytest.mark.skipif(
    not tracy_installed(), reason="iai_tracy_description is not installed"
)


@pytest.fixture
def layout():
    return ClutterLayoutSampler(np.random.default_rng(0)).sample()


@pytest.fixture
def surest_layout(layout):
    """
    The surest attempt there is: the highest friction level, and the target moved out to
    the edge of the clutter so no neighbour stands in the fingers' way.
    """
    layout.friction_coefficient = FrictionLadder().highest
    layout.target_index = 0
    layout.objects[0].x -= 0.15
    layout.objects[0].y -= 0.15
    return layout


def test_scene_stands_every_carton_on_the_table(layout):
    scene = MilkClutterWorld(layout)

    assert [milk.name.name for milk in scene.milks] == [
        MilkClutterWorld.milk_name(index) for index in range(len(layout.objects))
    ]
    assert scene.target is scene.milks[layout.target_index]
    assert len(scene.neighbours) == len(layout.objects) - 1
    assert scene.actuators


def test_high_friction_uncrowded_target_is_lifted(surest_layout):
    episode = PickEpisode()
    outcome = episode.run(surest_layout)

    assert outcome.lifted
    assert outcome.lift_height > episode.lift_threshold
    assert len(outcome.neighbour_displacements) == len(surest_layout.neighbours)


@pytest.mark.skipif(
    not MujocoSimulator.offscreen_rendering_available(),
    reason="MuJoCo has no OpenGL context to render offscreen with",
)
def test_screenshots_are_saved_before_and_after_the_pick(surest_layout, tmp_path):
    PickEpisode(screenshot_directory=tmp_path).run(surest_layout)

    assert (tmp_path / "before_pick.png").exists()
    assert (tmp_path / "after_pick.png").exists()
