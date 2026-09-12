"""
Tests for the synthetic attempts and their on-disk dataset.
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import numpy as np
import pytest
from requests import HTTPError

from experiments.causal_reasoning.tracy_clutter_picking.dataset import (
    ClutterPickDataset,
    HostedDataset,
)
from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutterEnvironment,
    ClutterPickSceneAggregations,
    DistanceBand,
    FrictionLadder,
)
from experiments.causal_reasoning.tracy_clutter_picking.layout_sampler import (
    ClutterLayoutSampler,
    EnvironmentDistribution,
)
from experiments.causal_reasoning.tracy_clutter_picking.run_pipeline import (
    ExperimentFiles,
)
from experiments.causal_reasoning.tracy_clutter_picking.synthetic import (
    SyntheticPickOutcomes,
    synthetic_clutter_pick_scenes,
)

CONTINUOUS_INTEGRATION_VARIABLE = "CI"
"""
The environment variable a continuous-integration run sets, where the network is not
relied on.
"""


@pytest.fixture
def scenes():
    return synthetic_clutter_pick_scenes(np.random.default_rng(0), scene_count=30)


# %% layout sampling


def test_sampled_layout_has_the_requested_object_count():
    layout = ClutterLayoutSampler(np.random.default_rng(0), object_count=7).sample()
    assert len(layout.objects) == 7
    assert len(layout.neighbours) == 6


def test_sampled_friction_comes_from_the_environments_own_levels():
    distributions = EnvironmentDistribution.of_mock_environments()
    bin_only = {ClutterEnvironment.BIN: distributions[ClutterEnvironment.BIN]}
    sampler = ClutterLayoutSampler(np.random.default_rng(0), distributions=bin_only)
    for _ in range(20):
        layout = sampler.sample()
        assert layout.environment == ClutterEnvironment.BIN
        assert (
            layout.friction_coefficient
            in distributions[ClutterEnvironment.BIN].friction_levels
        )


# %% synthetic outcomes


def test_highest_friction_without_adjacent_neighbours_always_lifts():
    distributions = EnvironmentDistribution.of_mock_environments()
    table_only = {ClutterEnvironment.TABLE: distributions[ClutterEnvironment.TABLE]}
    sampler = ClutterLayoutSampler(np.random.default_rng(1), distributions=table_only)
    outcomes = SyntheticPickOutcomes(np.random.default_rng(1))
    lifted_count = 0
    for _ in range(20):
        layout = sampler.sample()
        layout.friction_coefficient = FrictionLadder().highest
        outcome = outcomes.simulate(layout)
        scene = outcome.to_scene(layout)
        if ClutterPickSceneAggregations(instance=scene).crowding_count() == 0:
            assert outcome.lifted
            assert outcome.lift_height == outcomes.lifted_height
            lifted_count += 1
    assert lifted_count > 0


def test_synthetic_scenes_record_every_neighbour(scenes):
    assert len(scenes) == 30
    for scene in scenes:
        assert len(scene.neighbours) == 9


# %% dataset persistence


def test_dataset_round_trips_through_json(tmp_path, scenes):
    path = tmp_path / "attempts.json"
    ClutterPickDataset(scenes).save(path)

    loaded = ClutterPickDataset.load(path)

    assert loaded.scenes == scenes
    assert type(loaded.scenes[0].environment) is ClutterEnvironment
    assert type(loaded.scenes[0].neighbours[0].distance_band) is DistanceBand


def test_dataset_split_keeps_every_scene_once(scenes):
    dataset = ClutterPickDataset(scenes)

    first, second = dataset.split(0.8, np.random.default_rng(0))

    assert len(first.scenes) == 24
    assert len(second.scenes) == 6
    assert sorted(map(id, first.scenes + second.scenes)) == sorted(map(id, scenes))


def test_success_rate_is_the_share_of_lifted_scenes(scenes):
    dataset = ClutterPickDataset(scenes)
    assert dataset.success_rate == pytest.approx(
        sum(scene.lifted for scene in scenes) / len(scenes)
    )


def test_success_rate_by_environment_counts_every_scene_once(scenes):
    dataset = ClutterPickDataset(scenes)

    rates = dataset.success_rate_by(lambda scene: scene.environment)

    assert sum(rate.attempt_count for rate in rates.values()) == len(scenes)
    assert sum(rate.lifted_count for rate in rates.values()) == sum(
        scene.lifted for scene in scenes
    )
    for environment, rate in rates.items():
        assert rate.rate == pytest.approx(
            sum(scene.lifted for scene in scenes if scene.environment == environment)
            / rate.attempt_count
        )


# %% hosted dataset


@pytest.fixture
def hosted_file(scenes, tmp_path):
    """
    A dataset served over HTTP from a directory standing in for the host.
    """
    served_directory = tmp_path / "served"
    ClutterPickDataset(scenes).save(served_directory / "attempts.json")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=served_directory)
    )
    Thread(target=server.serve_forever, daemon=True).start()
    yield HostedDataset(
        url=f"http://127.0.0.1:{server.server_port}/attempts.json",
        cache_directory=tmp_path / "cache",
    )
    server.shutdown()


def test_hosted_dataset_is_fetched_into_the_cache(hosted_file, scenes):
    path = hosted_file.fetch()

    assert path == hosted_file.cache_directory / "attempts.json"
    assert hosted_file.load().scenes == scenes


def test_hosted_dataset_is_fetched_once(hosted_file):
    first = hosted_file.fetch()
    first.write_text("cached")

    assert hosted_file.fetch().read_text() == "cached"


def test_missing_hosted_dataset_raises(hosted_file):
    missing = HostedDataset(
        url=hosted_file.url.replace("attempts.json", "missing.json"),
        cache_directory=hosted_file.cache_directory,
    )

    with pytest.raises(HTTPError):
        missing.fetch()
    assert not missing.path.exists()


@pytest.mark.skipif(
    os.environ.get(CONTINUOUS_INTEGRATION_VARIABLE, "false").lower() == "true",
    reason="fetches the hosted attempts over the network",
)
def test_hosted_attempts_are_fetched_from_their_repository(tmp_path):
    hosted = replace(ExperimentFiles().recorded_attempts, cache_directory=tmp_path)

    dataset = hosted.load()

    assert len(dataset.scenes) == 300
    assert {len(scene.neighbours) for scene in dataset.scenes} == {9}
